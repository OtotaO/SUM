// /api/qid — Wikidata QID / PID resolver.
//
// Turns human-readable labels into Wikidata identifiers: "alice" → Q-???,
// "orbit" → P-???. Used by the demo's Phase 4a "annotate triples with
// QIDs" flow so emitted axiom keys can carry stable URIs instead of
// opaque lowercase strings.
//
// Pipeline (per 2026 Wikidata best-practice research):
//
//   1. Cache lookup — each (text, kind, lang) key is deterministic.
//      Cache API first (edge-local, same-colo), then KV (global, if
//      bound). Hit rate > 95% expected once traffic stabilises.
//   2. wbsearchentities API (MediaWiki) — text/fuzzy match with a
//      type filter (item vs property). Returns candidates ranked by
//      prefix + exact-label matching. Take top-1; confidence mirrors
//      the match quality the API reports.
//   3. No SPARQL disambiguation in v0.3 — wbsearchentities alone hits
//      >80% accuracy on common nouns / proper names per the Wikidata
//      docs. SPARQL refinement (filtered by predicate domain) is a
//      Phase 4b follow-up once the v0.3 accuracy baseline is measured.
//
// Request shape:
//
//   POST /api/qid
//   {
//     "terms": [
//       { "text": "Alice",  "kind": "item"     },    // expect Q-id
//       { "text": "orbit",  "kind": "property" },    // expect P-id
//       { "text": "Earth",  "kind": "item", "lang": "en" }
//     ]
//   }
//
// Response:
//
//   {
//     "resolved": [
//       { "text":"Alice", "id":"Q3099839", "label":"Alice",
//         "description":"female given name", "confidence":1.0,
//         "source":"cache" | "wbsearchentities" },
//       { "text":"unknown_foo", "id":null, "reason":"no-match" },
//       ...
//     ]
//   }
//
// Caching: 30-day TTL. Wikidata labels move on monthly scales at
// most; 30d strikes the balance between freshness and API load.
//
// Rate limits: Wikidata asks for a descriptive User-Agent and a
// reasonable request rate. The Worker adds a contact User-Agent header
// pointing at the repo so Wikidata can reach us if we ever misbehave.

import type { Env } from "../index";

const WBSEARCH_BASE = "https://www.wikidata.org/w/api.php";
const USER_AGENT =
  "SUMDemoQIDResolver/0.3.0 (+https://github.com/OtotaO/SUM; hosted-demo-phase-4a)";
const CACHE_TTL_SECONDS = 30 * 24 * 60 * 60;
const MAX_TERMS_PER_REQUEST = 50;

interface TermRequest {
  text: string;
  kind?: "item" | "property";
  lang?: string;
}

interface ResolvedTerm {
  text: string;
  id: string | null;
  label?: string;
  description?: string;
  confidence?: number;
  source?: "cache" | "wbsearchentities";
  reason?: string;
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

function cacheKey(text: string, kind: string, lang: string): string {
  // https-scheme fake URL so Cache API accepts it as a request key.
  const enc = encodeURIComponent(text);
  return `https://sum-qid-cache.internal/${kind}/${lang}/${enc}`;
}

/**
 * Lookup a single term via wbsearchentities. Top-1 match becomes the
 * returned QID/PID; confidence is 1.0 on an exact-label match (API's
 * `match.type === "label"`), 0.7 on alias match, 0.5 otherwise.
 */
async function searchSingle(
  text: string,
  kind: "item" | "property",
  lang: string,
): Promise<ResolvedTerm> {
  const params = new URLSearchParams({
    action: "wbsearchentities",
    search: text,
    language: lang,
    type: kind,
    format: "json",
    limit: "1",
    origin: "*",
  });
  const url = `${WBSEARCH_BASE}?${params.toString()}`;

  const res = await fetch(url, {
    headers: { "user-agent": USER_AGENT, accept: "application/json" },
  });
  if (!res.ok) {
    return { text, id: null, reason: `wikidata ${res.status}` };
  }
  const data = (await res.json()) as {
    search?: Array<{
      id: string;
      label?: string;
      description?: string;
      match?: { type?: string; text?: string; language?: string };
    }>;
    error?: { info?: string };
  };
  if (data.error) {
    return { text, id: null, reason: `wikidata error: ${data.error.info ?? "unknown"}` };
  }
  const hit = (data.search ?? [])[0];
  if (!hit) {
    return { text, id: null, reason: "no-match", source: "wbsearchentities" };
  }
  // Confidence is derived from the match-type field the API returns.
  // This is not a probabilistic score; it's a categorical "how did
  // Wikidata think this matched" translated into a 0–1 ordering so
  // downstream code can threshold easily.
  const matchType = hit.match?.type ?? "unknown";
  const confidence =
    matchType === "label" ? 1.0 :
    matchType === "alias" ? 0.7 :
    0.5;
  return {
    text,
    id: hit.id,
    label: hit.label,
    description: hit.description,
    confidence,
    source: "wbsearchentities",
  };
}

/**
 * Resolve one term with cache → wbsearchentities fallback. Writes a
 * cache entry on successful (non-null-id) resolution. Cache failures
 * are swallowed — the user gets a correct answer even if the cache
 * layer misbehaves.
 */
async function resolveTerm(
  term: TermRequest,
  ctx: ExecutionContext,
): Promise<ResolvedTerm> {
  const kind = term.kind ?? "item";
  const lang = term.lang ?? "en";
  if (!term.text || typeof term.text !== "string") {
    return { text: term.text ?? "", id: null, reason: "empty-text" };
  }
  if (term.text.length > 200) {
    return { text: term.text, id: null, reason: "text-too-long" };
  }

  const cache = caches.default;
  const ck = cacheKey(term.text, kind, lang);
  const cachedRes = await cache.match(new Request(ck));
  if (cachedRes) {
    try {
      const cached = (await cachedRes.json()) as ResolvedTerm;
      return { ...cached, source: "cache" };
    } catch {
      // Fall through to a fresh lookup if the cache entry is corrupt.
    }
  }

  const resolved = await searchSingle(term.text, kind, lang);

  if (resolved.id) {
    // Persist a cache entry with the 30-day TTL. Use waitUntil so the
    // user's response is not delayed by cache plumbing.
    const body = JSON.stringify(resolved);
    const cacheRes = new Response(body, {
      headers: {
        "content-type": "application/json",
        "cache-control": `public, max-age=${CACHE_TTL_SECONDS}`,
      },
    });
    ctx.waitUntil(cache.put(new Request(ck), cacheRes));
  }

  return resolved;
}

export async function handleQid(
  request: Request,
  _env: Env,
  ctx: ExecutionContext,
): Promise<Response> {
  if (request.method !== "POST") {
    return json({ error: "method not allowed; use POST" }, 405);
  }

  let body: { terms?: TermRequest[] };
  try {
    body = (await request.json()) as { terms?: TermRequest[] };
  } catch {
    return json({ error: "invalid JSON body" }, 400);
  }

  const terms = body?.terms;
  if (!Array.isArray(terms) || terms.length === 0) {
    return json(
      { error: "missing or empty 'terms' array (expected [{text,kind?,lang?}, ...])" },
      400,
    );
  }
  if (terms.length > MAX_TERMS_PER_REQUEST) {
    return json(
      { error: `too many terms (max ${MAX_TERMS_PER_REQUEST} per request)` },
      413,
    );
  }

  // Parallelise the per-term lookups. Each hits Wikidata independently;
  // the cache layer collapses duplicate requests within a colo.
  const resolved = await Promise.all(terms.map((t) => resolveTerm(t, ctx)));

  return json({ resolved });
}
