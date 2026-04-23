// /api/qid — Wikidata QID resolver (Phase 4a stub).
//
// CURRENT STATUS: stub. Returns 501 Not Implemented for every request.
// The contract (request/response shape + KV cache binding + MediaWiki
// search + SPARQL disambiguation fallback) is specified inline so the
// next session can drop in the real implementation without re-deriving
// the design.
//
// Contract when activated:
//
//   POST /api/qid
//   { "triples": [["alice","likes","cats"], ["bob","owns","dog"]] }
//
//   → { "resolved": [
//       { "subject": {"label":"alice","qid":"Q...","confidence":0.95},
//         "predicate": {"label":"likes","pid":"P...","confidence":0.90},
//         "object": {"label":"cats","qid":"Q146","confidence":0.99} },
//       ...
//     ] }
//
// Lookup strategy (per 2026 Wikidata best-practice research):
//   1. MediaWiki wbsearchentities API for text-to-QID fuzzy match
//      (GET https://www.wikidata.org/w/api.php?action=wbsearchentities
//       &search=<label>&language=en&type=item&format=json).
//      Returns candidate QIDs with descriptions; take top-k.
//   2. SPARQL disambiguation when multiple candidates — filter by
//      instance-of hints derived from the triple's predicate domain
//      (e.g. a predicate like "born_in" filters objects to Q17334923
//      "geographic location" descendants).
//   3. For predicates, search property-space (type=property) and map
//      common surface forms to canonical PIDs via a small in-code
//      dictionary plus the same search fallback.
//
// Cache: KV_CACHE binding, key = sha256(`${label}|${kind}|${lang}`),
// TTL 30 days. Labels are deterministic; QID mappings change on the
// scale of months at most. Expected hit rate > 95 %.

import type { Env } from "../index";

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

export async function handleQid(
  _request: Request,
  env: Env,
  _ctx: ExecutionContext,
): Promise<Response> {
  const kvBound = Boolean(env.QID_CACHE);
  return json(
    {
      error: "not implemented",
      status: "phase-4a-stub",
      next: {
        cache_binding_present: kvBound,
        action: kvBound
          ? "Implement the wbsearchentities + SPARQL pipeline in this file."
          : "Uncomment [[kv_namespaces]] in wrangler.toml and run " +
            "`wrangler kv:namespace create qid-cache` before activating.",
        contract_spec: "See header comment in worker/src/routes/qid.ts.",
      },
    },
    501,
  );
}
