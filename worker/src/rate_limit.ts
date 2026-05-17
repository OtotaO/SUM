// Per-IP rate limiter with BYO-key escape valve.
//
// Why this exists: the Worker is public-facing. Without rate limiting,
// any caller can drain the operator's LLM credits via /api/render or
// /api/transform off-centre slider calls (each one bills Anthropic).
// Without BYO-key gating, the operator is one bad actor away from a
// surprise bill on credits they may not have.
//
// Policy:
//
//   LLM-axis routes (/api/render off-centre, /api/transform off-centre,
//                    /api/complete):
//     - BYO key (X-Render-LLM-Key-Anthropic|OpenAI present):
//         100 calls per IP per hour. Defends the Worker's CPU + KV
//         budget; the caller is paying their own LLM bill.
//     - Operator-keyed demo:
//         5 calls per IP per 24 hours. Funders + first-time visitors
//         get a frictionless try; the operator's LLM credits are
//         shielded from abuse.
//
//   Canonical / cheap routes (/api/qid, canonical-path renders):
//     - 60-100 calls per IP per hour. CPU + KV protection only;
//       no LLM cost.
//
//   /.well-known/* (jwks, revoked-kids):
//     - No limit. Browser verifiers fetch these on every receipt
//       verify; rate-limiting them breaks the trust loop.
//
// Storage: RENDER_CACHE KV namespace, keyed by
// `rl:<scope>:<ip>:<window-index>`. TTL is the window length plus a
// small buffer. Free-tier-compatible: no paid Cloudflare Rate
// Limiting API, no Durable Objects required.

import type { KVNamespace } from "@cloudflare/workers-types";

export type RateLimitScope =
  | "llm-axis-byok"
  | "llm-axis-demo"
  | "canonical"
  | "qid";

export interface RateLimitResult {
  allowed: boolean;
  scope: RateLimitScope;
  limit: number;
  remaining: number;
  reset_seconds: number;
}

interface RateLimitConfig {
  limit: number;
  window_seconds: number;
}

// Policy table — single source of truth for limits. Tunable per
// scope without touching the dispatch code.
const POLICY: Record<RateLimitScope, RateLimitConfig> = {
  "llm-axis-byok": { limit: 100, window_seconds: 3600 },      // 100/hr per IP with BYO key
  "llm-axis-demo": { limit: 5, window_seconds: 86400 },        // 5/day per IP on operator key
  canonical: { limit: 100, window_seconds: 3600 },             // 100/hr per IP, no LLM
  qid: { limit: 60, window_seconds: 3600 },                     // 60/hr per IP, Wikidata upstream
};

/**
 * Classify the request's rate-limiting scope. The route handler calls
 * this BEFORE the LLM dispatch. Returns the policy scope + whether
 * the caller has supplied a BYO key for the LLM provider this route
 * may use.
 */
export function classifyScope(
  endpoint: "render" | "transform" | "complete" | "qid",
  request: Request,
): RateLimitScope {
  if (endpoint === "qid") return "qid";

  // BYO-key detection: any of the recognised LLM-key headers present
  // and non-empty puts the caller in the byok bucket. The route still
  // validates the key against the LLM call; we just classify here.
  const hasByoKey =
    !!request.headers.get("x-render-llm-key-anthropic") ||
    !!request.headers.get("x-render-llm-key-openai");

  return hasByoKey ? "llm-axis-byok" : "llm-axis-demo";
}

/**
 * Atomic-ish increment-and-check against the rate-limit bucket. KV
 * doesn't have true atomic increments, but for our throughput level
 * the read-then-write race is benign (worst case: a burst of
 * concurrent requests at the boundary briefly under-counts by ~1).
 * A counter that under-counts by 1 once an hour is not the threat
 * model.
 */
export async function checkRateLimit(
  request: Request,
  kv: KVNamespace,
  scope: RateLimitScope,
): Promise<RateLimitResult> {
  const policy = POLICY[scope];
  const ip = request.headers.get("cf-connecting-ip") || "unknown";
  const nowSec = Math.floor(Date.now() / 1000);
  const windowIndex = Math.floor(nowSec / policy.window_seconds);
  const key = `rl:${scope}:${ip}:${windowIndex}`;

  const current = await kv.get(key);
  const count = current ? parseInt(current, 10) : 0;

  const allowed = count < policy.limit;
  const reset_seconds = (windowIndex + 1) * policy.window_seconds - nowSec;

  if (allowed) {
    // Increment with TTL slightly past the window so abandoned buckets
    // self-clean. KV writes are eventually consistent within the region;
    // for our threat model this is acceptable.
    await kv.put(key, String(count + 1), {
      expirationTtl: policy.window_seconds + 60,
    });
  }

  return {
    allowed,
    scope,
    limit: policy.limit,
    remaining: Math.max(0, policy.limit - count - (allowed ? 1 : 0)),
    reset_seconds,
  };
}

/**
 * Build the standard 429 response with informative headers + a body
 * that points the caller at the BYO-key escape valve.
 */
export function rateLimitedResponse(result: RateLimitResult): Response {
  let remediation: string;
  switch (result.scope) {
    case "llm-axis-demo":
      remediation =
        "Operator-keyed demo allowance exhausted (5 / 24h per IP). " +
        "Supply X-Render-LLM-Key-Anthropic or X-Render-LLM-Key-OpenAI " +
        "header with your own key for 100/hr quota, retry after the " +
        "window resets, or run locally via `pip install sum-engine[openai]` " +
        "and `sum render` / `sum transform apply slider`.";
      break;
    case "llm-axis-byok":
      remediation =
        "BYO-key quota exhausted (100/hr per IP). The cap exists to " +
        "protect the Worker's CPU + KV budget; retry after the window " +
        "resets, or run locally with the same BYO key.";
      break;
    case "canonical":
      remediation =
        "Canonical-path quota exhausted (100/hr per IP). The canonical " +
        "path is purely deterministic; you can run it locally with " +
        "`pip install sum-engine[sieve]` for unlimited use.";
      break;
    case "qid":
      remediation =
        "Wikidata-resolver quota exhausted (60/hr per IP). Wikidata's " +
        "upstream API is rate-limited regardless; retry after the " +
        "window resets.";
      break;
  }

  return new Response(
    JSON.stringify({
      error: "rate limit exceeded",
      scope: result.scope,
      limit: result.limit,
      remaining: 0,
      reset_seconds: result.reset_seconds,
      remediation,
    }),
    {
      status: 429,
      headers: {
        "content-type": "application/json; charset=utf-8",
        "x-ratelimit-limit": String(result.limit),
        "x-ratelimit-remaining": "0",
        "x-ratelimit-reset": String(result.reset_seconds),
        "retry-after": String(result.reset_seconds),
      },
    },
  );
}
