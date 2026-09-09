// Best-effort KV quotas for cheap routes, plus shared quota policy and errors.
// Paid provider work uses SQLite transactions in llm_budget.ts; this module's
// KV read/modify/write helper must never be used as paid-work admission.
// Trust endpoints remain unlimited so receipt verification stays independent.

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
export const POLICY: Record<RateLimitScope, RateLimitConfig> = {
  "llm-axis-byok": { limit: 100, window_seconds: 3600 },      // 100/hr per IP with BYO key
  "llm-axis-demo": { limit: 5, window_seconds: 86400 },        // 5/day per IP on operator key
  canonical: { limit: 100, window_seconds: 3600 },             // 100/hr per IP, no LLM
  qid: { limit: 60, window_seconds: 3600 },                     // 60/hr per IP, Wikidata upstream
};

/**
 * Classify BEFORE dispatch, using only the BYO credential selected for
 * that dispatch. A key for a different provider cannot fund this call.
 * Omit selectedUserKey when the route cannot establish BYO funding.
 */
export function classifyScope(
  endpoint: "render" | "transform" | "complete" | "qid",
  selectedUserKey?: string,
): RateLimitScope {
  if (endpoint === "qid") return "qid";

  // /api/complete never honours a BYO key — it always calls the operator's
  // provider key (complete.ts uses env.ANTHROPIC_API_KEY / env.OPENAI_API_KEY
  // unconditionally). So a BYO header must NOT promote the caller out of the
  // operator-funded demo bucket: otherwise a stray header (even garbage) drains
  // operator credit at the 100/hr byok rate instead of 5/day (2026-07-31 #10).
  if (endpoint === "complete") return "llm-axis-demo";

  // /api/transform is deterministic today. If provider dispatch is added,
  // it must separately establish funding and use atomic LLM admission.
  if (endpoint === "transform") return "canonical";

  // Whitespace-only credentials must match dispatch's operator fallback.
  return selectedUserKey?.trim() ? "llm-axis-byok" : "llm-axis-demo";
}

/**
 * Increment-and-check against the rate-limit bucket.
 *
 * KNOWN LIMITATION — this is NOT atomic, and the bound it enforces is
 * weaker than the policy table implies. The read (`kv.get`) and the write
 * (`kv.put`) are separate awaits with nothing serializing them, and Workers
 * serves requests concurrently. N requests that arrive together all read the
 * same `count`, all evaluate `count < limit` against it, and all proceed;
 * the final `put` leaves the counter at `count + 1`. So a concurrent burst
 * is bounded by the attacker's concurrency, not by `policy.limit`.
 *
 * An earlier version of this comment claimed the race "briefly under-counts
 * by ~1". That is wrong, and it mattered: on `llm-axis-demo` (5/day) the
 * over-spend is on the OPERATOR's provider key. KV's eventual consistency
 * across colos widens the window further.
 *
 * Paid provider calls now use LLMBudget instead. This helper is retained
 * only for deterministic and QID routes, where it is best-effort CPU control.
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
    // this is best-effort protection for cheap routes only.
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
        "For /api/render, select a provider and supply its matching " +
        "X-Render-LLM-Key-Anthropic or X-Render-LLM-Key-OpenAI header " +
        "with your own key for 100/hr quota. Otherwise retry after the " +
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
