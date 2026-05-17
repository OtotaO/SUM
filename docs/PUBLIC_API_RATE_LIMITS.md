# PUBLIC_API_RATE_LIMITS.md

**Operator-facing.** Documents the per-IP rate-limit policy on the hosted Worker at `https://sum-demo.ototao.workers.dev`. Public callers (funders, journalists, agent builders, contributors poking at the demo) should also read this so they understand what they get for free, what BYO-keys unlocks, and how to retry under quota.

## Why this exists

The Worker is public-facing. Without rate limiting, any caller could drain the operator's LLM credits (Anthropic / OpenAI) via `/api/render`, `/api/transform` off-centre, or `/api/complete`. Without BYO-key gating, the operator is one bad actor away from a surprise bill.

This policy:
- Protects the operator's wallet from abuse on operator-keyed routes.
- Preserves a frictionless first-try experience for funders and casual visitors (the README's "verify in 60 seconds" example still works on a fresh IP).
- Gives heavy users a path to higher quota: BYO key.
- Keeps the trust-loop infrastructure (`/.well-known/*`) unlimited so verifiers don't get rate-limited.

## Policy table

| Route | Mode | Per-IP limit | Window |
|---|---|---:|---|
| `/api/render` | operator-keyed (no `X-Render-LLM-Key-*`) | **5** | 24 hours |
| `/api/render` | BYO-key (`X-Render-LLM-Key-Anthropic` OR `-OpenAI`) | **100** | 1 hour |
| `/api/transform` | operator-keyed | **5** | 24 hours |
| `/api/transform` | BYO-key | **100** | 1 hour |
| `/api/complete` | operator-keyed | **5** | 24 hours |
| `/api/qid` | n/a (no LLM cost; Wikidata upstream) | **60** | 1 hour |
| `/.well-known/jwks.json` | n/a (cheap, browser verifiers need it) | **no limit** | — |
| `/.well-known/revoked-kids.json` | n/a (cheap, browser verifiers need it) | **no limit** | — |

## What the 429 response looks like

When a limit is exceeded, the route returns HTTP `429 Too Many Requests` with:

**Headers:**
```
content-type: application/json; charset=utf-8
x-ratelimit-limit: <limit>
x-ratelimit-remaining: 0
x-ratelimit-reset: <seconds-until-window-resets>
retry-after: <same>
```

**Body:**
```json
{
  "error": "rate limit exceeded",
  "scope": "llm-axis-demo",
  "limit": 5,
  "remaining": 0,
  "reset_seconds": 73421,
  "remediation": "Operator-keyed demo allowance exhausted (5 / 24h per IP). Supply X-Render-LLM-Key-Anthropic or X-Render-LLM-Key-OpenAI header with your own key for 100/hr quota, retry after the window resets, or run locally via `pip install sum-engine[openai]` and `sum render` / `sum transform apply slider`."
}
```

The `scope` field distinguishes:
- `llm-axis-demo` — operator-keyed LLM call (the 5/day bucket)
- `llm-axis-byok` — caller-keyed LLM call (the 100/hr bucket)
- `canonical` — non-LLM route (reserved; not yet active)
- `qid` — Wikidata resolver

## How to use BYO keys

The Worker accepts two BYO-key headers on `/api/render` and `/api/transform`:

- `X-Render-LLM-Key-Anthropic: sk-ant-...` — for off-centre slider renders using Anthropic
- `X-Render-LLM-Key-OpenAI: sk-...` — for off-centre slider renders using OpenAI

Either header puts you in the 100/hr-per-IP BYO bucket. The header value is forwarded to the LLM provider's API; **the Worker does not log, persist, or proxy the key anywhere else**. The receipt's `provider` field reports what actually served (so an audit trail can distinguish operator-keyed from BYO-keyed renders).

Example using Anthropic with your own key:

```bash
curl -sS -X POST https://sum-demo.ototao.workers.dev/api/render \
  -H 'content-type: application/json' \
  -H "x-render-llm-key-anthropic: $YOUR_ANTHROPIC_KEY" \
  -d '{"triples":[["alice","graduated","2012"]],"slider_position":{"density":1.0,"length":0.5,"formality":0.7,"audience":0.5,"perspective":0.5}}'
```

## Local fallback (no Worker, no limit)

If you need higher quota than even the 100/hr BYO bucket gives, or you want privacy / offline, install locally:

```bash
pip install 'sum-engine[openai,sieve,receipt-verify]'
export OPENAI_API_KEY=sk-...
sum transform apply slider --input doc.json --parameters '{"density":1.0,"length":0.5,...}'
```

The CLI exposes the same transform-registry surface as `/api/transform`, signs receipts with your own key (set `SUM_TRANSFORM_SIGNING_JWK` + `SUM_TRANSFORM_SIGNING_KID`), and has no rate limit.

Free-provider routing (Hugging Face, Ollama, llama.cpp, custom OpenAI-compatible endpoints like Modal or Fireworks.ai) is wired in `sum_engine_internal/ensemble/llm_dispatch.get_adapter` — see `docs/BYOK_AND_FREE_PROVIDERS.md` (forthcoming) for the matrix.

## Operator notes

- **Storage:** the rate limiter uses the `RENDER_CACHE` KV namespace, key shape `rl:<scope>:<ip>:<window-index>`, with TTL slightly past the window length. No new KV namespace required; no Cloudflare Paid Rate Limiting API required. Free-tier-compatible.
- **Bypass during outage:** if `RENDER_CACHE` is unbound for any reason, the rate limit silently disables (route handler's `if (env.RENDER_CACHE)` guard). The route still serves; only the rate limit is degraded. This is intentional fail-open behaviour — the operator can choose to fail-closed by making `RENDER_CACHE` mandatory (would require touching each route).
- **Tuning:** all limits are in `worker/src/rate_limit.ts::POLICY`. Tunable per scope without touching the route handlers.
- **Headers respected:** `CF-Connecting-IP` (set automatically by Cloudflare). If absent (impossible in production but possible in local dev), the IP defaults to `"unknown"` — all-unknown traffic shares one bucket.

## What this does NOT defend against

- **Distributed abuse from many IPs.** A botnet can amplify within the per-IP limit. Mitigation: Cloudflare's WAF + Bot Management at the edge. Free Cloudflare plan ships some of this; paid tiers expand it. Out of scope for the in-Worker policy.
- **Application-layer logic abuse.** A caller within their quota can still craft expensive inputs (very large triples lists, very long source-chain arrays). Mitigation: `MAX_PROMPT_CHARS` on `/api/complete` already exists; similar caps belong on `/api/render` + `/api/transform` (deferred).
- **BYO-key abuse against the caller's own wallet.** If you give the Worker a key with broad scope, the Worker forwards calls to that provider. Use scoped keys.

## Pointers

- `worker/src/rate_limit.ts` — implementation.
- `worker/src/routes/{render,transform,complete,qid}.ts` — call sites.
- `worker/wrangler.toml` — RENDER_CACHE KV binding declaration.
- `docs/CHARTER_2026-05-17.md` §6 — constraint: "no shipping substrate that exposes the operator's wallet to abuse."
