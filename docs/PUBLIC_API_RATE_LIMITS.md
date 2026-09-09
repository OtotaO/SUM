# Public API resource and admission limits

The hosted Worker admits provider work through a single SQLite Durable Object.
One transaction checks the caller quota, the shared operator budget, and active
call limits before any provider request. Deterministic rendering remains usable
without provider credentials or the admission binding.

## Policy

| Route | Mode | Per-IP allowance | Enforcement |
|---|---|---:|---|
| `/api/render` | Provider call using the operator key | 5 per UTC day | Atomic admission |
| `/api/render` | Provider call using the selected provider's BYO key | 100 per UTC hour | Atomic admission |
| `/api/complete` | Operator key; BYO headers ignored | Shares 5 per UTC day with render | Atomic admission |
| `/api/render` | All non-density axes centered | 100 per UTC hour | Best-effort KV |
| `/api/transform` | Canonical slider transform | Shares canonical 100 per UTC hour | Best-effort KV |
| `/api/qid` | Wikidata resolution | 60 per UTC hour | Best-effort KV |
| `/.well-known/*` | Verification keys and revocations | Unlimited | No admission dependency |

The operator-funded routes together admit at most **100 provider attempts per
UTC day across all IPs**. All provider work shares **8 active leases globally**
and **2 per IP**, including BYO calls. Denied admission consumes no quota.
Admitted attempts consume quota even when a provider rejects or times out; there
is no automatic retry or refund. Cache hits consume no provider allowance.
Windows are fixed UTC windows, not rolling intervals.

A successful call releases its concurrency lease in `finally`; abandoned leases
expire after 35 seconds. The provider deadline is 30 seconds. Leases bound
admitted Worker calls, not work an upstream provider may continue after an abort.
These are request-count controls, not a dollar-denominated provider billing cap.

`/api/transform` currently implements canonical slider rendering only; off-center
requests return 501. It does not spend provider quota. Adding LLM dispatch there
must use the same admission service.

## Request and provider limits

Render, complete, and transform accept JSON bodies up to **192 KiB**. The Worker
counts streamed bytes before JSON parsing, including requests without
Content-Length, and stops a stalled read after 10 seconds.

- Render and slider transform: 1 to 256 triples, exactly three nonempty strings
  per triple, at most 512 characters per string and 40,000 characters total.
- Every supplied slider axis must be a finite number in [0, 1]. Render requires
  all five axes; the slider transform also requires them in its parameters.
- Complete: a nonempty prompt of at most 40,000 characters. A requested model
  must equal the configured model for the selected operator provider. There is
  no caller-controlled expensive-model escape hatch.
- Render: provider must be `anthropic` or `openai`; force_render must be boolean;
  an optional cache TTL must be an integer from 60 through 86,400 seconds.
- Transform source chains: at most 256 records, bounded claim and URI strings,
  nonnegative safe-integer byte offsets, and end no earlier than start.
- Provider output requests are capped at 2,048 tokens. Response JSON is capped
  at 256 KiB, and the 30-second deadline includes reading the response body.

Malformed or oversized requests are rejected before provider admission or
provider dispatch. An empty or missing matching provider key also fails before
admission. Body errors return 400, size limits 413, and stalled body reads 408.
Provider failures return 502 without returning provider error bodies to callers.

## BYO provider credentials

On `/api/render`, supply the header matching `provider` in the JSON body:

- `X-Render-LLM-Key-Anthropic` for `anthropic`.
- `X-Render-LLM-Key-OpenAI` for `openai`.

Without an explicit provider, configured Anthropic credentials are preferred,
then OpenAI. A key for the other provider cannot promote an operator-funded call
to the BYO allowance. Whitespace-only keys are absent. The exact selected,
trimmed credential funds dispatch; rejection never retries on the operator key.
BYO calls bypass the operator's AI Gateway. Credentials are not intentionally
logged or stored. Receipt provider fields identify service provenance, not payer.

## Failure responses

Per-IP quota exhaustion returns 429 with `scope`, `limit`, `remaining: 0`,
`reset_seconds`, and remediation in the JSON body. Headers include
`x-ratelimit-limit`, `x-ratelimit-remaining`, `x-ratelimit-reset`, and `retry-after`.
Shared daily-budget or concurrency exhaustion also returns 429 with a specific
error and `retry-after`; concurrency pressure asks callers to retry in 5 seconds.

Missing, failed, or malformed `LLM_BUDGET` responses return 503 and `retry-after:
30`. No provider call follows that failure. Canonical rendering and previously
cached renders can still succeed. An absent `RENDER_CACHE` only degrades caching
and best-effort cheap-route limits; it cannot bypass provider admission.

## Deployment and operations

`worker/wrangler.toml` declares the `LLM_BUDGET` Durable Object binding and the
`v1-llm-budget` migration with `new_sqlite_classes = ["LLMBudget"]`. Deployment
provisions the namespace with the Worker; no manual namespace ID or additional
secret is needed. Keep the migration history and class export on future deploys.
Cloudflare supports SQLite Durable Objects on both Free and Paid plans; account
quotas still apply. See [Cloudflare's Durable Objects overview](https://developers.cloudflare.com/durable-objects/)
and [transactional SQLite storage](https://developers.cloudflare.com/durable-objects/api/sqlite-storage-api/).

Admission uses one stable object name, `public-llm-v1`. Changing that name,
recreating its namespace, or deleting its storage resets budget history; do not
use those actions as a deployment workaround. The object stores daily IP digests
and expiry times, never raw IPs or provider keys. Expired rows are removed during
admission; bounded bucket capacity fails closed under excessive distinct callers.

Before deployment run `npm ci`, `npm run typecheck`, `npm run test:rate-limit`,
and `npm run test:hardening` in `worker/`. The hardening suite executes concurrent
requests against local workerd and SQLite, with all provider traffic mocked.
`npx wrangler deploy --dry-run` checks the bundle and binding declaration.
After deployment verify frontend bytes and perform a canonical render plus
receipt verification. Any paid smoke request consumes real quota and should be
intentional. An incident can disable generation by removing provider secrets
while leaving canonical rendering available. Preserve the admission class and
storage in a fix-forward deployment rather than deleting its migration.

## Limits of the controls

A distributed caller can exhaust the shared allowance and deny demo generation
to other visitors. KV limits on cheap routes remain non-atomic and eventually
consistent. Provider prices, operator changes to configured models, upstream
billing after cancellation, and unrelated uses of the same provider account are
outside the request counter. Maintain provider-account spending controls as an
independent layer. No claim of a universal spending cap or semantic safety is
made by successful admission.

For heavier use, run locally with your own provider credentials and signing
configuration; see [BYO and free providers](BYOK_AND_FREE_PROVIDERS.md).
Implementation: `worker/src/llm_budget.ts`, `worker/src/request_limits.ts`, and
`worker/src/rate_limit.ts`.
