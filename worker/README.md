# SUM hosted-demo Worker

Cloudflare Worker that serves the single-file demo + same-origin API
routes. Migrated from the previous Cloudflare Pages deployment
(`single_file_demo/functions/api/complete.ts`) per the April 2026
Cloudflare platform convergence guidance: Workers is where every new
capability lands; Pages is being absorbed.

Static assets live in `../single_file_demo/` (unchanged — the demo
file is still a standalone HTML page and still works as a Claude
artifact). This Worker is the routing shell around it.

## Routing

| Path                           | Handler                       | Purpose                                                                                                       |
|--------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------|
| `/api/render`                  | `src/routes/render.ts`        | Slider render → tome + `sum.render_receipt.v1` (Ed25519 / JCS / detached JWS). Public, per-IP rate-limited.    |
| `/api/transform`               | `src/routes/transform.ts`     | Transform-registry dispatch → `sum.transform_receipt.v1`. **Worker-side registry currently registers `slider` only** (`worker/src/transforms/_registry.ts`); `compose` + `extract` are Python-CLI-only today via `sum transform apply <name>`. Same signing path as `/api/render`. |
| `/api/complete`                | `src/routes/complete.ts`      | LLM proxy using operator credentials (Anthropic / OpenAI / AI Gateway). Shared 5/day demo allowance per IP; BYO headers ignored. |
| `/api/qid`                     | `src/routes/qid.ts`           | Wikidata QID/PID resolver (`wbsearchentities` + 30-day edge cache).                                            |
| `/.well-known/jwks.json`       | `src/routes/jwks.ts`          | Issuer's Ed25519 public-key set for offline receipt verification (RFC 7517).                                   |
| `/.well-known/revoked-kids.json` | `src/routes/revoked_kids.ts`| Operator-curated kid revocation list. Receipt verifiers should reject signed-with-revoked-kid envelopes.        |
| _everything else_              | `ASSETS` binding              | `../single_file_demo/` static files (the SUM hosted demo HTML).                                                |

The trust loop (`/api/render` + `/.well-known/jwks.json` + offline verifier) is reachable end-to-end from this Worker. Receipt format specs are in `../docs/RENDER_RECEIPT_FORMAT.md` and `../docs/TRANSFORM_RECEIPT_FORMAT.md`. [Public API limits](../docs/PUBLIC_API_RATE_LIMITS.md) document atomic SQLite admission: 5 operator calls per IP per UTC day, 100 BYO calls per IP per UTC hour, 100 operator calls across all IPs per day, and 8 global / 2 per-IP active calls. Missing admission infrastructure fails closed for provider work. Canonical renders need no provider key or admission binding; cache hits consume no provider allowance.

### `/api/qid` — Wikidata resolver

```bash
curl -X POST https://sum-demo.<account>.workers.dev/api/qid \
  -H 'content-type: application/json' \
  -d '{"terms":[{"text":"Alice","kind":"item"},{"text":"orbit","kind":"property"}]}'
```

Response:

```json
{"resolved":[
  {"text":"Alice","id":"Q3099839","label":"Alice",
   "description":"female given name","confidence":1.0,
   "source":"wbsearchentities"},
  {"text":"orbit","id":"P398","label":"orbits","confidence":0.7,
   "source":"wbsearchentities"}
]}
```

Cache: Cache API, 30-day TTL, `source:"cache"` on subsequent hits.
Optional KV second layer — uncomment `[[kv_namespaces]]` in
`wrangler.toml` after `wrangler kv:namespace create qid-cache`.

## First deploy (one-time, user-only)

Requires a Cloudflare account.

```bash
cd worker/
npm install
npx wrangler login              # OAuth flow in your browser
npx wrangler secret put ANTHROPIC_API_KEY   # paste the key when prompted
npx wrangler deploy             # ships to sum-demo.<account>.workers.dev
```

The deploy URL is printed at the end of `wrangler deploy`.

**After every deploy, confirm the live page matches HEAD.** The Worker
serves `../single_file_demo/index.html` verbatim (no build step), so the
only way it can drift from the repo is a missed/lagged deploy — the exact
failure that once hid PR #243's cascade panel from production. Guard
against it:

```bash
make verify-frontend-bytes        # SHA-256(live assets) == SHA-256(repo) for the page + browser verifier JS
# or, for a non-default account:
SUM_DEMO_URL=https://sum-demo.<account>.workers.dev make verify-frontend-bytes
```

To bind a custom subdomain (e.g. `sum-demo.sumequities.com`), uncomment
the `[[routes]]` block in `wrangler.toml` after associating the domain
with the Worker in the Cloudflare dashboard → Workers & Pages → sum-demo
→ Settings → Triggers.

## Subsequent deploys (CI)

`.github/workflows/deploy-worker.yml` runs `wrangler deploy` on
workflow_dispatch. Requires two GitHub secrets:

- `CLOUDFLARE_API_TOKEN` — create one at
  `dash.cloudflare.com/profile/api-tokens` with template "Edit
  Cloudflare Workers".
- `CLOUDFLARE_ACCOUNT_ID` — visible in the Cloudflare dashboard URL.

Manual trigger keeps deploy in your control. Enable push-on-tag later
if you want automatic releases.

## Local development

```bash
cd worker/
npm install
npm run dev         # wrangler dev on localhost:8787
```

The dev server serves `../single_file_demo/` as static assets and
runs the Worker code from `src/`. For API routes, set secrets in
a `.dev.vars` file (gitignored):

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Typecheck

```bash
npm run typecheck
```

Uses `@cloudflare/workers-types` for `Env` / `Fetcher` / `ExecutionContext`
bindings.

## Admission and resource-bound verification

```bash
npm ci
npm run typecheck
npm run test:rate-limit
npm run test:hardening
npx wrangler deploy --dry-run
```

The hardening suite runs local workerd with SQLite Durable Objects and mocked
provider traffic. The `LLM_BUDGET` binding and `v1-llm-budget` migration are
provisioned by the existing deploy workflow. Preserve that class and migration
history during future deploys. Provider calls are capped at 2,048 output tokens
and a 30-second deadline. Request bytes, triples, sliders, and model selection
are validated before admission. These request-count limits are independent of
provider-account billing controls.

Unicode sorting for new render hashes, density selection, transform inputs, and
cache keys compares code points to match Python, including supplementary
characters. The regression fixture is generated by
`python -m worker.test.generate_unicode_fixture` from the repository root.
Historical signed receipt bytes are never rewritten.
