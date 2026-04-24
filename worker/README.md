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

| Path            | Handler                     | Purpose                                     |
|-----------------|-----------------------------|---------------------------------------------|
| `/api/complete` | `src/routes/complete.ts`    | LLM proxy — Anthropic / OpenAI / AI Gateway |
| `/api/qid`      | `src/routes/qid.ts`         | Wikidata QID/PID resolver (wbsearchentities + 30-day edge cache) |
| _everything_    | `ASSETS` binding            | `../single_file_demo/` static files         |

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
