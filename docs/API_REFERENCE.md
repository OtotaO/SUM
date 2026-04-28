# SUM HTTP API reference

**Status:** wire spec for the SUM Cloudflare Worker, current at `worker/src/index.ts`. Source of truth for external systems integrating with the hosted endpoint at `https://sum.ototao.com` or any self-hosted deployment of the same Worker.

This is the second integration surface alongside [`MCP_INTEGRATION.md`](MCP_INTEGRATION.md). Pick by use case:

| If your caller is… | Use… |
|---|---|
| An MCP-aware LLM client (Claude Desktop, Cursor, custom agent) on the same host | the MCP server (`sum-mcp` console script) |
| A web app, mobile app, server-side service, or any HTTP client | this HTTP API |
| A Python/Node verifier needing only to check signatures on receipts produced by this API | this API for the receipt + JWKS endpoints, then a local verifier |

Both surfaces produce byte-identical canonical outputs for the operations they share, so a render receipt issued by the Worker verifies under the same Python / Node / browser verifiers regardless of which surface initiated the call.

---

## 1. Endpoint catalogue

| Method | Path | Purpose | Auth | Section |
|---|---|---|---|---|
| `POST` | `/api/render`               | Slider-conditioned tome rendering, optional signed receipt | none | [§3](#3-post-apirender) |
| `POST` | `/api/complete`             | LLM proxy for the demo UI (Anthropic-first, OpenAI fallback) | none | [§4](#4-post-apicomplete) |
| `POST` | `/api/qid`                  | Wikidata QID/PID resolver (edge-cached) | none | [§5](#5-post-apiqid) |
| `GET`  | `/.well-known/jwks.json`    | Render-receipt public keys (RFC 7517) | none, public CORS | [§6](#6-get-well-knownjwksjson) |
| `GET`  | `/.well-known/revoked-kids.json` | Receipt revocation list (`sum.revoked_kids.v1`) | none, public CORS | [§7](#7-get-well-knownrevoked-kidsjson) |

All other paths under the host serve static assets (the in-browser demo) via the `ASSETS` Worker binding.

---

## 2. Cross-cutting contract

### 2.1 Base URL

The hosted demo is at `https://sum.ototao.com`. Self-hosted deployments terminate at whatever hostname Cloudflare maps the Worker to. Examples in this doc use `https://sum.ototao.com`; substitute as needed.

### 2.2 Authentication

**None.** The API is unauthenticated and rate-limited at the Cloudflare edge. The hosted Worker never accepts secrets from clients — all server-side keys (Anthropic, OpenAI, Ed25519 signing JWK) are configured via `wrangler secret put` in the `worker/` directory.

If you operate your own deployment and want client auth, layer a Cloudflare Access policy or your own API gateway in front of the Worker. The Worker itself does not enforce auth.

### 2.3 Content type

All JSON request bodies must be `Content-Type: application/json`. Successful JSON responses are `Content-Type: application/json; charset=utf-8` (or `application/jwk-set+json` for JWKS). Error responses carry the same content type as their success counterparts so a generic JSON parser works on both.

### 2.4 Baseline security headers

Every response — JSON, JWKS, or static asset — passes through `applyBaselineHeaders()` in `worker/src/index.ts` and gets:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Referrer-Policy: no-referrer
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'none'; script-src 'unsafe-inline';
                         style-src 'unsafe-inline'; connect-src 'self';
                         img-src 'self' data:; font-src 'self';
                         base-uri 'none'; form-action 'none';
                         frame-ancestors 'none'
Permissions-Policy: <denies all major sensors + APIs except fullscreen=(self)>
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Resource-Policy: same-origin
```

The two `/.well-known/*` endpoints override `Cross-Origin-Resource-Policy: cross-origin` and add the CORS headers below. The three `/api/*` endpoints keep the baseline (same-origin only). If you need to call `/api/render` from a different origin, deploy a proxy on your origin or call the Worker server-side.

### 2.5 Error response shape

Every documented error returns JSON of the form:

```json
{ "error": "human-readable explanation of what went wrong" }
```

Plus an HTTP status code from the per-route documentation. There is no machine-readable error code field today; status code + path + the substring of the `error` value should be the basis for branching in caller code.

### 2.6 Caching

Mutating endpoints (`/api/complete`, `/api/qid`, `/api/render`) set `Cache-Control: no-store` on the response so intermediaries do not cache them. The `/api/render` endpoint maintains its own KV cache keyed by `sha256(sliders + triples)`; that cache is **server-side and transparent to callers** (response is identical whether `cache_status` is `"hit"` or `"miss"`, only the latency and the `cache_status` field differ).

The two `/.well-known/*` endpoints are read-mostly and set `Cache-Control: public, max-age=3600` (1 hour). Verifiers should respect this and re-fetch on cache miss; do not pin a specific JWKS forever, since key rotation will silently break verification when the hosted JWK rotates out.

### 2.7 Rate limiting

Cloudflare's edge defaults apply. There is no documented per-IP rate limit at the application layer today. If you need predictable throughput for a production integration, contact the maintainer before launching to get a dedicated rate-limit policy.

---

## 3. `POST /api/render`

The primary integration verb. Takes a list of `(subject, predicate, object)` triples and a 5-axis slider position, returns a slider-conditioned tome plus an optional signed receipt that proves the tome was rendered from those exact triples under those exact slider settings.

### 3.1 Request

```http
POST /api/render HTTP/1.1
Host: sum.ototao.com
Content-Type: application/json

{
  "triples": [
    ["alice", "graduated", "2012"],
    ["alice", "born", "1990"],
    ["bob", "owns", "dog"]
  ],
  "slider_position": {
    "density":     0.7,
    "length":      0.4,
    "formality":   0.6,
    "audience":    0.5,
    "perspective": 0.5
  },
  "force_render":      false,
  "cache_ttl_seconds": 86400
}
```

Required fields:

| Field | Type | Constraint |
|---|---|---|
| `triples` | `Array<[string, string, string]>` | Non-empty. Each tuple must be exactly three strings. |
| `slider_position.density`     | `number` | `[0, 1]` |
| `slider_position.length`      | `number` | `[0, 1]` |
| `slider_position.formality`   | `number` | `[0, 1]` |
| `slider_position.audience`    | `number` | `[0, 1]` |
| `slider_position.perspective` | `number` | `[0, 1]` |

Optional:

| Field | Type | Default | Effect |
|---|---|---|---|
| `force_render` | `boolean` | `false` | If `true`, bypass the Worker KV cache and always re-render. |
| `cache_ttl_seconds` | `number` | `86400` (24h) | KV cache TTL for the entry this render produces. |

### 3.2 Response (200)

```json
{
  "tome": "Alice was born in 1990 and graduated in 2012. Bob owns a dog.",
  "triples_used": [["alice", "born", "1990"], ["alice", "graduated", "2012"]],
  "drift": [
    { "axis": "density", "value": 0.05, "threshold": 0.10, "classification": "ok" },
    { "axis": "length", "value": 0.12, "threshold": 0.20, "classification": "ok" }
  ],
  "cache_status": "miss",
  "llm_calls_made": 1,
  "wall_clock_ms": 842,
  "quantized_sliders": {
    "density": 0.7, "length": 0.5, "formality": 0.7,
    "audience": 0.5, "perspective": 0.5
  },
  "render_id": "a1b2c3d4e5f60718",
  "render_receipt": {
    "schema": "sum.render_receipt.v1",
    "kid": "sum-receipt-2026-04",
    "payload": {
      "render_id": "a1b2c3d4e5f60718",
      "sliders_quantized": { "density": 0.7, "length": 0.5, "formality": 0.7, "audience": 0.5, "perspective": 0.5 },
      "triples_hash": "sha256-9f8e...",
      "tome_hash": "sha256-3a4b...",
      "model": "claude-haiku-4-5-20251001",
      "provider": "anthropic",
      "signed_at": "2026-04-28T02:55:14.117Z",
      "digital_source_type": "trainedAlgorithmicMedia"
    },
    "jws": "eyJhbGciOiJFZERTQSIsImtpZCI6InN1bS1yZWNlaXB0LTIwMjYtMDQiLCJiNjQiOmZhbHNlLCJjcml0IjpbImI2NCJdfQ..<signature>"
  }
}
```

Field semantics:

| Field | Notes |
|---|---|
| `tome` | The rendered prose. UTF-8. The signed `tome_hash` covers exactly these bytes. |
| `triples_used` | The subset of input triples that survived the `density` slider. May be smaller than the input. |
| `drift` | Per-axis fact-preservation drift measurements. Empty if the render hit the cache. See `docs/SLIDER_CONTRACT.md` for the per-axis formulas and thresholds. |
| `cache_status` | `"hit"`, `"miss"`, or `"bypass"` (when `force_render: true`). |
| `llm_calls_made` | `0` for the canonical-deterministic path, `1` for the LLM path. |
| `wall_clock_ms` | Server-side wall clock for the render (excludes network latency). |
| `quantized_sliders` | The sliders after binning (each non-density axis snaps to 5 bins). The signed receipt covers these, **not** the original request sliders. |
| `render_id` | `sha256(cache_key + tome)[:16]`. Stable identifier for this exact (sliders, triples, tome) tuple. |
| `render_receipt` | **Optional.** Present iff the operator has configured `RENDER_RECEIPT_SIGNING_JWK` + `RENDER_RECEIPT_SIGNING_KID`. Absent for self-hosted deployments without signing keys. The hosted demo always produces this field. |

### 3.3 The render receipt

When present, `render_receipt` is the load-bearing field for any caller that wants cryptographic proof of provenance. Its shape is fixed by `sum.render_receipt.v1` ([`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) is the wire spec):

```typescript
{
  schema: "sum.render_receipt.v1",
  kid: string,                    // matches a key in /.well-known/jwks.json
  payload: ReceiptPayload,        // the signed-over data (see below)
  jws: string,                    // detached JWS (RFC 7515 §A.5)
}
```

`ReceiptPayload`:

| Field | Type | Meaning |
|---|---|---|
| `render_id` | `string` | Same value as the top-level `render_id`. |
| `sliders_quantized` | object | The 5-axis slider position **after** binning. |
| `triples_hash` | `string` | `sha256-{hex}` over the JCS-canonical bytes of the lex-sorted triples. |
| `tome_hash` | `string` | `sha256-{hex}` over the UTF-8 bytes of `tome`. |
| `model` | `string` | The actual model identifier the LLM provider returned, or `"canonical-deterministic-v0"` for the no-LLM path. |
| `provider` | `"anthropic"` \| `"openai"` \| `"cf-ai-gateway-anthropic"` \| `"canonical-path"` | Which backend produced the tome. |
| `signed_at` | `string` | ISO 8601 UTC timestamp at the moment the JWS was produced. |
| `digital_source_type` | `"trainedAlgorithmicMedia"` \| `"algorithmicMedia"` | C2PA terminology. The first means LLM-generated; the second means deterministic. |

The `jws` is a [detached JWS](https://www.rfc-editor.org/rfc/rfc7515#appendix-A.5) per RFC 7515, with `b64=false` per [RFC 7797](https://www.rfc-editor.org/rfc/rfc7797). Protected header:

```json
{ "alg": "EdDSA", "kid": "<kid>", "b64": false, "crit": ["b64"] }
```

### 3.4 Verifying a receipt client-side

The full verifier algorithm is documented in [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md). Quick reference:

1. **Fetch the JWKS** from `https://sum.ototao.com/.well-known/jwks.json` (or the publisher's equivalent). Pick the JWK whose `kid` matches the receipt's `kid`. Reject if not found.
2. **Check revocation**: fetch `/.well-known/revoked-kids.json`. If the receipt's `kid` is in `revoked` and the receipt's `signed_at` ≥ `effective_revocation_at`, reject.
3. **JCS-canonicalise** `receipt.payload`. (Use `canonicalize` in JS, `sum_engine_internal/infrastructure/jcs.py` in Python — both are byte-identical.)
4. **Verify the JWS** as detached over those canonical bytes. In Node:
   ```js
   import { flattenedVerify } from "jose";
   const [protectedHeader, , signature] = jws.split(".");
   await flattenedVerify(
     { protected: protectedHeader, payload: canonicalBytes, signature },
     publicKey,
     { algorithms: ["EdDSA"] }
   );
   ```
5. **Re-derive** `triples_hash` from the triples you sent and `tome_hash` from the response's `tome`. Compare to the receipt's claimed values. Mismatch ⇒ the receipt does not bind the bytes you have.
6. **Optional**: pin a max-age window on `signed_at` to refuse very old receipts.

A working browser implementation is in [`single_file_demo/receipt_verifier.js`](../single_file_demo/receipt_verifier.js). A Node implementation is in [`standalone_verifier/`](../standalone_verifier/). A Python implementation is `sum_engine_internal/render_receipt/verifier.py`. All three accept the same receipt bytes.

### 3.5 Errors

| Status | Trigger | Body |
|---|---|---|
| 400 | Body is not valid JSON | `{ "error": "invalid JSON body" }` |
| 400 | `triples` missing or empty | `{ "error": "missing or empty 'triples' array" }` |
| 400 | `slider_position` missing | `{ "error": "missing 'slider_position'" }` |
| 400 | Slider axis out of `[0, 1]` | `{ "error": "slider value out of [0, 1]: <value>" }` |
| 405 | Non-POST method | `{ "error": "method not allowed; use POST" }` |
| 502 | LLM upstream failure | `{ "error": "render failed: <provider> <status>: <upstream-body-prefix>", "cache_key": "<key>" }` |

A 502 leaves the cache untouched — the next request with the same `(triples, sliders)` will retry.

---

## 4. `POST /api/complete`

Thin LLM proxy for the in-browser demo's slider UI. Anthropic-first, OpenAI fallback, 503 if neither is configured. **Not intended as a general LLM proxy for third-party integrations** — use the model providers directly for that. This endpoint exists so the static demo can render text without leaking API keys to the client.

### 4.1 Request

```http
POST /api/complete HTTP/1.1
Content-Type: application/json

{
  "prompt": "string, max 40000 chars",
  "model":  "string (optional)"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `prompt` | `string` | yes | Non-empty after trim, ≤40 000 chars. |
| `model` | `string` | no | Provider-specific model tag. Default is `claude-haiku-4-5-20251001` for Anthropic, `gpt-4o-mini` for OpenAI. |

### 4.2 Response (200)

```json
{
  "completion": "Generated text from the LLM.",
  "source":     "anthropic",
  "model":      "claude-haiku-4-5-20251001"
}
```

`source` is the provider that actually answered (Anthropic takes priority if both keys are set). `model` is the model identifier the provider returned (may differ from the `model` you requested if the provider mapped it).

### 4.3 Errors

| Status | Trigger |
|---|---|
| 400 | Invalid JSON, missing/empty `prompt`, or `prompt` over 40 000 chars |
| 405 | Non-POST method |
| 502 | Upstream provider returned non-2xx; body includes the provider name, status, and a 500-char prefix of the upstream response |
| 503 | Neither `ANTHROPIC_API_KEY` nor `OPENAI_API_KEY` is configured. The demo's JS detects 503 and falls back to a deterministic naive tokeniser. |

---

## 5. `POST /api/qid`

Wikidata QID/PID resolver. Takes a list of free-text terms, returns the best Wikidata match for each, with edge caching. Used by the demo to attach Q-numbers to extracted axioms; useful for any caller that wants to resolve terms to Wikidata IDs without paying the round-trip cost.

### 5.1 Request

```json
{
  "terms": [
    { "text": "Marie Curie",     "kind": "item",     "lang": "en" },
    { "text": "discovered",       "kind": "property", "lang": "en" }
  ]
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `terms` | `Array<Term>` | yes | 1–50 entries. |
| `terms[].text` | `string` | yes | 1–200 chars after trim. |
| `terms[].kind` | `"item"` \| `"property"` | no | Default `"item"`. `"property"` searches the P-namespace (predicates). |
| `terms[].lang` | `string` | no | Default `"en"`. Wikidata language code. |

### 5.2 Response (200)

```json
{
  "resolved": [
    {
      "text": "Marie Curie",
      "id": "Q7186",
      "label": "Marie Curie",
      "description": "Polish-French physicist and chemist (1867–1934)",
      "confidence": 1.0,
      "source": "wbsearchentities"
    },
    {
      "text": "discovered",
      "id": null,
      "reason": "no-match"
    }
  ]
}
```

Each resolved term has either an `id` (a Wikidata QID like `Q7186` or PID like `P575`) or a null `id` with a `reason`. Confidence is `1.0` for an exact label match, `0.7` for an alias, `0.5` otherwise.

`source` is `"cache"` for an edge-cached resolution and `"wbsearchentities"` for a fresh hit against Wikidata.

### 5.3 Errors

| Status | Trigger |
|---|---|
| 400 | Invalid JSON or missing/empty `terms` |
| 405 | Non-POST method |
| 413 | More than 50 terms |

The Worker uses Cloudflare's Cache API (not KV) to memoise resolutions for 30 days. Cache misses or Wikidata transient errors surface as `id: null` with a descriptive `reason`; the request itself still returns 200.

---

## 6. `GET /.well-known/jwks.json`

Public verification keys for render receipts. RFC 7517 JWKS shape, served with permissive CORS so any origin can fetch it.

### 6.1 Request

`GET`, `HEAD`, or `OPTIONS`. No body.

### 6.2 Response (200)

```json
{
  "keys": [
    {
      "kty": "OKP",
      "crv": "Ed25519",
      "x":   "<base64url of the public key bytes>",
      "kid": "sum-receipt-2026-04",
      "alg": "EdDSA",
      "use": "sig"
    }
  ]
}
```

May contain multiple keys during rotation windows. Use `kid` to pick the one matching a receipt's `kid` field.

### 6.3 Headers

```
Content-Type: application/jwk-set+json
Cache-Control: public, max-age=3600
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, HEAD, OPTIONS
Access-Control-Max-Age: 86400
Cross-Origin-Resource-Policy: cross-origin
```

The `Access-Control-Allow-Credentials` header is **deliberately not set** — JWKS reads no credentials, and the negative pinning guards against future regressions.

### 6.4 Errors

| Status | Trigger |
|---|---|
| 503 | `RENDER_RECEIPT_PUBLIC_JWKS` env var not configured. Self-hosted deployments without signing return this. |
| 500 | `RENDER_RECEIPT_PUBLIC_JWKS` is set but is not valid JSON. |

---

## 7. `GET /.well-known/revoked-kids.json`

Receipt revocation list. `sum.revoked_kids.v1` schema. CORS-permissive like the JWKS endpoint. Verifiers must consult this before trusting a receipt.

### 7.1 Request

`GET`, `HEAD`, or `OPTIONS`. No body.

### 7.2 Response (200)

```json
{
  "schema":     "sum.revoked_kids.v1",
  "issued_at":  "2026-04-28T02:55:14.117Z",
  "revoked": [
    {
      "kid":                      "sum-receipt-2025-11",
      "effective_revocation_at":  "2026-01-15T00:00:00Z"
    }
  ]
}
```

A receipt with `kid: "sum-receipt-2025-11"` whose `signed_at` is on or after `2026-01-15T00:00:00Z` must be rejected. Receipts signed before that timestamp remain valid (historical receipts under a since-revoked key are not retroactively invalidated unless the operator explicitly rolls `effective_revocation_at` back, which is documented in `docs/INCIDENT_RESPONSE.md` case 1).

If `RENDER_RECEIPT_REVOKED_KIDS` is unset, the response is a valid empty list with the current timestamp. **Verifiers should still call this endpoint and check** — silently skipping the revocation check defeats the surface.

### 7.3 Headers

Same CORS-permissive baseline as `/.well-known/jwks.json`. `Cache-Control: public, max-age=3600`.

### 7.4 Errors

| Status | Trigger |
|---|---|
| 500 | `RENDER_RECEIPT_REVOKED_KIDS` is set but is not valid JSON or fails the `sum.revoked_kids.v1` schema check. |

There is no 404 on this endpoint — absence of an env var produces an empty-list response, not a missing-resource error. This is so verifiers can have a single fetch path that always succeeds when the Worker is up.

---

## 8. Operator: configuring a self-hosted Worker

The hosted demo at `sum.ototao.com` is one deployment. To run your own:

```bash
cd worker/
wrangler deploy

# Required for the LLM render path:
echo "$ANTHROPIC_KEY" | wrangler secret put ANTHROPIC_API_KEY

# Required for receipt signing (without these, /api/render returns
# a tome but no `render_receipt` field, and /.well-known/jwks.json
# returns 503):
echo "$ED25519_PRIVATE_JWK_JSON" | wrangler secret put RENDER_RECEIPT_SIGNING_JWK
echo "sum-receipt-yyyy-mm"        | wrangler secret put RENDER_RECEIPT_SIGNING_KID

# Plaintext vars (in wrangler.toml [vars] block, not secrets):
RENDER_RECEIPT_PUBLIC_JWKS = '{"keys": [...]}'
RENDER_RECEIPT_REVOKED_KIDS = '{"schema": "sum.revoked_kids.v1", "issued_at": "...", "revoked": []}'

# Optional KV bindings (referenced by name in wrangler.toml):
# RENDER_CACHE — caches /api/render results, keyed by sha256(sliders+triples)
```

| Variable | Type | Effect when absent |
|---|---|---|
| `ANTHROPIC_API_KEY` | secret | `/api/render` LLM path returns 502; `/api/complete` falls back to OpenAI |
| `OPENAI_API_KEY` | secret | `/api/complete` returns 503 if Anthropic also missing |
| `RENDER_RECEIPT_SIGNING_JWK` + `RENDER_RECEIPT_SIGNING_KID` | secrets | `/api/render` returns no `render_receipt` field |
| `RENDER_RECEIPT_PUBLIC_JWKS` | plaintext var | `/.well-known/jwks.json` returns 503 |
| `RENDER_RECEIPT_REVOKED_KIDS` | plaintext var | `/.well-known/revoked-kids.json` returns an empty list |
| `RENDER_CACHE` | KV binding | `/api/render` re-runs every call; `cache_status` is always `"miss"` |
| `CF_AI_GATEWAY_BASE` | plaintext var | LLM calls go direct to provider instead of through the gateway |

Generating an Ed25519 JWK pair: `bun scripts/cert/gen_render_keypair.ts` writes the private JWK to `/tmp/private.jwk` and the public JWKS to `/tmp/render_receipt_public.jwks`. Then:

```bash
wrangler secret put RENDER_RECEIPT_SIGNING_JWK < /tmp/private.jwk
wrangler secret put RENDER_RECEIPT_SIGNING_KID  # paste the kid string at the prompt
# Public JWKS goes in the Cloudflare dashboard as a plaintext variable
# (escaping inline JSON in wrangler.toml is fragile):
#   Worker → Settings → Variables → add RENDER_RECEIPT_PUBLIC_JWKS
#   paste the contents of /tmp/render_receipt_public.jwks
```

The same procedure is documented in `worker/wrangler.toml` comments.

---

## 9. Working integration examples

### 9.1 Render + verify in Node

```js
import { flattenedVerify, importJWK } from "jose";
import canonicalize from "canonicalize";
import { createHash } from "node:crypto";

const BASE = "https://sum.ototao.com";

async function renderAndVerify(triples, sliders) {
  // 1. Render
  const renderRes = await fetch(`${BASE}/api/render`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ triples, slider_position: sliders }),
  });
  if (!renderRes.ok) throw new Error(`render failed: ${renderRes.status}`);
  const result = await renderRes.json();

  if (!result.render_receipt) {
    return { tome: result.tome, verified: false, reason: "no receipt issued" };
  }
  const { kid, payload, jws } = result.render_receipt;

  // 2. JWKS
  const jwks = await (await fetch(`${BASE}/.well-known/jwks.json`)).json();
  const jwk = jwks.keys.find((k) => k.kid === kid);
  if (!jwk) throw new Error(`no JWK for kid=${kid}`);

  // 3. Revocation
  const revoked = await (await fetch(`${BASE}/.well-known/revoked-kids.json`)).json();
  const hit = revoked.revoked.find((r) => r.kid === kid);
  if (hit && payload.signed_at >= hit.effective_revocation_at) {
    return { tome: result.tome, verified: false, reason: "kid revoked" };
  }

  // 4. Verify the detached JWS over the canonical payload bytes
  const canonicalBytes = new TextEncoder().encode(canonicalize(payload));
  const [protectedHeader, , signature] = jws.split(".");
  const key = await importJWK(jwk, "EdDSA");
  await flattenedVerify(
    { protected: protectedHeader, payload: canonicalBytes, signature },
    key,
    { algorithms: ["EdDSA"] }
  );

  // 5. Re-derive hashes; compare
  // Componentwise tuple-lex sort — matches Python's
  // `sorted(tuple(t) for t in triples)` byte-for-byte. The Worker's
  // hashTriples helper uses the same comparator. Default `.sort()`
  // works for triples without separator-collisions but the explicit
  // version is safe under all string contents.
  const sortedTriples = [...triples].map((t) => [...t]).sort((a, b) => {
    for (let i = 0; i < 3; i++) {
      if (a[i] < b[i]) return -1;
      if (a[i] > b[i]) return 1;
    }
    return 0;
  });
  const triplesHash = "sha256-" + createHash("sha256")
    .update(canonicalize(sortedTriples)).digest("hex");
  const tomeHash = "sha256-" + createHash("sha256")
    .update(result.tome, "utf-8").digest("hex");

  if (triplesHash !== payload.triples_hash) {
    return { verified: false, reason: "triples_hash mismatch" };
  }
  if (tomeHash !== payload.tome_hash) {
    return { verified: false, reason: "tome_hash mismatch" };
  }

  return { tome: result.tome, verified: true, payload };
}
```

### 9.2 Resolve QIDs in Python

```python
import httpx

def resolve_qids(terms: list[str], lang: str = "en") -> dict[str, str | None]:
    body = {"terms": [{"text": t, "kind": "item", "lang": lang} for t in terms]}
    r = httpx.post("https://sum.ototao.com/api/qid", json=body, timeout=10.0)
    r.raise_for_status()
    return {row["text"]: row.get("id") for row in r.json()["resolved"]}
```

### 9.3 Render-only (no verification) in Python

```python
import httpx

def render(triples, sliders) -> dict:
    r = httpx.post("https://sum.ototao.com/api/render", json={
        "triples": triples,
        "slider_position": sliders,
    }, timeout=30.0)
    r.raise_for_status()
    return r.json()  # dict with tome, drift, render_receipt, ...
```

For Python receipt verification, use `sum_engine_internal.render_receipt.verifier.verify_receipt()` from this repo (`pip install sum-engine[receipt-verify]`). It accepts the same receipt bytes the JS verifier does.

---

## 10. Cross-references

- [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) — wire spec for `sum.render_receipt.v1`. Source of truth for the receipt payload schema and the six-step verifier algorithm.
- [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) §1.3.1 — what the cross-runtime Ed25519 trust triangle proves and what it doesn't.
- [`docs/MCP_INTEGRATION.md`](MCP_INTEGRATION.md) — the MCP server companion to this HTTP API. Use MCP for local LLM clients, this HTTP API for everything else.
- [`docs/INCIDENT_RESPONSE.md`](INCIDENT_RESPONSE.md) — operator runbook for kid revocation (case 1), JWKS rollback, and related contingencies.
- [`docs/SLIDER_CONTRACT.md`](SLIDER_CONTRACT.md) — the 5-axis slider contract: per-axis drift formulas, fact-preservation thresholds, what each axis controls.
- [`docs/COMPATIBILITY_POLICY.md`](COMPATIBILITY_POLICY.md) — schema-version policy. The `sum.*.v1` schemas in this doc are pinned; future incompatible changes ship as `v2`.
- [`worker/src/index.ts`](../worker/src/index.ts), [`worker/src/routes/`](../worker/src/routes/) — the implementation. If this doc and the code disagree, the code wins; please file an issue.
