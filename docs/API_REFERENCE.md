# SUM HTTP API reference

**Status:** wire spec for the SUM Cloudflare Worker, current at `worker/src/index.ts`. Source of truth for external systems integrating with the hosted endpoint at `https://sum-demo.ototao.workers.dev` or any self-hosted deployment of the same Worker.

This is the second integration surface alongside [`MCP_INTEGRATION.md`](MCP_INTEGRATION.md). Pick by use case:

| If your caller is… | Use… |
|---|---|
| An MCP-aware LLM client (Claude Desktop, Cursor, custom agent) on the same host | the MCP server (`sum-mcp` console script) |
| A web app, mobile app, server-side service, or any HTTP client | this HTTP API |
| A Python/Node verifier needing only to check signatures on receipts produced by this API | this API for the receipt + JWKS endpoints, then a local verifier |

The two surfaces do not produce the same bytes: the MCP server's `render` tool renders locally (deterministic density only) in the canonical tome format, while the Worker's canonical tome is plain sentences, and only the Worker issues render receipts. A render receipt from the Worker verifies under the Python, Node and browser verifiers.

---

## 1. Endpoint catalogue

| Method | Path | Purpose | Auth | Section |
|---|---|---|---|---|
| `POST` | `/api/render`               | Slider-conditioned tome rendering, optional signed receipt | none | [§3](#3-post-apirender) |
| `POST` | `/api/complete`             | LLM proxy for the demo UI (Anthropic-first, OpenAI fallback) | none | [§4](#4-post-apicomplete) |
| `POST` | `/api/qid`                  | Wikidata QID/PID resolver (edge-cached) | none | [§5](#5-post-apiqid) |
| `POST` | `/api/transform`            | Transform-registry dispatch → `sum.transform_receipt.v1` (Worker: `slider` canonical path only; LLM-axis returns 501) | none | [§8](#8-post-apitransform) |
| `GET`  | `/.well-known/jwks.json`    | Render-receipt public keys (RFC 7517) | none, public CORS | [§6](#6-get-well-knownjwksjson) |
| `GET`  | `/.well-known/revoked-kids.json` | Receipt revocation list (`sum.revoked_kids.v1`) | none, public CORS | [§7](#7-get-well-knownrevoked-kidsjson) |

All other paths under the host serve static assets (the in-browser demo) via the `ASSETS` Worker binding.

---

## 2. Cross-cutting contract

### 2.1 Base URL

The hosted demo is at `https://sum-demo.ototao.workers.dev`. Self-hosted deployments terminate at whatever hostname Cloudflare maps the Worker to. Examples in this doc use `https://sum-demo.ototao.workers.dev`; substitute as needed.

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

The two `/.well-known/*` endpoints override `Cross-Origin-Resource-Policy: cross-origin` and add the CORS headers below. The four `/api/*` endpoints keep the baseline (same-origin only). If you need to call `/api/render` from a different origin, deploy a proxy on your origin or call the Worker server-side.

### 2.5 Error response shape

Every documented error returns JSON of the form:

```json
{ "error": "human-readable explanation of what went wrong" }
```

Plus an HTTP status code from the per-route documentation. There is no machine-readable error code field today; status code + path + the substring of the `error` value should be the basis for branching in caller code.

### 2.6 Caching

Mutating endpoints (`/api/complete`, `/api/qid`, `/api/render`, `/api/transform`) set `Cache-Control: no-store` on the response so intermediaries do not cache them. The `/api/render` endpoint maintains its own KV cache keyed by `sha256(sliders + triples)`; that cache is **server-side and transparent to callers** (response is identical whether `cache_status` is `"hit"` or `"miss"`, only the latency and the `cache_status` field differ).

The two `/.well-known/*` endpoints are read-mostly and set `Cache-Control: public, max-age=3600` (1 hour). Verifiers should respect this and re-fetch on cache miss; do not pin a specific JWKS forever, since key rotation will silently break verification when the hosted JWK rotates out.

### 2.7 Rate limiting

The hosted Worker applies a **per-IP application-layer rate limit** on the LLM-spending routes (it protects the operator's API credits). Full policy in [`docs/PUBLIC_API_RATE_LIMITS.md`](PUBLIC_API_RATE_LIMITS.md); summary:

| Route | Mode | Limit | Window |
|---|---|---|---|
| `/api/render`, `/api/transform` | operator-keyed (no BYO header) | **5** | 24 h |
| `/api/render`, `/api/transform` | BYO-key (`X-Render-LLM-Key-Anthropic` or `-OpenAI`) | **100** | 1 h |
| `/api/complete` | operator-keyed | **5** | 24 h |
| `/.well-known/*` | — | **no limit** | — (verifiers must not be throttled) |

Over-quota responses return **`429`** with `x-ratelimit-limit` / `-remaining` / `-reset` headers and an `error` + `remediation` body (supply a BYO key for the 100/h bucket, wait for the window to reset, or run locally via `pip install 'sum-engine[openai]'`). The `/.well-known/*` trust-loop endpoints are deliberately unlimited so receipt verification never gets rate-limited.

---

## 3. `POST /api/render`

The primary integration verb. Takes a list of `(subject, predicate, object)` triples and a 5-axis slider position, returns a slider-conditioned tome plus an optional signed receipt that proves the tome was rendered from those exact triples under those exact slider settings.

### 3.1 Request

```http
POST /api/render HTTP/1.1
Host: sum-demo.ototao.workers.dev
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
  "drift": [],
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
    "kid": "sum-render-2026-04-27-1",
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
| `drift` | Per-axis fact-preservation drift measurements — shape `[{ "axis", "value", "threshold", "classification" }]`. **The hosted Worker always returns this empty (`[]`):** it does not re-extract triples from the rendered tome, so it computes no drift. The Python slider bench is the canonical source for per-axis fact-preservation/drift (see `docs/SLIDER_CONTRACT.md` for the formulas and thresholds). The field is present for schema stability and may be populated by non-Worker render paths. |
| `cache_status` | `"hit"`, `"miss"`, or `"bypass"` (when `force_render: true`). |
| `llm_calls_made` | `0` for the canonical-deterministic path, `1` for the LLM path. |
| `wall_clock_ms` | Server-side wall clock for the render (excludes network latency). |
| `quantized_sliders` | The sliders after binning (each non-density axis snaps to 5 bins). The signed receipt covers these, **not** the original request sliders. |
| `render_id` | `sha256(cache_key + tome)[:16]`. Stable identifier for this exact (sliders, triples, tome) tuple. |
| `render_receipt` | **Optional.** Present when the operator has configured `RENDER_RECEIPT_SIGNING_JWK` + `RENDER_RECEIPT_SIGNING_KID` and signing succeeds. Absent for self-hosted deployments without signing keys. If signing fails, the render is returned without this field and is not cached. |

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
| `signed_at` | `string` | ISO 8601 UTC timestamp at the moment the JWS was produced. Verifiers read it in the receipt timestamp form of [`RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) §6.1 (see that file's §6.2). |
| `digital_source_type` | `"trainedAlgorithmicMedia"` \| `"algorithmicMedia"` | C2PA terminology. The first means LLM-generated; the second means deterministic. |

The `jws` is a [detached JWS](https://www.rfc-editor.org/rfc/rfc7515#appendix-A.5) per RFC 7515, with `b64=false` per [RFC 7797](https://www.rfc-editor.org/rfc/rfc7797). Protected header:

```json
{ "alg": "EdDSA", "kid": "<kid>", "b64": false, "crit": ["b64"] }
```

### 3.4 Verifying a receipt client-side

The full verifier algorithm is documented in [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md). Quick reference:

1. **Fetch the JWKS** from `https://sum-demo.ototao.workers.dev/.well-known/jwks.json` (or the publisher's equivalent). Pick the JWK whose `kid` matches the receipt's `kid`. Reject if not found.
2. **Check revocation**: fetch `/.well-known/revoked-kids.json`. Reject if, for any entry whose `kid` is the receipt's `kid`, the receipt's `signed_at` ≥ that entry's `effective_revocation_at`, or either time is not in the receipt timestamp form ([`RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) §6.1).
3. **JCS-canonicalise** `receipt.payload`. (Use `canonicalize` in JS, `sum_engine_internal/infrastructure/jcs.py` in Python — both are byte-identical.)
4. **Verify the JWS** as detached over those canonical bytes. First require the protected header, the signature and the key's `x` to be canonical unpadded base64url (base64url characters only, no `=`, spare low bits of the last character zero; signature 86 characters, `x` 43): `flattenedVerify` accepts a respelled or padded signature, WebCrypto a respelled or padded `x`, and a header the issuer signed in such a spelling verifies; SUM's own verifiers reject all of them. Also require the key's other members to allow EdDSA verification (no private member `d`; `key_ops`, when present, distinct strings that include `verify`; `use`, when present, `sig`; `alg`, when present, `EdDSA` or `Ed25519`), and build the public key from `kty`, `crv` and `x` alone: some JOSE libraries build the key from `d` when it is present, without comparing it with `x`. Apply the protected-header rule of [`RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) §2.1 step 3 before verifying (at most 512 characters, UTF-8 JSON without a BOM, NaN or Infinity, the registered header names' types, `b64` false and `crit` including `b64`), and refuse a canonical payload over 128,000 bytes: `flattenedVerify` checks none of these. In Node:
   ```js
   import { flattenedVerify, importJWK } from "jose";
   const publicKey = await importJWK({ kty: jwk.kty, crv: jwk.crv, x: jwk.x }, "EdDSA");
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
| 400 | Body missing, not valid JSON or UTF-8, or not an object | `{ "error": "invalid JSON body" }` or `{ "error": "body must be an object" }` |
| 400 | `triples` missing, empty or over 256 items | `{ "error": "triples must contain 1 to 256 items" }` |
| 400 | A triple that is not three nonempty strings of at most 512 characters | `{ "error": "each triple must contain three nonempty strings of at most 512 characters" }` |
| 400 | A triple string contains a lone UTF-16 surrogate (not well-formed Unicode) | `{ "error": "triple text must be well-formed Unicode (no lone surrogates)" }` |
| 400 | `slider_position` missing or not an object | `{ "error": "sliders must be an object" }` |
| 400 | A slider axis missing, not a finite number, or outside `[0, 1]` | `{ "error": "<axis> must be a finite number in [0, 1]" }` |
| 400 | `provider`, `force_render` or `cache_ttl_seconds` invalid | `{ "error": "provider must be anthropic or openai" }`, `{ "error": "force_render must be boolean" }` or `{ "error": "cache_ttl_seconds must be an integer from 60 to 86400" }` |
| 405 | Non-POST method | `{ "error": "method not allowed; use POST" }` |
| 408 | Body not fully received within 10 seconds | `{ "error": "JSON body read timed out" }` |
| 413 | Body over 192 KiB (196,608 bytes), or triple text over 40,000 characters in all | `{ "error": "JSON body exceeds 196608 bytes" }` or `{ "error": "triple text exceeds 40000 characters" }` |
| 429 | Canonical path: the per-IP limit of 100 an hour; LLM path: generation admission refused (§2.7) | `{ "error": "rate limit exceeded", ... }`, `{ "error": "shared operator daily budget exhausted", ... }` or `{ "error": "generation capacity busy; retry shortly" }` |
| 502 | LLM provider failure | `{ "error": "render failed: <reason>", "cache_key": "<key>" }`, where the reason is, for example, `provider returned HTTP 529` |
| 503 | LLM path with no provider key configured, or with a named `provider` whose key is missing, or generation admission unavailable | `{ "error": "render requires at least one LLM provider configured (ANTHROPIC_API_KEY or OPENAI_API_KEY)" }`, `{ "error": "no credential configured for <provider>" }` or `{ "error": "generation admission unavailable; retry later or use canonical rendering" }` |

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
| `model` | `string` | no | If given, it must be the configured model (by default `claude-haiku-4-5-20251001` for Anthropic, `gpt-4o-mini` for OpenAI); a value that is not a string is a 400, and another string is a 400 on a deployment with a provider key (one without returns the 503 in §4.3). |

### 4.2 Response (200)

```json
{
  "completion": "Generated text from the LLM.",
  "source":     "anthropic",
  "model":      "claude-haiku-4-5-20251001"
}
```

`source` is the provider that actually answered (Anthropic takes priority if both keys are set). `model` is the configured model the request ran with.

### 4.3 Errors

| Status | Trigger |
|---|---|
| 400 | Invalid JSON or UTF-8, a body that is not an object, missing/empty `prompt`, a `prompt` containing a lone UTF-16 surrogate (not well-formed Unicode), or a `model` that is not a string or not the configured one |
| 408 | Body not fully received within 10 seconds |
| 413 | `prompt` over 40 000 UTF-16 code units, or a body over 192 KiB (196,608 bytes) |
| 429 | Generation admission refused: the per-IP demo allowance, the shared daily budget, or calls in flight |
| 405 | Non-POST method |
| 502 | Upstream provider failure; the body is `{ "error": "upstream failure: <reason>" }`, where the reason is, for example, `provider returned HTTP 529` or `anthropic: empty completion` |
| 503 | Neither `ANTHROPIC_API_KEY` nor `OPENAI_API_KEY` is configured (checked after the `prompt` checks above), or generation admission is unavailable. The demo's JS treats any non-2xx as no claims; the local comparison stays usable. |

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
| `terms[].text` | `string` | yes | 1 to 200 UTF-16 code units, not trimmed. Missing, empty or well-formed longer text is a per-term result, not an error; a lone surrogate is a 400 at any length (§5.3). |
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
| 400 | Invalid JSON or UTF-8, a body that is not an object, or missing/empty `terms` |
| 400 | A term that is not an object (`null`, a string, a number) |
| 400 | `kind` set, and not `null`, `"item"` or `"property"` |
| 400 | `lang` set, and neither `null` nor shaped like a language code (2 to 8 letters with optional hyphenated subtags, such as `en` or `zh-hans`) |
| 400 | `text` contains a lone UTF-16 surrogate (not well-formed Unicode), at any length |
| 405 | Non-POST method |
| 408 | Body not fully received within 10 seconds |
| 413 | Body over 192 KiB (196,608 bytes) |
| 413 | More than 50 terms |
| 429 | The per-IP limit of 60 an hour (scope `qid`) |

Every term is checked before any lookup, so one bad term refuses the whole request. A `kind` or `lang` of `null` means the default (`item`, `en`). A term whose `text` is missing or empty, or well-formed and over 200 UTF-16 code units, is not an error: it comes back with `id: null` and `reason` `empty-text` or `text-too-long` in a 200 response.

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
      "kid": "sum-render-2026-04-27-1",
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

## 8. `POST /api/transform`

Generic transform-registry dispatch — the substrate sibling of `/api/render`, emitting `sum.transform_receipt.v1` instead of `sum.render_receipt.v1`. Same signing path (Ed25519 / JCS / detached JWS), same per-IP rate limiting and BYO-key headers as `/api/render`.

**Worker registry scope (important).** The Worker registers the **`slider`** transform only, and only its **canonical (deterministic) path** — `density` quantization with all four LLM axes at the neutral `0.5`. The `extract` and `compose` transforms are **Python-CLI-only** (`sum transform apply extract|compose`); they are not on the Worker. A `slider` request with any LLM axis off-centre returns **`501`** (the Worker's LLM-axis dispatch is not yet wired) — use [`/api/render`](#3-post-apirender) for LLM-conditioned renders, which produces a render receipt rather than a transform receipt.

### 8.1 Request

```json
POST /api/transform HTTP/1.1
content-type: application/json

{
  "transform": "slider",
  "input":      { "...": "transform-specific" },
  "parameters": { "...": "transform-specific" },
  "source_chain": [
    { "claim": "Alice graduated in 2012",
      "provenance": { "source_uri": "src1.txt", "byte_start": 0, "byte_end": 42 } }
  ]
}
```

`source_chain` is optional (T4): when present, the receipt's `source_chain_hash` binds the receipt to specific source byte ranges (`docs/TRANSFORM_RECEIPT_FORMAT.md` §1.1). BYO-key headers `X-Render-LLM-Key-Anthropic` / `X-Render-LLM-Key-OpenAI` are honoured (falling back to operator keys), same as `/api/render`.

### 8.2 Response (200)

```json
{
  "output":            { "...": "transform-specific" },
  "transform_id":      "a1b2c3d4e5f60718",
  "wall_clock_ms":     842,
  "model":             "claude-haiku-4-5-20251001",
  "provider":          "anthropic",
  "transform_receipt": { "schema": "sum.transform_receipt.v1", "kid": "…", "payload": { "…": "…" }, "jws": "…" }
}
```

`transform_receipt` is **absent** when the Worker has no signing config (`RENDER_RECEIPT_SIGNING_JWK` / `_KID`) — same behaviour as `/api/render`. Wire spec: [`docs/TRANSFORM_RECEIPT_FORMAT.md`](TRANSFORM_RECEIPT_FORMAT.md).

### 8.3 Errors

| Status | When |
|---|---|
| `400` | invalid JSON or UTF-8, a body that is not an object, a missing or unknown `transform` name, or a missing `input` (`input must be an object`) |
| `400` | `triples` empty or over 256 items (`triples must contain 1 to 256 items`), a triple that is not three nonempty strings of at most 512 characters, `parameters` not an object or an axis that is not a finite number in [0, 1] |
| `400` | a `source_chain` that is not an array or holds more than 256 records (`source_chain must contain at most 256 records`), a record that is not an object (`source_chain record must be an object`), a record whose `provenance` is missing or not an object (`provenance must be an object`), or a malformed record (`invalid source_chain record`) |
| `400` | a triple, a `source_chain` claim or a `source_uri` containing a lone UTF-16 surrogate (not well-formed Unicode) |
| `405` | non-`POST` method |
| `408` | body not fully received within 10 seconds |
| `413` | body over 192 KiB (196,608 bytes), or triple text over 40,000 characters in all (`triple text exceeds 40000 characters`) |
| `429` | the per-IP limit of 100 an hour (scope `canonical`, the limit for credential-free rendering) |
| `501` | transform recognised but the requested path is not yet wired on the Worker (e.g. `slider` with an off-centre LLM axis); the error message contains `"not yet wired"`. Use `/api/render` for that case. |

---

## 9. Operator: configuring a self-hosted Worker

The hosted demo at `sum-demo.ototao.workers.dev` is one deployment. To run your own:

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
| `ANTHROPIC_API_KEY` | secret | `/api/render` (LLM path) and `/api/complete` fall back to OpenAI when `OPENAI_API_KEY` is set; an `/api/render` request that names `provider: "anthropic"` returns 503 unless it carries `X-Render-LLM-Key-Anthropic` |
| `OPENAI_API_KEY` | secret | with Anthropic also missing, `/api/complete` returns 503, and so does the `/api/render` LLM path unless the request carries a BYO key header (`X-Render-LLM-Key-Anthropic` or `-OpenAI`); an `/api/render` request that names `provider: "openai"` returns 503 unless it carries `X-Render-LLM-Key-OpenAI` |
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

## 10. Working integration examples

### 10.1 Render + verify in Node

```js
import { flattenedVerify, importJWK } from "jose";
import canonicalize from "canonicalize";
import { createHash } from "node:crypto";
// The receipt timestamp parser SUM's verifiers use (a strict profile of
// RFC 3339, RENDER_RECEIPT_FORMAT.md §6.1): copy
// single_file_demo/receipt_verifier.js with its vendor/ folder.
import { parseInstantMs } from "./receipt_verifier.js";

const BASE = "https://sum-demo.ototao.workers.dev";

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

  // 3. Revocation. Every entry for this kid counts, whatever the list order,
  //    and a time not in the receipt timestamp form fails closed, as in SUM's
  //    verifiers. Compare instants, not strings ("...16.849Z" sorts before
  //    "...16Z"); Date.parse also reads timezone-less and other forms, and
  //    its NaN compares false, which would accept.
  const revoked = await (await fetch(`${BASE}/.well-known/revoked-kids.json`)).json();
  for (const entry of revoked.revoked.filter((r) => r.kid === kid)) {
    const signed = parseInstantMs(payload.signed_at);
    const effective = parseInstantMs(entry.effective_revocation_at);
    if (signed === null || effective === null || signed >= effective) {
      return { tome: result.tome, verified: false, reason: "kid revoked" };
    }
  }

  // 4. Verify the detached JWS over the canonical payload bytes. Before this,
  //    require protectedHeader, signature and jwk.x to be canonical unpadded
  //    base64url, and the key to carry no d and only key_ops, use and alg
  //    values that allow EdDSA verification, and apply the protected-header
  //    rule and the 128,000-byte payload limit, as SUM's verifiers do (see
  //    the format docs, steps 2 and 3).
  const canonicalBytes = new TextEncoder().encode(canonicalize(payload));
  const [protectedHeader, , signature] = jws.split(".");
  // Build the key from x alone, so a d or other member cannot change it.
  const key = await importJWK({ kty: jwk.kty, crv: jwk.crv, x: jwk.x }, "EdDSA");
  await flattenedVerify(
    { protected: protectedHeader, payload: canonicalBytes, signature },
    key,
    { algorithms: ["EdDSA"] }
  );

  // 5. Re-derive hashes; compare
  // Componentwise tuple-lex sort by Unicode CODE POINT — matches Python's
  // `sorted(tuple(t) for t in triples)` byte-for-byte, and matches the
  // Worker's hashTriples helper (worker/src/receipt/sign.ts, which sorts
  // with compareTriples from worker/src/unicode_order.ts).
  //
  // Do NOT use `<` or default `.sort()` here. JavaScript compares UTF-16
  // code units, Python compares code points, and the two disagree whenever
  // a supplementary character (U+10000 and up) meets a BMP character above
  // U+DFFF — fullwidth forms, U+FFFD, CJK compatibility ideographs. Getting
  // this wrong makes your recomputed triples_hash differ from the signed
  // one on exactly that text, while every ASCII fixture still passes.
  const cmpCodepoints = (left, right) => {
    const a = Array.from(left, (c) => c.codePointAt(0));
    const b = Array.from(right, (c) => c.codePointAt(0));
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      if (a[i] !== b[i]) return a[i] - b[i];
    }
    return a.length - b.length;
  };
  const sortedTriples = [...triples].map((t) => [...t]).sort((a, b) => {
    for (let i = 0; i < 3; i++) {
      const order = cmpCodepoints(a[i], b[i]);
      if (order) return order;
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

### 10.2 Resolve QIDs in Python

```python
import httpx

def resolve_qids(terms: list[str], lang: str = "en") -> dict[str, str | None]:
    body = {"terms": [{"text": t, "kind": "item", "lang": lang} for t in terms]}
    r = httpx.post("https://sum-demo.ototao.workers.dev/api/qid", json=body, timeout=10.0)
    r.raise_for_status()
    return {row["text"]: row.get("id") for row in r.json()["resolved"]}
```

### 10.3 Render-only (no verification) in Python

```python
import httpx

def render(triples, sliders) -> dict:
    r = httpx.post("https://sum-demo.ototao.workers.dev/api/render", json={
        "triples": triples,
        "slider_position": sliders,
    }, timeout=30.0)
    r.raise_for_status()
    return r.json()  # dict with tome, drift, render_receipt, ...
```

For Python receipt verification, use `sum_engine_internal.render_receipt.verifier.verify_receipt()` from this repo (`pip install sum-engine[receipt-verify]`). It accepts the same receipt bytes the JS verifier does.

---

## 11. Cross-references

- [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) — wire spec for `sum.render_receipt.v1`. Source of truth for the receipt payload schema and the six-step verifier algorithm.
- [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) §1.3.1 — what the cross-runtime Ed25519 trust triangle proves and what it doesn't.
- [`docs/MCP_INTEGRATION.md`](MCP_INTEGRATION.md) — the MCP server companion to this HTTP API. Use MCP for local LLM clients, this HTTP API for everything else.
- [`docs/INCIDENT_RESPONSE.md`](INCIDENT_RESPONSE.md) — operator runbook for kid revocation (case 1), JWKS rollback, and related contingencies.
- [`docs/SLIDER_CONTRACT.md`](SLIDER_CONTRACT.md) — the 5-axis slider contract: per-axis drift formulas, fact-preservation thresholds, what each axis controls.
- [`docs/COMPATIBILITY_POLICY.md`](COMPATIBILITY_POLICY.md) — schema-version policy. The `sum.*.v1` schemas in this doc are pinned; future incompatible changes ship as `v2`.
- [`worker/src/index.ts`](../worker/src/index.ts), [`worker/src/routes/`](../worker/src/routes/) — the implementation. If this doc and the code disagree, the code wins; please file an issue.
