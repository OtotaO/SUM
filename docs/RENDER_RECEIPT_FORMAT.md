# Render Receipt Format (v0.9.A wire spec)

**Status:** stable; produced by Phase E.1 v0.9.A `worker/src/receipt/sign.ts`.
**Schema identifier:** `sum.render_receipt.v1`
**Cryptography:** Ed25519 (RFC 8032) over JCS-canonical bytes (RFC 8785), wrapped as a detached JWS (RFC 7515 §A.5) with public keys distributed via JWKS (RFC 7517).

A render receipt is a signed event log: "this Worker rendered this tome from these triples at these slider positions at this time, using this model, on this provider." It is NOT a truth claim about the tome's content — that's the bench's NLI audit's job, not the receipt's. Read §5 (Trust Scope) before assuming what a verified receipt proves.

The bar for this document: a consumer should be able to implement a working receipt verifier from this spec without reading the Worker source. If you find yourself reaching for `worker/src/receipt/sign.ts`, that's a defect in this spec — please file an issue.

---

## 1. Wire format

Every successful `/api/render` response carries a `render_receipt` field with this shape:

```json
{
  "render_receipt": {
    "schema": "sum.render_receipt.v1",
    "kid": "sum-render-2026-04-26-1",
    "payload": {
      "render_id": "<sha256-trunc-16-hex>",
      "sliders_quantized": {
        "audience": 0.5,
        "density": 1.0,
        "formality": 0.5,
        "length": 0.5,
        "perspective": 0.5
      },
      "triples_hash": "sha256-<hex>",
      "tome_hash": "sha256-<hex>",
      "model": "claude-haiku-4-5-20251001",
      "provider": "anthropic",
      "signed_at": "2026-04-26T14:23:01.000Z",
      "digital_source_type": "trainedAlgorithmicMedia"
    },
    "jws": "<protected-b64>..<signature-b64>"
  }
}
```

The four top-level keys are siblings, not nested:

| Field | Type | Purpose |
|---|---|---|
| `schema` | string | Schema identifier. v1 today; future versions add new identifiers, never mutate this value's meaning. |
| `kid` | string | Key ID matching one entry in the issuer's JWKS at `/.well-known/jwks.json`. Used by the verifier to pick the right public key. |
| `payload` | object | The signed-over object. Verifier JCS-canonicalises this exact object and verifies `jws` against the resulting bytes. |
| `jws` | string | Detached JWS: `<protected-header-base64url>..<signature-base64url>`. Middle segment is empty (the JCS-canonical payload bytes are the detached payload). |

### 1.1 Payload field semantics

| Field | Type | Source of truth |
|---|---|---|
| `render_id` | string | First 16 hex chars of `sha256(cache_key ‖ tome_bytes)`. Stable per (triples, sliders, tome) tuple; not a freshness token. |
| `sliders_quantized` | object | The post-quantize slider values. Density passes through unchanged; LLM axes are snapped to bin centres `{0.1, 0.3, 0.5, 0.7, 0.9}`. See [`SLIDER_CONTRACT.md`](SLIDER_CONTRACT.md). |
| `triples_hash` | string | `sha256-<hex>` of JCS-canonicalised triples. Sort order: **componentwise tuple-lex** (matches Python's default `sorted()` on tuples) — NOT a string-join-sort. The componentwise rule keeps separator characters from leaking into the comparison space when triple components contain `|`. Cross-runtime byte-stable: Python `jcs` on `sorted(tuple(t) for t in triples)` and TypeScript `canonicalize` on the componentwise-sorted array produce the same bytes. |
| `tome_hash` | string | `sha256-<hex>` of the tome's UTF-8 bytes. Verifier rehashes the response's `tome` field and compares. |
| `model` | string | The model that ACTUALLY served the call, taken from the LLM API response's reported `model` field — NOT the configured-default. May be a more specific snapshot id than the requested tag (e.g., Anthropic resolves `claude-haiku-4-5-20251001` → a dated snapshot). When the API doesn't echo a model, the value is `<requested>_inferred` so the inference itself is visible. For canonical-path renders (no LLM), the value is the literal string `canonical-deterministic-v0`. |
| `provider` | string | One of the values enumerated in §1.2. |
| `signed_at` | string | ISO-8601 UTC timestamp at issuance time. See §1.3 for cache-HIT durability semantics. |
| `digital_source_type` | string | C2PA `digitalSourceType` per spec.c2pa.org v2.2. See §7. |

### 1.2 Provider taxonomy

The `provider` field distinguishes how the tome was produced:

| Value | Meaning |
|---|---|
| `anthropic` | Direct call to `https://api.anthropic.com/v1/messages`. |
| `cf-ai-gateway-anthropic` | Call routed through Cloudflare AI Gateway (`CF_AI_GATEWAY_BASE` env was set at issuance). The Anthropic model still served; the gateway is a transparent proxy with caching and observability. |
| `openai` | Reserved for future fallback. v0.9.A does not produce this value; the field's union includes it as a forward-compat extension point. |
| `canonical-path` | No LLM call. The tome was produced by `deterministicTome(triples)` — pure algorithmic prose composition from the post-density triple set. Pairs with `digital_source_type: "algorithmicMedia"`. |

A consumer MUST treat unknown provider values as opaque metadata (signed, but interpretable only by issuer-aware tools). See §1.4 forward-compat policy.

### 1.3 Cache-HIT durability

Receipts are stamped at issuance time — i.e., during the original cache miss. Subsequent cache HITs on the same key serve the receipt verbatim, INCLUDING the original `signed_at`. This means:

- A response returned today may carry `signed_at` from hours or days ago. That is correct, not stale: the receipt asserts "I, the issuer, attested to this render at the timestamp shown."
- The receipt's `kid` MUST remain queryable in JWKS as long as cached responses can be returned with that kid. See §6 rotation cadence.
- A consumer requiring freshness (e.g., for time-bound trust) MUST compare `signed_at` against their own clock and reject receipts older than their freshness window. The receipt does not enforce a freshness policy.

### 1.4 Forward-compat policy

The v1 schema is stable. Future revisions:
- MAY add new fields to `payload`. Verifiers MUST treat unknown fields as opaque-but-signed metadata.
- MUST NOT remove or rename existing fields.
- MUST bump the `schema` identifier to `sum.render_receipt.v2` (etc.) for breaking changes.
- New protected-header fields that change verification semantics MUST land in the JWS `crit` array, forcing older verifiers to fail closed per RFC 7515 §4.1.11.

A v1-aware verifier will continue to verify v1 receipts forever. A v1-aware verifier that encounters a v2 receipt will succeed if the protected header's `crit` array is empty or contains only fields the verifier understands; otherwise it MUST fail closed.

---

## 2. Verifier algorithm

The verifier needs three inputs:
- The full `render_receipt` block from a `/api/render` response.
- The issuer's JWKS, fetched once from `/.well-known/jwks.json` and cached per the response's `Cache-Control: max-age` header (default 3600 seconds).
- A JOSE library that supports detached JWS verification with `b64: false` (RFC 7797).

### 2.1 Six-step procedure

```
1. Parse receipt.kid. Look up the matching key in jwks.keys by kid.
   If no match → REJECT (unknown key id).

2. Take receipt.payload (the object, not the string). JCS-canonicalise
   it (RFC 8785). Encode to UTF-8 bytes. This is the detached payload.

3. Split receipt.jws on "." → [protected_b64, middle, signature_b64].
   The middle segment MUST be empty (detached payload encoding).
   If middle is non-empty → REJECT (malformed JWS).

4. Construct the flattened JWS object:
   { protected: protected_b64, payload: <JCS bytes from step 2>, signature: signature_b64 }

5. Verify with jose.flattenedVerify(flattened, key). Catch
   ERR_JWS_SIGNATURE_VERIFICATION_FAILED → REJECT.

6. Inspect the verified protected header. It MUST contain:
     alg: "EdDSA"
     kid: <matching receipt.kid>
     b64: false
     crit: ["b64"]
   Any deviation → REJECT (header tampering).
```

Note the choice of `flattenedVerify` (not `compactVerify`): for `b64: false` detached payloads, the flattened API accepts the protected/payload/signature as separate inputs without requiring the consumer to manually splice the unencoded payload into a compact serialisation. Cleaner contract.

### 2.2 Positive example (TypeScript / Node)

```typescript
import { flattenedVerify, createRemoteJWKSet } from "jose";
import canonicalize from "canonicalize";

const JWKS = createRemoteJWKSet(
  new URL("https://sum-demo.ototao.workers.dev/.well-known/jwks.json")
);

async function verifyReceipt(receipt: any): Promise<boolean> {
  const canonicalBytes = new TextEncoder().encode(canonicalize(receipt.payload)!);
  const [protected_b64, middle, signature_b64] = receipt.jws.split(".");
  if (middle !== "") throw new Error("not a detached JWS");
  const result = await flattenedVerify(
    { protected: protected_b64, payload: canonicalBytes, signature: signature_b64 },
    JWKS
  );
  // result.protectedHeader.{alg, kid, b64, crit} are inspectable.
  return result.protectedHeader.alg === "EdDSA"
      && result.protectedHeader.kid === receipt.kid
      && result.protectedHeader.b64 === false
      && Array.isArray(result.protectedHeader.crit)
      && result.protectedHeader.crit.includes("b64");
}
```

### 2.3 Negative example — tampered receipt

A consumer that mutates ANY signed field MUST get a verification error. Concrete:

```typescript
// Take a valid receipt and tamper with the tome_hash.
const tampered = JSON.parse(JSON.stringify(originalReceipt));
tampered.payload.tome_hash = "sha256-deadbeef" + tampered.payload.tome_hash.slice(15);

try {
  await verifyReceipt(tampered);
  throw new Error("should not have verified");
} catch (e) {
  // Expected: ERR_JWS_SIGNATURE_VERIFICATION_FAILED
  console.log(e.code); // "ERR_JWS_SIGNATURE_VERIFICATION_FAILED"
}
```

Same expected error if `sliders_quantized.formality` is changed, if any character of `model` is altered, if `signed_at` is shifted, if `digital_source_type` is rewritten, etc. Every payload field is signed; tampering any one bit invalidates the signature.

---

## 3. Library recommendations

| Runtime | JOSE | JCS |
|---|---|---|
| TypeScript / JavaScript (Node ≥ 18, browsers, Cloudflare Workers, Deno, Bun) | [`jose@>=6`](https://github.com/panva/jose) (panva, MIT) | [`canonicalize@>=3`](https://www.npmjs.com/package/canonicalize) (Erdtman, Apache 2.0) |
| Python (3.10+) | [`joserfc`](https://pypi.org/project/joserfc/) or [`authlib`](https://pypi.org/project/Authlib/) (both active, both EdDSA + detached JWS) | [`jcs`](https://pypi.org/project/jcs/) (Erdtman, Apache 2.0) |
| Go | `github.com/go-jose/go-jose/v3` | `github.com/cyberphone/json-canonicalization/go/src/webpki.org/jsoncanonicalizer` |
| Rust | `josekit` or `jsonwebtoken` | `serde-jcs` |

The TypeScript pair is what the Worker uses to produce receipts; the Python pair is what `sum_engine_internal/` uses to verify them in the bench (v0.9.C). All four pairs are tested against the JCS interop suite and produce byte-identical canonical bytes for the receipt's payload shape — Probe 3 of the v0.9.A research pass confirmed 10/10 edge cases byte-identical between TS `canonicalize@3` and Python `jcs`.

Avoid: hand-rolled JSON canonicalisers (the integer-vs-float-zero edge case bites every implementation), and JOSE libraries that don't support `b64: false` per RFC 7797 (most modern libraries do; check before committing).

---

## 4. Cross-runtime canonicalisation rule

**The single most surprising rule:** JCS normalises integer-valued floats. `1.0` becomes `1`. `1.00` becomes `1`. `-0` becomes `0`. This follows RFC 8785's adoption of ECMAScript's `Number.prototype.toString` algorithm (RFC 8785 §3.2.2.3).

For the slider's `density: 1.0`, this means the canonical signed-payload bytes contain `"density":1`, not `"density":1.0`.

**The correctness implication:** consumers MUST run the receipt's `payload` through their language's JCS implementation before computing the signed-over bytes. A naive `JSON.stringify(receipt.payload)` will diverge on this exact rule and the signature will fail to verify.

Concrete:

```javascript
// WRONG — diverges from issuer canonicalisation
const bytes_wrong = new TextEncoder().encode(JSON.stringify(receipt.payload));

// RIGHT
import canonicalize from "canonicalize";
const bytes_right = new TextEncoder().encode(canonicalize(receipt.payload));

// Same input, different bytes:
//   JSON.stringify({a: 1.0}) === '{"a":1}'   on V8 (incidentally correct)
//   JSON.stringify({a: 1.5}) === '{"a":1.5}' (also correct)
//   JSON.stringify({b: 1, a: 2}) === '{"b":1,"a":2}'   (WRONG — keys not sorted)
//   canonicalize({b: 1, a: 2})    === '{"a":2,"b":1}'  (right)
```

Property-sorted, then each value canonicalised per the RFC. Strings are UTF-8 with the standard JSON escapes; arrays preserve order; objects sort keys lexicographically.

If your verifier's bytes don't match the issuer's bytes, your signature won't verify even with a correct key. The canonicalisation step is where ~all verifier bugs hide.

---

## 5. Trust scope

A verified receipt PROVES:

| Claim | Defence mechanism |
|---|---|
| The issuer (holder of the `kid`'s private key) attested to this render. | Ed25519 signature; verified with public JWK at the kid. |
| The tome bytes match `tome_hash` at issuance time. | `tome_hash` is signed; consumer rehashes response.tome and compares. |
| The slider position is `sliders_quantized`. | Signed field. |
| The triple set hash is `triples_hash`. | Signed field. Note: this is the post-density set, not the pre-density input. |
| The model that served was `model` on provider `provider`. | Signed; sourced from the API's reported `model`, not the configured-default — see §1.1. |
| The render was issued no later than `signed_at`. | Signed timestamp; only as trustworthy as the issuer's clock. |
| The tome is AI-generated (or canonical-path). | `digital_source_type` per C2PA — see §7. |

A verified receipt DOES NOT PROVE:

| Non-claim | Why not |
|---|---|
| The tome's content is factually accurate. | The receipt is a render attestation, not a truth oracle. Fact preservation across slider axes is verified by the bench's NLI audit (see [`SLIDER_CONTRACT.md`](SLIDER_CONTRACT.md)). |
| The render is fresh. | `signed_at` is at issuance, not retrieval. Cache-HIT responses serve old timestamps verbatim — see §1.3. Consumers requiring freshness must enforce it themselves. |
| The issuer is honest. | Receipt verifies the issuer signed it; if the issuer's environment was misconfigured (wrong model in `SUM_DEFAULT_MODEL_ANTHROPIC`), the receipt records the misconfiguration faithfully — including the misconfigured value. The receipt is honest about what the issuer asserted; it does not validate the issuer's beliefs. |
| The tome was generated FROM these triples. | Hashes are consistent at issuance time. The receipt asserts "I observed this tome and these triples together"; the relationship between them (LLM-rendered, canonical-path, etc.) is asserted by `provider` + `digital_source_type` but not cryptographically bound. |

### 5.1 Threat model

**Defends against:**
- *Tome substitution.* The signed `tome_hash` lets verifiers detect any byte change to the rendered prose.
- *Slider lying.* `sliders_quantized` is signed; consumer can reject if the slider claim doesn't match what they requested.
- *Model claim drift.* `model` + `provider` are sourced from the served call, not configuration.
- *Replay across renders.* `render_id` is content-addressed; replaying receipt A against tome B will fail at the rehash step.

**Does NOT defend against:**
- *Issuer key compromise.* If the private key leaks, an attacker can forge receipts. Mitigation: rotate (§6) and monitor for anomalous receipt issuance. Not eliminated.
- *Freshness replay.* An old, valid receipt remains a valid receipt — that's the durability semantics. Consumers requiring freshness must check `signed_at` against their own clock and reject older than their window. Receipt does not carry a freshness token.
- *Issuer collusion.* If the issuer signs intentionally false claims, the receipt is internally consistent but externally wrong. This is the trust-the-issuer assumption every signature scheme makes.

### 5.2 Privacy

Receipts carry hashes only — never raw triples or tome bytes. `tome_hash` is preimage-resistant (sha-256); the receipt cannot be used to recover the rendered content.

This means receipts are SAFE to:
- Log to plaintext audit trails.
- Share publicly (e.g., as a "this content was AI-generated, here's the proof" attestation).
- Archive long-term without exposing the underlying content.

The receipt is also a privacy-preserving alternative to logging full responses: a downstream system can keep the receipt forever and discard the response body, retaining the ability to verify (against a future re-fetch) without retaining the prose itself.

---

## 6. Key rotation

Rotation is the standard JWKS pattern:

1. Generate a new keypair with `npx tsx scripts/cert/gen_render_keypair.ts <new-kid>`.
2. ADD the new public JWK to the existing JWKS — DO NOT replace. The JWKS now has two `keys[]` entries.
3. Upload the new private JWK as the secret: `wrangler secret put RENDER_RECEIPT_SIGNING_JWK < /tmp/render_receipt_private.jwk`
4. Update `RENDER_RECEIPT_SIGNING_KID` to the new kid.
5. Wipe the tempfile: `rm -P /tmp/render_receipt_private.jwk` (macOS / BSD) or `shred -u` (GNU/Linux).
6. Deploy. New renders sign with the new kid; old receipts continue to verify against the old kid in JWKS.
7. After the rotation grace window passes (see below), remove the old kid from JWKS and redeploy.

**Rotation cadence ≥ render cache TTL.** The render cache TTL is 24 hours by default (`bin_cache.ts::DEFAULT_TTL_SECONDS`). A kid rotated out of JWKS sooner than 24 hours could leave cached responses serving receipts whose key is no longer queryable, breaking verifiers.

For extended trust windows (e.g., long-term audit logs), the grace window is whatever your downstream consumers expect — there is no fixed number. The conservative rule: never remove a kid from JWKS while any consumer might still try to verify a receipt that references it.

---

## 7. C2PA `digital_source_type` alignment

The `digital_source_type` field uses the [C2PA AI/ML specification v2.2](https://spec.c2pa.org/specifications/specifications/2.2/ai-ml/ai_ml.html) `digitalSourceType` taxonomy. This aligns SUM render receipts with the broader content-provenance ecosystem (Adobe, Microsoft, OpenAI, Google) that uses C2PA for AI-vs-human content labelling.

| Value | Meaning | When emitted |
|---|---|---|
| `trainedAlgorithmicMedia` | Generated by a trained AI model. | LLM-path renders (any non-default LLM axis). |
| `algorithmicMedia` | Generated by a deterministic algorithm — not a trained model. | Canonical-path renders (all LLM axes at neutral, only density non-default). |

Both labels are honest about what produced the bytes. A consumer reading a receipt cannot mistake an AI-generated tome for a deterministic one or vice versa. C2PA-aware tooling can correctly attribute AI-generated content without SUM-specific knowledge.

---

## Appendix A: Worked end-to-end example

This is a complete trace from JWKS publication through receipt verification. Cryptographic byte values (`x`, signature) are illustrative — your run will produce different bytes, but the structural shapes will match exactly.

### A.1 Issuer's JWKS at `/.well-known/jwks.json`

```json
{
  "keys": [
    {
      "kty": "OKP",
      "crv": "Ed25519",
      "alg": "EdDSA",
      "use": "sig",
      "kid": "sum-render-2026-04-26-1",
      "x": "11qYAYKxCrfVS_7TyWQHOg7hcvPapiMlrwIaaPcHURo"
    }
  ]
}
```

Headers:
```
HTTP/2 200
content-type: application/jwk-set+json
cache-control: public, max-age=3600
```

### A.2 Render request

```http
POST /api/render HTTP/1.1
Host: sum-demo.ototao.workers.dev
Content-Type: application/json

{
  "triples": [
    ["alice", "graduated", "2012"],
    ["alice", "born", "1990"]
  ],
  "slider_position": {
    "density": 1.0,
    "length": 0.5,
    "formality": 0.7,
    "audience": 0.5,
    "perspective": 0.5
  }
}
```

### A.3 Render response (truncated for clarity)

```json
{
  "tome": "Alice was born in 1990. She graduated in 2012.",
  "triples_used": [["alice", "born", "1990"], ["alice", "graduated", "2012"]],
  "drift": [],
  "cache_status": "miss",
  "llm_calls_made": 1,
  "wall_clock_ms": 1453,
  "quantized_sliders": {
    "density": 1.0,
    "length": 0.5,
    "formality": 0.7,
    "audience": 0.5,
    "perspective": 0.5
  },
  "render_id": "e34df444f6ea1c92",
  "render_receipt": {
    "schema": "sum.render_receipt.v1",
    "kid": "sum-render-2026-04-26-1",
    "payload": {
      "render_id": "e34df444f6ea1c92",
      "sliders_quantized": {
        "audience": 0.5,
        "density": 1.0,
        "formality": 0.7,
        "length": 0.5,
        "perspective": 0.5
      },
      "triples_hash": "sha256-7d3decf66362edbafbb397c1aa5af525e76df3bd666128fcc53aa3baf42e4618",
      "tome_hash": "sha256-7979a8d1f307bd269a7ea7fb2ecfc121ef80a08fc5b7dfbdb07841d3aabb6b63",
      "model": "claude-haiku-4-5-20251001",
      "provider": "anthropic",
      "signed_at": "2026-04-26T14:23:01.000Z",
      "digital_source_type": "trainedAlgorithmicMedia"
    },
    "jws": "eyJhbGciOiJFZERTQSIsImtpZCI6InN1bS1yZW5kZXItMjAyNi0wNC0yNi0xIiwiYjY0IjpmYWxzZSwiY3JpdCI6WyJiNjQiXX0..AbCdEf1234...XYZ"
  }
}
```

### A.4 JCS-canonical bytes of the payload

The `payload` object from §A.3, run through `canonicalize()`, produces exactly these UTF-8 bytes (no whitespace, keys lex-sorted at every level):

```
{"digital_source_type":"trainedAlgorithmicMedia","model":"claude-haiku-4-5-20251001","provider":"anthropic","render_id":"e34df444f6ea1c92","signed_at":"2026-04-26T14:23:01.000Z","sliders_quantized":{"audience":0.5,"density":1,"formality":0.7,"length":0.5,"perspective":0.5},"tome_hash":"sha256-7979a8d1f307bd269a7ea7fb2ecfc121ef80a08fc5b7dfbdb07841d3aabb6b63","triples_hash":"sha256-7d3decf66362edbafbb397c1aa5af525e76df3bd666128fcc53aa3baf42e4618"}
```

Note `density:1` (not `1.0`) — the JCS integer-float normalisation rule from §4 applied to a real payload. The hashes are computable from §A.3's literals: `tome_hash = sha256(utf8("Alice was born in 1990. She graduated in 2012."))`, and `triples_hash = sha256(canonicalize(componentwise-tuple-lex-sorted §A.3 triples))`. Reproduce both in any JOSE-aware runtime; if your bytes differ, the canonicalisation step is the most likely culprit.

### A.5 Verification command

```typescript
// verify-receipt.ts — minimal verifier.
import { flattenedVerify, createRemoteJWKSet } from "jose";
import canonicalize from "canonicalize";

const ISSUER = "https://sum-demo.ototao.workers.dev";
const JWKS = createRemoteJWKSet(new URL(`${ISSUER}/.well-known/jwks.json`));

async function main() {
  const response = await fetch(`${ISSUER}/api/render`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      triples: [["alice", "graduated", "2012"], ["alice", "born", "1990"]],
      slider_position: { density: 1.0, length: 0.5, formality: 0.7, audience: 0.5, perspective: 0.5 },
    }),
  });
  const body = await response.json();
  const r = body.render_receipt;

  const canonical = new TextEncoder().encode(canonicalize(r.payload)!);
  const [proto, middle, sig] = r.jws.split(".");
  if (middle !== "") throw new Error("not a detached JWS");

  const result = await flattenedVerify(
    { protected: proto, payload: canonical, signature: sig },
    JWKS,
  );
  console.log("VERIFIED:", {
    kid: result.protectedHeader.kid,
    alg: result.protectedHeader.alg,
    b64: result.protectedHeader.b64,
    crit: result.protectedHeader.crit,
    payload: r.payload,
  });
}
main().catch((e) => { console.error("REJECTED:", e.code ?? e.message); process.exit(1); });
```

Expected stdout on success:
```
VERIFIED: {
  kid: 'sum-render-2026-04-26-1',
  alg: 'EdDSA',
  b64: false,
  crit: [ 'b64' ],
  payload: { ... }
}
```

### A.6 Tampered receipt — expected failure

```typescript
// Take r, mutate any signed field, re-verify.
r.payload.sliders_quantized.formality = 0.6;  // was 0.7

// Repeat steps A.5. Expected stderr:
//   REJECTED: ERR_JWS_SIGNATURE_VERIFICATION_FAILED
```

Same expected outcome from mutating any other signed field: `tome_hash`, `triples_hash`, `model`, `provider`, `signed_at`, `digital_source_type`, or any nested slider value. Every payload field is signed; tampering any one bit invalidates the signature.

If your verifier accepts a receipt with a mutated signed field, it has a defect — the canonicalisation step is the most common culprit. Re-read §4.

---

## Document history

| Version | Date | Notes |
|---|---|---|
| v0.9.A | 2026-04-26 | Initial spec landing alongside the v0.9.A Worker implementation. |
