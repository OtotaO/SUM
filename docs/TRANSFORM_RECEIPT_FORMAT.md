# Transform Receipt Format (`sum.transform_receipt.v1` wire spec)

**Status:** designed; first implementation lands with the slider migration (T1).
**Schema identifier:** `sum.transform_receipt.v1`
**Cryptography:** Ed25519 (RFC 8032) over JCS-canonical bytes (RFC 8785), wrapped as a detached JWS (RFC 7515 §A.5) with public keys distributed via JWKS (RFC 7517).
**Relationship to `sum.render_receipt.v1`:** the render receipt is the *prior art*. The transform receipt is its generalization. The slider render is the first registered transform; its receipt is a transform receipt with `transform: "slider"`. Existing render receipts remain verifiable forever; new transforms produce transform receipts.

A transform receipt is a signed event log: *"this Worker applied this transform with these parameters to this input at this time, using this model, on this provider — and the output hashes to this."* It is **not** a truth claim about the content of the output — that's the bench's NLI audit's job, not the receipt's. Read §5 (Trust Scope) before assuming what a verified transform receipt proves.

The bar for this document: a consumer must be able to implement a working transform-receipt verifier from this spec without reading the Worker source. If you find yourself reaching for `worker/src/transforms/`, that's a defect in this spec — please file an issue.

---

## 1. Wire format

Every successful `POST /api/transform` response carries a `transform_receipt` field with this shape:

```json
{
  "transform_receipt": {
    "schema": "sum.transform_receipt.v1",
    "kid": "sum-render-2026-04-27-1",
    "payload": {
      "transform_id": "<sha256-trunc-16-hex>",
      "transform": "slider",
      "parameters_hash": "sha256-<hex>",
      "input_hash": "sha256-<hex>",
      "output_hash": "sha256-<hex>",
      "model": "claude-haiku-4-5-20251001",
      "provider": "anthropic",
      "signed_at": "2026-05-12T14:23:01.000Z",
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
| `transform_id` | string | First 16 hex chars of `sha256(transform ‖ parameters_hash ‖ input_hash ‖ output_hash)`. Stable per `(transform, parameters, input, output)` tuple; not a freshness token. |
| `transform` | string | The registered transform identifier (e.g. `slider`, `extract`, `compose`, `translate`, `expand`). MUST be a member of the active transform registry; verifiers treat unknown values as opaque-but-signed metadata. See §1.2 for the v1 registry. |
| `parameters_hash` | string | `sha256-<hex>` of JCS-canonicalised parameters object. Sort: alphabetical keys; numeric values rendered with the transform's documented precision rule. Cross-runtime byte-stable: Python `jcs` and TypeScript `canonicalize` produce identical bytes given the same parameter object. |
| `input_hash` | string | `sha256-<hex>` of the input artifact's canonical-form bytes. For a `CanonicalBundle` input, this is the bundle's `state_integer ‖ canonical_tome` bytes per RFC 8785. For raw text input, this is `sha256(utf8_bytes)`. The transform's documentation pins which input shape it accepts and how the hash is computed. |
| `output_hash` | string | `sha256-<hex>` of the output artifact's canonical-form bytes. Output shape varies by transform: tome (UTF-8 bytes), tag set (JCS-canonicalised), merged bundle (state integer ‖ canonical tome), etc. The transform pins its output canonicalisation. |
| `model` | string | The model that ACTUALLY served the call, taken from the LLM API response's reported `model` field — NOT the configured-default. May be a more specific snapshot id than the requested tag. For non-LLM transforms (e.g. deterministic compose), the value is `canonical-deterministic-v0` or the transform's documented marker. |
| `provider` | string | One of the values enumerated in §1.2 (inherited from `sum.render_receipt.v1`'s provider taxonomy). |
| `signed_at` | string | ISO-8601 UTC timestamp at issuance time. See §1.3 for cache-HIT durability semantics. |
| `digital_source_type` | string | C2PA `digitalSourceType` per spec.c2pa.org v2.2. `trainedAlgorithmicMedia` for LLM-served transforms; `algorithmicMedia` for deterministic ones. |

### 1.2 Transform registry (v1)

The `transform` field MUST be one of:

| Transform id | Input shape | Output shape | Parameters | LLM-mediated? |
|---|---|---|---|---|
| `slider` | `CanonicalBundle` (post-extract) | tome string | `{density, length, formality, audience, perspective}` ∈ [0,1]⁵ | when any non-density axis ≠ 0.5 |
| `extract` | text OR `CanonicalBundle` | tag set (sorted unique triple components, JCS) | `{ontology, max_tags, multi_school}` | configurable (off by default) |
| `compose` | array of `CanonicalBundle` | merged `CanonicalBundle` (LCM-merged state integer + concatenated tome) | `{merge_strategy}` ∈ `"lcm" | "intersect" | "diff"` | no — pure Gödel-state algebra |
| `translate` *(reserved)* | tome string | tome string | `{source_lang, target_lang}` | yes |
| `expand` *(reserved)* | tag set | tome string | `{voice, target_words}` | yes |
| `restyle` *(reserved)* | tome string | tome string | `{register, genre, cultural_frame}` | yes |

Reserved entries are not yet implemented; their inclusion here pins the namespace so v1 verifiers don't mis-classify future transforms as adversarial. A verifier MUST treat any `transform` value it doesn't recognise as opaque-but-signed metadata (see §1.5 forward-compat).

### 1.3 Provider taxonomy

Same as `sum.render_receipt.v1` §1.2:

| Value | Meaning |
|---|---|
| `anthropic` | Direct call to `https://api.anthropic.com/v1/messages`. |
| `cf-ai-gateway-anthropic` | Call routed through Cloudflare AI Gateway (`CF_AI_GATEWAY_BASE` set at issuance). |
| `openai` | Direct call to `https://api.openai.com/v1/chat/completions`. |
| `cf-ai-gateway-openai` | OpenAI via Cloudflare AI Gateway. |
| `canonical-path` | Pure algorithmic; no LLM call. Pairs with `digital_source_type: "algorithmicMedia"`. |

### 1.4 Cache-HIT durability

Receipts are stamped at issuance time. Subsequent cache HITs on the same `(transform, parameters, input)` key serve the receipt verbatim, INCLUDING the original `signed_at`. This means:

- A response returned today may carry `signed_at` from hours or days ago. That is correct, not stale: the receipt asserts "I, the issuer, attested to this transform at the timestamp shown."
- The receipt's `kid` MUST remain queryable in JWKS as long as cached responses can be returned with that kid. See `docs/RENDER_RECEIPT_FORMAT.md` §6 for rotation cadence.
- A consumer requiring freshness MUST compare `signed_at` against their own clock and reject receipts older than their freshness window. The receipt does not enforce a freshness policy.

### 1.5 Forward-compat policy

The v1 schema is stable. Future revisions:
- MAY add new fields to `payload`. Verifiers MUST treat unknown fields as opaque-but-signed metadata.
- MAY add new entries to the `transform` registry. Verifiers MUST treat unknown transform values as opaque-but-signed metadata.
- MUST NOT remove or rename existing fields or registered transform values.
- MUST bump the `schema` identifier to `sum.transform_receipt.v2` (etc.) for breaking changes.
- New protected-header fields that change verification semantics MUST land in the JWS `crit` array, forcing older verifiers to fail closed per RFC 7515 §4.1.11.

A v1-aware verifier will continue to verify v1 receipts forever, including receipts whose `transform` value or payload fields it doesn't know about.

---

## 2. Verifier algorithm

The verifier needs three inputs:
- The full `transform_receipt` block from a `POST /api/transform` response.
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
   { protected: protected_b64, payload: <JCS bytes from step 2>,
     signature: signature_b64 }

5. Verify with jose.flattenedVerify(flattened, key). Catch
   ERR_JWS_SIGNATURE_VERIFICATION_FAILED → REJECT.

6. Inspect the verified protected header. It MUST contain:
     alg: "EdDSA"
     kid: <matching receipt.kid>
     b64: false
     crit: ["b64"]
   Reject if any field is missing or differs. Reject if `crit` contains
   any value the verifier does not understand.

Accept iff all six steps complete without rejecting.
```

### 2.2 Optional integrity checks (post-verify)

A verified receipt proves the issuer signed the payload. To bind the receipt to a *served* artifact, the consumer must additionally:

- Recompute `sha256(served_output_bytes)` (per the transform's documented canonicalisation) and compare to `payload.output_hash`. If different, the served artifact does not match the receipt — REJECT at the application layer.
- For transforms that accept user-supplied input (rather than re-fetching it from a content-addressed store), recompute `sha256(supplied_input_bytes)` and compare to `payload.input_hash`. Mismatch means the consumer's local input is not what the issuer signed.
- Recompute `sha256(JCS(parameters_object))` and compare to `payload.parameters_hash`. Used when replaying a transform deterministically.

The verifier algorithm in §2.1 does NOT do these comparisons automatically — they require the consumer to know which transform produced the receipt and its canonicalisation rules. The reference Python verifier exposes `verify_transform_receipt(receipt, jwks)` for the signature check and separate `compare_input_hash(...)`, `compare_output_hash(...)`, `compare_parameters_hash(...)` helpers for the application-layer checks.

---

## 3. Cross-runtime byte-equivalence

The transform-receipt format is byte-identical across Python / Node / browser verifiers, the same way `sum.render_receipt.v1` is. Specifically:

- JCS canonicalisation of `payload` produces identical bytes in `sum_engine_internal/infrastructure/jcs.py` (Python), `scripts/vendor/canonicalize` (Node, bundled into `worker/`), and `single_file_demo/vendor/sum-verify-deps.js` (browser).
- Ed25519 verification produces identical accept/reject across `cryptography` (Python), `jose` (Node), and `SubtleCrypto` (browser).
- The K1 / K1-multiword / K2 / K3 / K4 / A1–A6 cross-runtime gate matrix that locks `sum.render_receipt.v1` is extended in T1 to also lock `sum.transform_receipt.v1` — same fixtures, same outcomes, same byte-equivalence guarantee.

---

## 4. Migration from `sum.render_receipt.v1`

When the slider migrates to the transform registry (T1):

| Aspect | `sum.render_receipt.v1` (legacy) | `sum.transform_receipt.v1` (new) |
|---|---|---|
| Endpoint | `POST /api/render` | `POST /api/transform` (with `transform: "slider"`) |
| `schema` field | `"sum.render_receipt.v1"` | `"sum.transform_receipt.v1"` |
| Slider-specific fields (`sliders_quantized`, `triples_hash`, `tome_hash`) | Present in payload | Replaced by `parameters_hash` + `input_hash` + `output_hash` |
| Verifier code path | `sum_engine_internal.render_receipt.verify_receipt` | `sum_engine_internal.transform_receipt.verify_transform_receipt` |
| Backwards compatibility | Existing receipts remain verifiable forever via the legacy verifier | New receipts use the new verifier |
| `POST /api/render` route | Stays live as a thin adapter that calls `POST /api/transform` with `transform=slider` internally | — |

Existing pre-T1 render receipts are NOT migrated automatically. They remain valid forever under their original schema. Consumers verifying both old and new receipts must dispatch on `receipt.schema`.

---

## 5. Trust scope

**A verified transform receipt PROVES:**

1. The issuer (identified by the `kid` in the JWKS) signed this exact `payload` tuple at the `signed_at` timestamp.
2. The payload's bytes are JCS-canonical and the JWS signature covers them.
3. The `transform`, `parameters_hash`, `input_hash`, and `output_hash` values bind the receipt to one specific (transform, parameters, input, output) tuple.

**A verified transform receipt DOES NOT PROVE:**

1. **The factual truth of the output.** A `transform: "extract"` receipt does not prove the extracted tags are *correct* — only that the issuer extracted them. Truth claims about the output are the bench's NLI-audit's job (see `docs/SLIDER_CONTRACT.md` for the slider's preservation contract).
2. **Freshness on cache HIT.** The receipt carries `signed_at` from the original miss; the consumer enforces freshness policy.
3. **Soundness of the transform implementation.** If a future bug in `transform: "extract"` produced empty tag sets, the receipt would still verify — what's signed is "this issuer ran this transform with these parameters and got this output," not "the implementation is correct."
4. **That the input was unmodified before the transform.** `input_hash` proves what the issuer received; it does not prove the upstream chain of custody. Consumers who need that should use the evidence-chain layer (`docs/EVIDENCE_CHAIN.md`) to bind `input_hash` to a source byte range.

---

## 6. Related specifications

- [`RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) — the prior-art receipt format that this generalizes. Existing render receipts remain valid forever under their original schema.
- [`TRANSFORM_REGISTRY.md`](TRANSFORM_REGISTRY.md) — design doc for the transform-registry pattern: how transforms are registered, what contracts they implement, where they live in the codebase.
- [`PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) — proved-vs-measured discipline. The trust-scope above maps to §1.3.1 (cross-runtime Ed25519 trust triangle).
- [`THREAT_MODEL.md`](THREAT_MODEL.md) — attack surface. New attack vectors specific to the transform registry (e.g., transform-name spoofing, parameter-hash collision) are added in the T1 implementation PR.
- [`AUDIT_LOG_FORMAT.md`](AUDIT_LOG_FORMAT.md) — every `POST /api/transform` operation emits one `sum.audit_log.v1` row with `operation: "transform"` and `transform_name` populated.
- [`CANONICAL_ABI_SPEC.md`](CANONICAL_ABI_SPEC.md) — JCS canonicalisation rules; the binding contract for `parameters_hash`, `input_hash`, `output_hash`.
