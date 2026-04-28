# Receipt-fixture set — shared between v0.9.B (browser) and v0.9.C (Python) verifiers

19 runtime-neutral JSON fixtures pinning every named failure mode of the
render-receipt verifier algorithm from
[`docs/RENDER_RECEIPT_FORMAT.md`](../../docs/RENDER_RECEIPT_FORMAT.md) §2.1.

Both verifiers — the v0.9.B browser-side ESM verifier (in
[`single_file_demo/`](../../single_file_demo/)) and the v0.9.C Python
verifier (under `sum_engine_internal/` once landed) — consume these same
files. Any divergence in outcome between the two, on any fixture, is the
exact cross-runtime trust-triangle bug the K-style harness is designed to
catch.

## Fixture format

Every file has the same shape:

```jsonc
{
  "name": "<fixture-name>",
  "description": "<plain-English what was mutated and why>",
  "expected_outcome": "verify" | "reject",
  "expected_error_class": "<err-class>" | null,
  "receipt":      { schema, kid, payload, jws },
  "jwks":         { keys: [...] },
  "revoked_kids": [...]    // OPTIONAL — present only on G3
                           // revocation fixtures. Verifiers MUST
                           // pass this to verify_receipt's
                           // optional revoked_kids parameter when
                           // present; absent fixtures verify
                           // without revocation (backwards-compat
                           // with v0.9.C).
}
```

`expected_outcome` is `"verify"` only for the positive control. Every other
fixture is `"reject"` with a specific `expected_error_class`.

## Error classes (verifier-runtime-neutral)

| Class | Meaning |
|---|---|
| `signature_invalid` | JWS signature does not verify against the JCS-canonical payload bytes. Catches every tampered-signed-field case. |
| `malformed_jws` | Detached JWS contract violated (middle segment non-empty, etc.). Pre-signature check. |
| `unknown_kid` | `receipt.kid` not present in the supplied JWKS. Pre-signature check. |
| `kid_mismatch` | Protected-header `kid` differs from top-level `receipt.kid` (header tampering pattern). |
| `schema_unknown` | `receipt.schema` is not a value this verifier accepts. Forward-compat lever. |
| `crit_unknown_extension` | Protected header `crit` array contains an extension this verifier doesn't understand. Per RFC 7515 §4.1.11, fail closed. Forward-compat lever. |
| `revoked_kid` | (G3) Receipt's kid is on the supplied revocation list with `effective_revocation_at` ≤ `receipt.payload.signed_at`. Distinct from `signature_invalid` so the operator-side distinction between "tampered" and "issued under a now-revoked key" is visible at the consumer. See [`docs/RENDER_RECEIPT_FORMAT.md`](../../docs/RENDER_RECEIPT_FORMAT.md) §6.1. |
| `unsupported_alg` | (G3) Protected header `alg` claim is not in the in-tree algorithm registry under `current`. Distinct from `header_invariant_violated` so an alg-downgrade-attempt or unsupported algorithm fails with a precise classification. See [`docs/ALGORITHM_REGISTRY.md`](../../docs/ALGORITHM_REGISTRY.md). |

## The 15 fixtures

| Name | Class | What's mutated |
|---|---|---|
| `positive_control` | — verify ✓ | nothing |
| `tampered_tome_hash` | `signature_invalid` | `payload.tome_hash` |
| `tampered_triples_hash` | `signature_invalid` | `payload.triples_hash` |
| `tampered_sliders_quantized` | `signature_invalid` | one slider value |
| `tampered_model` | `signature_invalid` | `payload.model` |
| `tampered_provider` | `signature_invalid` | `payload.provider` |
| `tampered_signed_at` | `signature_invalid` | `payload.signed_at` |
| `tampered_digital_source_type` | `signature_invalid` | `payload.digital_source_type` |
| `tampered_render_id` | `signature_invalid` | `payload.render_id` |
| `tampered_signature` | `signature_invalid` | last char of JWS signature segment |
| `tampered_kid_header` | `signature_invalid` | kid claim inside the JWS protected header |
| `malformed_jws_middle_nonempty` | `malformed_jws` | middle JWS segment populated (must be empty) |
| `unknown_kid` | `unknown_kid` | JWKS empty; receipt's kid not findable |
| `schema_unknown` | `schema_unknown` | `receipt.schema` bumped to `sum.render_receipt.v99` |
| `crit_unknown_extension` | `crit_unknown_extension` | protected header `crit` extended with an unknown extension |
| `revoked_kid_active` | `revoked_kid` | (G3) revocation list names the receipt's kid with `effective_revocation_at` = receipt's signed_at |
| `revoked_kid_historical` | — verify ✓ | (G3) revocation list names the kid but `effective_revocation_at` is in the future; receipt predates revocation |
| `revoked_kid_unrelated` | — verify ✓ | (G3) revocation list mentions a different kid; receipt's kid not on the list |
| `unsupported_alg` | `unsupported_alg` | (G3) protected header `alg` mutated to `HS256` (not in the EdDSA-only v1 registry) |

## How to regenerate

The fixtures derive from `source_render.json` (a captured `/api/render`
response) plus `jwks_at_capture.json` (the JWKS as of capture time). To
regenerate after a new live capture:

```
# 1. Capture fresh inputs (must be at the same moment so the kid lines up)
curl -sS https://sum-demo.ototao.workers.dev/.well-known/jwks.json \
  -o fixtures/render_receipts/jwks_at_capture.json
curl -sS -X POST https://sum-demo.ototao.workers.dev/api/render \
  -H 'content-type: application/json' \
  -d '{"triples":[["alice","graduated","2012"],["alice","born","1990"]],"slider_position":{"density":1.0,"length":0.5,"formality":0.7,"audience":0.5,"perspective":0.5}}' \
  -o fixtures/render_receipts/source_render.json

# 2. Regenerate the derived fixtures
python fixtures/render_receipts/generate_fixtures.py
```

Generation is idempotent: same inputs produce byte-identical outputs. CI
asserts this via the smoke test in `single_file_demo/`.

## The trust contract this enables

A receipt is trustworthy when **every** verifier — Python `joserfc`, Node
`jose`, browser `jose` (vendored ESM), and any independent third-party
verifier (G2 governance track) — produces the same outcome on every
fixture. The K-style cross-runtime equivalence we already have for
CanonicalBundle, applied to render receipts.

PROOF_BOUNDARY §1.8 currently says the negative path is "exercised in
worker-local TS tests but not yet locked across runtimes." This fixture
set is what closes that gap once both v0.9.B and v0.9.C land.
