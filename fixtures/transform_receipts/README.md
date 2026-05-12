# Transform-receipt fixture set — shared between Python + Node + browser verifiers

20 runtime-neutral JSON fixtures pinning every named failure mode of the
`sum.transform_receipt.v1` verifier from
[`docs/TRANSFORM_RECEIPT_FORMAT.md`](../../docs/TRANSFORM_RECEIPT_FORMAT.md),
parallel to the render-receipt fixture set under
[`fixtures/render_receipts/`](../render_receipts/README.md).

Three verifiers consume these same files:

  - Python: `sum_engine_internal.transform_receipt.verify_transform_receipt`
    via [`Tests/test_transform_receipt_verifier_fixtures.py`](../../Tests/test_transform_receipt_verifier_fixtures.py).
  - Browser/Node: `single_file_demo/transform_receipt_verifier.js` via
    [`single_file_demo/test_transform_receipt_fixtures.js`](../../single_file_demo/test_transform_receipt_fixtures.js).
  - Worker (TypeScript): the Worker is the producer in this loop — its
    sign-side byte-equivalence is empirically proved by Python + browser
    accepting Worker-signed receipts (the existing live trust-loop probe).

Any divergence between the three verifiers on any fixture is the cross-
runtime trust-triangle bug the K-style harness is designed to catch.

## Why deterministic-test-key, not live-capture?

The render-receipt fixture set derives from a captured live `/api/render`
response so the kid + JWKS are operator-real. The transform-receipt
fixture set instead uses a deterministic test key (RFC 8032 §7.1 test
vector 1 — 32 zero bytes as the seed) so:

  - Regeneration is byte-identical without needing a live network capture
    or a fresh operator key rotation.
  - The fixture key cannot be confused with any operator key. The kid
    `transform-fixture-key-2026` is the canonical "this is a test fixture"
    marker. Verifiers that surface this kid in production logs have a
    fixture leaking into runtime.
  - CI doesn't depend on the live Worker being online.

Both patterns prove the same thing — cross-runtime byte-equivalence on
adversarial inputs — at different layers (operator-real vs schema-real).

## Fixture format

Every file has the same shape as the render-receipt set:

```jsonc
{
  "name": "<fixture-name>",
  "description": "<plain-English what was mutated and why>",
  "expected_outcome": "verify" | "reject",
  "expected_error_class": "<err-class>" | null,
  "receipt":      { schema, kid, payload, jws },
  "jwks":         { keys: [...] }
}
```

Two fixtures `expected_outcome: "verify"` (positive control + the T4
source-chain-bound positive control); every other fixture is `"reject"`
with a specific `expected_error_class`.

## Error classes

Same taxonomy as render-receipt (see that README for full table). For
transform-receipt the receipt-specific covered fields are:

  - `parameters_hash`, `input_hash`, `output_hash` — the three hashes
    binding the transform invocation to its canonical inputs + output.
  - `transform`, `transform_id` — what was invoked + its derived ID.
  - `source_chain_hash` — T4 evidence-chain binding (optional field;
    when present, signature-covered).

The forward-compat lever also includes a `schema_confusion_render_receipt`
fixture that locks the schema gate as a cross-receipt-type firewall: a
cryptographically-valid `sum.render_receipt.v1` envelope MUST NOT
verify against the transform-receipt verifier.

## The 20 fixtures

| Name | Outcome / class | What's mutated |
|---|---|---|
| `positive_control` | verify ✓ | nothing |
| `positive_control_with_source_chain` | verify ✓ | nothing (T4 source_chain_hash bound) |
| `tampered_parameters_hash` | `signature_invalid` | `payload.parameters_hash` |
| `tampered_input_hash` | `signature_invalid` | `payload.input_hash` |
| `tampered_output_hash` | `signature_invalid` | `payload.output_hash` |
| `tampered_transform` | `signature_invalid` | `payload.transform` (slider → extract) |
| `tampered_transform_id` | `signature_invalid` | `payload.transform_id` |
| `tampered_model` | `signature_invalid` | `payload.model` |
| `tampered_provider` | `signature_invalid` | `payload.provider` |
| `tampered_signed_at` | `signature_invalid` | `payload.signed_at` |
| `tampered_digital_source_type` | `signature_invalid` | `payload.digital_source_type` |
| `tampered_source_chain_hash` | `signature_invalid` | `payload.source_chain_hash` (T4) |
| `tampered_signature` | `signature_invalid` | last char of JWS signature segment |
| `tampered_kid_header` | `signature_invalid` | kid inside protected header |
| `malformed_jws_middle_nonempty` | `malformed_jws` | middle JWS segment populated |
| `unknown_kid` | `unknown_kid` | JWKS empty |
| `schema_unknown` | `schema_unknown` | `receipt.schema` → `sum.transform_receipt.v99` |
| `schema_confusion_render_receipt` | `schema_unknown` | `receipt.schema` → `sum.render_receipt.v1` |
| `crit_unknown_extension` | `crit_unknown_extension` | protected header `crit` extended |
| `unsupported_alg` | `unsupported_alg` | protected header `alg` → `HS256` |

## How to regenerate

```
python fixtures/transform_receipts/generate_fixtures.py
```

The script:

  1. Derives the Ed25519 keypair from the fixed 32-byte seed.
  2. Signs one canonical-path slider transform-receipt → writes
     `source_receipt.json` + `jwks_at_capture.json`.
  3. Signs a second receipt with a T4 source_chain_hash → that's the
     basis for the source-chain-bound positive control + tamper.
  4. Derives every mutation fixture from those two source receipts.

Idempotent: same seed → same key → same signatures (Ed25519 is
deterministic per RFC 8032 §5.1.6).

## The trust contract this enables

A transform-receipt is trustworthy when **every** verifier — Python
`joserfc`, Node + browser `jose` (vendored ESM), the Worker's own
TypeScript verifier — produces the same outcome on every fixture. The
same K-style cross-runtime equivalence that locks CanonicalBundle and
render receipts, now extended to the transform substrate.
