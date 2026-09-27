"""Transform-receipt signing + verification.

Mirrors ``sum_engine_internal.render_receipt`` for the new
``sum.transform_receipt.v1`` schema. Same JOSE-envelope core; the only
difference is the supported schema string and the receipt-specific
field invariants.

Public surface:

    sign_transform_receipt(...) → signed envelope dict
    verify_transform_receipt(...)
    SUPPORTED_SCHEMA            = "sum.transform_receipt.v1"
    VerifyError                 — single exception class for failures
    ErrorClass                  — string enum mirrored across runtimes
    VerifyResult                — JoseEnvelopeResult re-export

The render-receipt format's cross-runtime checks extend to this format
unchanged: same JCS canonicalisation, same Ed25519, same detached JWS,
same JWKS distribution (CI exercises Python and the JS verifier under
Node; no browser engine runs in CI). The 20-fixture set in
``fixtures/transform_receipts/`` is consumed by both the Python
verifier here and the JS verifier under ``single_file_demo/`` (run
under Node in CI); both produce byte-identical accept/reject +
error_class outcomes on every fixture.
"""
from sum_engine_internal.transform_receipt.format import (
    SUPPORTED_SCHEMA,
    TransformReceiptPayload,
    build_payload,
    canonical_hash,
    compute_source_chain_hash,
)
from sum_engine_internal.transform_receipt.sign import sign_transform_receipt
from sum_engine_internal.transform_receipt.verifier import (
    ErrorClass,
    KNOWN_CRIT_EXTENSIONS,
    VerifyError,
    VerifyResult,
    verify_transform_receipt,
)


__all__ = [
    "ErrorClass",
    "KNOWN_CRIT_EXTENSIONS",
    "SUPPORTED_SCHEMA",
    "TransformReceiptPayload",
    "VerifyError",
    "VerifyResult",
    "build_payload",
    "canonical_hash",
    "compute_source_chain_hash",
    "sign_transform_receipt",
    "verify_transform_receipt",
]
