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

The cross-runtime byte-equivalence guarantee that locks the render-
receipt format (K1/K1-mw/K2/K3/K4 + A1-A6 gate matrix) extends to
this format unchanged: same JCS canonicalisation, same Ed25519, same
detached JWS, same JWKS distribution. The K-matrix gates a new
fixture set against sum.transform_receipt.v1 in T1b (Worker port).
"""
from sum_engine_internal.transform_receipt.format import (
    SUPPORTED_SCHEMA,
    TransformReceiptPayload,
    build_payload,
    canonical_hash,
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
    "sign_transform_receipt",
    "verify_transform_receipt",
]
