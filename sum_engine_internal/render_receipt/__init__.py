"""Render-receipt verification (Phase E.1 v0.9.C).

The Python counterpart to ``single_file_demo/receipt_verifier.js``.
Implements the six-step verifier algorithm from
``docs/RENDER_RECEIPT_FORMAT.md`` §2.1 plus the two §1.4
forward-compat levers (schema gate, RFC 7515 §4.1.11
crit-extension fail-closed). Same error-class taxonomy as the JS
verifier — fixtures under ``fixtures/render_receipts/`` are
consumed unchanged by both runtimes; cross-runtime equivalence
on every fixture is what closes the PROOF_BOUNDARY §1.8
"proved on adversarial inputs across runtimes" gap.

Optional dependency: install with ``pip install sum-engine[receipt-verify]``
which pulls in ``joserfc`` (the existing ``cryptography`` hard dep
covers Ed25519). Importing this module without joserfc installed
raises a clear ImportError pointing at the extra.

Public surface re-exported here:
    verify_receipt(receipt, jwks) -> VerifyResult
    VerifyError                   — single exception class for failures
    ErrorClass                    — string enum mirrored across runtimes
    SUPPORTED_SCHEMA              — "sum.render_receipt.v1"
    KNOWN_CRIT_EXTENSIONS         — frozenset({"b64"})
"""
from sum_engine_internal.render_receipt.verifier import (
    ErrorClass,
    KNOWN_CRIT_EXTENSIONS,
    SUPPORTED_SCHEMA,
    VerifyError,
    VerifyResult,
    verify_receipt,
)

__all__ = [
    "ErrorClass",
    "KNOWN_CRIT_EXTENSIONS",
    "SUPPORTED_SCHEMA",
    "VerifyError",
    "VerifyResult",
    "verify_receipt",
]
