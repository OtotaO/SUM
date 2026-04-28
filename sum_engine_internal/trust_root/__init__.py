"""Trust-root manifest verification (Phase R0.2).

Companion to ``sum_engine_internal.render_receipt``: same JOSE
envelope shape, same six-step verifier algorithm, same error class
taxonomy. The two surfaces sit on the same shared core under
``sum_engine_internal.infrastructure.jose_envelope``; only the
supported schema string differs.

A trust-root manifest is the single signed artifact a downstream
consumer verifies first when adopting a SUM release. It binds the
source commit, the release version, the published artifact hashes,
the JWKS state, and the algorithm registry into one verifiable
object. See ``docs/TRUST_ROOT_FORMAT.md`` for the wire spec.

Public surface re-exported here:
    verify_trust_manifest(manifest, jwks) -> VerifyResult
    VerifyError                   — raised on verification failure
    ErrorClass                    — string-constant taxonomy
    SUPPORTED_SCHEMA              — "sum.trust_root.v1"
    KNOWN_CRIT_EXTENSIONS         — frozenset({"b64"})

Optional dependency: install with
``pip install sum-engine[receipt-verify]`` (the same extra v0.9.C
introduced — joserfc covers both surfaces).
"""
from sum_engine_internal.trust_root.verifier import (
    KNOWN_CRIT_EXTENSIONS,
    SUPPORTED_SCHEMA,
    ErrorClass,
    VerifyError,
    VerifyResult,
    verify_trust_manifest,
)

__all__ = [
    "ErrorClass",
    "KNOWN_CRIT_EXTENSIONS",
    "SUPPORTED_SCHEMA",
    "VerifyError",
    "VerifyResult",
    "verify_trust_manifest",
]
