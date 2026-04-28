"""Trust-root manifest verifier (Phase R0.2).

Schema-aware wrapper around the shared JOSE-envelope core in
``sum_engine_internal.infrastructure.jose_envelope``. Mirrors the
``render_receipt`` verifier surface exactly — the two share the
same algorithm and error taxonomy; only the supported schema
differs.

See ``docs/TRUST_ROOT_FORMAT.md`` for the wire spec, the verifier
algorithm, and the trust-scope boundary.
"""
from __future__ import annotations

from sum_engine_internal.infrastructure.jose_envelope import (
    DEFAULT_KNOWN_CRIT_EXTENSIONS,
    JoseEnvelopeError,
    JoseEnvelopeErrorClass,
    JoseEnvelopeResult,
    verify_jose_envelope,
)


SUPPORTED_SCHEMA = "sum.trust_root.v1"
KNOWN_CRIT_EXTENSIONS = DEFAULT_KNOWN_CRIT_EXTENSIONS


class ErrorClass:
    """Trust-root-surface error class names. Identical string values
    to ``sum_engine_internal.render_receipt.ErrorClass`` and
    ``sum_engine_internal.infrastructure.jose_envelope.JoseEnvelopeErrorClass``;
    consumers asserting by string match across surfaces work without
    modification."""
    MALFORMED_MANIFEST = JoseEnvelopeErrorClass.MALFORMED_ENVELOPE
    MALFORMED_JWS = JoseEnvelopeErrorClass.MALFORMED_JWS
    MALFORMED_JWKS = JoseEnvelopeErrorClass.MALFORMED_JWKS
    UNKNOWN_KID = JoseEnvelopeErrorClass.UNKNOWN_KID
    KID_MISMATCH = JoseEnvelopeErrorClass.KID_MISMATCH
    SCHEMA_UNKNOWN = JoseEnvelopeErrorClass.SCHEMA_UNKNOWN
    CRIT_UNKNOWN_EXTENSION = JoseEnvelopeErrorClass.CRIT_UNKNOWN_EXTENSION
    HEADER_INVARIANT_VIOLATED = JoseEnvelopeErrorClass.HEADER_INVARIANT_VIOLATED
    SIGNATURE_INVALID = JoseEnvelopeErrorClass.SIGNATURE_INVALID
    UNSUPPORTED_ALG = JoseEnvelopeErrorClass.UNSUPPORTED_ALG


class VerifyError(JoseEnvelopeError):
    """Trust-root-specific subclass. ``isinstance(e, VerifyError)``
    and ``isinstance(e, JoseEnvelopeError)`` both hold."""


VerifyResult = JoseEnvelopeResult


def verify_trust_manifest(manifest, jwks) -> VerifyResult:
    """Verify a SUM trust-root manifest against a JWKS.

    Parameters
    ----------
    manifest
        The signed trust-root manifest. Dict with keys ``schema``,
        ``kid``, ``payload``, ``jws``.
    jwks
        Dict with key ``keys`` containing JWK dicts. Typically the
        parsed body of the trust-root JWKS endpoint.

    Returns
    -------
    VerifyResult on success. ``.payload`` carries the verified
    manifest payload (issued_at, repo, commit, release, artifacts,
    render_receipt_jwks, algorithm_registry).

    Raises
    ------
    VerifyError on any failure. ``.error_class`` per ``ErrorClass``.
    """
    try:
        return verify_jose_envelope(
            manifest,
            jwks,
            supported_schema=SUPPORTED_SCHEMA,
            known_crit_extensions=KNOWN_CRIT_EXTENSIONS,
        )
    except JoseEnvelopeError as e:
        raise VerifyError(e.error_class, str(e)) from e
