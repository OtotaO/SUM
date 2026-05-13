"""Verifier for ``sum.transform_receipt.v1``.

Schema-aware wrapper around
``sum_engine_internal.infrastructure.jose_envelope.verify_jose_envelope``
mirroring ``sum_engine_internal.render_receipt.verifier``. Same six-
step algorithm, same forward-compat levers (schema gate + RFC 7515
§4.1.11 crit-extension fail-closed), same error class taxonomy.

Public API:

    SUPPORTED_SCHEMA              "sum.transform_receipt.v1"
    KNOWN_CRIT_EXTENSIONS         frozenset[str]  = {"b64"}
    ErrorClass                    class — string-constant enum
    VerifyError                   exception — has .error_class
    VerifyResult                  dataclass (JoseEnvelopeResult alias)
    verify_transform_receipt(receipt, jwks) → VerifyResult

The K-matrix cross-runtime fixture set under
``fixtures/transform_receipts/`` is consumed by this Python verifier
AND the browser/Node verifier in ``single_file_demo/`` — 20 fixtures
covering every named failure mode (positive control with/without T4
source-chain binding, every signed payload field tampered, malformed
JWS, unknown kid, schema_unknown, schema-confusion firewall against
sum.render_receipt.v1, crit-unknown-extension, unsupported_alg).
Both runtimes must produce byte-identical outcomes on every fixture.
"""
from __future__ import annotations

from sum_engine_internal.infrastructure.jose_envelope import (
    DEFAULT_KNOWN_CRIT_EXTENSIONS,
    JoseEnvelopeError,
    JoseEnvelopeErrorClass,
    JoseEnvelopeResult,
    verify_jose_envelope,
)
from sum_engine_internal.transform_receipt.format import SUPPORTED_SCHEMA


KNOWN_CRIT_EXTENSIONS = DEFAULT_KNOWN_CRIT_EXTENSIONS


class ErrorClass:
    """String constants mirroring ``JoseEnvelopeErrorClass`` with
    receipt-specific names where relevant. Re-exported into the
    ``transform_receipt`` namespace so consumers can branch on
    ``e.error_class == ErrorClass.SIGNATURE_INVALID`` without
    reaching into the underlying envelope module."""
    MALFORMED_RECEIPT = JoseEnvelopeErrorClass.MALFORMED_ENVELOPE
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
    """Transform-receipt-specific subclass of JoseEnvelopeError.
    ``isinstance(e, VerifyError)`` and
    ``isinstance(e, JoseEnvelopeError)`` both hold so consumers
    catching either work. ``.error_class`` carries identical string
    values across both surfaces — the receipt and trust-root manifest
    surfaces share an error taxonomy."""


# Type alias preserved for symmetry with render_receipt.verifier.
VerifyResult = JoseEnvelopeResult


def verify_transform_receipt(
    receipt: dict,
    jwks: dict,
) -> VerifyResult:
    """Verify a ``sum.transform_receipt.v1`` envelope.

    Returns a ``VerifyResult`` on accept; raises ``VerifyError`` on
    reject. The error class is one of ``ErrorClass`` and matches the
    same enum the render-receipt verifier uses, so downstream tooling
    that branches on error class works unchanged across both receipt
    types.

    Parameters
    ----------
    receipt
        The full envelope dict (with ``schema``, ``kid``, ``payload``,
        ``jws`` keys) as it appears in a ``/api/transform`` response
        body.
    jwks
        The issuer's JWKS — a dict with a ``keys`` array. Fetch this
        once from the operator's ``/.well-known/jwks.json`` and cache
        per ``Cache-Control: max-age`` as documented in §1.4 of the
        render-receipt format (same cache cadence applies here).

    Application-layer integrity checks
    ----------------------------------
    A verified receipt does NOT automatically check that ``parameters_
    hash`` / ``input_hash`` / ``output_hash`` match locally-recomputed
    hashes of the served artifacts. Callers that want those checks
    should:

        from sum_engine_internal.transform_receipt import canonical_hash

        # recompute over what was served / supplied
        assert canonical_hash(transform.canonicalize_output(served_output))
               == receipt["payload"]["output_hash"]
    """
    try:
        return verify_jose_envelope(
            receipt,
            jwks=jwks,
            supported_schema=SUPPORTED_SCHEMA,
            known_crit_extensions=KNOWN_CRIT_EXTENSIONS,
        )
    except JoseEnvelopeError as e:
        # Re-raise the underlying envelope error as a transform-
        # receipt-specific VerifyError so consumers branching on
        # `isinstance(e, transform_receipt.VerifyError)` work, while
        # `e.error_class` continues to carry the same string-enum
        # value the render-receipt verifier uses.
        if isinstance(e, VerifyError):
            raise
        new_err = VerifyError(e.error_class, str(e))
        raise new_err from e
