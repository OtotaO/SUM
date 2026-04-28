"""Render-receipt verifier (Phase E.1 v0.9.C).

Schema-aware wrapper around the shared JOSE-envelope core in
``sum_engine_internal.infrastructure.jose_envelope``. The trust-root
manifest verifier (Phase R0.2) uses the same core with a different
supported schema; both surfaces share the same six-step algorithm,
the same forward-compat levers (schema gate + RFC 7515 §4.1.11
crit-extension fail-closed), and the same error class taxonomy.

Public API preserved across the v0.9.C → R0.2 refactor:

    SUPPORTED_SCHEMA              str  — "sum.render_receipt.v1"
    KNOWN_CRIT_EXTENSIONS         frozenset[str] — {"b64"}
    ErrorClass                    class — string-constant enum
    VerifyError                   exception — has .error_class
    VerifyResult                  dataclass
    verify_receipt(receipt, jwks) → VerifyResult

Existing 16 receipt-fixture tests under ``Tests/test_render_receipt_
verifier.py`` cover this surface; the refactor MUST preserve every
fixture's expected error class so the cross-runtime equivalence
PROOF_BOUNDARY §1.8 claims still holds.
"""
from __future__ import annotations

from sum_engine_internal.infrastructure.jose_envelope import (
    DEFAULT_KNOWN_CRIT_EXTENSIONS,
    JoseEnvelopeError,
    JoseEnvelopeErrorClass,
    JoseEnvelopeResult,
    verify_jose_envelope,
)


SUPPORTED_SCHEMA = "sum.render_receipt.v1"
KNOWN_CRIT_EXTENSIONS = DEFAULT_KNOWN_CRIT_EXTENSIONS


class ErrorClass:
    """String constants — identical to JoseEnvelopeErrorClass but
    kept as a separate class so the public name `ErrorClass` lands
    cleanly in the render_receipt namespace. Receipt-specific naming:
    MALFORMED_RECEIPT mirrors MALFORMED_ENVELOPE."""
    MALFORMED_RECEIPT = JoseEnvelopeErrorClass.MALFORMED_ENVELOPE
    MALFORMED_JWS = JoseEnvelopeErrorClass.MALFORMED_JWS
    MALFORMED_JWKS = JoseEnvelopeErrorClass.MALFORMED_JWKS
    UNKNOWN_KID = JoseEnvelopeErrorClass.UNKNOWN_KID
    KID_MISMATCH = JoseEnvelopeErrorClass.KID_MISMATCH
    SCHEMA_UNKNOWN = JoseEnvelopeErrorClass.SCHEMA_UNKNOWN
    CRIT_UNKNOWN_EXTENSION = JoseEnvelopeErrorClass.CRIT_UNKNOWN_EXTENSION
    HEADER_INVARIANT_VIOLATED = JoseEnvelopeErrorClass.HEADER_INVARIANT_VIOLATED
    SIGNATURE_INVALID = JoseEnvelopeErrorClass.SIGNATURE_INVALID
    REVOKED_KID = JoseEnvelopeErrorClass.REVOKED_KID


class VerifyError(JoseEnvelopeError):
    """Receipt-specific subclass of JoseEnvelopeError. ``isinstance(e,
    VerifyError)`` and ``isinstance(e, JoseEnvelopeError)`` both hold,
    so consumers catching either work. ``.error_class`` carries the
    same string values across both surfaces — receipts and trust-root
    manifests share an error taxonomy that downstream cross-runtime
    fixtures assert by string compare."""


# Type alias preserved for backwards compat with v0.9.C.
VerifyResult = JoseEnvelopeResult


def _check_revoked_kid(receipt: dict, revoked_kids: list[dict]) -> None:
    """Raise VerifyError(REVOKED_KID) if the receipt's kid is on the
    revocation list AND the receipt's signed_at is at or after the
    revocation's effective_revocation_at.

    Per docs/RENDER_RECEIPT_FORMAT.md §6.1:

    * A receipt with signed_at BEFORE effective_revocation_at retains
      its original validity (was signed legitimately before the
      compromise window). Continue verification normally.
    * A receipt with signed_at AT OR AFTER effective_revocation_at is
      rejected with the revoked_kid error class.

    Comparison is on ISO-8601 string lex-order, which matches
    timestamp ordering for UTC strings. The signed_at field is
    required by the receipt spec; missing or unparseable signed_at
    is treated as "cannot determine — fail closed" (reject).
    """
    kid = receipt.get("kid")
    if not isinstance(kid, str):
        return  # not our problem here; envelope-shape check catches it later
    payload = receipt.get("payload") or {}
    signed_at = payload.get("signed_at")

    for entry in revoked_kids:
        if not isinstance(entry, dict):
            continue
        if entry.get("kid") != kid:
            continue
        effective_at = entry.get("effective_revocation_at")
        if not isinstance(effective_at, str):
            # Malformed revocation entry; defensive fail-closed.
            raise VerifyError(
                ErrorClass.REVOKED_KID,
                f"kid {kid!r} appears on revocation list with malformed "
                f"effective_revocation_at={effective_at!r}; failing closed",
            )
        if not isinstance(signed_at, str):
            # Receipt has no signed_at; can't compare; fail closed.
            raise VerifyError(
                ErrorClass.REVOKED_KID,
                f"kid {kid!r} on revocation list and receipt has no "
                f"parseable signed_at; failing closed",
            )
        # ISO-8601 UTC strings compare correctly via lex-order.
        if signed_at >= effective_at:
            raise VerifyError(
                ErrorClass.REVOKED_KID,
                f"kid {kid!r} revoked effective {effective_at}; "
                f"receipt signed at {signed_at} (>= effective time)",
            )
        # signed_at < effective_at: legitimate historical receipt,
        # continue verification.
        return


def verify_receipt(receipt, jwks, revoked_kids=None) -> VerifyResult:
    """Verify a SUM render receipt against a JWKS.

    Parameters
    ----------
    receipt
        The ``render_receipt`` block from an ``/api/render`` response.
        Must be a dict with keys ``schema``, ``kid``, ``payload``,
        ``jws``.
    jwks
        A dict with key ``keys`` containing JWK dicts. Typically the
        parsed body of ``/.well-known/jwks.json``.
    revoked_kids
        Optional list of revocation entries
        ``[{"kid": ..., "effective_revocation_at": ..., "reason": ...}]``
        as served at ``/.well-known/revoked-kids.json`` (see
        docs/RENDER_RECEIPT_FORMAT.md §6.1). When provided, kids
        with signed_at >= effective_revocation_at are rejected with
        the ``revoked_kid`` error class. Pass ``None`` (default) to
        skip revocation entirely. Pass an empty list ``[]`` to
        explicitly assert "fetched the list, no kids revoked."

    Returns
    -------
    VerifyResult on success. ``.payload`` carries the verified
    receipt payload (render_id, sliders_quantized, model, etc.).

    Raises
    ------
    VerifyError on any failure. ``.error_class`` distinguishes
    between failure modes per ``ErrorClass``.
    """
    # G3 revocation check runs BEFORE the cryptographic verify so a
    # kid that was both revoked AND tampered surfaces as
    # `revoked_kid` (the more actionable error class for an operator
    # — points at "rotate + revoke" rather than "investigate the
    # signature").
    if revoked_kids is not None:
        _check_revoked_kid(receipt, revoked_kids)

    try:
        return verify_jose_envelope(
            receipt,
            jwks,
            supported_schema=SUPPORTED_SCHEMA,
            known_crit_extensions=KNOWN_CRIT_EXTENSIONS,
        )
    except JoseEnvelopeError as e:
        # Re-raise as the receipt-specific subclass so callers
        # importing only `VerifyError` still get a useful match.
        raise VerifyError(e.error_class, str(e)) from e
