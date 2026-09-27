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

import re
from typing import Any

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
    MALFORMED_RECEIPT mirrors MALFORMED_RECEIPT."""
    MALFORMED_RECEIPT = JoseEnvelopeErrorClass.MALFORMED_RECEIPT
    MALFORMED_JWS = JoseEnvelopeErrorClass.MALFORMED_JWS
    MALFORMED_JWKS = JoseEnvelopeErrorClass.MALFORMED_JWKS
    UNKNOWN_KID = JoseEnvelopeErrorClass.UNKNOWN_KID
    KID_MISMATCH = JoseEnvelopeErrorClass.KID_MISMATCH
    SCHEMA_UNKNOWN = JoseEnvelopeErrorClass.SCHEMA_UNKNOWN
    CRIT_UNKNOWN_EXTENSION = JoseEnvelopeErrorClass.CRIT_UNKNOWN_EXTENSION
    HEADER_INVARIANT_VIOLATED = JoseEnvelopeErrorClass.HEADER_INVARIANT_VIOLATED
    SIGNATURE_INVALID = JoseEnvelopeErrorClass.SIGNATURE_INVALID
    REVOKED_KID = JoseEnvelopeErrorClass.REVOKED_KID
    UNSUPPORTED_ALG = JoseEnvelopeErrorClass.UNSUPPORTED_ALG
    SIGNED_AT_OUT_OF_WINDOW = JoseEnvelopeErrorClass.SIGNED_AT_OUT_OF_WINDOW


class VerifyError(JoseEnvelopeError):
    """Receipt-specific subclass of JoseEnvelopeError. ``isinstance(e,
    VerifyError)`` and ``isinstance(e, JoseEnvelopeError)`` both hold,
    so consumers catching either work. ``.error_class`` carries the
    same string values across both surfaces — receipts and trust-root
    manifests share an error taxonomy that downstream cross-runtime
    fixtures assert by string compare."""


# Type alias preserved for backwards compat with v0.9.C.
VerifyResult = JoseEnvelopeResult


# RFC 3339 instant, seconds required, fraction truncated to milliseconds.
# The grammar, range checks and arithmetic are shared line for line with
# parseInstantMs in single_file_demo/receipt_verifier.js, so Python and the
# browser give the same revocation verdict for every input (datetime
# parsing differed across runtimes and Python versions, and could raise
# OverflowError). Cases: Tests/fixtures/revocation_instants.json.
_INSTANT = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})"
    r"(?:\.([0-9]{1,9}))?(Z|([+-])([0-9]{2}):([0-9]{2}))"
)


def _days_from_civil(y: int, m: int, d: int) -> int:
    y -= 1 if m <= 2 else 0
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _instant_ms(value: Any) -> int | None:
    """Epoch milliseconds for an RFC 3339 instant, or None if malformed."""
    if not isinstance(value, str):
        return None
    m = _INSTANT.fullmatch(value)
    if m is None:
        return None
    y, mo, d, h, mi, s = (int(m.group(i)) for i in range(1, 7))
    ms = int((m.group(7) or "").ljust(3, "0")[:3])
    leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
    days_in_month = (31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    if not (1 <= mo <= 12 and 1 <= d <= days_in_month[mo - 1]
            and h <= 23 and mi <= 59 and s <= 59):
        return None
    offset = 0
    if m.group(8) != "Z":
        oh, om = int(m.group(10)), int(m.group(11))
        if oh > 23 or om > 59:
            return None
        offset = (oh * 60 + om) * (1 if m.group(9) == "+" else -1)
    return (((_days_from_civil(y, mo, d) * 24 + h) * 60 + mi) * 60 + s) * 1000 + ms - offset * 60000


def _revocation_entries(revoked_kids: Any) -> list:
    """Return the revocation entries from a list or a served document,
    failing closed on any other shape."""
    if isinstance(revoked_kids, dict):
        # The served document, or the bare {"revoked": [...]} form that
        # RENDER_RECEIPT_FORMAT §6.1 describes; any other object fails closed.
        if (revoked_kids.get("schema", "sum.revoked_kids.v1") == "sum.revoked_kids.v1"
                and isinstance(revoked_kids.get("revoked"), list)):
            revoked_kids = revoked_kids["revoked"]
        else:
            raise VerifyError(
                ErrorClass.REVOKED_KID,
                "revoked_kids must be a list of revocation entries or the "
                "sum.revoked_kids.v1 document served at "
                "/.well-known/revoked-kids.json; failing closed",
            )
    if not isinstance(revoked_kids, (list, tuple)):
        raise VerifyError(
            ErrorClass.REVOKED_KID,
            f"revoked_kids must be a list of revocation entries, got "
            f"{type(revoked_kids).__name__}; failing closed",
        )
    for entry in revoked_kids:
        if not isinstance(entry, dict):
            raise VerifyError(
                ErrorClass.REVOKED_KID,
                f"revocation list contains a non-object entry "
                f"({type(entry).__name__}); failing closed",
            )
        if not isinstance(entry.get("kid"), str) or not entry["kid"]:
            raise VerifyError(
                ErrorClass.REVOKED_KID,
                "revocation list entry has no non-empty string kid; failing closed",
            )
    return list(revoked_kids)


def _check_revoked_kid(receipt: dict, revoked_kids: Any) -> None:
    """Raise VerifyError(REVOKED_KID) if the receipt's kid is on the
    revocation list AND the receipt's signed_at is at or after the
    revocation's effective_revocation_at.

    Per docs/RENDER_RECEIPT_FORMAT.md §6.1:

    * A receipt with signed_at BEFORE effective_revocation_at retains
      its original validity (was signed legitimately before the
      compromise window). Continue verification normally.
    * A receipt with signed_at AT OR AFTER effective_revocation_at is
      rejected with the revoked_kid error class.

    ``revoked_kids`` may be the entry list or the whole
    ``sum.revoked_kids.v1`` document served at
    ``/.well-known/revoked-kids.json``. Any other shape, and any entry
    that is not an object, fails closed: passing the served document
    used to iterate its keys, skip them all, and verify a revoked kid.

    Both timestamps are parsed and compared as instants (a string
    compare put ``...16.849Z`` before ``...16Z``). The signed_at field is
    required by the receipt spec; a missing or unparseable signed_at or
    effective time is treated as "cannot determine — fail closed".
    """
    entries = _revocation_entries(revoked_kids)
    if not isinstance(receipt, dict):
        return  # not our problem here; envelope-shape check catches it later
    kid = receipt.get("kid")
    if not isinstance(kid, str):
        return  # not our problem here; envelope-shape check catches it later
    payload = receipt.get("payload")
    signed_at = payload.get("signed_at") if isinstance(payload, dict) else None

    for entry in entries:
        if entry.get("kid") != kid:
            continue
        effective_at = entry.get("effective_revocation_at")
        effective = _instant_ms(effective_at)
        if effective is None:
            # Malformed revocation entry; defensive fail-closed.
            raise VerifyError(
                ErrorClass.REVOKED_KID,
                f"kid {kid!r} appears on revocation list with malformed "
                f"effective_revocation_at={effective_at!r}; failing closed",
            )
        signed = _instant_ms(signed_at)
        if signed is None:
            # Receipt has no parseable signed_at; can't compare; fail closed.
            raise VerifyError(
                ErrorClass.REVOKED_KID,
                f"kid {kid!r} on revocation list and receipt has no "
                f"parseable signed_at; failing closed",
            )
        if signed >= effective:
            raise VerifyError(
                ErrorClass.REVOKED_KID,
                f"kid {kid!r} revoked effective {effective_at}; "
                f"receipt signed at {signed_at} (>= effective time)",
            )
        # signed_at < effective_at: legitimate historical receipt,
        # continue verification.
        return


# Payload fields REQUIRED by sum.render_receipt.v1, per
# docs/RENDER_RECEIPT_FORMAT.md 1.1. The `schema` field sits OUTSIDE the
# signature, so it is attacker-editable: relabelling another receipt family
# to this schema makes the schema check pass on a payload this verifier has
# never validated. Checking the payload shape closes that, because a foreign
# payload cannot carry this family's fields.
REQUIRED_PAYLOAD_FIELDS = frozenset({
    "render_id",
    "sliders_quantized",
    "triples_hash",
    "tome_hash",
    "model",
    "provider",
    "signed_at",
    "digital_source_type",
})


def _check_payload_shape(receipt: dict) -> None:
    """Reject a payload that does not carry this receipt family's fields.

    Receipt-type confusion is the same bug class as JWT alg-confusion: an
    unsigned discriminator decides which validator runs. The signature alone
    does not prevent it, because a genuine signature over a DIFFERENT
    family's payload is still a genuine signature.
    """
    payload = receipt.get("payload") if isinstance(receipt, dict) else None
    if not isinstance(payload, dict):
        raise VerifyError(
            ErrorClass.MALFORMED_RECEIPT,
            "payload must be a JSON object, got "
            f"{type(payload).__name__}",
        )
    missing = sorted(REQUIRED_PAYLOAD_FIELDS - payload.keys())
    if missing:
        raise VerifyError(
            ErrorClass.MALFORMED_RECEIPT,
            f"payload declares schema {SUPPORTED_SCHEMA!r} but is missing "
            f"required field(s) {missing}: refusing to verify a payload of "
            "another receipt family (schema is not covered by the signature)",
        )


def verify_receipt(
    receipt,
    jwks,
    revoked_kids=None,
    *,
    max_age_seconds=None,
    max_future_skew_seconds=60,
) -> VerifyResult:
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
        result = verify_jose_envelope(
            receipt,
            jwks,
            supported_schema=SUPPORTED_SCHEMA,
            known_crit_extensions=KNOWN_CRIT_EXTENSIONS,
            max_age_seconds=max_age_seconds,
            max_future_skew_seconds=max_future_skew_seconds,
        )
    except JoseEnvelopeError as e:
        # Re-raise as the receipt-specific subclass so callers
        # importing only `VerifyError` still get a useful match.
        raise VerifyError(e.error_class, str(e)) from e

    # AFTER the signature is proven. Ordering is deliberate twice over: it
    # preserves the malformed_jws / signature_invalid precedence the error
    # taxonomy already guarantees, and it refuses to leak payload-shape
    # information to a caller who has not yet produced a valid signature.
    _check_payload_shape(receipt)
    return result
