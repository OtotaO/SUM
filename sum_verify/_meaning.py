"""Dependency-light verify for ``sum.meaning_risk_receipt.v1``.

A verify-side mirror of
``sum_engine_internal.research.meaning.receipt.verify_meaning_risk_receipt``
that performs the identical cryptographic, disclosure, and replay checks
but routes the bound recompute through the pure-Python kernel in
``sum_verify._conformal`` instead of the numpy/scipy one. The result: an
integrator can replay a meaning-risk certificate offline with only
``cryptography`` + ``joserfc`` installed — no ``[research]`` stack.

The trust root is shared, not reimplemented: JCS canonicalisation and the
Ed25519/JWS envelope verifier are imported from
``sum_engine_internal.infrastructure`` unchanged. Only the conformal
*arithmetic* is re-derived (see ``_conformal`` for the divergence guard).

The micro-unit wire helpers below are copied verbatim from the canonical
``receipt`` module because they ARE the wire contract — the exact
quantisation a producer committed to. ``Tests/test_sum_verify_sdk.py``
pins this copy to the original by replaying the committed golden receipts
through both paths and asserting identical verdicts.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import hashlib
import math
import unicodedata
from typing import Any, Sequence

from sum_engine_internal.infrastructure.jcs import canonicalize
from sum_engine_internal.infrastructure.jose_envelope import verify_jose_envelope
from sum_verify._conformal import certify_meaning_risk

SUPPORTED_SCHEMA = "sum.meaning_risk_receipt.v1"

# Must match research.meaning.receipt exactly: 6-decimal / 1e6 micro grid.
_LOSS_DECIMALS = 6
_MICRO_SCALE = 10 ** _LOSS_DECIMALS


class MeaningReceiptReplayError(Exception):
    """Raised when a meaning-risk receipt is cryptographically valid but
    the supplied losses do not reproduce its committed hash, bound, point
    estimate, ``n``, or ``controlled`` flag. The signature is genuine, but
    the side-band evidence does not back the claim."""


class MeaningReceiptDisclosureError(Exception):
    """Raised when a meaning-risk receipt is cryptographically valid but
    omits a required disclosure field (``not_covered`` non-empty,
    ``disclosure`` non-empty). A receipt that passes every cryptographic
    check yet discloses nothing reads as a bare bound — the exact failure
    the receipt family exists to prevent. Enforced on every verify, with
    or without side-band losses."""


def _unwrap_loss_vector(obj: Any) -> Any:
    """Accept BOTH a bare ``[..]`` loss vector and the metadata-wrapped shape
    the committed fixtures use (``{"losses": [..], "judge": .., "note": ..}``),
    so ``verify(..., losses=json.load(open("losses.json")))`` runs verbatim on
    a committed losses file — the contract ``VERIFY_SDK.md`` documents ("bare
    list or {'losses': [...]}") and the ``sum verify-meaning`` / ``python -m
    sum_verify`` CLIs already honour. Without this, the documented library
    snippet crashes on the project's own golden (the wrapper's string keys hit
    ``float()``). Idempotent on a bare list."""
    if isinstance(obj, dict):
        for key in ("losses", "values"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
    return obj


def _to_micro(x: float) -> int:
    return int(round(float(x) * _MICRO_SCALE))


def _from_micro(m: int) -> float:
    return int(m) / _MICRO_SCALE


def _losses_micro(losses: Sequence[float]) -> list[int]:
    return [_to_micro(x) for x in losses]


def _quantized(losses: Sequence[float]) -> list[float]:
    return [_from_micro(m) for m in _losses_micro(losses)]


def losses_hash(losses: Sequence[float]) -> str:
    """``"sha256-<hex>"`` over the JCS-canonical bytes of the INTEGER
    micro-unit loss vector — the replay anchor committed in the payload."""
    return "sha256-" + hashlib.sha256(
        canonicalize(_losses_micro(losses))
    ).hexdigest()


def _has_visible_text(s: str) -> bool:
    """True iff ``s`` has a character that is neither whitespace nor a
    zero-width / format / control character. ``str.strip()`` does not remove
    U+200B / U+FEFF / other Cf/Cc code points, so ``"​"`` passes a naive
    ``not s.strip()`` test while rendering blank — a way to satisfy the
    disclosure invariant yet disclose nothing. Must match research.meaning."""
    return any(
        not (ch.isspace() or unicodedata.category(ch) in ("Cf", "Cc"))
        for ch in s
    )


def _require_int_micro(payload: dict[str, Any], key: str) -> int:
    """Return ``payload[key]`` only if a genuine JSON integer. The conformal
    wire is float-free integer micro-units: this rejects ``None``/missing with
    a clean replay error (vs an unhandled ``TypeError`` from ``int(None)``) and
    a coercible string like ``"645438"`` that a strict cross-runtime verifier
    would refuse — keeping runtimes in lockstep. ``bool`` is excluded."""
    v = payload.get(key)
    if type(v) is not int:
        raise MeaningReceiptReplayError(
            f"payload.{key} must be an integer micro-unit (float-free wire); "
            f"got {v!r}"
        )
    return v


def _validate_side_band_losses(losses: Sequence[float]) -> None:
    """Reject non-finite or out-of-[0,1] side-band losses with a clean replay
    error before ``losses_hash`` / ``_quantized``, where a NaN/inf would raise
    an unhandled ``ValueError``/``OverflowError`` from ``int(round(...))``."""
    for i, x in enumerate(losses):
        if (
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or not (0.0 <= float(x) <= 1.0)
        ):
            raise MeaningReceiptReplayError(
                f"side-band loss[{i}] must be a finite number in [0, 1]; "
                f"got {x!r}"
            )


def verify_meaning_risk_receipt(
    envelope: Any,
    jwks: Any,
    *,
    losses: Sequence[float] | None = None,
    max_age_seconds: int | None = None,
) -> dict[str, Any]:
    """Verify a ``sum.meaning_risk_receipt.v1`` envelope (dependency-light).

    Behaviour is identical to the canonical verifier:

      - always performs full JOSE verification (signature, schema, header
        invariants) and the structural disclosure checks;
      - when ``losses`` are supplied side-band, also replays the bound:
        confirms the losses hash matches, re-certifies over the quantised
        committed vector, and confirms ``risk_upper_bound_micro``,
        ``point_estimate_micro``, ``n``, and ``controlled`` all reproduce
        by exact integer equality.

    Returns the verified payload dict on success.
    """
    result = verify_jose_envelope(
        envelope,
        jwks,
        supported_schema=SUPPORTED_SCHEMA,
        max_age_seconds=max_age_seconds,
    )
    payload = result.payload

    # ---- structural disclosure invariants (always, losses or not) ----
    not_covered = payload.get("not_covered")
    if not isinstance(not_covered, list) or not not_covered:
        raise MeaningReceiptDisclosureError(
            "payload.not_covered must be a non-empty list declaring the "
            f"proxy's structural blind spots; got {not_covered!r}"
        )
    disclosure = payload.get("disclosure")
    if not isinstance(disclosure, str) or not _has_visible_text(disclosure):
        raise MeaningReceiptDisclosureError(
            "payload.disclosure must be a non-empty string with visible text "
            "stating the proxy / marginal / exchangeability caveat; got "
            f"{disclosure!r}"
        )

    if losses is None:
        return payload
    # Normalise the metadata-wrapped fixture shape to a bare vector, so the
    # documented snippet (a committed losses file loaded straight in) replays.
    losses = _unwrap_loss_vector(losses)

    # ---- replay step 0: reject malformed side-band losses cleanly ----
    _validate_side_band_losses(losses)

    # ---- replay step 1: hash anchor ----
    recomputed_hash = losses_hash(losses)
    if recomputed_hash != payload.get("losses_hash"):
        raise MeaningReceiptReplayError(
            f"losses_hash mismatch: supplied losses hash to "
            f"{recomputed_hash} but receipt commits "
            f"{payload.get('losses_hash')!r}"
        )

    # ---- replay step 2: re-certify over the ROUNDED committed vector ----
    rounded = _quantized(losses)
    try:
        replay = certify_meaning_risk(
            rounded,
            scorer_name=str(payload.get("scorer", "")),
            scorer_version=str(payload.get("scorer_version", "")),
            delta=_from_micro(_require_int_micro(payload, "delta_micro")),
            method=payload["method"],
        )
    except ValueError as e:
        raise MeaningReceiptReplayError(
            f"committed losses are not valid [0,1] data: {e}"
        ) from e

    # ---- replay step 3: bound + point estimate match (integer micro) ----
    want_ub = _require_int_micro(payload, "risk_upper_bound_micro")
    got_ub = _to_micro(replay.risk_upper_bound)
    if want_ub != got_ub:
        raise MeaningReceiptReplayError(
            f"risk_upper_bound does not replay: receipt claims {want_ub} "
            f"micro but re-certification yields {got_ub} micro"
        )
    want_pe = _require_int_micro(payload, "point_estimate_micro")
    got_pe = _to_micro(replay.point_estimate)
    if want_pe != got_pe:
        raise MeaningReceiptReplayError(
            f"point_estimate does not replay: receipt claims {want_pe} "
            f"micro but re-certification yields {got_pe} micro"
        )

    # ---- replay step 4: sample size matches the committed losses ----
    if _require_int_micro(payload, "n") != replay.n:
        raise MeaningReceiptReplayError(
            f"n does not replay: receipt claims n={payload['n']} but the "
            f"committed losses contain {replay.n} samples"
        )

    # ---- replay step 5: the operational pass/fail decision ----
    if "alpha_target_micro" in payload:
        alpha = _from_micro(_require_int_micro(payload, "alpha_target_micro"))
        expected_controlled = replay.controls(alpha)
        if bool(payload.get("controlled")) != expected_controlled:
            raise MeaningReceiptReplayError(
                f"controlled does not replay: receipt claims "
                f"controlled={payload.get('controlled')} at "
                f"alpha_target_micro={payload['alpha_target_micro']}, but "
                f"the replayed bound {_to_micro(replay.risk_upper_bound)} "
                f"micro gives controlled={expected_controlled}"
            )

    return payload
