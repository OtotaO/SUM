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
    if not isinstance(disclosure, str) or not disclosure.strip():
        raise MeaningReceiptDisclosureError(
            "payload.disclosure must be a non-empty string stating the "
            f"proxy / marginal / exchangeability caveat; got {disclosure!r}"
        )

    if losses is None:
        return payload

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
            delta=_from_micro(int(payload["delta_micro"])),
            method=payload["method"],
        )
    except ValueError as e:
        raise MeaningReceiptReplayError(
            f"committed losses are not valid [0,1] data: {e}"
        ) from e

    # ---- replay step 3: bound + point estimate match (integer micro) ----
    want_ub = int(payload["risk_upper_bound_micro"])
    got_ub = _to_micro(replay.risk_upper_bound)
    if want_ub != got_ub:
        raise MeaningReceiptReplayError(
            f"risk_upper_bound does not replay: receipt claims {want_ub} "
            f"micro but re-certification yields {got_ub} micro"
        )
    want_pe = int(payload["point_estimate_micro"])
    got_pe = _to_micro(replay.point_estimate)
    if want_pe != got_pe:
        raise MeaningReceiptReplayError(
            f"point_estimate does not replay: receipt claims {want_pe} "
            f"micro but re-certification yields {got_pe} micro"
        )

    # ---- replay step 4: sample size matches the committed losses ----
    if int(payload["n"]) != replay.n:
        raise MeaningReceiptReplayError(
            f"n does not replay: receipt claims n={payload['n']} but the "
            f"committed losses contain {replay.n} samples"
        )

    # ---- replay step 5: the operational pass/fail decision ----
    if "alpha_target_micro" in payload:
        alpha = _from_micro(int(payload["alpha_target_micro"]))
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
