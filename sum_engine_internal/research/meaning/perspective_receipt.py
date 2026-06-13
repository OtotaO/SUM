"""Wire format + sign/verify for ``sum.perspective_risk_receipt.v1``.

A **Perspective Receipt**: the signed, replayable artifact form of
group-conditional meaning-risk control. Where ``sum.meaning_risk_receipt.v1``
certifies one *marginal* bound (average meaning-loss), this certifies the
marginal bound PLUS a separate, valid-within-its-cohort bound for each
declared perspective / cohort (language, genre, audience) — and seals all
of them in one signature. It is the receipt that says "…and within *every*
named perspective, not just on average."

It reuses the meaning-risk receipt's machinery wholesale — the integer
micro-unit float-free encoding, JCS canonicalisation, Ed25519/detached-JWS
signing, the disclosure-invariant and replay error classes — and adds one
thing: a per-cohort sub-bound array + a single ``evidence_hash`` that
commits the (loss, cohort) pairing, so a verifier handed the losses and
their cohort labels can replay every cohort bound byte-for-byte.

What a verified perspective receipt proves — and does NOT:
PROVES (signature + replay): authentic signature; required disclosure
present; the committed (loss, cohort) evidence hashes to ``evidence_hash``;
re-running the group-conditional certifier reproduces the marginal bound
AND every cohort's bound (and ``controlled`` / ``controls_all`` when an
``alpha_target`` is set).
Does NOT prove: meaning was preserved (a named PROXY, per-cohort
*marginally within the cohort*, under exchangeability); the per-cohort
bounds pay full finite-sample cost (small cohorts are wide); nothing about
the ``not_covered`` layers.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Sequence

from sum_engine_internal.infrastructure.jcs import canonicalize
from sum_engine_internal.infrastructure.jose_envelope import (
    sign_jose_envelope,
    verify_jose_envelope,
)
from sum_engine_internal.research.meaning.conformal_meaning import (
    GroupedMeaningRisk,
    certify_meaning_risk_by_group,
)
# Reuse the meaning-risk receipt's float-free micro encoding + disclosure
# constants + error classes — same trust discipline, one place.
from sum_engine_internal.research.meaning.receipt import (
    DEFAULT_NOT_COVERED,
    MeaningReceiptDisclosureError,
    MeaningReceiptReplayError,
    _DEFAULT_DISCLOSURE,
    _has_visible_text,
    _quantized,
    _require_int_micro,
    _to_micro,
    _from_micro,
    _validate_side_band_losses,
)

SUPPORTED_SCHEMA = "sum.perspective_risk_receipt.v1"


def evidence_hash(losses: Sequence[float], group_ids: Sequence[str]) -> str:
    """``"sha256-<hex>"`` over the JCS-canonical bytes of the
    ``[[micro_loss, cohort], …]`` pairing — float-free (integer micro-units
    + string cohort ids), so it is byte-stable cross-runtime and commits
    BOTH the losses and their cohort assignment in one anchor."""
    pairs = [[_to_micro(x), str(g)] for x, g in zip(losses, group_ids)]
    return "sha256-" + hashlib.sha256(canonicalize(pairs)).hexdigest()


def _group_block(guarantee, *, with_controlled: bool, alpha_q: float | None) -> dict:
    block = {
        "n": guarantee.n,
        "point_estimate_micro": _to_micro(guarantee.point_estimate),
        "risk_upper_bound_micro": _to_micro(guarantee.risk_upper_bound),
    }
    if with_controlled and alpha_q is not None:
        block["controlled"] = bool(guarantee.controls(alpha_q))
    return block


def build_perspective_payload(
    *,
    grouped: GroupedMeaningRisk,
    losses: Sequence[float],
    group_ids: Sequence[str],
    corpus_id: str,
    transform: str,
    loss_definition: str,
    alpha_target: float | None = None,
    not_covered: Sequence[str] = DEFAULT_NOT_COVERED,
    disclosure: str = _DEFAULT_DISCLOSURE,
    signed_at: str | None = None,
) -> dict[str, Any]:
    """Assemble a ``sum.perspective_risk_receipt.v1`` payload.

    ``grouped`` supplies the certification parameters (scorer, delta,
    method, simultaneous); the bounds are **(re)certified over the quantised
    committed losses** — exactly as the marginal receipt does — so producer
    and verifier compute over byte-identical inputs and replay is exact.
    """
    if not not_covered:
        raise ValueError(
            "not_covered must be non-empty — declare the proxy's blind spots"
        )
    if not (len(losses) == len(group_ids) == grouped.marginal.n):
        raise ValueError(
            f"losses ({len(losses)}), group_ids ({len(group_ids)}) and "
            f"grouped.marginal.n ({grouped.marginal.n}) must all agree"
        )
    if signed_at is None:
        now = datetime.now(timezone.utc)
        signed_at = (
            now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
        )

    rounded = _quantized(losses)
    # Certify at the SAME quantised delta verify recomputes from
    # delta_micro — else an off-grid delta (>6 dp, e.g. a Bonferroni
    # threshold or 1/30) shifts every bound by ~1 micro and false-rejects a
    # genuine receipt. This is the #275 bug class; the loss vector is
    # already quantised via _quantized, the scalar delta must be too.
    delta = _from_micro(_to_micro(grouped.marginal.delta))
    canonical = certify_meaning_risk_by_group(
        rounded, group_ids,
        scorer_name=grouped.marginal.scorer_name,
        scorer_version=grouped.marginal.scorer_version,
        delta=delta, method=grouped.marginal.method,
        simultaneous=grouped.simultaneous,
    )
    alpha_q = _from_micro(_to_micro(alpha_target)) if alpha_target is not None else None
    with_ctrl = alpha_target is not None

    payload: dict[str, Any] = {
        "scorer": grouped.marginal.scorer_name,
        "scorer_version": grouped.marginal.scorer_version,
        "loss_definition": loss_definition,
        "method": grouped.marginal.method,
        "delta_micro": _to_micro(delta),
        "n": grouped.marginal.n,
        "simultaneous": bool(grouped.simultaneous),
        "evidence_hash": evidence_hash(losses, group_ids),
        "marginal_point_estimate_micro": _to_micro(canonical.marginal.point_estimate),
        "marginal_risk_upper_bound_micro": _to_micro(canonical.marginal.risk_upper_bound),
        "groups": [
            {"group_id": gid, **_group_block(canonical.groups[gid],
                                             with_controlled=with_ctrl, alpha_q=alpha_q)}
            for gid in sorted(canonical.groups)
        ],
        "corpus_id": corpus_id,
        "transform": transform,
        "not_covered": list(not_covered),
        "disclosure": disclosure,
        "signed_at": signed_at,
    }
    if alpha_target is not None:
        payload["alpha_target_micro"] = _to_micro(alpha_target)
        payload["controls_all"] = bool(canonical.controls_all(alpha_q))
    return payload


def sign_perspective_risk_receipt(
    payload: dict[str, Any], *, private_jwk: dict[str, Any], kid: str,
) -> dict[str, Any]:
    """Produce a signed ``sum.perspective_risk_receipt.v1`` envelope."""
    envelope = sign_jose_envelope(payload, private_jwk=private_jwk, kid=kid)
    envelope["schema"] = SUPPORTED_SCHEMA
    return envelope


def verify_perspective_risk_receipt(
    envelope: Any,
    jwks: Any,
    *,
    losses: Sequence[float] | None = None,
    group_ids: Sequence[str] | None = None,
    max_age_seconds: int | None = None,
) -> dict[str, Any]:
    """Verify a ``sum.perspective_risk_receipt.v1`` envelope (signature +
    disclosure always; per-cohort replay when ``losses`` + ``group_ids``
    are supplied side-band).

    Raises ``MeaningReceiptDisclosureError`` (missing disclosure) or
    ``MeaningReceiptReplayError`` (evidence hash / any bound / controlled /
    controls_all does not reproduce).
    """
    result = verify_jose_envelope(
        envelope, jwks, supported_schema=SUPPORTED_SCHEMA,
        max_age_seconds=max_age_seconds,
    )
    payload = result.payload

    # disclosure invariants (always)
    nc = payload.get("not_covered")
    if not isinstance(nc, list) or not nc:
        raise MeaningReceiptDisclosureError(
            f"payload.not_covered must be a non-empty list; got {nc!r}"
        )
    disc = payload.get("disclosure")
    if not isinstance(disc, str) or not _has_visible_text(disc):
        raise MeaningReceiptDisclosureError(
            f"payload.disclosure must be a non-empty string with visible "
            f"text; got {disc!r}"
        )

    if losses is None or group_ids is None:
        return payload

    # Reject malformed side-band losses cleanly before _quantized / evidence_hash
    # (a NaN/inf would otherwise raise an unhandled numeric exception).
    _validate_side_band_losses(losses)

    # Guard the zip in evidence_hash against silent truncation on a
    # length mismatch (the hash would still differ, but fail clearly here).
    if len(losses) != len(group_ids):
        raise MeaningReceiptReplayError(
            f"losses ({len(losses)}) and group_ids ({len(group_ids)}) "
            f"side-band lengths disagree"
        )

    # ---- replay 1: evidence anchor (losses + cohort assignment) ----
    got = evidence_hash(losses, group_ids)
    if got != payload.get("evidence_hash"):
        raise MeaningReceiptReplayError(
            f"evidence_hash mismatch: supplied evidence hashes to {got} but "
            f"receipt commits {payload.get('evidence_hash')!r}"
        )

    # ---- replay 2: re-certify group-conditional over the quantised vector ----
    rounded = _quantized(losses)
    try:
        replay = certify_meaning_risk_by_group(
            rounded, group_ids,
            scorer_name=str(payload.get("scorer", "")),
            scorer_version=str(payload.get("scorer_version", "")),
            delta=_from_micro(_require_int_micro(payload, "delta_micro")),
            method=payload["method"],
            simultaneous=bool(payload.get("simultaneous", False)),
        )
    except ValueError as e:
        raise MeaningReceiptReplayError(
            f"committed losses are not valid [0,1] data: {e}"
        ) from e

    # ---- replay 3: marginal + total n ----
    if _require_int_micro(payload, "marginal_risk_upper_bound_micro") != _to_micro(replay.marginal.risk_upper_bound):
        raise MeaningReceiptReplayError("marginal_risk_upper_bound does not replay")
    if _require_int_micro(payload, "marginal_point_estimate_micro") != _to_micro(replay.marginal.point_estimate):
        raise MeaningReceiptReplayError("marginal_point_estimate does not replay")
    if _require_int_micro(payload, "n") != replay.marginal.n:
        raise MeaningReceiptReplayError(
            f"n does not replay: receipt claims {payload['n']}, evidence has "
            f"{replay.marginal.n}"
        )

    # ---- replay 4: every cohort sub-bound ----
    has_alpha = "alpha_target_micro" in payload
    alpha_q = _from_micro(_require_int_micro(payload, "alpha_target_micro")) if has_alpha else None
    # Reject duplicate cohort ids BEFORE collapsing to a dict: otherwise a
    # forged duplicate entry (e.g. a second 'legalese' with a zeroed bound)
    # would be silently dropped (last-wins) and evade replay, while a
    # downstream reader iterating payload["groups"] sees the forged bound.
    _ids = [g["group_id"] for g in payload["groups"]]
    if len(_ids) != len(set(_ids)):
        raise MeaningReceiptReplayError(
            f"duplicate cohort id(s) in payload.groups: {_ids}"
        )
    payload_groups = {g["group_id"]: g for g in payload["groups"]}
    if set(payload_groups) != set(replay.groups):
        raise MeaningReceiptReplayError(
            f"cohort set does not replay: receipt {sorted(payload_groups)} vs "
            f"evidence {sorted(replay.groups)}"
        )
    for gid, g in payload_groups.items():
        rg = replay.groups[gid]
        if _require_int_micro(g, "n") != rg.n:
            raise MeaningReceiptReplayError(f"cohort {gid!r}: n does not replay")
        if _require_int_micro(g, "risk_upper_bound_micro") != _to_micro(rg.risk_upper_bound):
            raise MeaningReceiptReplayError(
                f"cohort {gid!r}: risk_upper_bound does not replay"
            )
        if _require_int_micro(g, "point_estimate_micro") != _to_micro(rg.point_estimate):
            raise MeaningReceiptReplayError(
                f"cohort {gid!r}: point_estimate does not replay"
            )
        if has_alpha and bool(g.get("controlled")) != rg.controls(alpha_q):
            raise MeaningReceiptReplayError(
                f"cohort {gid!r}: controlled does not replay"
            )

    # ---- replay 5: the headline decision ----
    if has_alpha:
        if bool(payload.get("controls_all")) != replay.controls_all(alpha_q):
            raise MeaningReceiptReplayError(
                "controls_all does not replay: the all-cohorts decision "
                "contradicts the re-certified cohort bounds"
            )

    return payload
