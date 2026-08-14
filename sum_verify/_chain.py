"""Dependency-light verify for ``sum.chain_receipt.v1`` — the certified
chain: an ordered sequence of meaning-risk receipts bound into one signed,
replayable certificate.

What a chain receipt binds (and the honest split it enforces):

- **Structure**: an ordered list of per-hop receipt hashes (``sha256`` over
  the RFC 8785 JCS bytes of each full hop envelope) plus a ``chain_id``
  derived from that order — reordering, dropping, or substituting a hop
  breaks replay.
- **The composed budget (provable)**: ``budget_micro`` = the integer-exact
  sum of the hops' ``risk_upper_bound_micro``; ``joint_delta_micro`` = the
  sum of their ``delta_micro``. By the Bonferroni union bound this bounds
  the SUM of per-hop expected proxy losses with confidence
  ``>= 1 - joint_delta``. Composition rule is named in the payload
  (``bonferroni_additive.v1``).
- **The end-to-end leg (measured, optional)**: a direct certification over
  source→final losses, with its own ``losses_hash`` replay anchor. It is
  carried SEPARATELY because the additive budget does NOT bound end-to-end
  loss — the proxy is a directed loss, not a metric; no triangle
  inequality holds in either direction (``drift_budget.py`` measures both
  over- and under-counting regimes). A verifier here never lets one leg
  masquerade as the other.

This is the single implementation (no numpy anywhere in the chain path);
the research namespace re-exports it. Issuance lives in
``sum_engine_internal.research.meaning.chain_receipt``.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

from sum_engine_internal.infrastructure.jcs import canonicalize
from sum_engine_internal.infrastructure.jose_envelope import (
    SumVerifyError,
    verify_jose_envelope,
)
from sum_verify._conformal import certify_meaning_risk
from sum_verify._meaning import (
    SUPPORTED_SCHEMA as MEANING_RISK_SCHEMA,
)
from sum_verify._meaning import (
    MeaningReceiptDisclosureError,
    MeaningReceiptReplayError,
    _from_micro,
    _has_visible_text,
    _quantized,
    _require_int_micro,
    _require_str,
    _to_micro,
    _unwrap_loss_vector,
    _validate_side_band_losses,
    losses_hash,
    verify_meaning_risk_receipt,
)

SUPPORTED_SCHEMA = "sum.chain_receipt.v1"
COMPOSITION_RULE = "bonferroni_additive.v1"

# The per-hop payload fields the chain mirrors verbatim. Mirroring lets a
# consumer read the whole chain from the chain receipt alone; replay then
# proves each mirror equals the referenced hop payload exactly.
_HOP_MIRROR_INT_FIELDS = ("risk_upper_bound_micro", "delta_micro", "n")
_HOP_MIRROR_STR_FIELDS = ("method", "corpus_id", "transform", "scorer")


class ChainReceiptReplayError(SumVerifyError):
    """Cryptographically valid chain receipt whose side-band hop envelopes
    or end-to-end losses do not reproduce its committed hashes, mirrors,
    sums, or bound."""


class ChainReceiptDisclosureError(SumVerifyError):
    """Cryptographically valid chain receipt missing a required disclosure
    (``not_covered`` non-empty, ``disclosure`` visible text, or the
    ``budget_scope`` statement that keeps the additive budget from
    masquerading as an end-to-end bound)."""


def _chain_int(obj: dict, key: str) -> int:
    """``_require_int_micro`` re-raised in the CHAIN error taxonomy: a
    malformed integer field on a chain payload/hop mirror is a chain replay
    failure, and a caller catching ``ChainReceiptReplayError`` must see it
    (both classes derive from ``SumVerifyError``, but the specific class is
    part of the documented taxonomy)."""
    try:
        return _require_int_micro(obj, key)
    except MeaningReceiptReplayError as e:
        raise ChainReceiptReplayError(str(e)) from e


def _chain_str(obj: dict, key: str) -> str:
    """``_require_str`` re-raised in the CHAIN error taxonomy — the string
    sibling of ``_chain_int`` (used for the end-to-end leg's ``method``)."""
    try:
        return _require_str(obj, key)
    except MeaningReceiptReplayError as e:
        raise ChainReceiptReplayError(str(e)) from e


def canonical_receipt_hash(envelope: Any) -> str:
    """``sha256-<hex>`` over the RFC 8785 canonical bytes of a full receipt
    envelope — the hash a chain hop commits."""
    return "sha256-" + hashlib.sha256(canonicalize(envelope)).hexdigest()


def chain_id_for(hop_hashes: Sequence[str]) -> str:
    """Order-binding chain id: first 16 hex of sha256 over the JCS bytes of
    the ordered hop-hash list (same derivation shape as transform ids)."""
    return hashlib.sha256(canonicalize(list(hop_hashes))).hexdigest()[:16]


# Fields that make a payload a CHAIN receipt rather than a sibling family.
# See the note in _meaning.py: `schema` is outside the signature, so a genuine
# signature over another family's payload still verifies, and the shared
# not_covered / disclosure gate cannot discriminate.
REQUIRED_PAYLOAD_FIELDS = (
    "chain_id",
    "hops",
    "n_hops",
    "composition_rule",
    "budget_micro",
    "joint_delta_micro",
)


def _check_payload_shape(payload: object) -> None:
    """Reject a payload that is not of this receipt family. Fails closed."""
    if not isinstance(payload, dict):
        raise ChainReceiptDisclosureError(
            f"payload must be a JSON object, got {type(payload).__name__}"
        )
    missing = [f for f in REQUIRED_PAYLOAD_FIELDS if f not in payload]
    if missing:
        raise ChainReceiptDisclosureError(
            "payload declares schema sum.chain_receipt.v1 but is missing "
            f"required field(s) {missing}: refusing to verify a payload of "
            "another receipt family (schema is not covered by the signature)"
        )


def _check_disclosures(payload: dict) -> None:
    _check_payload_shape(payload)
    not_covered = payload.get("not_covered")
    if not isinstance(not_covered, list) or not not_covered:
        raise ChainReceiptDisclosureError(
            "payload.not_covered must be a non-empty list; got "
            f"{not_covered!r}"
        )
    for key in ("disclosure", "budget_scope"):
        v = payload.get(key)
        if not isinstance(v, str) or not _has_visible_text(v):
            raise ChainReceiptDisclosureError(
                f"payload.{key} must be a non-empty string with visible "
                f"text; got {v!r}"
            )


def _check_internal_consistency(payload: dict) -> list[dict]:
    """Payload-internal checks that need no side-band: hop shape, ordered
    indices, integer sums, chain id. Returns the hops list."""
    hops = payload.get("hops")
    if not isinstance(hops, list) or len(hops) < 2:
        raise ChainReceiptReplayError(
            f"payload.hops must be a list of >= 2 hops; got "
            f"{type(hops).__name__ if not isinstance(hops, list) else len(hops)}"
        )
    if _chain_int(payload, "n_hops") != len(hops):
        raise ChainReceiptReplayError(
            f"n_hops={payload.get('n_hops')} does not match len(hops)="
            f"{len(hops)}"
        )
    if payload.get("composition_rule") != COMPOSITION_RULE:
        raise ChainReceiptReplayError(
            f"unknown composition_rule {payload.get('composition_rule')!r}; "
            f"this verifier implements {COMPOSITION_RULE!r}"
        )
    budget = 0
    joint_delta = 0
    hop_hashes = []
    for i, hop in enumerate(hops):
        if not isinstance(hop, dict):
            raise ChainReceiptReplayError(f"hops[{i}] is not an object")
        if hop.get("index") != i + 1:
            raise ChainReceiptReplayError(
                f"hops[{i}].index must be {i + 1} (1-based, ordered); got "
                f"{hop.get('index')!r}"
            )
        if hop.get("schema") != MEANING_RISK_SCHEMA:
            raise ChainReceiptReplayError(
                f"hops[{i}].schema must be {MEANING_RISK_SCHEMA!r} in v1; "
                f"got {hop.get('schema')!r}"
            )
        h = hop.get("receipt_hash")
        if (
            not isinstance(h, str)
            or not h.startswith("sha256-")
            or len(h) != 71
        ):
            raise ChainReceiptReplayError(
                f"hops[{i}].receipt_hash malformed: {h!r}"
            )
        hop_hashes.append(h)
        budget += _chain_int(hop, "risk_upper_bound_micro")
        joint_delta += _chain_int(hop, "delta_micro")
    if _chain_int(payload, "budget_micro") != budget:
        raise ChainReceiptReplayError(
            f"budget_micro does not replay: receipt claims "
            f"{payload.get('budget_micro')} but the hop mirrors sum to "
            f"{budget}"
        )
    if _chain_int(payload, "joint_delta_micro") != joint_delta:
        raise ChainReceiptReplayError(
            f"joint_delta_micro does not replay: receipt claims "
            f"{payload.get('joint_delta_micro')} but the hop mirrors sum "
            f"to {joint_delta}"
        )
    if payload.get("chain_id") != chain_id_for(hop_hashes):
        raise ChainReceiptReplayError(
            f"chain_id does not replay: receipt claims "
            f"{payload.get('chain_id')!r} but the ordered hop hashes "
            f"derive {chain_id_for(hop_hashes)!r}"
        )
    return hops


def verify_chain_receipt(
    envelope: Any,
    jwks: Any,
    *,
    hop_envelopes: Sequence[Any] | None = None,
    end_to_end_losses: Sequence[float] | None = None,
    max_age_seconds: int | None = None,
) -> dict[str, Any]:
    """Verify a ``sum.chain_receipt.v1`` envelope.

    Always: full JOSE verification, disclosure invariants, and the
    payload-internal replay (ordered indices, integer-exact budget and
    joint-delta sums, chain-id derivation).

    With ``hop_envelopes`` (the per-hop receipt envelopes, in order): each
    hop envelope must hash to its committed ``receipt_hash``, must itself
    verify against ``jwks`` (a JWKS may carry multiple kids — multi-issuer
    chains supply one JWKS containing every issuer's key), and every
    mirrored field must equal the hop payload exactly. ``max_age_seconds``
    windows the CHAIN envelope only — hops predate the chain by
    construction and are verified without a window. Note this verifies
    each hop's signature and disclosures; each hop's own LOSS replay
    remains available independently via ``verify_meaning_risk_receipt``
    with that hop's side-band losses.

    With ``end_to_end_losses`` (requires the payload to carry an
    ``end_to_end`` leg): replays the direct source→final certification
    exactly like a meaning-risk receipt (hash anchor, re-certify over the
    quantised committed vector, integer-exact bound / point estimate / n).

    Returns the verified payload dict on success.
    """
    result = verify_jose_envelope(
        envelope,
        jwks,
        supported_schema=SUPPORTED_SCHEMA,
        max_age_seconds=max_age_seconds,
    )
    payload = result.payload

    _check_disclosures(payload)
    hops = _check_internal_consistency(payload)

    if hop_envelopes is not None:
        if not isinstance(hop_envelopes, (list, tuple)):
            raise ChainReceiptReplayError(
                f"hop_envelopes must be a list of receipt envelopes; got "
                f"{hop_envelopes!r}"
            )
        if len(hop_envelopes) != len(hops):
            raise ChainReceiptReplayError(
                f"supplied {len(hop_envelopes)} hop envelopes but the "
                f"receipt commits {len(hops)} hops"
            )
        for i, (hop, env) in enumerate(zip(hops, hop_envelopes)):
            # A caller-supplied hop envelope is untrusted input: a value JCS
            # cannot canonicalize (NaN, which json.load accepts by default; a
            # non-JSON-serialisable object) must fail closed in the chain
            # replay taxonomy, not as a raw ValueError/TypeError out of
            # canonicalize — the same guard jose_envelope.py wraps its own
            # canonicalize call in.
            try:
                actual_hash = canonical_receipt_hash(env)
            except (TypeError, ValueError) as e:
                raise ChainReceiptReplayError(
                    f"hops[{i}]: supplied envelope is not JCS-canonicalizable: "
                    f"{e}"
                ) from e
            if actual_hash != hop["receipt_hash"]:
                raise ChainReceiptReplayError(
                    f"hops[{i}]: supplied envelope hashes to {actual_hash} "
                    f"but the chain commits {hop['receipt_hash']}"
                )
            try:
                # No replay window on the hops: hops predate the chain BY
                # CONSTRUCTION, so a caller's max_age_seconds (freshness of
                # the CHAIN attestation) must not false-reject a fresh chain
                # over legitimately old hop receipts. A caller who cares
                # about a hop's own age verifies that hop directly.
                hop_payload = verify_meaning_risk_receipt(env, jwks)
            except SumVerifyError as e:
                raise ChainReceiptReplayError(
                    f"hops[{i}]: referenced receipt fails verification: {e}"
                ) from e
            for f in _HOP_MIRROR_INT_FIELDS:
                if hop.get(f) != hop_payload.get(f):
                    raise ChainReceiptReplayError(
                        f"hops[{i}].{f} mirror mismatch: chain says "
                        f"{hop.get(f)!r}, hop payload says "
                        f"{hop_payload.get(f)!r}"
                    )
            for f in _HOP_MIRROR_STR_FIELDS:
                if hop.get(f) != hop_payload.get(f):
                    raise ChainReceiptReplayError(
                        f"hops[{i}].{f} mirror mismatch: chain says "
                        f"{hop.get(f)!r}, hop payload says "
                        f"{hop_payload.get(f)!r}"
                    )

    if end_to_end_losses is not None:
        leg = payload.get("end_to_end")
        if not isinstance(leg, dict):
            raise ChainReceiptReplayError(
                "end_to_end losses supplied but the receipt carries no "
                "end_to_end leg"
            )
        losses = _unwrap_loss_vector(end_to_end_losses)
        _validate_side_band_losses(losses)
        recomputed = losses_hash(losses)
        if recomputed != leg.get("losses_hash"):
            raise ChainReceiptReplayError(
                f"end_to_end.losses_hash mismatch: supplied losses hash to "
                f"{recomputed} but receipt commits {leg.get('losses_hash')!r}"
            )
        try:
            replay = certify_meaning_risk(
                _quantized(losses),
                scorer_name=str(leg.get("scorer", "")),
                scorer_version=str(leg.get("scorer_version", "")),
                delta=_from_micro(_chain_int(leg, "delta_micro")),
                method=_chain_str(leg, "method"),
            )
        except ValueError as e:
            raise ChainReceiptReplayError(
                f"end_to_end losses are not valid [0,1] data: {e}"
            ) from e
        if _chain_int(leg, "risk_upper_bound_micro") != _to_micro(
            replay.risk_upper_bound
        ):
            raise ChainReceiptReplayError(
                f"end_to_end.risk_upper_bound does not replay: receipt "
                f"claims {leg['risk_upper_bound_micro']} micro but "
                f"re-certification yields "
                f"{_to_micro(replay.risk_upper_bound)} micro"
            )
        if _chain_int(leg, "point_estimate_micro") != _to_micro(
            replay.point_estimate
        ):
            raise ChainReceiptReplayError(
                f"end_to_end.point_estimate does not replay: receipt "
                f"claims {leg['point_estimate_micro']} micro but "
                f"re-certification yields "
                f"{_to_micro(replay.point_estimate)} micro"
            )
        if _chain_int(leg, "n") != replay.n:
            raise ChainReceiptReplayError(
                f"end_to_end.n does not replay: receipt claims "
                f"n={leg['n']} but the losses contain {replay.n} samples"
            )

    return payload


# Re-exported error aliases so ``except`` clauses read naturally whether a
# caller thinks in meaning- or chain-receipt terms.
__all__ = [
    "SUPPORTED_SCHEMA",
    "COMPOSITION_RULE",
    "ChainReceiptReplayError",
    "ChainReceiptDisclosureError",
    "canonical_receipt_hash",
    "chain_id_for",
    "verify_chain_receipt",
    "MeaningReceiptReplayError",
    "MeaningReceiptDisclosureError",
]
