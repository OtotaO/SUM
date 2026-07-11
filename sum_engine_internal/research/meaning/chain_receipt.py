"""Issuance for ``sum.chain_receipt.v1`` — the certified chain.

The convergent next move the drift-budget work seeded
(``docs/DRIFT_BUDGET.md`` named an unshipped receipt "binding the ordered
list of per-hop receipt hashes + the Bonferroni joint_delta"): in the
agentic era the unit of meaning-motion is a CHAIN — summarised, then
translated, then re-summarised — and occurrence-level receipts prove the
hops *happened* while nothing proves what the chain did to meaning. This
receipt binds:

1. the ordered per-hop ``sum.meaning_risk_receipt.v1`` envelopes, by
   canonical hash (reorder / drop / substitute → replay breaks);
2. the composed additive budget, integer-exact
   (``budget_micro = Σ risk_upper_bound_micro``,
   ``joint_delta_micro = Σ delta_micro`` — Bonferroni, the same math as
   ``drift_budget.compose_drift_budget_from_payloads``);
3. optionally, a DIRECTLY measured end-to-end leg (source→final losses
   certified like any meaning-risk bound, with its own replay anchor).

The honest split is structural: the additive budget bounds the SUM of
per-hop expected proxy losses; it does NOT bound end-to-end loss (the
proxy is a directed loss, not a metric — both over- and under-counting
regimes are real and measured; see ``drift_budget.py``). The payload
carries that statement in a mandatory ``budget_scope`` field a verifier
fails closed without.

Verification is implemented ONCE, dependency-light, in
``sum_verify._chain`` (the chain path needs no numpy); this module
re-exports it so research callers have one import site.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from sum_engine_internal.infrastructure.jose_envelope import (
    sign_jose_envelope,
)
from sum_engine_internal.research.meaning.receipt import (
    DEFAULT_NOT_COVERED,
)
from sum_engine_internal.research.meaning.receipt import (
    SUPPORTED_SCHEMA as MEANING_RISK_SCHEMA,
)
from sum_verify._chain import (
    COMPOSITION_RULE,
    SUPPORTED_SCHEMA,
    ChainReceiptDisclosureError,
    ChainReceiptReplayError,
    canonical_receipt_hash,
    chain_id_for,
    verify_chain_receipt,
)
from sum_verify._conformal import certify_meaning_risk
from sum_verify._meaning import (
    _quantized,
    _require_int_micro,
    _to_micro,
    losses_hash,
)

__all__ = [
    "SUPPORTED_SCHEMA",
    "COMPOSITION_RULE",
    "DEFAULT_CHAIN_DISCLOSURE",
    "BUDGET_SCOPE_STATEMENT",
    "build_end_to_end_leg",
    "build_chain_payload",
    "sign_chain_receipt",
    "verify_chain_receipt",
    "canonical_receipt_hash",
    "chain_id_for",
    "ChainReceiptReplayError",
    "ChainReceiptDisclosureError",
]

DEFAULT_CHAIN_DISCLOSURE = (
    "This certificate binds an ordered chain of per-hop meaning-risk "
    "certificates and their composed additive budget. Every per-hop "
    "caveat applies unchanged: each bound is over a NAMED PROXY for "
    "meaning-loss, marginal over its calibration corpus, valid under "
    "exchangeability. Composition adds no new knowledge about any "
    "single document's fate across the chain."
)

# Mandatory, verifier-enforced. THE honesty line for composition: without
# it a reader will read Σ(bounds) as an end-to-end guarantee — the exact
# overclaim the drift-budget audit measured both failure directions of.
BUDGET_SCOPE_STATEMENT = (
    "budget_micro bounds the SUM of per-hop expected proxy losses "
    "(Bonferroni union bound: joint confidence >= 1 - joint_delta). It "
    "does NOT bound the end-to-end loss: the proxy is a directed loss, "
    "not a metric, and no triangle inequality holds in either direction. "
    "The end_to_end leg, when present, is a separate DIRECT measurement "
    "over source-to-final pairs with its own replay anchor."
)


def build_end_to_end_leg(
    losses: Sequence[float],
    *,
    scorer_name: str,
    scorer_version: str,
    loss_definition: str,
    delta: float = 0.05,
    method: str = "hoeffding",
) -> dict[str, Any]:
    """Certify the DIRECT source→final losses into the optional
    ``end_to_end`` leg (same machinery, quantisation, and float-free wire
    as a meaning-risk receipt's bound)."""
    quantised = _quantized(losses)
    delta_q = _to_micro(delta) / 1_000_000
    g = certify_meaning_risk(
        quantised,
        scorer_name=scorer_name,
        scorer_version=scorer_version,
        delta=delta_q,
        method=method,
    )
    return {
        "scorer": scorer_name,
        "scorer_version": scorer_version,
        "loss_definition": loss_definition,
        "n": g.n,
        "method": g.method,
        "delta_micro": _to_micro(g.delta),
        "point_estimate_micro": _to_micro(g.point_estimate),
        "risk_upper_bound_micro": _to_micro(g.risk_upper_bound),
        "losses_hash": losses_hash(losses),
    }


def build_chain_payload(
    hop_envelopes: Sequence[dict],
    *,
    end_to_end: dict | None = None,
    not_covered: Sequence[str] = DEFAULT_NOT_COVERED,
    disclosure: str = DEFAULT_CHAIN_DISCLOSURE,
    budget_scope: str = BUDGET_SCOPE_STATEMENT,
    signed_at: str | None = None,
) -> dict[str, Any]:
    """Assemble a ``sum.chain_receipt.v1`` payload from >= 2 ordered,
    already-signed per-hop meaning-risk envelopes. Mirrors are read
    straight from each hop payload (so they cannot be built wrong), sums
    are integer-exact, and the chain id binds the order."""
    if len(hop_envelopes) < 2:
        raise ValueError(
            f"a chain needs >= 2 hops; got {len(hop_envelopes)} (a 1-hop "
            f"chain is just the hop receipt)"
        )
    if not not_covered:
        raise ValueError("not_covered must be non-empty")
    hops = []
    for i, env in enumerate(hop_envelopes):
        if not isinstance(env, dict) or env.get("schema") != MEANING_RISK_SCHEMA:
            raise ValueError(
                f"hop {i}: expected a signed {MEANING_RISK_SCHEMA} "
                f"envelope; got schema={env.get('schema') if isinstance(env, dict) else type(env).__name__!r}"
            )
        p = env.get("payload")
        if not isinstance(p, dict):
            raise ValueError(f"hop {i}: envelope has no payload object")
        hops.append(
            {
                "index": i + 1,
                "receipt_hash": canonical_receipt_hash(env),
                "schema": MEANING_RISK_SCHEMA,
                "risk_upper_bound_micro": _require_int_micro(
                    p, "risk_upper_bound_micro"
                ),
                "delta_micro": _require_int_micro(p, "delta_micro"),
                "n": _require_int_micro(p, "n"),
                "method": str(p.get("method", "")),
                "corpus_id": str(p.get("corpus_id", "")),
                "transform": str(p.get("transform", "")),
                "scorer": str(p.get("scorer", "")),
            }
        )
    if signed_at is None:
        now = datetime.now(timezone.utc)
        signed_at = (
            now.strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{now.microsecond // 1000:03d}Z"
        )
    payload: dict[str, Any] = {
        "chain_id": chain_id_for([h["receipt_hash"] for h in hops]),
        "n_hops": len(hops),
        "hops": hops,
        "composition_rule": COMPOSITION_RULE,
        "budget_micro": sum(h["risk_upper_bound_micro"] for h in hops),
        "joint_delta_micro": sum(h["delta_micro"] for h in hops),
        "budget_scope": budget_scope,
        "not_covered": list(not_covered),
        "disclosure": disclosure,
        "signed_at": signed_at,
    }
    if end_to_end is not None:
        payload["end_to_end"] = end_to_end
    return payload


def sign_chain_receipt(
    payload: dict[str, Any],
    *,
    private_jwk: dict[str, Any],
    kid: str,
) -> dict[str, Any]:
    """Sign a chain payload into the four-key envelope
    ``{schema, kid, payload, jws}``."""
    envelope = sign_jose_envelope(payload, private_jwk=private_jwk, kid=kid)
    envelope["schema"] = SUPPORTED_SCHEMA
    return envelope
