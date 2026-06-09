"""Compose meaning-loss across a CHAIN of transforms — the drift budget.

A single ``sum.meaning_risk_receipt.v1`` certifies ONE hop: the expected
meaning-loss of one transform, over one corpus, under one named proxy. But
text in the wild rarely stops at one hop. An autonomous agent loop
(propose → transform → measure → commit, the "loopy era") and a human
editing pipeline both run a document through MANY transforms, and the
question that matters is cumulative: *how much meaning has drifted after
N hops, and is the chain still within budget?*

This module composes the per-hop pieces into a chain-level statement. It
keeps TWO quantities rigorously separate, because they are different and
conflating them is the easy overclaim here:

  Leg A — per-document MEASUREMENT (`measure_chain_drift`).
    Run one document x0 → x1 → … → xN through the named scorer. Report
    each hop's measured loss L_i, the **additive budget** Σ L_i (the drift
    consumed hop-by-hop), AND the **directly-measured end-to-end loss**
    L_e2e = loss(x0, xN). The gap between them — the *slack* Σ L_i − L_e2e
    — is itself informative and is NOT assumed to have a sign (see below).
    This is the multi-hop analogue of ``sum meaning-diff``: a measurement
    for one chain, honestly labelled, not a certified bound.

  Leg B — corpus-level CERTIFIED composition (`compose_drift_budget`).
    Given a per-hop ``MeaningRiskGuarantee`` for each transform (each
    certifying E[L_i] ≤ U_i at confidence 1 − δ_i), the union bound gives:
    with confidence ≥ 1 − Σ δ_i, EVERY per-hop bound holds simultaneously,
    hence the cumulative expected per-hop loss Σ_i E[L_i] ≤ Σ_i U_i. That
    sum is the **certified drift budget**. It is provable (Bonferroni +
    monotonicity of summation) and it *cannot disagree* with the
    single-hop receipts because it is literally their sum, carried at the
    joint confidence. ``compose_drift_budget_from_payloads`` does the same
    over verified receipt payloads in integer micro-units, so the chain
    budget is byte-exact against the receipts it composes.

Why additive Σ L_i is NOT claimed to upper-bound L_e2e
------------------------------------------------------
It is tempting to assert a triangle inequality — that hop-by-hop loss can
only over-count end-to-end loss — and ship Σ L_i as a guaranteed ceiling
on L_e2e. We do NOT, because the entailment proxy is not a metric and
claim-survival along a chain is not monotone:

  - **Recovery (additive over-counts, conservative).** Hop 1 drops claim
    A; hop 2 re-derives A. L_1 > 0 but L_e2e ≈ 0, so Σ L_i > L_e2e.
  - **Compounding brittleness (additive UNDER-counts).** Each hop is a
    faithful paraphrase the judge scores at loss ≈ 0, but the compounded
    rewrite x0→xN drifts far enough that the judge scores L_e2e > 0. Then
    Σ L_i < L_e2e — the additive budget misses drift the end-to-end
    measurement catches.

So the relationship between the additive budget and end-to-end loss is a
proxy-geometry question to be **measured, not asserted**
(``audit_additive_vs_end_to_end`` — the discipline the slider's T4
drift-composition audit applied to drift_pct). The certified leg (B)
sidesteps this entirely: it bounds Σ E[L_i], a quantity that IS additive
by definition of the sum, never E[L_e2e].

Honest boundary (inherited from the receipt family): every loss is a
NAMED PROXY, marginal, valid under exchangeability with each hop's
calibration corpus; nothing here covers arrangement / sound / connotation
/ implicature.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    # Imported for types only. `MeaningRiskGuarantee` lives in
    # conformal_meaning, which pulls numpy; keeping the import behind
    # TYPE_CHECKING means drift_budget itself stays numpy-free, so the
    # certified composition can run over verified receipt payloads
    # (sum_verify) without the research numeric stack. compose_drift_budget
    # only reads `.risk_upper_bound` / `.delta` / `.method`, so it accepts
    # any guarantee-shaped object by duck typing.
    from sum_engine_internal.research.meaning.conformal_meaning import (
        MeaningRiskGuarantee,
    )
    from sum_engine_internal.research.meaning.meaning_loss import MeaningScorer

# Micro-unit grid shared with the meaning-risk receipt wire (1e-6), so a
# budget composed from receipt payloads stays integer-exact against them.
_MICRO_SCALE = 1_000_000


def _to_micro(x: float) -> int:
    return int(round(float(x) * _MICRO_SCALE))


def _from_micro(m: int) -> float:
    return int(m) / _MICRO_SCALE


# ── Leg A: per-document chain measurement ─────────────────────────────


@dataclass(frozen=True, slots=True)
class HopLoss:
    """One hop's measured meaning-loss in a chain. ``index`` is 1-based
    (hop 1 is x0 → x1)."""

    index: int
    loss: float


@dataclass(frozen=True, slots=True)
class ChainDriftReadout:
    """A per-document drift readout over a chain x0 → … → xN — a
    MEASUREMENT, not a certified bound.

    ``additive_budget`` is the hop-by-hop drift consumed (Σ L_i);
    ``end_to_end_loss`` is the SAME proxy measured directly between the
    chain's endpoints; ``slack`` is their difference. The sign of the
    slack is reported, not assumed (see the module docstring).
    """

    hops: tuple[HopLoss, ...]
    additive_budget: float
    end_to_end_loss: float
    judge: str
    judge_version: str
    end_to_end_dropped: tuple[str, ...] = ()

    @property
    def n_hops(self) -> int:
        return len(self.hops)

    @property
    def slack(self) -> float:
        """``additive_budget − end_to_end_loss``. Positive ⇒ the additive
        budget is conservative (over-counts) on this chain; negative ⇒ it
        UNDER-counts the directly-measured end-to-end drift (compounding
        brittleness)."""
        return self.additive_budget - self.end_to_end_loss

    @property
    def additive_is_conservative(self) -> bool:
        """True iff Σ L_i ≥ L_e2e on this chain — i.e. the additive budget
        did not miss drift the end-to-end measurement caught. Per-chain;
        NOT a guarantee across chains."""
        return self.slack >= 0.0

    def most_expensive_hop(self) -> HopLoss:
        """The hop that consumed the most budget — where to look first."""
        return max(self.hops, key=lambda h: h.loss)

    @property
    def scope(self) -> str:
        return (
            "measured for THIS chain under the named judge — a per-document "
            "MEASUREMENT, not a certified bound. The additive budget Σ Lᵢ is "
            "the drift consumed hop-by-hop; its relationship to the "
            "end-to-end loss is proxy-dependent and not guaranteed in either "
            "direction (see slack). For a (1-δ) bound on cumulative EXPECTED "
            "per-hop loss, compose per-hop meaning_risk receipts "
            "(compose_drift_budget)"
        )


def measure_chain_drift(
    texts: Sequence[str],
    scorer: MeaningScorer,
) -> ChainDriftReadout:
    """Measure meaning drift along a chain of texts x0 → x1 → … → xN.

    ``texts`` is the ordered chain (length ≥ 2). Each adjacent pair is one
    hop; the readout carries per-hop losses, the additive budget Σ L_i, and
    the directly-measured end-to-end loss between the first and last text.
    When the scorer exposes ``explain`` (the EntailmentScorer path), the
    end-to-end DROPPED claims are included — the legible "what did the whole
    chain lose".

    This is a MEASUREMENT for one chain, not a certified bound — see
    ``ChainDriftReadout.scope``.
    """
    if len(texts) < 2:
        raise ValueError(
            f"a chain needs at least 2 texts (one hop); got {len(texts)}"
        )

    hops = tuple(
        HopLoss(index=i + 1, loss=float(scorer.loss(texts[i], texts[i + 1])))
        for i in range(len(texts) - 1)
    )
    additive = sum(h.loss for h in hops)
    end_to_end = float(scorer.loss(texts[0], texts[-1]))

    dropped: tuple[str, ...] = ()
    explain = getattr(scorer, "explain", None)
    if callable(explain):
        readout = explain(texts[0], texts[-1])
        dropped = tuple(readout.dropped_claims)

    return ChainDriftReadout(
        hops=hops,
        additive_budget=max(0.0, min(float(len(hops)), additive)),
        end_to_end_loss=end_to_end,
        judge=getattr(scorer, "name", "unknown"),
        judge_version=getattr(scorer, "version", "unspecified"),
        end_to_end_dropped=dropped,
    )


# ── Leg B: corpus-level certified composition (provable) ──────────────


@dataclass(frozen=True, slots=True)
class CertifiedDriftBudget:
    """A certified ceiling on the cumulative EXPECTED per-hop meaning-loss
    across a chain, holding JOINTLY.

    Reads as: "with confidence ≥ ``joint_confidence``, EVERY hop's expected
    meaning-loss is at or below its own certified bound simultaneously,
    hence the sum of expected per-hop losses Σ E[Lᵢ] ≤ ``budget``." The
    budget is the sum of the per-hop ``risk_upper_bound``s; the joint
    confidence pays the Bonferroni union (1 − Σ δᵢ).

    What this does NOT say: it does not bound the end-to-end expected loss
    E[L_e2e] (that is not additive — see the module docstring), and it is
    not a per-document statement. It bounds the additive sum of per-hop
    expectations, each within its own corpus's exchangeability scope.
    """

    per_hop_bounds: tuple[float, ...]
    per_hop_deltas: tuple[float, ...]
    budget: float
    joint_delta: float
    methods: tuple[str, ...]

    @property
    def n_hops(self) -> int:
        return len(self.per_hop_bounds)

    @property
    def joint_confidence(self) -> float:
        return 1.0 - self.joint_delta

    def within(self, total_budget: float) -> bool:
        """Is the certified chain budget at or below an operator-set
        ceiling? The operational pass/fail for a drift budget."""
        return self.budget <= total_budget

    def most_expensive_hop(self) -> int:
        """1-based index of the hop contributing the most to the budget."""
        return 1 + max(
            range(self.n_hops), key=lambda i: self.per_hop_bounds[i]
        )

    @property
    def scope(self) -> str:
        return (
            f"certified ceiling on cumulative EXPECTED per-hop meaning-loss "
            f"Σ E[Lᵢ] ≤ {self.budget:.6f}, holding JOINTLY across all "
            f"{self.n_hops} hops with confidence ≥ {self.joint_confidence:.4f} "
            f"(Bonferroni union of the per-hop receipts). Each per-hop bound "
            f"is marginal, valid under exchangeability with that hop's "
            f"calibration corpus, over a NAMED proxy; this composition does "
            f"NOT bound end-to-end expected loss and is NOT per-document"
        )


def compose_drift_budget(
    guarantees: Sequence[MeaningRiskGuarantee],
) -> CertifiedDriftBudget:
    """Compose per-hop meaning-risk guarantees into a certified chain
    drift budget via the Bonferroni union bound.

    Each ``MeaningRiskGuarantee`` certifies E[Lᵢ] ≤ ``risk_upper_bound`` at
    confidence 1 − ``delta``. The composition is provable: by the union
    bound, all N per-hop bounds hold simultaneously with confidence
    ≥ 1 − Σ δᵢ, and on that event Σ E[Lᵢ] ≤ Σ ``risk_upper_bound``. The
    returned budget is exactly that sum — it never disagrees with the
    single-hop guarantees because it IS their sum.
    """
    if not guarantees:
        raise ValueError("a chain budget needs at least one hop guarantee")
    bounds = tuple(float(g.risk_upper_bound) for g in guarantees)
    deltas = tuple(float(g.delta) for g in guarantees)
    return CertifiedDriftBudget(
        per_hop_bounds=bounds,
        per_hop_deltas=deltas,
        budget=sum(bounds),
        joint_delta=sum(deltas),
        methods=tuple(str(g.method) for g in guarantees),
    )


def compose_drift_budget_from_payloads(
    payloads: Sequence[dict[str, Any]],
) -> CertifiedDriftBudget:
    """Compose a certified chain budget directly from VERIFIED
    ``sum.meaning_risk_receipt.v1`` payloads (e.g. the dicts returned by
    ``sum_verify.verify_meaning_risk_receipt``).

    Reads ``risk_upper_bound_micro`` and ``delta_micro`` and sums them in
    integer micro-units, so the chain budget is byte-exact against the
    receipts it composes — the same wire grid, no float reintroduced. The
    caller is responsible for having verified each payload first; this
    function trusts the (already-checked) numbers.
    """
    if not payloads:
        raise ValueError("a chain budget needs at least one receipt payload")
    bounds_micro = [int(p["risk_upper_bound_micro"]) for p in payloads]
    deltas_micro = [int(p["delta_micro"]) for p in payloads]
    return CertifiedDriftBudget(
        per_hop_bounds=tuple(_from_micro(m) for m in bounds_micro),
        per_hop_deltas=tuple(_from_micro(m) for m in deltas_micro),
        budget=_from_micro(sum(bounds_micro)),
        joint_delta=_from_micro(sum(deltas_micro)),
        methods=tuple(str(p.get("method", "unknown")) for p in payloads),
    )


# ── The honest audit: additive budget vs measured end-to-end ──────────


@dataclass(frozen=True, slots=True)
class AdditiveAuditResult:
    """Empirical characterisation of how the additive budget Σ Lᵢ relates
    to the directly-measured end-to-end loss across a set of chains — the
    measured answer to a question we refuse to assert (see module
    docstring). Mirrors the slider T4 drift-composition audit's discipline.
    """

    n_chains: int
    conservative_fraction: float   # share of chains with Σ Lᵢ ≥ L_e2e
    mean_slack: float
    min_slack: float               # most negative ⇒ worst under-count
    max_slack: float

    @property
    def worst_undercount(self) -> float:
        """How far the additive budget under-counted end-to-end drift on the
        worst chain (0.0 if it never under-counted). The honest headline:
        the additive budget is NOT a safe end-to-end ceiling when this is
        > 0."""
        return max(0.0, -self.min_slack)


def audit_additive_vs_end_to_end(
    chains: Sequence[Sequence[str]],
    scorer: MeaningScorer,
) -> AdditiveAuditResult:
    """Measure, over many chains, whether the additive budget Σ Lᵢ bounds
    the directly-measured end-to-end loss — and by how much it fails when
    it does not. Returns the slack distribution and the conservative
    fraction; makes NO claim that the additive budget is a guaranteed
    end-to-end ceiling (it characterises exactly when it is not)."""
    if not chains:
        raise ValueError("audit needs at least one chain")
    slacks = [measure_chain_drift(c, scorer).slack for c in chains]
    conservative = sum(1 for s in slacks if s >= 0.0)
    return AdditiveAuditResult(
        n_chains=len(slacks),
        conservative_fraction=conservative / len(slacks),
        mean_slack=sum(slacks) / len(slacks),
        min_slack=min(slacks),
        max_slack=max(slacks),
    )
