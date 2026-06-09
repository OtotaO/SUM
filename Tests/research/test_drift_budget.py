"""Tests for the multi-hop drift budget (`research.meaning.drift_budget`).

Two things this suite holds:

  1. **The provable leg is exactly the union bound.** `compose_drift_budget`
     and `compose_drift_budget_from_payloads` sum the per-hop bounds and
     Bonferroni-sum the deltas — and the payload path is integer-exact
     against the receipts it composes (the "never disagrees with the
     single-hop receipts" property).

  2. **The measured leg is honest about the additive-vs-end-to-end gap.**
     We deliberately exhibit BOTH regimes with deterministic judges:
       - a *transitive* (subset) judge ⇒ additive Σ Lᵢ is conservative
         (over-counts) — slack ≥ 0;
       - a *non-transitive* (overlap-threshold) judge ⇒ compounding
         brittleness makes additive UNDER-count — slack < 0.
     The module never claims additive is a safe end-to-end ceiling; the
     audit measures exactly when it is not. These tests pin that.

Deterministic injected judges keep the suite torch-free.
"""
from __future__ import annotations

import pytest

from sum_engine_internal.research.meaning.drift_budget import (
    audit_additive_vs_end_to_end,
    compose_drift_budget,
    compose_drift_budget_from_payloads,
    measure_chain_drift,
)
from sum_engine_internal.research.meaning.meaning_loss import (
    EntailmentScorer,
    LexicalCoverageScorer,
)

# --- deterministic judges (no NLI model) ------------------------------


def _subset_judge(premise: str, hypothesis: str) -> bool:
    """Transitive: hypothesis entailed iff its words ⊆ premise's words.
    A⊇B and B⊇C ⇒ A⊇C, so along a chain losses can only accumulate —
    additive Σ Lᵢ is always conservative under this judge."""
    return set(hypothesis.lower().split()) <= set(premise.lower().split())


def _overlap_judge(premise: str, hypothesis: str) -> bool:
    """Non-transitive: entailed iff ≥ half of the hypothesis's words appear
    in the premise. Adjacent texts can each entail while the endpoints do
    not — the compounding-brittleness regime real NLI judges exhibit."""
    hw = set(hypothesis.lower().split())
    if not hw:
        return True
    pw = set(premise.lower().split())
    return len(hw & pw) / len(hw) >= 0.5


def _scorer(judge) -> EntailmentScorer:
    return EntailmentScorer(entails=judge, judge_name="stub", judge_version="t")


# --- Leg A: per-document chain measurement ----------------------------


def test_identity_chain_has_zero_drift():
    s = _scorer(_subset_judge)
    text = "alpha beta. gamma delta."
    r = measure_chain_drift([text, text, text], s)
    assert r.n_hops == 2
    assert all(h.loss == 0.0 for h in r.hops)
    assert r.additive_budget == 0.0
    assert r.end_to_end_loss == 0.0
    assert r.slack == 0.0
    assert r.additive_is_conservative


def test_single_text_is_not_a_chain():
    s = _scorer(_subset_judge)
    with pytest.raises(ValueError):
        measure_chain_drift(["just one"], s)


def test_per_hop_losses_and_attribution():
    s = LexicalCoverageScorer()  # deterministic, dependency-free
    chain = ["alpha beta gamma. delta epsilon.", "alpha beta. delta epsilon.", "alpha. delta."]
    r = measure_chain_drift(chain, s)
    assert r.n_hops == 2
    # hop 2 drops more than hop 1, so it is the most expensive
    assert r.hops[1].loss > r.hops[0].loss
    assert r.most_expensive_hop().index == 2
    # additive budget is the sum of the per-hop losses
    assert r.additive_budget == pytest.approx(sum(h.loss for h in r.hops))


def test_transitive_judge_additive_is_conservative():
    """Subset judge ⇒ Σ Lᵢ ≥ L_e2e on a lossy chain (over-counts)."""
    s = _scorer(_subset_judge)
    # each hop drops a token; endpoints share fewer ⇒ e2e loss is real but
    # never exceeds the accumulated per-hop loss.
    chain = ["a b c d", "a b c", "a b", "a"]
    r = measure_chain_drift(chain, s)
    assert r.end_to_end_loss > 0.0
    assert r.slack >= 0.0
    assert r.additive_is_conservative


def test_nontransitive_judge_can_undercount():
    """Overlap judge ⇒ each hop entails (loss 0) yet the endpoints drift —
    additive UNDER-counts. The honest failure mode the module refuses to
    assert away."""
    s = _scorer(_overlap_judge)
    chain = ["a b c d", "b c d e", "c d e f", "d e f g"]
    r = measure_chain_drift(chain, s)
    assert r.additive_budget == 0.0          # every hop looked faithful
    assert r.end_to_end_loss > 0.0           # but the chain drifted
    assert r.slack < 0.0
    assert not r.additive_is_conservative


def test_end_to_end_dropped_claims_surface_for_entailment_scorer():
    s = _scorer(_subset_judge)
    chain = ["keep this. drop that.", "keep this.", "keep this."]
    r = measure_chain_drift(chain, s)
    # "drop that." is a source claim the final text does not preserve
    assert any("drop" in d.lower() for d in r.end_to_end_dropped)


# --- Leg B: certified composition (provable) --------------------------


class _G:
    """Minimal guarantee-shaped object (duck-typed; no numpy)."""

    def __init__(self, ub, delta, method="hoeffding"):
        self.risk_upper_bound = ub
        self.delta = delta
        self.method = method


def test_compose_is_sum_and_bonferroni():
    b = compose_drift_budget([_G(0.10, 0.05), _G(0.20, 0.05), _G(0.05, 0.05)])
    assert b.n_hops == 3
    assert b.budget == pytest.approx(0.35)               # Σ Uᵢ
    assert b.joint_delta == pytest.approx(0.15)          # Σ δᵢ (union bound)
    assert b.joint_confidence == pytest.approx(0.85)     # 1 − Σ δᵢ
    assert b.most_expensive_hop() == 2                   # the 0.20 hop
    assert b.within(0.40)
    assert not b.within(0.30)


def test_compose_requires_at_least_one_hop():
    with pytest.raises(ValueError):
        compose_drift_budget([])
    with pytest.raises(ValueError):
        compose_drift_budget_from_payloads([])


def test_compose_from_payloads_is_integer_exact():
    """The chain budget composed from receipt payloads is byte-exact (in
    micro-units) against the receipts — it cannot disagree with them."""
    payloads = [
        {"risk_upper_bound_micro": 123456, "delta_micro": 50000, "method": "hoeffding"},
        {"risk_upper_bound_micro": 200001, "delta_micro": 33333, "method": "hoeffding"},
    ]
    b = compose_drift_budget_from_payloads(payloads)
    # sum in micro then back to float — no float reintroduced mid-sum
    assert round(b.budget * 1_000_000) == 123456 + 200001
    assert round(b.joint_delta * 1_000_000) == 50000 + 33333


def test_compose_matches_real_guarantees():
    """Against the real conformal certifier (numpy path): the composed
    budget equals the sum of the per-hop certified bounds exactly."""
    pytest.importorskip("numpy")
    from sum_engine_internal.research.meaning.conformal_meaning import (
        certify_meaning_risk,
    )

    g1 = certify_meaning_risk(
        [0.0, 0.1, 0.2, 0.05], scorer_name="x", scorer_version="1", delta=0.05
    )
    g2 = certify_meaning_risk(
        [0.3, 0.1, 0.0, 0.4], scorer_name="x", scorer_version="1", delta=0.05
    )
    b = compose_drift_budget([g1, g2])
    assert b.budget == pytest.approx(g1.risk_upper_bound + g2.risk_upper_bound)
    assert b.joint_delta == pytest.approx(0.10)


# --- The honest audit -------------------------------------------------


def test_audit_characterises_both_regimes():
    s = _scorer(_overlap_judge)
    conservative_chain = ["a b c d", "a b c", "a b"]   # subset-ish under overlap ⇒ slack ≥ 0
    undercount_chain = ["a b c d", "b c d e", "c d e f", "d e f g"]
    res = audit_additive_vs_end_to_end([conservative_chain, undercount_chain], s)
    assert res.n_chains == 2
    # exactly one chain under-counts ⇒ conservative_fraction == 0.5
    assert res.conservative_fraction == pytest.approx(0.5)
    assert res.min_slack < 0.0
    assert res.worst_undercount > 0.0


def test_audit_needs_a_chain():
    s = _scorer(_subset_judge)
    with pytest.raises(ValueError):
        audit_additive_vs_end_to_end([], s)
