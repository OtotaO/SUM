"""Group-conditional meaning-risk control — the Perspective Receipts substrate.

The marginal bound answers "on average, how much meaning is lost?" The
group-conditional bound answers "…and within EACH cohort (language / genre
/ named perspective)?" — surfacing the worst cohort the average hides.
These pin: per-cohort = the kernel applied per subset, marginal
consistency, the worst-cohort surfacing, the small-cohort-wide-bound
honesty, and the Bonferroni simultaneous mode.
"""
from __future__ import annotations

import numpy as np
import pytest

from sum_engine_internal.research.meaning import (
    certify_meaning_risk,
    certify_meaning_risk_by_group,
)

_S = dict(scorer_name="lexical-coverage-bidirectional", scorer_version="1")


def test_per_group_equals_kernel_on_the_subset():
    losses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    groups = ["a", "a", "a", "b", "b", "b"]
    g = certify_meaning_risk_by_group(losses, groups, **_S)
    # each cohort's guarantee IS certify_meaning_risk over that cohort
    a_direct = certify_meaning_risk([0.1, 0.2, 0.3], **_S)
    b_direct = certify_meaning_risk([0.4, 0.5, 0.6], **_S)
    assert g.groups["a"].risk_upper_bound == a_direct.risk_upper_bound
    assert g.groups["b"].risk_upper_bound == b_direct.risk_upper_bound
    assert g.groups["a"].n == 3 and g.groups["b"].n == 3


def test_marginal_matches_plain_certification():
    losses = [0.1, 0.2, 0.3, 0.9, 0.9, 0.9]
    groups = ["x", "x", "x", "y", "y", "y"]
    g = certify_meaning_risk_by_group(losses, groups, **_S)
    marg = certify_meaning_risk(losses, **_S)
    assert g.marginal.risk_upper_bound == marg.risk_upper_bound


def test_surfaces_the_cohort_the_average_hides():
    """The whole point: a marginal bound that looks acceptable can hide a
    cohort that is not. weakest_group + controls_all catch it."""
    # cohort 'good' all-preserved; cohort 'bad' heavy loss. 200 each so the
    # finite-sample radius is small enough that the gap is real, not noise.
    losses = [0.0] * 200 + [0.85] * 200
    groups = ["good"] * 200 + ["bad"] * 200
    g = certify_meaning_risk_by_group(losses, groups, **_S)

    worst_id, worst = g.weakest_group()
    assert worst_id == "bad"
    assert worst.risk_upper_bound > g.groups["good"].risk_upper_bound
    # the marginal (≈0.425 mean) can be 'controlled' at 0.6 while the bad
    # cohort is NOT — controls_all is the honest, stricter gate.
    assert g.marginal.controls(0.6) is True
    assert g.controls_all(0.6) is False


def test_small_cohort_gets_a_wider_bound():
    """Honest: no free conditional coverage. A small cohort has a wider
    finite-sample radius than a large cohort with the same observed mean."""
    losses = [0.3, 0.3] + [0.3] * 200          # cohort 'small' (n=2), 'big' (n=200)
    groups = ["small", "small"] + ["big"] * 200
    g = certify_meaning_risk_by_group(losses, groups, **_S)
    assert g.groups["small"].slack > g.groups["big"].slack


def test_simultaneous_bonferroni_widens_bounds():
    """simultaneous=True certifies each cohort at delta/G so all hold
    jointly — strictly wider per-cohort bounds than the per-cohort-marginal
    default."""
    losses = [0.2, 0.3, 0.4] * 20
    groups = (["a"] + ["b"] + ["c"]) * 20
    indep = certify_meaning_risk_by_group(losses, groups, delta=0.05, **_S)
    joint = certify_meaning_risk_by_group(
        losses, groups, delta=0.05, simultaneous=True, **_S
    )
    for gid in ("a", "b", "c"):
        assert joint.groups[gid].risk_upper_bound >= indep.groups[gid].risk_upper_bound
    assert joint.simultaneous is True and indep.simultaneous is False


def test_controls_all_true_when_every_cohort_controlled():
    losses = [0.0] * 100 + [0.0] * 100
    groups = ["a"] * 100 + ["b"] * 100
    g = certify_meaning_risk_by_group(losses, groups, **_S)
    assert g.controls_all(0.2) is True


# ── Bonferroni simultaneous JOINT coverage (the Monte-Carlo receipt) ──
#
# test_simultaneous_bonferroni_widens_bounds proves the per-cohort radii
# WIDEN under simultaneous=True. That is necessary but not sufficient: the
# claim the mode actually makes is that ALL cohort bounds hold *jointly*
# with confidence ≥ 1−δ. Widening without measuring joint coverage could
# still under-cover. This Monte-Carlo measures the joint event directly —
# it is the empirical receipt for the simultaneous guarantee, the audit's
# named highest-value missing test.


def _joint_simultaneous_coverage(
    true_rates, n_per, delta, method, *, n_trials=2000, seed=0
):
    """Fraction of trials in which EVERY cohort's certified ceiling holds
    at once, under the Bonferroni (simultaneous=True) split. ``true_rates``
    maps cohort id → its true per-pair loss rate (the data-generating
    Bernoulli mean)."""
    rng = np.random.RandomState(seed)
    cohorts = sorted(true_rates)
    all_held = 0
    for _ in range(n_trials):
        losses: list[float] = []
        gids: list[str] = []
        for c in cohorts:
            draws = (rng.uniform(size=n_per) < true_rates[c]).astype(float)
            losses.extend(draws.tolist())
            gids.extend([c] * n_per)
        g = certify_meaning_risk_by_group(
            losses, gids, delta=delta, method=method, simultaneous=True, **_S
        )
        if all(g.groups[c].risk_upper_bound >= true_rates[c] for c in cohorts):
            all_held += 1
    return all_held / n_trials


@pytest.mark.parametrize(
    "method", ["hoeffding", "clopper_pearson", "empirical_bernstein"]
)
def test_simultaneous_bonferroni_joint_coverage_holds(method):
    """The named receipt: under simultaneous=True the probability that ALL
    cohort ceilings hold at once is ≥ 1−δ. The Bonferroni δ/G split is what
    converts G separate per-cohort guarantees into one joint guarantee —
    this measures that the joint event actually clears the target, for
    every method (the variance-adaptive eB included)."""
    true_rates = {"novice": 0.10, "expert": 0.20, "regulator": 0.30}
    delta = 0.05
    coverage = _joint_simultaneous_coverage(
        true_rates, n_per=60, delta=delta, method=method, n_trials=2000, seed=17
    )
    assert coverage >= (1 - delta) - 0.01, (
        f"simultaneous={method} joint coverage {coverage:.4f} < target "
        f"{1 - delta:.2f} — the Bonferroni split failed to deliver the "
        f"all-cohorts guarantee"
    )


def test_independent_mode_need_not_hold_jointly():
    """The contrast that justifies Bonferroni: with simultaneous=False each
    cohort holds at 1−δ on its OWN, so the joint event is only guaranteed
    at ~(1−δ)^G < 1−δ. We assert the structural fact (independent joint
    coverage ≤ Bonferroni joint coverage) rather than a brittle threshold,
    since the conservative kernels can still over-cover."""
    true_rates = {f"c{i}": 0.25 for i in range(8)}  # many cohorts → product bites
    delta = 0.10
    rng = np.random.RandomState(5)
    cohorts = sorted(true_rates)

    def joint(simultaneous):
        held = 0
        for _ in range(1500):
            losses: list[float] = []
            gids: list[str] = []
            for c in cohorts:
                draws = (rng.uniform(size=40) < true_rates[c]).astype(float)
                losses.extend(draws.tolist())
                gids.extend([c] * 40)
            g = certify_meaning_risk_by_group(
                losses, gids, delta=delta, simultaneous=simultaneous, **_S
            )
            if all(g.groups[c].risk_upper_bound >= true_rates[c] for c in cohorts):
                held += 1
        return held / 1500

    assert joint(simultaneous=True) >= joint(simultaneous=False) - 0.01


def test_empirical_bernstein_routes_through_by_group():
    """The eB method threads cleanly through the group-conditional path —
    marginal and every cohort certified with the variance-adaptive bound."""
    losses = [0.03] * 120 + [0.04] * 120
    groups = ["a"] * 120 + ["b"] * 120
    g = certify_meaning_risk_by_group(
        losses, groups, method="empirical_bernstein", **_S
    )
    assert g.marginal.method == "empirical_bernstein"
    assert all(grp.method == "empirical_bernstein" for grp in g.groups.values())


# ── validation ────────────────────────────────────────────────────────


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        certify_meaning_risk_by_group([0.1, 0.2], ["a"], **_S)


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        certify_meaning_risk_by_group([], [], **_S)


def test_single_cohort_matches_marginal():
    losses = [0.1, 0.2, 0.3, 0.4]
    g = certify_meaning_risk_by_group(losses, ["only"] * 4, **_S)
    assert g.groups["only"].risk_upper_bound == g.marginal.risk_upper_bound
