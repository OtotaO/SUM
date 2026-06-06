"""Group-conditional meaning-risk control — the Perspective Receipts substrate.

The marginal bound answers "on average, how much meaning is lost?" The
group-conditional bound answers "…and within EACH cohort (language / genre
/ named perspective)?" — surfacing the worst cohort the average hides.
These pin: per-cohort = the kernel applied per subset, marginal
consistency, the worst-cohort surfacing, the small-cohort-wide-bound
honesty, and the Bonferroni simultaneous mode.
"""
from __future__ import annotations

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
