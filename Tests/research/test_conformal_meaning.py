"""Contract for the distribution-free meaning-loss upper bound.

Two layers:
  - Contract layer: empirical coverage ≥ 1-δ (the provable dual
    guarantee — a valid (1-δ) upper bound must sit above the true loss
    rate at least 1-δ of the time).
  - Algebra layer: the bound is the exact dual of the rate kernel
    (upper-bound on E[loss] = 1 - lower-bound on E[1-loss]), is
    deterministic, rejects garbage, and is conservative (ub ≥ mean).
"""
from __future__ import annotations

import numpy as np
import pytest

from sum_engine_internal.research.conformal.risk_control import (
    hoeffding_lower_bound,
)
from sum_engine_internal.research.meaning.conformal_meaning import (
    MeaningRiskGuarantee,
    certify_meaning_risk,
    empirical_risk_coverage,
)


_SCORER = dict(scorer_name="lexical-coverage-bidirectional", scorer_version="1")


# ── Contract: empirical coverage ──────────────────────────────────────


@pytest.mark.parametrize("method", ["hoeffding", "clopper_pearson"])
@pytest.mark.parametrize(
    ("true_loss_rate", "n", "delta"),
    [
        (0.1, 50, 0.05),
        (0.3, 100, 0.05),
        (0.5, 80, 0.10),
        (0.05, 200, 0.05),
    ],
)
def test_upper_bound_covers_true_loss_rate(true_loss_rate, n, delta, method):
    """A valid (1-δ) upper bound must achieve coverage ≥ 1-δ."""
    coverage = empirical_risk_coverage(
        true_loss_rate, n, delta, method=method, n_trials=4000, seed=11
    )
    # ±0.01 flake band (≈4σ at n_trials=4000), matching the rate kernel's
    # coverage-test tolerance.
    assert coverage >= (1 - delta) - 0.01


# ── Algebra: duality with the rate kernel ─────────────────────────────


def test_bound_is_dual_of_rate_kernel():
    losses = [0.0, 0.1, 0.2, 0.3, 0.4, 0.05, 0.15, 0.25]
    g = certify_meaning_risk(losses, delta=0.05, method="hoeffding", **_SCORER)
    preservations = [1.0 - x for x in losses]
    lb = hoeffding_lower_bound(preservations, 0.05)
    assert g.risk_upper_bound == pytest.approx(1.0 - lb, abs=1e-12)


def test_point_estimate_is_mean_loss():
    losses = [0.0, 0.2, 0.4, 0.6]
    g = certify_meaning_risk(losses, **_SCORER)
    assert g.point_estimate == pytest.approx(np.mean(losses))


def test_bound_is_conservative():
    """The certified ceiling sits at or above the observed mean — slack
    is the price of rigour."""
    losses = [0.1, 0.2, 0.15, 0.05, 0.25, 0.1]
    g = certify_meaning_risk(losses, **_SCORER)
    assert g.risk_upper_bound >= g.point_estimate
    assert g.slack >= 0.0


def test_more_data_tightens_bound():
    rng = np.random.RandomState(3)
    small = rng.uniform(0.0, 0.3, size=20).tolist()
    large = rng.uniform(0.0, 0.3, size=500).tolist()
    g_small = certify_meaning_risk(small, **_SCORER)
    g_large = certify_meaning_risk(large, **_SCORER)
    assert g_large.slack < g_small.slack


def test_scorer_identity_is_carried():
    g = certify_meaning_risk([0.1, 0.2], **_SCORER)
    assert g.scorer_name == "lexical-coverage-bidirectional"
    assert g.scorer_version == "1"


def test_confidence_is_one_minus_delta():
    g = certify_meaning_risk([0.1, 0.2], delta=0.05, **_SCORER)
    assert g.confidence == pytest.approx(0.95)


def test_controls_at_level():
    # 400 perfectly-preserved pairs: Hoeffding radius ≈ 0.061, so the
    # certified ceiling clears 0.10. Fewer samples could not — control
    # is a function of sample size, not just observed loss.
    g = certify_meaning_risk([0.0] * 400, **_SCORER)
    assert g.controls(0.10) is True
    g2 = certify_meaning_risk([0.9] * 20, **_SCORER)  # heavy loss
    assert g2.controls(0.10) is False


def test_small_n_cannot_certify_tight_control():
    """The proof-boundary in action: even perfect observed preservation
    over only 4 pairs cannot certify control at 0.5 — the finite-sample
    radius is wider than that. The certificate refuses to overclaim."""
    g = certify_meaning_risk([0.0] * 4, **_SCORER)
    assert g.point_estimate == 0.0       # proxy sees zero loss
    assert g.controls(0.5) is False      # but 4 samples can't prove it


# ── Hardening: garbage in → error, not a fabricated bound ─────────────


def test_rejects_nan():
    with pytest.raises(ValueError, match="finite"):
        certify_meaning_risk([0.1, float("nan"), 0.2], **_SCORER)


def test_rejects_out_of_range():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        certify_meaning_risk([0.1, 1.5], **_SCORER)


def test_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        certify_meaning_risk([], **_SCORER)


def test_clopper_pearson_requires_binary():
    with pytest.raises(ValueError, match="binary"):
        certify_meaning_risk([0.1, 0.2], method="clopper_pearson", **_SCORER)


def test_auto_picks_clopper_pearson_for_binary():
    g = certify_meaning_risk([0.0, 1.0, 0.0, 0.0], method="auto", **_SCORER)
    assert g.method == "clopper_pearson"


def test_auto_picks_hoeffding_for_fractional():
    g = certify_meaning_risk([0.0, 0.5, 0.3], method="auto", **_SCORER)
    assert g.method == "hoeffding"


def test_is_deterministic():
    losses = [0.1, 0.2, 0.3, 0.15]
    a = certify_meaning_risk(losses, **_SCORER)
    b = certify_meaning_risk(losses, **_SCORER)
    assert a == b
