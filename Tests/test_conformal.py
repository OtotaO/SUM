"""Split conformal prediction contract tests.

Layers:
  1. **Coverage guarantee** — the load-bearing finite-sample claim
     from Vovk-Gammerman-Shafer / Angelopoulos-Bates 2023. Empirical
     coverage on synthetic exchangeable data must hit 1-α to
     within Monte-Carlo tolerance.
  2. **Determinism** — same calibration data → same q̂ → same intervals.
  3. **API surface** — ConformalInterval shape, score_type variants,
     edge cases, error paths.
  4. **Diagnostics** — empirical_coverage and average_interval_width
     return sensible numbers on toy inputs.

Coverage tolerance: with n_test = 2000 calibration + 2000 test,
α=0.1 has Monte-Carlo SD ≈ √(α(1-α)/n_test) ≈ 0.0067; ±0.03 is a
~4σ band so flake-tolerant for CI.
"""
from __future__ import annotations

import numpy as np
import pytest

from sum_engine_internal.research.conformal import (
    ConformalInterval,
    SplitConformal,
    average_interval_width,
    empirical_coverage,
)


# -- Coverage guarantee on synthetic data ------------------------------


def _synthetic(n=4000, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-5, 5, n)
    y = 2 * x + rng.normal(0, 1, n)
    pred = np.full_like(y, 0.0)  # constant-mean predictor (poor)
    return pred, y


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_coverage_matches_target_within_band(alpha):
    """The provable kernel: empirical coverage ≥ 1-α
    finite-sample under exchangeability. We allow a ±0.03
    band (~4σ at n_test=2000)."""
    pred, y = _synthetic(n=4000, seed=42)
    cal_pred, test_pred = pred[:2000], pred[2000:]
    cal_y, test_y = y[:2000], y[2000:]

    sc = SplitConformal(alpha=alpha)
    sc.calibrate(cal_pred, cal_y)
    intervals = sc.predict_batch(test_pred)
    coverage = empirical_coverage(intervals, test_y)
    assert abs(coverage - (1 - alpha)) <= 0.03, (
        f"empirical coverage {coverage:.3f} outside ±0.03 of target {1-alpha:.2f}"
    )


def test_signed_score_type_also_meets_coverage():
    pred, y = _synthetic(seed=99)
    cal_pred, test_pred = pred[:2000], pred[2000:]
    cal_y, test_y = y[:2000], y[2000:]
    sc = SplitConformal(alpha=0.1, score_type="signed")
    sc.calibrate(cal_pred, cal_y)
    intervals = sc.predict_batch(test_pred)
    coverage = empirical_coverage(intervals, test_y)
    assert abs(coverage - 0.9) <= 0.03


# -- Determinism -------------------------------------------------------


def test_same_calibration_data_yields_same_intervals():
    pred, y = _synthetic(n=200, seed=1)
    sc1 = SplitConformal(alpha=0.1)
    sc1.calibrate(pred, y)
    sc2 = SplitConformal(alpha=0.1)
    sc2.calibrate(pred, y)
    iv1 = sc1.predict(0.0)
    iv2 = sc2.predict(0.0)
    assert iv1.lower == iv2.lower
    assert iv1.upper == iv2.upper


def test_recalibration_overrides_prior_state():
    pred, y = _synthetic(n=200, seed=1)
    sc = SplitConformal(alpha=0.1)
    sc.calibrate(pred, y)
    iv_first = sc.predict(0.0)

    # Calibrate again on tighter data → tighter interval
    tight_pred = np.zeros(200)
    tight_y = np.zeros(200)
    sc.calibrate(tight_pred, tight_y)
    iv_after = sc.predict(0.0)
    assert iv_after.width < iv_first.width


# -- API + edge cases --------------------------------------------------


def test_predict_before_calibrate_raises():
    sc = SplitConformal(alpha=0.1)
    with pytest.raises(RuntimeError, match="before calibrate"):
        sc.predict(0.0)


def test_alpha_outside_unit_interval_raises():
    with pytest.raises(ValueError, match="alpha"):
        SplitConformal(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        SplitConformal(alpha=1.0)
    with pytest.raises(ValueError, match="alpha"):
        SplitConformal(alpha=-0.1)


def test_unknown_score_type_raises():
    with pytest.raises(ValueError, match="score_type"):
        SplitConformal(alpha=0.1, score_type="quadratic")


def test_calibrate_shape_mismatch_raises():
    sc = SplitConformal(alpha=0.1)
    with pytest.raises(ValueError, match="shape mismatch"):
        sc.calibrate(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_calibrate_2d_raises():
    sc = SplitConformal(alpha=0.1)
    with pytest.raises(ValueError, match="1-D"):
        sc.calibrate(np.zeros((10, 2)), np.zeros((10, 2)))


def test_calibrate_empty_raises():
    sc = SplitConformal(alpha=0.1)
    with pytest.raises(ValueError, match="non-empty"):
        sc.calibrate(np.array([]), np.array([]))


def test_interval_contains_and_width():
    iv = ConformalInterval(point=5.0, lower=4.0, upper=6.5, alpha=0.1, score_type="absolute")
    assert iv.contains(5.0)
    assert iv.contains(4.0)
    assert iv.contains(6.5)
    assert not iv.contains(3.99)
    assert not iv.contains(6.51)
    assert iv.width == 2.5


# -- Diagnostics -------------------------------------------------------


def test_empirical_coverage_on_perfect_intervals():
    intervals = [
        ConformalInterval(point=0.0, lower=-1.0, upper=1.0, alpha=0.1, score_type="absolute"),
        ConformalInterval(point=0.0, lower=-1.0, upper=1.0, alpha=0.1, score_type="absolute"),
    ]
    assert empirical_coverage(intervals, np.array([0.5, -0.5])) == 1.0


def test_empirical_coverage_on_zero_hits():
    intervals = [
        ConformalInterval(point=0.0, lower=-1.0, upper=1.0, alpha=0.1, score_type="absolute"),
    ]
    assert empirical_coverage(intervals, np.array([10.0])) == 0.0


def test_empirical_coverage_length_mismatch_raises():
    intervals = [ConformalInterval(0.0, -1.0, 1.0, 0.1, "absolute")]
    with pytest.raises(ValueError, match="length mismatch"):
        empirical_coverage(intervals, np.array([0.0, 0.0]))


def test_average_width_on_known_intervals():
    intervals = [
        ConformalInterval(0.0, -1.0, 1.0, 0.1, "absolute"),  # width 2
        ConformalInterval(0.0, -2.0, 2.0, 0.1, "absolute"),  # width 4
    ]
    assert average_interval_width(intervals) == 3.0


def test_average_width_on_empty_list_returns_zero():
    assert average_interval_width([]) == 0.0


def test_empirical_coverage_on_empty_returns_zero():
    assert empirical_coverage([], np.array([])) == 0.0
