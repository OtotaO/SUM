"""Multiplier bootstrap contract tests.

Layers:
  1. **Coverage on a known mean** — empirical coverage of the
     bootstrapped mean's CI should track 1-α. The headline
     CCK 2013 guarantee, simplified.
  2. **Vector-valued statistic** — multi-component CIs return
     correct shape and per-component intervals.
  3. **Determinism via rng** — same generator state → same
     replicates → same CIs.
  4. **API surface** — multiplier-type variants, score-type
     edge cases, error paths.
"""
from __future__ import annotations

import numpy as np
import pytest

from sum_engine_internal.research.bootstrap import (
    BootstrapInterval,
    bootstrap_ci,
    gaussian_multipliers,
    multiplier_bootstrap,
    rademacher_multipliers,
)


# -- Coverage on a known scalar ---------------------------------------


def _trial_coverage(true_mean=3.0, n_trials=200, n_per=200, alpha=0.1, seed_base=0):
    """Repeat: draw N(true_mean, 1) of size n_per, bootstrap a
    CI on the mean, check whether it contains true_mean.
    Returns the empirical coverage rate."""
    rng = np.random.default_rng(seed_base)
    hits = 0
    for _ in range(n_trials):
        x = rng.normal(true_mean, 1.0, size=n_per).reshape(-1, 1)
        point, reps = multiplier_bootstrap(
            x, lambda s: np.array([s.mean()]),
            B=300, rng=np.random.default_rng(),
        )
        ivs = bootstrap_ci(point, reps, alpha=alpha)
        if ivs[0].contains(true_mean):
            hits += 1
    return hits / n_trials


@pytest.mark.parametrize("alpha", [0.10, 0.20])
def test_coverage_on_mean_estimation(alpha):
    """Bootstrap CI on the sample mean tracks 1-α within
    Monte-Carlo error."""
    cov = _trial_coverage(alpha=alpha, n_trials=200, n_per=200)
    # n_trials=200 → SE ≈ √(α(1-α)/200) ≈ 0.02-0.03; ±0.06 is
    # ~2-3σ, flake-tolerant for CI
    assert abs(cov - (1 - alpha)) < 0.06, (
        f"empirical coverage {cov:.3f} outside ±0.06 of target {1-alpha:.2f}"
    )


def test_coverage_with_rademacher_multipliers_is_comparable():
    """Both multiplier types should give similar coverage at
    standard sizes."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, size=200).reshape(-1, 1)
    point_g, reps_g = multiplier_bootstrap(
        x, lambda s: np.array([s.mean()]), B=500, multiplier="gaussian",
        rng=np.random.default_rng(0),
    )
    point_r, reps_r = multiplier_bootstrap(
        x, lambda s: np.array([s.mean()]), B=500, multiplier="rademacher",
        rng=np.random.default_rng(0),
    )
    iv_g = bootstrap_ci(point_g, reps_g, alpha=0.10)[0]
    iv_r = bootstrap_ci(point_r, reps_r, alpha=0.10)[0]
    # Widths should agree to within ~2x — they're both unbiased
    # estimators of the bootstrap distribution width
    assert 0.5 < iv_g.width / iv_r.width < 2.0


# -- Vector-valued statistic -------------------------------------------


def test_returns_one_interval_per_statistic_component():
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, size=(100, 3))
    point, reps = multiplier_bootstrap(
        samples, lambda s: s.mean(axis=0), B=200,
    )
    assert point.shape == (3,)
    assert reps.shape == (200, 3)
    intervals = bootstrap_ci(point, reps, alpha=0.10)
    assert len(intervals) == 3
    for iv in intervals:
        assert isinstance(iv, BootstrapInterval)


def test_eigenvalue_bootstrap_returns_meaningful_intervals():
    """Bootstrap top eigenvalues of a covariance matrix. Each
    interval is non-degenerate (lower < upper)."""
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 1, size=(300, 5))

    def top_3_eigvals(s):
        cov = np.cov(s.T)
        return np.linalg.eigvalsh(cov)[::-1][:3]

    point, reps = multiplier_bootstrap(samples, top_3_eigvals, B=200)
    intervals = bootstrap_ci(point, reps, alpha=0.05)
    for iv in intervals:
        assert iv.lower < iv.upper, f"degenerate interval {iv}"
        assert iv.width > 0


# -- Determinism -------------------------------------------------------


def test_same_rng_seed_yields_same_replicates():
    samples = np.arange(100, dtype=float).reshape(-1, 1)
    fn = lambda s: np.array([s.mean()])
    p1, r1 = multiplier_bootstrap(samples, fn, B=50, rng=np.random.default_rng(7))
    p2, r2 = multiplier_bootstrap(samples, fn, B=50, rng=np.random.default_rng(7))
    assert np.array_equal(p1, p2)
    assert np.array_equal(r1, r2)


def test_different_rng_seeds_yield_different_replicates():
    samples = np.arange(100, dtype=float).reshape(-1, 1)
    fn = lambda s: np.array([s.mean()])
    _, r1 = multiplier_bootstrap(samples, fn, B=50, rng=np.random.default_rng(1))
    _, r2 = multiplier_bootstrap(samples, fn, B=50, rng=np.random.default_rng(2))
    assert not np.array_equal(r1, r2)


# -- API + edge cases --------------------------------------------------


def test_alpha_outside_unit_interval_raises():
    point = np.array([0.0])
    reps = np.zeros((10, 1))
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci(point, reps, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci(point, reps, alpha=1.0)


def test_bootstrap_ci_shape_mismatch_raises():
    point = np.array([0.0, 1.0])
    reps = np.zeros((10, 3))  # wrong width
    with pytest.raises(ValueError, match="column count"):
        bootstrap_ci(point, reps, alpha=0.1)


def test_bootstrap_ci_1d_replicates_raises():
    point = np.array([0.0])
    reps = np.zeros(10)  # 1-D not 2-D
    with pytest.raises(ValueError, match="2-D"):
        bootstrap_ci(point, reps, alpha=0.1)


def test_unknown_multiplier_raises():
    with pytest.raises(ValueError, match="multiplier"):
        multiplier_bootstrap(
            np.zeros((10, 1)), lambda s: np.array([0.0]),
            B=5, multiplier="cauchy",  # not supported
        )


def test_empty_samples_raises():
    with pytest.raises(ValueError, match="non-empty"):
        multiplier_bootstrap(
            np.zeros((0, 1)), lambda s: np.array([0.0]), B=5,
        )


def test_inconsistent_statistic_shape_raises():
    """If statistic_fn returns different shapes across calls, the
    bootstrap loop should error rather than silently mis-aggregate."""
    samples = np.zeros((10, 1))
    call_count = [0]
    def bad_fn(s):
        # First call returns shape (3,); second call returns shape (5,)
        call_count[0] += 1
        return np.zeros(3) if call_count[0] == 1 else np.zeros(5)
    with pytest.raises(ValueError, match="returned shape"):
        multiplier_bootstrap(samples, bad_fn, B=2)


# -- Multiplier helpers ------------------------------------------------


def test_gaussian_multipliers_have_correct_distribution():
    rng = np.random.default_rng(0)
    eps = gaussian_multipliers(10000, rng=rng)
    assert eps.shape == (10000,)
    # Sanity: sample mean ≈ 0, sample std ≈ 1
    assert abs(eps.mean()) < 0.05
    assert abs(eps.std() - 1.0) < 0.05


def test_rademacher_multipliers_are_pm_one():
    rng = np.random.default_rng(0)
    eps = rademacher_multipliers(1000, rng=rng)
    assert eps.shape == (1000,)
    assert set(np.unique(eps)) <= {-1.0, 1.0}


# -- BootstrapInterval -------------------------------------------------


def test_interval_contains_and_width():
    iv = BootstrapInterval(point=5.0, lower=4.0, upper=6.5, alpha=0.05)
    assert iv.width == 2.5
    assert iv.contains(5.0)
    assert iv.contains(4.0)
    assert iv.contains(6.5)
    assert not iv.contains(3.99)
