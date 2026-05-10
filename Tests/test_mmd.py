"""MMD math contract tests.

Layers:
  1. **Provable kernel** — identical samples produce MMD² = 0;
     shifted distributions produce MMD² strictly larger than
     same-distribution different-draw MMD². The Gretton 2012
     theorem in action.
  2. **Numerical robustness** — non-negativity, symmetry,
     bandwidth handling.
  3. **Edge cases** — single-sample, dimension mismatch, bad
     bandwidth.
  4. **Baseline computer** — calibrates from corpora; cold-start
     returns None; predict_mmd returns sensible shape.
"""
from __future__ import annotations

import numpy as np
import pytest

from sum_engine_internal.research.mmd import (
    BaselineMMDComputer,
    median_heuristic_bandwidth,
    mmd_squared,
    rbf_kernel_matrix,
)


# -- Provable kernel: identical → 0; shifted → strictly larger ---------


def test_identical_samples_yield_zero_mmd():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (50, 5))
    sigma = median_heuristic_bandwidth(X, X)
    K = rbf_kernel_matrix(X, X, sigma)
    assert mmd_squared(K, K, K) < 1e-10


def test_shifted_distribution_yields_larger_mmd_than_same_distribution():
    """Headline provable property: MMD discriminates shift from
    no-shift. Should hold even at small n."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (60, 5))
    X2 = rng.normal(0, 1, (60, 5))   # same distribution, different draw
    Y = rng.normal(5, 1, (60, 5))    # mean-shifted

    sigma = median_heuristic_bandwidth(X, Y)
    K_xx = rbf_kernel_matrix(X, X, sigma)
    K_x2 = rbf_kernel_matrix(X, X2, sigma)
    K_xy = rbf_kernel_matrix(X, Y, sigma)
    K_yy = rbf_kernel_matrix(Y, Y, sigma)
    K_x2x2 = rbf_kernel_matrix(X2, X2, sigma)

    mmd_same = mmd_squared(K_xx, K_x2, K_x2x2)
    mmd_shift = mmd_squared(K_xx, K_xy, K_yy)
    assert mmd_shift > mmd_same * 5, (
        f"shifted MMD² {mmd_shift:.4f} should be ≥5× same-dist "
        f"MMD² {mmd_same:.4f}"
    )


def test_mmd_is_non_negative():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (20, 3))
    Y = rng.normal(2, 0.5, (20, 3))
    sigma = median_heuristic_bandwidth(X, Y)
    val = mmd_squared(
        rbf_kernel_matrix(X, X, sigma),
        rbf_kernel_matrix(X, Y, sigma),
        rbf_kernel_matrix(Y, Y, sigma),
    )
    assert val >= 0


def test_mmd_is_symmetric():
    """MMD²(A, B) == MMD²(B, A) — kernel matrices swap, math
    is invariant."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (30, 4))
    Y = rng.normal(1, 1, (30, 4))
    sigma = median_heuristic_bandwidth(X, Y)
    K_xx = rbf_kernel_matrix(X, X, sigma)
    K_xy = rbf_kernel_matrix(X, Y, sigma)
    K_yx = rbf_kernel_matrix(Y, X, sigma)
    K_yy = rbf_kernel_matrix(Y, Y, sigma)
    a = mmd_squared(K_xx, K_xy, K_yy)
    b = mmd_squared(K_yy, K_yx, K_xx)
    assert abs(a - b) < 1e-12


# -- RBF kernel matrix --------------------------------------------------


def test_rbf_diagonal_is_one():
    """k(x, x) = exp(-0/2σ²) = 1 for any σ."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (10, 3))
    K = rbf_kernel_matrix(X, X, sigma=1.5)
    assert np.allclose(np.diag(K), 1.0)


def test_rbf_values_in_unit_interval():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (20, 4))
    Y = rng.normal(0, 1, (20, 4))
    K = rbf_kernel_matrix(X, Y, sigma=2.0)
    assert (K >= 0).all() and (K <= 1.0).all()


def test_rbf_dimension_mismatch_raises():
    with pytest.raises(ValueError, match="feature dim"):
        rbf_kernel_matrix(np.zeros((5, 3)), np.zeros((5, 4)), sigma=1.0)


def test_rbf_negative_sigma_raises():
    with pytest.raises(ValueError, match="sigma"):
        rbf_kernel_matrix(np.zeros((2, 2)), np.zeros((2, 2)), sigma=-1.0)


def test_rbf_zero_sigma_raises():
    with pytest.raises(ValueError, match="sigma"):
        rbf_kernel_matrix(np.zeros((2, 2)), np.zeros((2, 2)), sigma=0.0)


# -- Median heuristic --------------------------------------------------


def test_median_heuristic_returns_positive():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (20, 3))
    sigma = median_heuristic_bandwidth(X)
    assert sigma > 0


def test_median_heuristic_falls_back_to_one_on_degenerate_input():
    """All-identical points produce zero distances → fall back to
    σ = 1.0 rather than returning 0 (which would crash the kernel)."""
    X = np.zeros((5, 3))
    sigma = median_heuristic_bandwidth(X)
    assert sigma == 1.0


def test_median_heuristic_single_sample_returns_one():
    sigma = median_heuristic_bandwidth(np.zeros((1, 3)))
    assert sigma == 1.0


# -- mmd_squared edge cases --------------------------------------------


def test_mmd_squared_empty_sample_raises():
    with pytest.raises(ValueError, match="non-empty"):
        mmd_squared(np.zeros((0, 0)), np.zeros((0, 5)), np.zeros((5, 5)))


def test_mmd_squared_kernel_shape_mismatch_raises():
    K_xx = np.zeros((5, 5))
    K_yy = np.zeros((3, 3))
    K_xy_wrong = np.zeros((5, 4))  # should be (5, 3)
    with pytest.raises(ValueError, match="K_xy"):
        mmd_squared(K_xx, K_xy_wrong, K_yy)


# -- BaselineMMDComputer -----------------------------------------------


def test_uncalibrated_computer_returns_none_for_predict():
    c = BaselineMMDComputer()
    assert not c.is_calibrated
    assert c.predict_mmd([]) is None
    # even with sample triples, returns None when uncalibrated
    from sum_engine_internal.graph_store import Triple
    assert c.predict_mmd([Triple("a", "b", "c")]) is None


def test_default_computer_calibrates_from_seed_corpora():
    """Lazy singleton initialisation runs `calibrate_from_corpora`
    on first access. Should succeed against the existing seed_*
    corpora."""
    from sum_engine_internal.research.mmd import get_default_mmd_computer
    c = get_default_mmd_computer()
    assert c.is_calibrated
    assert c.n_baseline_samples >= 100  # 6 corpora produces ~300+


def test_default_computer_predict_mmd_returns_expected_shape():
    from sum_engine_internal.graph_store import Triple
    from sum_engine_internal.research.mmd import get_default_mmd_computer
    c = get_default_mmd_computer()
    result = c.predict_mmd([
        Triple("alice", "build", "house"),
        Triple("bob", "write", "book"),
    ])
    assert isinstance(result, dict)
    for key in ("mmd_squared", "bandwidth", "n_baseline_samples", "n_bundle_samples"):
        assert key in result
    assert result["mmd_squared"] >= 0
    assert result["bandwidth"] > 0
    assert result["n_bundle_samples"] == 2


def test_predict_mmd_returns_none_for_empty_triples(codec_unused=None):
    from sum_engine_internal.research.mmd import get_default_mmd_computer
    c = get_default_mmd_computer()
    assert c.predict_mmd([]) is None
