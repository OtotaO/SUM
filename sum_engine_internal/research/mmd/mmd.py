"""Maximum Mean Discrepancy core.

Two functions form the load-bearing kernel:

  - ``rbf_kernel_matrix(X, Y, sigma)`` — pairwise RBF kernel matrix
    K[i,j] = exp(−‖x_i − y_j‖² / (2σ²))
  - ``mmd_squared(K_xx, K_xy, K_yy)`` — biased empirical MMD²
    estimator (Gretton et al., *JMLR* 13:723–773, 2012, Eq. 3)

Bandwidth selection uses the **median heuristic** (Gretton 2012, §8):
σ = median{‖x_i − x_j‖} over all pairs in X ∪ Y. Heuristic but
universally adopted; deterministic given a fixed sample.

The biased estimator (vs. the unbiased Eq. 5) is the simpler
choice for substrate metadata: it's strictly non-negative, has
the same asymptotic properties, and its O(1/n²) bias on small
samples is dominated by the substrate's other measurement
uncertainties.
"""
from __future__ import annotations

import numpy as np


def rbf_kernel_matrix(
    X: np.ndarray, Y: np.ndarray, sigma: float,
) -> np.ndarray:
    """Pairwise RBF kernel matrix K[i, j] = exp(-‖X_i − Y_j‖² / (2σ²)).

    Args:
        X: shape ``(n, d)``
        Y: shape ``(m, d)``
        sigma: bandwidth (>0). Use ``median_heuristic_bandwidth`` if
            unsure.

    Returns:
        ``(n, m)`` kernel matrix; values in [0, 1].
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0; got {sigma}")
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(
            f"X and Y must be 2-D; got shapes {X.shape}, {Y.shape}"
        )
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"feature dim mismatch: X has {X.shape[1]}, Y has {Y.shape[1]}"
        )
    # ‖x − y‖² = ‖x‖² + ‖y‖² − 2 ⟨x, y⟩
    sq_x = np.sum(X ** 2, axis=1, keepdims=True)         # (n, 1)
    sq_y = np.sum(Y ** 2, axis=1, keepdims=True).T       # (1, m)
    sq_dist = sq_x + sq_y - 2.0 * (X @ Y.T)
    np.maximum(sq_dist, 0.0, out=sq_dist)  # numerical floor
    return np.exp(-sq_dist / (2.0 * sigma * sigma))


def median_heuristic_bandwidth(
    X: np.ndarray, Y: np.ndarray | None = None,
) -> float:
    """Median pairwise distance over the joint sample.

    Standard MMD bandwidth choice (Gretton 2012, §8): σ = median
    {‖x_i − x_j‖} over all pairs in X ∪ Y. Returns a positive
    scalar; falls back to 1.0 if all pairs have zero distance
    (degenerate input).
    """
    X = np.asarray(X, dtype=np.float64)
    if Y is not None:
        Y = np.asarray(Y, dtype=np.float64)
        joint = np.vstack([X, Y])
    else:
        joint = X
    n = len(joint)
    if n < 2:
        return 1.0
    # Compute upper-triangular pairwise distances
    sq_norms = np.sum(joint ** 2, axis=1)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (joint @ joint.T)
    np.maximum(sq_dists, 0.0, out=sq_dists)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(sq_dists[iu])
    if dists.size == 0:
        return 1.0
    median = float(np.median(dists))
    return median if median > 0 else 1.0


def mmd_squared(
    K_xx: np.ndarray, K_xy: np.ndarray, K_yy: np.ndarray,
) -> float:
    """Biased empirical MMD² (Gretton 2012, Eq. 3):

        MMD² = (1/n²)·Σ K_xx + (1/m²)·Σ K_yy − (2/nm)·Σ K_xy

    Always non-negative when the kernel is PSD (RBF is). Two
    identical samples give MMD² = 0.

    Args:
        K_xx: (n, n) within-X kernel matrix
        K_xy: (n, m) cross kernel matrix
        K_yy: (m, m) within-Y kernel matrix
    """
    n = K_xx.shape[0]
    m = K_yy.shape[0]
    if n == 0 or m == 0:
        raise ValueError("MMD² requires non-empty samples")
    if K_xx.shape != (n, n) or K_yy.shape != (m, m):
        raise ValueError(
            f"K_xx must be (n, n) and K_yy (m, m); got {K_xx.shape}, "
            f"{K_yy.shape}"
        )
    if K_xy.shape != (n, m):
        raise ValueError(f"K_xy must be (n, m) = ({n}, {m}); got {K_xy.shape}")
    s_xx = K_xx.sum() / (n * n)
    s_yy = K_yy.sum() / (m * m)
    s_xy = K_xy.sum() / (n * m)
    val = float(s_xx + s_yy - 2.0 * s_xy)
    # Numerical floor — biased estimator is always ≥ 0 in theory
    return max(val, 0.0)
