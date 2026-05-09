"""Principal Component Pursuit via ADMM.

Convex robust-PCA solver. Recovers the unique (L₀, S₀) decomposition
of M = L₀ + S₀ where L₀ is low-rank and S₀ is sparse, by minimising

    ‖L‖_∗ + λ ‖S‖_1    s.t.   M = L + S

via the Augmented Lagrangian Method of Multipliers (ADMM, equivalent
to the Inexact ALM in Lin et al., arXiv:1009.5055, 2010).

The primal updates are:

  1. L ← SVT_{1/μ}(M − S + Y/μ)        (singular-value thresholding)
  2. S ← soft_thresh_{λ/μ}(M − L + Y/μ) (entry-wise soft thresholding)
  3. Y ← Y + μ (M − L − S)              (dual ascent)

with μ adaptive (μ ← min(ρ μ, μ_max)) and convergence measured by
‖M − L − S‖_F / ‖M‖_F < tol.

The ~60-line scope and the convergence guarantee (under the Candès et
al. 2011 incoherence + uniform-random-support assumptions) make this
the right *minimum kernel* per the deep-research article's §9.1
"smallest experiment" framing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


class PCPConvergenceError(RuntimeError):
    """Raised when PCP iteration does not converge within max_iter."""


@dataclass(frozen=True, slots=True)
class PCPResult:
    """Result of a successful PCP decomposition.

    Invariant: M ≈ L + S to within ``residual_norm`` (Frobenius).
    """
    L: np.ndarray            # low-rank component
    S: np.ndarray            # sparse component
    n_iter: int              # iterations until convergence
    residual_norm: float     # final ‖M − L − S‖_F / ‖M‖_F
    rank_estimate: int       # number of non-zero singular values in L (above tol)
    sparsity_estimate: float # fraction of S entries above zero-tolerance
    lam: float               # λ used (matches Candès default unless overridden)


def _soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    """Element-wise soft-thresholding: sign(x) · max(|x| − τ, 0)."""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def _svt(X: np.ndarray, tau: float) -> tuple[np.ndarray, int]:
    """Singular-value thresholding: SVD, soft-threshold the
    singular values, reconstruct. Returns (X_thresholded, rank).

    Rank is counted with a relative threshold (max-singular-value /
    1e6) rather than absolute, so numerical-noise singular values
    don't inflate the count when ADMM has converged.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    if s_thresh.size > 0 and s_thresh[0] > 0:
        rank_threshold = s_thresh[0] * 1e-6
        rank = int(np.sum(s_thresh > rank_threshold))
    else:
        rank = 0
    return (U * s_thresh) @ Vt, rank


def pcp(
    M: np.ndarray,
    *,
    lam: Optional[float] = None,
    mu: Optional[float] = None,
    rho: float = 1.5,
    mu_max: Optional[float] = None,
    tol: float = 1e-7,
    max_iter: int = 500,
    nonzero_tol: float = 1e-9,
) -> PCPResult:
    """Solve M = L + S via Principal Component Pursuit (ADMM).

    Args:
        M: Input matrix (n × d). Real-valued.
        lam: Regulariser. Default: 1/√(max(n, d)) — Candès et al. 2011's
            theoretically-justified value. Override for adversarial
            workloads.
        mu: Initial penalty parameter for the augmented Lagrangian.
            Default: 1.25 / ‖M‖₂ (Lin et al. 2010, Algorithm 5). This
            initialisation matters: an order-of-magnitude wrong μ₀
            converges to a high-rank "fits-but-doesn't-separate"
            solution where M = L + S holds but the rank/sparsity
            split is wrong.
        rho: Geometric increase rate for μ each iteration. > 1.
        mu_max: Cap on μ to avoid numerical blow-up. Default 10⁷.
        tol: Convergence tolerance on ‖M − L − S‖_F / ‖M‖_F.
        max_iter: Hard iteration cap; raises PCPConvergenceError if hit.
        nonzero_tol: Threshold below which S entries / L singular values
            are reported as zero (only affects the reported `rank_estimate`
            and `sparsity_estimate`, not the primal solution).

    Returns:
        PCPResult with L, S, n_iter, residual_norm, rank_estimate,
        sparsity_estimate, and the λ used.

    Raises:
        PCPConvergenceError if the residual does not fall below `tol`
        within `max_iter` iterations.
        ValueError if M is empty or non-finite.
    """
    if M.size == 0:
        raise ValueError("PCP input M is empty")
    if not np.all(np.isfinite(M)):
        raise ValueError("PCP input M contains non-finite values")

    n, d = M.shape
    M_norm = np.linalg.norm(M, "fro")
    if M_norm == 0:
        # All-zero input: trivial decomposition
        return PCPResult(
            L=np.zeros_like(M), S=np.zeros_like(M), n_iter=0,
            residual_norm=0.0, rank_estimate=0, sparsity_estimate=0.0,
            lam=0.0,
        )

    # Defaults from the literature
    lam = lam if lam is not None else 1.0 / np.sqrt(max(n, d))
    if mu is None:
        spectral_norm = np.linalg.norm(M, 2)
        mu = 1.25 / spectral_norm if spectral_norm > 0 else 1.0
    mu_max = mu_max if mu_max is not None else 1e7

    L = np.zeros_like(M)
    S = np.zeros_like(M)
    Y = M / max(np.linalg.norm(M, 2), np.linalg.norm(M, np.inf) / lam)

    rank = 0
    for it in range(1, max_iter + 1):
        L, rank = _svt(M - S + Y / mu, 1.0 / mu)
        S = _soft_threshold(M - L + Y / mu, lam / mu)
        residual = M - L - S
        Y = Y + mu * residual
        mu = min(rho * mu, mu_max)
        rel_residual = np.linalg.norm(residual, "fro") / M_norm
        if rel_residual < tol:
            sparsity = float(np.mean(np.abs(S) > nonzero_tol))
            return PCPResult(
                L=L, S=S, n_iter=it, residual_norm=float(rel_residual),
                rank_estimate=rank, sparsity_estimate=sparsity, lam=float(lam),
            )

    raise PCPConvergenceError(
        f"PCP did not converge within {max_iter} iterations "
        f"(final relative residual: {rel_residual:.3e}, target: {tol:.3e}). "
        f"Try increasing max_iter, raising mu_max, or relaxing tol."
    )


def corruption_score(M: np.ndarray, **pcp_kwargs) -> np.ndarray:
    """Per-row L1 norm of the sparse component S after PCP decomposition.

    High score ⇒ row is flagged as corruption (or otherwise off-manifold).
    Low score ⇒ row lives on the consensus low-rank manifold.

    Convenience wrapper around `pcp`. Pass-through kwargs for tuning.
    """
    result = pcp(M, **pcp_kwargs)
    return np.linalg.norm(result.S, ord=1, axis=1)
