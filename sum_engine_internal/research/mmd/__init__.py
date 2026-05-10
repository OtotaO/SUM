"""Maximum Mean Discrepancy — provable kernel-distance between
axiom-set distributions.

Hilbert-space framing: each axiom set ``A = {a_1, ..., a_n}``
embeds into a Reproducing Kernel Hilbert Space via the kernel
mean

    μ_A = (1/n) Σ_i φ(a_i)

where ``φ`` is the canonical feature map of a chosen kernel.
The squared MMD between two distributions ``P`` and ``Q``
sampled by ``A`` and ``B`` is

    MMD²(A, B) = ‖μ_A − μ_B‖²_RKHS

which expands (Gretton et al., *JMLR* 13:723–773, 2012, Eq. 3) to

    MMD² = (1/n²) Σ_ij k(a_i, a_j)
         − (2/nm)  Σ_ij k(a_i, b_j)
         +   (1/m²) Σ_ij k(b_i, b_j)

**Provable kernel** (Gretton 2012, Theorem 5): for a *characteristic*
kernel (RBF/Gaussian is characteristic on ℝ^d), MMD = 0 ⟺ P = Q.
This makes MMD a true metric on probability distributions —
ad-hoc Jaccard / cosine substitutes get a mathematically-grounded
replacement.

Substrate use: every signed bundle gets an
``axiom_distribution_mmd`` metadata field measuring the bundle's
distance from the substrate's calibration baseline. Cross-bundle
distribution-shift detection becomes a single scalar on bundle
metadata. Compounds with PR #185 (multiplier bootstrap for CIs
on MMD), PR #183 (conformal for calibrated thresholds), PR #184
(vN entropy as a different scalar of the same axiom set).
"""
from sum_engine_internal.research.mmd.mmd import (
    rbf_kernel_matrix,
    mmd_squared,
    mmd_permutation_pvalue,
    median_heuristic_bandwidth,
)
from sum_engine_internal.research.mmd.baseline import (
    BaselineMMDComputer,
    get_default_mmd_computer,
)

__all__ = [
    "rbf_kernel_matrix",
    "mmd_squared",
    "mmd_permutation_pvalue",
    "median_heuristic_bandwidth",
    "BaselineMMDComputer",
    "get_default_mmd_computer",
]
