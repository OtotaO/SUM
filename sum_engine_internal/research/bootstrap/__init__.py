"""Gaussian multiplier bootstrap — distribution-free CIs on
vector-valued statistics.

Replaces threshold-magic decisions on existing scalar / spectral
detectors (sheaf-Laplacian eigenvalues, von Neumann graph
entropy, RPCA corruption_score) with statistically-grounded CIs.

Core kernel: given iid samples X_1, ..., X_n and a statistic
T(X_1, ..., X_n), the multiplier bootstrap approximates the
distribution of T − E[T] by

    T*_b = (1/n) Σ_i ε_i (X_i − X̄),    ε_i ~iid N(0, 1)

Quantiles of {T*_b}_{b=1..B} give a (1-α) CI on T. The Gaussian-
multiplier variant has the strongest theoretical guarantees in
high dimensions:

  Chernozhukov, Chetverikov & Kato, *Annals of Statistics*
  41(6):2786-2819 (2013) — Gaussian multiplier bootstrap is
  valid for sup-norm functionals of high-dimensional means with
  rate (log p / n)^{1/2}, no Gaussian-noise assumption.

The substrate use case: each existing scalar detector becomes
"value ± CI" instead of "value vs threshold". Three concrete
wrap targets are surfaced in the spike findings doc.
"""
from sum_engine_internal.research.bootstrap.multiplier_bootstrap import (
    BootstrapInterval,
    bootstrap_ci,
    multiplier_bootstrap,
    gaussian_multipliers,
    rademacher_multipliers,
)

__all__ = [
    "BootstrapInterval",
    "bootstrap_ci",
    "multiplier_bootstrap",
    "gaussian_multipliers",
    "rademacher_multipliers",
]
