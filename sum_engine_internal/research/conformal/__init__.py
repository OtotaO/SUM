"""Distribution-free uncertainty quantification via conformal prediction.

Provides finite-sample, distribution-free coverage guarantees for
arbitrary point-predictor outputs. The substrate's slider axes,
ridge readouts, and any future per-axiom score can be wrapped to
emit calibrated prediction intervals instead of bare point values.

Core kernel (split conformal): given exchangeable
(X_i, Y_i)_{i=1..n} and a regressor f trained on a disjoint
training fold, compute non-conformity scores s_i = |Y_i - f(X_i)|
on the calibration fold; for a new X_{n+1}, the prediction set

    C(X_{n+1}) = [f(X_{n+1}) - q̂, f(X_{n+1}) + q̂]

with q̂ = the ⌈(n+1)(1-α)⌉/n empirical quantile of {s_i} satisfies

    P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 - α                  [provable]

— Vovk-Gammerman-Shafer, *Algorithmic Learning in a Random World*
(Springer 2005); modern treatment Angelopoulos & Bates,
*Foundations and Trends in ML* (2023).

The guarantee holds *finite-sample* and *distribution-free* — no
Gaussian-noise or i.i.d. assumptions, just exchangeability.
"""
from sum_engine_internal.research.conformal.split_conformal import (
    SplitConformal,
    ConformalInterval,
    empirical_coverage,
    average_interval_width,
)
from sum_engine_internal.research.conformal.entropy_baseline import (
    BaselineEntropyPredictor,
    get_default_predictor,
)

__all__ = [
    "SplitConformal",
    "ConformalInterval",
    "empirical_coverage",
    "average_interval_width",
    "BaselineEntropyPredictor",
    "get_default_predictor",
]
