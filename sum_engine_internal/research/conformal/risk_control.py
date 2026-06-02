"""Distribution-free lower confidence bounds on a preservation rate.

The split-conformal kernel (`split_conformal.py`) wraps a point
predictor in a calibrated *interval*. This module answers the
complementary, one-sided question that SUM's slider contract actually
asks:

    "With confidence ≥ 1 - δ, what is the largest X such that the
     fact-preservation rate ≥ X?"

That is a one-sided lower confidence bound on the mean of bounded
[0, 1] observations (per-cell preservation fractions) or on a binomial
proportion (per-fact preserved / lost). It is the certifier shape the
bench-hardening plan's T3 names — "fact preservation ≥ X with 95 %
confidence over the tested envelope" — expressed as a finite-sample,
distribution-free guarantee rather than a tail percentile of an
empirical distribution.

Two bounds ship, both finite-sample and distribution-free:

  - **Hoeffding** — for any observations in [0, 1]. From Hoeffding's
    inequality P(μ̂ - μ ≥ t) ≤ exp(-2 n t²), the (1-δ) one-sided lower
    bound is μ̂ - sqrt(ln(1/δ) / (2n)). Always valid; conservative.   [provable]

  - **Clopper–Pearson** — exact one-sided lower limit for a binomial
    proportion (per-fact preserved/lost), the β-quantile
    Beta(δ; k, n-k+1). Tighter than Hoeffding for the binary view and
    the most interpretable framing ("≥ X % of facts preserved").      [provable]

Relationship to DKW (the other T3 tool): DKW bounds the *entire* drift
CDF uniformly, which is the right tool for a quantile statement over a
distribution. For a single *rate* (a mean / proportion), the bounds
here are the tighter, purpose-built instrument. Use DKW for the
full-distribution worst-case envelope and these for the headline rate;
they are complementary, not redundant.

Honest boundary: like all conformal-family guarantees, validity rests
on **exchangeability** between the calibration sample and deployment —
i.e. the bound holds *within the tested envelope* (the T2 capability
region), degrading on out-of-distribution inputs. State the envelope
alongside the bound; never quote the rate without it.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class RateGuarantee:
    """A finite-sample, distribution-free lower bound on a rate.

    Reads as: "with confidence ≥ ``confidence``, the true rate is
    ≥ ``rate_lower_bound``", valid under exchangeability of the
    sample with deployment (i.e. within the tested envelope).
    """
    rate_lower_bound: float   # the certified floor X
    point_estimate: float     # observed mean / proportion
    n: int                    # sample size
    delta: float              # miscoverage allowance (confidence = 1 - delta)
    method: str               # "hoeffding" | "clopper_pearson"

    @property
    def confidence(self) -> float:
        return 1.0 - self.delta

    @property
    def slack(self) -> float:
        """Gap between the point estimate and the certified floor —
        the price of finite-sample, distribution-free rigour."""
        return self.point_estimate - self.rate_lower_bound


def _validate_delta(delta: float) -> None:
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1); got {delta}")


def hoeffding_lower_bound(values: Sequence[float], delta: float = 0.05) -> float:
    """One-sided (1-δ) lower confidence bound on the mean of [0, 1]
    observations, via Hoeffding's inequality. Distribution-free,
    finite-sample. Clamped to [0, 1]."""
    _validate_delta(delta)
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"values must be 1-D; got shape {arr.shape}")
    n = arr.size
    if n < 1:
        raise ValueError("values must be non-empty")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("Hoeffding bound requires all values in [0, 1]")
    mean = float(arr.mean())
    radius = math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    return max(0.0, min(1.0, mean - radius))


def clopper_pearson_lower_bound(successes: int, n: int, delta: float = 0.05) -> float:
    """Exact one-sided (1-δ) lower confidence limit for a binomial
    proportion (``successes`` of ``n`` Bernoulli trials).

    The limit is the δ-quantile of Beta(successes, n - successes + 1),
    with the standard convention that the bound is 0 when there are no
    successes. Tighter than Hoeffding for binary data and exact (never
    under-covers)."""
    _validate_delta(delta)
    if n < 1:
        raise ValueError("n must be >= 1")
    if not (0 <= successes <= n):
        raise ValueError(f"successes must be in [0, n]; got {successes} of {n}")
    if successes == 0:
        return 0.0
    # Lazy import: keeps the module usable (Hoeffding path) without scipy.
    from scipy.stats import beta  # type: ignore
    return float(beta.ppf(delta, successes, n - successes + 1))


def certify_rate(
    observations: Sequence[float],
    delta: float = 0.05,
    method: Literal["auto", "hoeffding", "clopper_pearson"] = "auto",
) -> RateGuarantee:
    """Certify a distribution-free lower bound on the preservation rate.

    ``method="auto"`` picks Clopper–Pearson when every observation is
    exactly 0 or 1 (the per-fact preserved/lost view — exact and
    tightest) and Hoeffding otherwise (the per-cell [0, 1] fraction
    view — always valid).
    """
    arr = np.asarray(observations, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"observations must be 1-D; got shape {arr.shape}")
    n = arr.size
    if n < 1:
        raise ValueError("observations must be non-empty")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("observations must lie in [0, 1]")

    is_binary = bool(np.all(np.isin(arr, (0.0, 1.0))))
    chosen = method
    if method == "auto":
        chosen = "clopper_pearson" if is_binary else "hoeffding"

    if chosen == "clopper_pearson":
        if not is_binary:
            raise ValueError(
                "clopper_pearson requires binary (0/1) observations; "
                "use 'hoeffding' for fractional [0, 1] values"
            )
        successes = int(round(float(arr.sum())))
        lb = clopper_pearson_lower_bound(successes, n, delta)
    elif chosen == "hoeffding":
        lb = hoeffding_lower_bound(arr, delta)
    else:
        raise ValueError(f"unknown method {method!r}")

    return RateGuarantee(
        rate_lower_bound=lb,
        point_estimate=float(arr.mean()),
        n=n,
        delta=float(delta),
        method=chosen,
    )


# -- Diagnostics --------------------------------------------------------


def empirical_bound_coverage(
    true_rate: float,
    n: int,
    delta: float,
    method: Literal["hoeffding", "clopper_pearson"],
    n_trials: int = 2000,
    seed: int = 0,
) -> float:
    """Fraction of trials in which the certified lower bound does not
    exceed ``true_rate``. A valid (1-δ) bound must achieve coverage
    ≥ 1-δ. This is the empirical check of the provable guarantee."""
    if not (0.0 <= true_rate <= 1.0):
        raise ValueError("true_rate must be in [0, 1]")
    rng = np.random.RandomState(seed)
    covered = 0
    for _ in range(n_trials):
        draws = (rng.uniform(size=n) < true_rate).astype(np.float64)
        if method == "clopper_pearson":
            lb = clopper_pearson_lower_bound(int(draws.sum()), n, delta)
        else:
            lb = hoeffding_lower_bound(draws, delta)
        if lb <= true_rate:
            covered += 1
    return covered / n_trials
