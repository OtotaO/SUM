"""Pure-Python, dependency-free replay of SUM's distribution-free bounds.

This module exists so that **verifying** a ``sum.meaning_risk_receipt.v1``
needs nothing heavier than the standard library. The canonical
*generation*-side kernels live in
``sum_engine_internal.research.conformal.risk_control`` and
``sum_engine_internal.research.meaning.conformal_meaning`` and depend on
numpy (and, for Clopper–Pearson, scipy). An integrator who only wants to
**check** a receipt should not have to install a 200 MB numeric stack to
do it, so the verify SDK re-derives the same three bounds here in plain
Python.

Two implementations of the same arithmetic is a divergence hazard, and
this project's discipline is to make divergence *loud*, not to trust that
it won't happen. The guard is ``Tests/test_sum_verify_sdk.py``, which
asserts these functions agree with the canonical numpy/scipy kernels to
well within the receipt's micro-unit (1e-6) wire resolution across a grid
of sample sizes, deltas, and loss patterns — and that the committed
golden receipts replay **identically** through both paths. The wire
contract (integer micro-units, exact-integer replay comparison) is the
stable interface both sides honour; this module reproduces the float math
*upstream* of that quantisation closely enough that the quantised results
are bit-identical.

What is NOT re-derived here: the RFC-8785 JCS canonicaliser and the
Ed25519/JWS envelope verifier. Those are the cryptographic trust root and
are imported from ``sum_engine_internal.infrastructure`` unchanged —
reimplementing them would be a real divergence risk with no upside.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import math
from typing import NamedTuple, Sequence

__all__ = [
    "MeaningRiskGuarantee",
    "certify_meaning_risk",
    "hoeffding_lower_bound",
    "empirical_bernstein_lower_bound",
    "clopper_pearson_lower_bound",
]


class MeaningRiskGuarantee(NamedTuple):
    """Verify-side mirror of
    ``research.meaning.conformal_meaning.MeaningRiskGuarantee`` — carries
    exactly the fields the replay step needs to compare against the
    receipt payload. A plain ``NamedTuple`` so this module stays
    dependency-free and the SDK has no shared mutable state."""

    risk_upper_bound: float
    point_estimate: float
    n: int
    delta: float
    method: str
    scorer_name: str
    scorer_version: str

    def controls(self, alpha: float) -> bool:
        """True iff the certified ceiling sits at or below ``alpha`` —
        the operational pass/fail the ``controlled`` field commits."""
        return self.risk_upper_bound <= alpha


def _validate_delta(delta: float) -> None:
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1); got {delta}")


def _validate_unit_interval(values: Sequence[float], label: str) -> list[float]:
    """Mirror the canonical kernel's hardening: reject non-finite BEFORE
    the range check (a NaN slips past every ``<`` / ``>`` comparison and
    would silently poison the mean into a maximal bound from garbage)."""
    arr = [float(x) for x in values]
    if len(arr) < 1:
        raise ValueError(f"{label} must be non-empty")
    if not all(math.isfinite(x) for x in arr):
        raise ValueError(f"{label} must all be finite (no NaN/inf)")
    if any(x < 0.0 or x > 1.0 for x in arr):
        raise ValueError(f"{label} must lie in [0, 1]")
    return arr


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _sample_variance(values: Sequence[float], mean: float) -> float:
    """Unbiased (ddof=1) sample variance, matching numpy's ``var(ddof=1)``
    to within float noise far finer than the micro wire resolution."""
    n = len(values)
    return math.fsum((x - mean) ** 2 for x in values) / (n - 1)


def hoeffding_lower_bound(values: Sequence[float], delta: float = 0.05) -> float:
    """One-sided (1-δ) lower confidence bound on the mean of [0, 1]
    observations via Hoeffding's inequality. Distribution-free,
    finite-sample. Clamped to [0, 1]. Pure-Python mirror of
    ``risk_control.hoeffding_lower_bound``."""
    _validate_delta(delta)
    arr = _validate_unit_interval(values, "values")
    n = len(arr)
    mean = _mean(arr)
    radius = math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    return max(0.0, min(1.0, mean - radius))


def empirical_bernstein_lower_bound(
    values: Sequence[float], delta: float = 0.05
) -> float:
    """One-sided (1-δ) lower bound via the empirical-Bernstein inequality
    (Maurer & Pontil, COLT 2009, Thm 11). Falls back to Hoeffding at
    ``n == 1`` (sample variance undefined), matching the canonical
    kernel. Pure-Python mirror of
    ``risk_control.empirical_bernstein_lower_bound``."""
    _validate_delta(delta)
    arr = _validate_unit_interval(values, "values")
    n = len(arr)
    if n == 1:
        return hoeffding_lower_bound(arr, delta)
    mean = _mean(arr)
    var = _sample_variance(arr, mean)
    t = math.log(2.0 / delta)
    radius = math.sqrt(2.0 * var * t / n) + 7.0 * t / (3.0 * (n - 1))
    return max(0.0, min(1.0, mean - radius))


# --- Regularised incomplete beta (pure Python) for Clopper–Pearson -----
# Replaces the canonical kernel's lazy ``scipy.stats.beta.ppf`` so the
# binary-loss receipts verify without scipy. Numerical Recipes' Lentz
# continued fraction for I_x(a, b), then bisection to invert it. The
# parity test pins this to scipy to ~1e-9, far inside the 1e-6 wire grid.


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction expansion for the incomplete beta function,
    evaluated by the modified Lentz algorithm."""
    MAXIT = 300
    EPS = 3.0e-16
    FPMIN = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b) ∈ [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _beta_ppf(p: float, a: float, b: float) -> float:
    """Inverse of I_x(a, b): the x where I_x(a, b) == p. I_x is strictly
    increasing in x, so bisect [0, 1]. 100 iterations ⇒ |error| < 2**-100,
    far inside the micro grid."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _betai(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson_lower_bound(successes: int, n: int, delta: float = 0.05) -> float:
    """Exact one-sided (1-δ) lower confidence limit for a binomial
    proportion: the δ-quantile of Beta(successes, n - successes + 1), 0
    when there are no successes. Pure-Python mirror of
    ``risk_control.clopper_pearson_lower_bound`` (scipy-free)."""
    _validate_delta(delta)
    if n < 1:
        raise ValueError("n must be >= 1")
    if not (0 <= successes <= n):
        raise ValueError(f"successes must be in [0, n]; got {successes} of {n}")
    if successes == 0:
        return 0.0
    return _beta_ppf(delta, float(successes), float(n - successes + 1))


def certify_meaning_risk(
    losses: Sequence[float],
    *,
    scorer_name: str,
    scorer_version: str,
    delta: float = 0.05,
    method: str = "hoeffding",
) -> MeaningRiskGuarantee:
    """Pure-Python mirror of
    ``research.meaning.conformal_meaning.certify_meaning_risk``.

    Certifies a distribution-free upper bound on expected meaning-loss as
    the dual of the rate kernel: ``risk_upper_bound = 1 - lb(1 - losses)``.
    Deterministic in ``losses``; on the quantised committed vector it
    reproduces the receipt's bound to the last micro-unit. Kept
    field-for-field identical to the canonical certifier so the verify
    SDK's replay compares like with like.
    """
    arr = _validate_unit_interval(losses, "losses")
    n = len(arr)

    is_binary = all(x in (0.0, 1.0) for x in arr)
    chosen = method
    if method == "auto":
        chosen = "clopper_pearson" if is_binary else "hoeffding"

    preservations = [1.0 - x for x in arr]
    if chosen == "clopper_pearson":
        if not is_binary:
            raise ValueError(
                "clopper_pearson requires binary (0/1) losses; "
                "use 'hoeffding' for fractional [0, 1] values"
            )
        successes = int(round(math.fsum(preservations)))
        preservation_lb = clopper_pearson_lower_bound(successes, n, delta)
    elif chosen == "hoeffding":
        preservation_lb = hoeffding_lower_bound(preservations, delta)
    elif chosen == "empirical_bernstein":
        preservation_lb = empirical_bernstein_lower_bound(preservations, delta)
    else:
        raise ValueError(f"unknown method {method!r}")

    risk_ub = max(0.0, min(1.0, 1.0 - preservation_lb))
    return MeaningRiskGuarantee(
        risk_upper_bound=risk_ub,
        point_estimate=_mean(arr),
        n=n,
        delta=float(delta),
        method=chosen,
        scorer_name=scorer_name,
        scorer_version=scorer_version,
    )
