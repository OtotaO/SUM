"""Contract tests for the distribution-free rate lower bound.

Layers (mirroring test_conformal.py):
  1. **Coverage guarantee** — the load-bearing finite-sample claim:
     a valid (1-δ) lower bound must not exceed the true rate more than
     a δ fraction of the time, i.e. empirical coverage ≥ 1-δ. Verified
     on synthetic Bernoulli data with a fixed seed (deterministic CI).
  2. **Tightness ordering** — Clopper–Pearson ≥ Hoeffding on binary
     data (exact beats the generic concentration bound).
  3. **Monotonicity** — more data / more successes / lower confidence
     all move the bound up toward the point estimate.
  4. **API surface** — RateGuarantee shape, auto-dispatch, edge cases,
     error paths.

Coverage is evaluated with a fixed seed, so these assertions are
deterministic; the ±0.01 slack only guards Monte-Carlo discreteness.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sum_engine_internal.research.conformal import (
    RateGuarantee,
    certify_rate,
    clopper_pearson_lower_bound,
    empirical_bound_coverage,
    hoeffding_lower_bound,
)


# -- 1. Coverage guarantee (the provable claim) ------------------------


@pytest.mark.parametrize("method", ["clopper_pearson", "hoeffding"])
@pytest.mark.parametrize(
    ("true_rate", "n", "delta"),
    [
        (0.90, 200, 0.05),
        (0.95, 200, 0.10),
        (0.80, 100, 0.20),
        (0.99, 400, 0.05),
        (0.70, 300, 0.05),
    ],
)
def test_lower_bound_covers_true_rate(method, true_rate, n, delta):
    """A valid (1-δ) lower bound undershoots the true rate with
    probability ≥ 1-δ. Both bounds are conservative, so coverage
    should comfortably clear the target."""
    coverage = empirical_bound_coverage(
        true_rate, n, delta, method, n_trials=4000, seed=7
    )
    assert coverage >= (1 - delta) - 0.01, (
        f"{method} under-covered: coverage={coverage:.4f} < target "
        f"{1 - delta:.2f} at p={true_rate}, n={n}, δ={delta}"
    )


# -- 2. Tightness ordering ---------------------------------------------


@pytest.mark.parametrize(("successes", "n"), [(198, 200), (95, 100), (270, 300)])
def test_clopper_pearson_at_least_as_tight_as_hoeffding(successes, n):
    """Exact binomial limit is never looser than the generic
    Hoeffding concentration bound on the same binary sample."""
    obs = [1.0] * successes + [0.0] * (n - successes)
    cp = clopper_pearson_lower_bound(successes, n, 0.05)
    ho = hoeffding_lower_bound(obs, 0.05)
    assert cp >= ho - 1e-12


# -- 3. Monotonicity ---------------------------------------------------


def test_more_data_tightens_bound():
    """Same observed rate, larger n → bound moves up toward it."""
    small = clopper_pearson_lower_bound(90, 100, 0.05)
    large = clopper_pearson_lower_bound(900, 1000, 0.05)
    assert large > small


def test_more_successes_raises_bound():
    assert clopper_pearson_lower_bound(195, 200, 0.05) < clopper_pearson_lower_bound(
        199, 200, 0.05
    )


def test_lower_confidence_raises_bound():
    """Larger δ (less confidence demanded) → higher (less conservative) bound."""
    strict = clopper_pearson_lower_bound(190, 200, 0.01)
    loose = clopper_pearson_lower_bound(190, 200, 0.20)
    assert loose > strict


# -- 4. Edge cases + error paths ---------------------------------------


def test_zero_successes_floor_is_zero():
    assert clopper_pearson_lower_bound(0, 50, 0.05) == 0.0


def test_all_successes_bound_in_open_interval():
    lb = clopper_pearson_lower_bound(200, 200, 0.05)
    assert 0.0 < lb < 1.0


def test_hoeffding_rejects_out_of_range():
    with pytest.raises(ValueError):
        hoeffding_lower_bound([0.5, 1.5], 0.05)


@pytest.mark.parametrize("bad_delta", [0.0, 1.0, -0.1, 2.0])
def test_delta_must_be_open_unit(bad_delta):
    with pytest.raises(ValueError):
        hoeffding_lower_bound([1.0, 0.0], bad_delta)
    with pytest.raises(ValueError):
        clopper_pearson_lower_bound(1, 2, bad_delta)


def test_empty_observations_rejected():
    with pytest.raises(ValueError):
        certify_rate([], 0.05)


def test_successes_out_of_range_rejected():
    with pytest.raises(ValueError):
        clopper_pearson_lower_bound(201, 200, 0.05)


# -- 5. certify_rate dispatch + RateGuarantee shape --------------------


def test_auto_dispatch_binary_to_clopper_pearson():
    g = certify_rate([1.0, 0.0, 1.0, 1.0] * 25, delta=0.05)
    assert g.method == "clopper_pearson"


def test_auto_dispatch_fractional_to_hoeffding():
    g = certify_rate([0.9, 0.8, 0.95, 1.0, 0.7] * 20, delta=0.05)
    assert g.method == "hoeffding"


def test_clopper_pearson_refuses_fractional():
    with pytest.raises(ValueError):
        certify_rate([0.9, 0.8, 0.95], delta=0.05, method="clopper_pearson")


def test_rate_guarantee_fields():
    g = certify_rate([1.0] * 99 + [0.0], delta=0.05)
    assert isinstance(g, RateGuarantee)
    assert g.n == 100
    assert g.confidence == pytest.approx(0.95)
    assert g.point_estimate == pytest.approx(0.99)
    assert 0.0 <= g.rate_lower_bound <= g.point_estimate
    assert g.slack == pytest.approx(g.point_estimate - g.rate_lower_bound)


def test_determinism():
    obs = [1.0] * 180 + [0.0] * 20
    a = certify_rate(obs, delta=0.05)
    b = certify_rate(obs, delta=0.05)
    assert a == b


# -- 6. Adversarial hardening ------------------------------------------
#
# A guarantee-emitting function's worst failure is a *silently invalid*
# bound. These guard the failure modes an adversarial reviewer / auditor
# would probe: poisoned input, extreme regimes, and dispatch confusion.


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_input_rejected(poison):
    """Regression: NaN evades the [0,1] range check (NaN comparisons are
    always False) and was observed to poison the mean into LCB=1.0 — a
    maximal 'guarantee' from garbage. Must reject, never compute."""
    with pytest.raises(ValueError):
        certify_rate([1.0, poison, 1.0] * 10, delta=0.05)
    with pytest.raises(ValueError):
        hoeffding_lower_bound([1.0, poison], delta=0.05)


def test_adversarial_coverage_stress():
    """Coverage must hold across extreme regimes — small n, near-boundary
    rates, aggressive δ — not just the comfortable middle. Both bounds are
    conservative, so every cell must clear 1-δ."""
    grid_p = [0.01, 0.5, 0.90, 0.99]
    grid_n = [2, 5, 10, 50]
    grid_delta = [0.05, 0.20]
    failures = []
    for method in ("clopper_pearson", "hoeffding"):
        for p in grid_p:
            for n in grid_n:
                for delta in grid_delta:
                    cov = empirical_bound_coverage(
                        p, n, delta, method, n_trials=3000, seed=23
                    )
                    if cov < (1 - delta) - 0.01:
                        failures.append((method, p, n, delta, cov))
    assert not failures, f"under-coverage in extreme regimes: {failures}"


def test_boundary_rates_do_not_overclaim():
    """At p=1.0 the bound must stay < 1 (finite-sample honesty: you can
    never certify 100% from a finite sample); at p=0 it must be 0."""
    all_ones = [1.0] * 100
    g = certify_rate(all_ones, delta=0.05)
    assert g.point_estimate == 1.0
    assert g.rate_lower_bound < 1.0, "certified 100% from a finite sample — overclaim"
    g0 = certify_rate([0.0] * 100, delta=0.05)
    assert g0.rate_lower_bound == 0.0


def test_near_binary_floats_not_misrouted_to_exact_bound():
    """0.9999999 is NOT 1.0 — must route to Hoeffding (valid for [0,1]),
    never to the exact-binary Clopper–Pearson path which assumes 0/1."""
    g = certify_rate([0.9999999, 1.0, 0.5] * 30, delta=0.05)
    assert g.method == "hoeffding"


def test_randomized_invariant_fuzz():
    """Fixed-seed fuzz: across many random valid inputs the bound must be
    finite, bracketed 0 ≤ LCB ≤ point ≤ 1, and CP ≥ Hoeffding on binary.
    Deterministic (no hypothesis dep) so it runs in the main CI job."""
    rng = np.random.RandomState(99)
    for _ in range(600):
        n = int(rng.randint(1, 300))
        delta = float(rng.uniform(0.001, 0.5))
        if rng.rand() < 0.5:
            obs = (rng.rand(n) < rng.uniform(0, 1)).astype(float)  # binary
        else:
            obs = rng.rand(n)  # fractional [0,1)
        g = certify_rate(obs, delta=delta)
        assert math.isfinite(g.rate_lower_bound)
        assert 0.0 <= g.rate_lower_bound <= g.point_estimate <= 1.0 + 1e-12
        if set(np.unique(obs)).issubset({0.0, 1.0}):
            k = int(obs.sum())
            cp = clopper_pearson_lower_bound(k, n, delta)
            ho = hoeffding_lower_bound(obs, delta)
            assert cp >= ho - 1e-9


@pytest.mark.parametrize("delta", [1e-6, 0.4999, 0.999])
def test_extreme_delta_stays_valid(delta):
    """Extreme-but-legal δ must not crash or produce out-of-range bounds."""
    g = certify_rate([1.0] * 50 + [0.0] * 10, delta=delta)
    assert 0.0 <= g.rate_lower_bound <= 1.0
    assert math.isfinite(g.rate_lower_bound)


def test_single_observation_is_valid_not_crash():
    """n=1 is a legal (if near-vacuous) sample — must return a valid
    bound, not raise or NaN."""
    assert clopper_pearson_lower_bound(1, 1, 0.05) == pytest.approx(0.05)
    assert hoeffding_lower_bound([1.0], 0.05) == 0.0  # radius > 1 → clamps to 0
