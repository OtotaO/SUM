"""SPRT contract tests.

Layers:
  1. **Wald 1947 error bounds** — empirical Type-I ≤ α and
     Type-II ≤ β across many trials (the load-bearing claim).
  2. **Boundary computation** — log A and log B match the
     closed-form Wald formulas exactly.
  3. **Sample-size savings** — SPRT terminates strictly earlier
     than fixed-N when the effect is clearly in either hypothesis.
  4. **API surface + edge cases** — observation validation, state
     transitions, reset, error paths.
"""
from __future__ import annotations

import numpy as np
import pytest

from sum_engine_internal.research.sequential import (
    BinomialSPRT,
    SPRTDecision,
)


# -- Wald error bounds ------------------------------------------------


def _trial_error_rates(true_p, n_trials, p0, p1, alpha, beta, n_per, seed=42):
    rng = np.random.default_rng(seed)
    accept = reject = no_dec = 0
    for _ in range(n_trials):
        sprt = BinomialSPRT(p0=p0, p1=p1, alpha=alpha, beta=beta)
        obs = rng.binomial(1, true_p, size=n_per)
        state = sprt.run_until_decision(obs)
        if state.decision == SPRTDecision.ACCEPT_H0: accept += 1
        elif state.decision == SPRTDecision.REJECT_H0: reject += 1
        else: no_dec += 1
    return accept, reject, no_dec


def test_type_I_error_below_alpha_under_H0():
    """Under p = p_0, P(reject H_0) ≤ α + Monte-Carlo noise."""
    accept, reject, no_dec = _trial_error_rates(
        true_p=0.5, n_trials=1000, p0=0.5, p1=0.8,
        alpha=0.05, beta=0.05, n_per=200, seed=42,
    )
    type_I = reject / 1000
    # Wald guarantees ≤ α; allow Monte-Carlo over-shoot up to ~3σ
    se = np.sqrt(0.05 * 0.95 / 1000)
    assert type_I <= 0.05 + 3 * se, (
        f"Type-I rate {type_I:.4f} > bound 0.05 + 3σ ({0.05 + 3*se:.4f})"
    )


def test_type_II_error_below_beta_under_H1():
    """Under p = p_1, P(accept H_0) ≤ β + Monte-Carlo noise."""
    accept, reject, no_dec = _trial_error_rates(
        true_p=0.8, n_trials=1000, p0=0.5, p1=0.8,
        alpha=0.05, beta=0.05, n_per=200, seed=99,
    )
    type_II = accept / 1000
    se = np.sqrt(0.05 * 0.95 / 1000)
    assert type_II <= 0.05 + 3 * se, (
        f"Type-II rate {type_II:.4f} > bound 0.05 + 3σ ({0.05 + 3*se:.4f})"
    )


def test_sprt_makes_a_decision_with_high_probability():
    """For a clear effect (p_0 = 0.5 vs p_1 = 0.8 with n_per=200),
    SPRT should almost always decide before running out of samples."""
    accept, reject, no_dec = _trial_error_rates(
        true_p=0.7, n_trials=500, p0=0.5, p1=0.8,
        alpha=0.05, beta=0.05, n_per=200, seed=1,
    )
    decided = accept + reject
    assert decided / 500 > 0.99


# -- Boundary computation ---------------------------------------------


def test_log_A_matches_wald_formula():
    sprt = BinomialSPRT(p0=0.5, p1=0.8, alpha=0.05, beta=0.10)
    expected = np.log(0.10 / (1 - 0.05))
    assert abs(sprt.log_A - expected) < 1e-12


def test_log_B_matches_wald_formula():
    sprt = BinomialSPRT(p0=0.5, p1=0.8, alpha=0.05, beta=0.10)
    expected = np.log((1 - 0.10) / 0.05)
    assert abs(sprt.log_B - expected) < 1e-12


def test_log_A_is_negative_log_B_is_positive_at_symmetric_alpha_beta():
    """At α = β, log A = -log B (boundaries symmetric around 0)."""
    sprt = BinomialSPRT(p0=0.5, p1=0.8, alpha=0.05, beta=0.05)
    assert abs(sprt.log_A + sprt.log_B) < 1e-12


# -- Sample-size savings ----------------------------------------------


def test_mean_sample_size_smaller_than_fixed_n_at_clear_effect():
    """Headline savings: SPRT uses far fewer observations when the
    truth is clearly H_0 or H_1."""
    rng = np.random.default_rng(42)
    sample_sizes = []
    n_per = 200
    for _ in range(200):
        sprt = BinomialSPRT(p0=0.5, p1=0.8, alpha=0.05, beta=0.05)
        obs = rng.binomial(1, 0.5, size=n_per)
        state = sprt.run_until_decision(obs)
        sample_sizes.append(state.n)
    mean_n = float(np.mean(sample_sizes))
    # For p_0=0.5, p_1=0.8, true p=0.5, with α=β=0.05 the expected
    # sample size is well below 50 (Wald approximation: ~14)
    assert mean_n < 50, f"mean SPRT sample size {mean_n:.1f} too large"
    assert mean_n < n_per, "SPRT should terminate before n_per"


# -- API + edge cases -------------------------------------------------


def test_observation_must_be_binary():
    sprt = BinomialSPRT(p0=0.5, p1=0.8)
    with pytest.raises(ValueError, match="0 or 1"):
        sprt.observe(2)
    with pytest.raises(ValueError, match="0 or 1"):
        sprt.observe(-1)


def test_p0_outside_unit_interval_raises():
    with pytest.raises(ValueError, match="p0"):
        BinomialSPRT(p0=0.0, p1=0.8)
    with pytest.raises(ValueError, match="p0"):
        BinomialSPRT(p0=1.0, p1=0.8)


def test_p1_outside_unit_interval_raises():
    with pytest.raises(ValueError, match="p1"):
        BinomialSPRT(p0=0.5, p1=0.0)


def test_p0_equal_p1_raises():
    """Test is degenerate when null = alternative."""
    with pytest.raises(ValueError, match="must differ"):
        BinomialSPRT(p0=0.5, p1=0.5)


def test_alpha_plus_beta_at_or_above_one_raises():
    """log A or log B becomes degenerate."""
    with pytest.raises(ValueError, match="alpha \\+ beta"):
        BinomialSPRT(p0=0.5, p1=0.8, alpha=0.5, beta=0.5)


def test_initial_state_is_continue_with_zero_observations():
    sprt = BinomialSPRT(p0=0.5, p1=0.8)
    state = sprt.state()
    assert state.n == 0
    assert state.n_successes == 0
    assert state.log_likelihood_ratio == 0.0
    assert state.decision == SPRTDecision.CONTINUE


def test_observe_increments_n_and_n_successes_correctly():
    sprt = BinomialSPRT(p0=0.5, p1=0.8)
    sprt.observe(1)
    sprt.observe(0)
    sprt.observe(1)
    state = sprt.state()
    assert state.n == 3
    assert state.n_successes == 2


def test_reset_clears_state():
    sprt = BinomialSPRT(p0=0.5, p1=0.8)
    for _ in range(10):
        sprt.observe(1)
    assert sprt.n == 10
    sprt.reset()
    assert sprt.n == 0
    assert sprt.n_successes == 0


def test_run_until_decision_stops_at_first_decision():
    """Observation stream long enough for many decisions; the
    runner should stop at the first one."""
    sprt = BinomialSPRT(p0=0.5, p1=0.8, alpha=0.05, beta=0.05)
    # Stream of 100 ones — strongly favours H_1, decision in a few obs
    obs = np.ones(100, dtype=int)
    state = sprt.run_until_decision(obs)
    assert state.decision == SPRTDecision.REJECT_H0
    assert state.n < 20  # decision well before stream end


def test_run_until_decision_returns_continue_if_stream_too_short():
    """If the stream ends before the test decides, the state's
    decision is CONTINUE."""
    sprt = BinomialSPRT(p0=0.5, p1=0.51, alpha=0.05, beta=0.05)
    # Tiny effect → very large expected sample size; 5 obs not enough
    obs = np.array([1, 0, 1, 0, 1])
    state = sprt.run_until_decision(obs)
    assert state.decision == SPRTDecision.CONTINUE


def test_run_until_decision_respects_max_n():
    sprt = BinomialSPRT(p0=0.5, p1=0.8, alpha=0.05, beta=0.05)
    obs = np.ones(100, dtype=int)
    state = sprt.run_until_decision(obs, max_n=3)
    assert state.n <= 3


# -- Detection direction ----------------------------------------------


def test_strong_one_evidence_rejects_H0():
    sprt = BinomialSPRT(p0=0.5, p1=0.8, alpha=0.05, beta=0.05)
    state = sprt.run_until_decision(np.ones(100, dtype=int))
    assert state.decision == SPRTDecision.REJECT_H0


def test_strong_zero_evidence_accepts_H0():
    sprt = BinomialSPRT(p0=0.5, p1=0.8, alpha=0.05, beta=0.05)
    state = sprt.run_until_decision(np.zeros(100, dtype=int))
    assert state.decision == SPRTDecision.ACCEPT_H0
