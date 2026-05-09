"""Sequential Probability Ratio Test (Wald 1947), Bernoulli case.

The simplest substrate-shaped implementation: each observation is a
binary (success / failure) draw of a parameter p, and we test

    H_0: p = p_0    vs    H_1: p = p_1

with operator-chosen Type-I / Type-II error rates (α, β).

The log-likelihood-ratio statistic after n observations is

    log Λ_n = Σ_i log [ p_1^{x_i} (1-p_1)^{1-x_i} / p_0^{x_i} (1-p_0)^{1-x_i} ]
            = (Σ x_i) · log(p_1/p_0) + (n - Σ x_i) · log((1-p_1)/(1-p_0))

Wald's stopping boundaries:

    log A = log( β / (1 - α) )       # accept H_0 (lower)
    log B = log( (1 - β) / α )       # reject H_0, accept H_1 (upper)

The test is provably (Wald 1947):
  - Type-I error ≤ α
  - Type-II error ≤ β
  - Expected sample size minimised at given (α, β) among tests
    with the same error rates (Wald-Wolfowitz optimality, 1948)

For SUM: each round-trip iteration emits a binary "passes ≥ τ
faithfulness threshold" signal; SPRT stops as soon as accept/reject
boundaries are crossed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

import numpy as np


class SPRTDecision(Enum):
    """Result of a single ``observe`` step."""
    CONTINUE = "continue"        # below both boundaries; keep sampling
    ACCEPT_H0 = "accept_h0"      # log Λ ≤ log A
    REJECT_H0 = "reject_h0"      # log Λ ≥ log B (= accept H_1)


@dataclass(frozen=True, slots=True)
class SPRTState:
    """Snapshot of the SPRT's running state.

    Useful for receipt logging / checkpointing — every value here
    is recoverable from the (n, n_successes) pair plus the test's
    configured (p_0, p_1, α, β)."""
    n: int                 # observations so far
    n_successes: int       # successes among them
    log_likelihood_ratio: float
    decision: SPRTDecision


class BinomialSPRT:
    """Wald 1947 SPRT for a binomial proportion.

    Args:
        p0: Null-hypothesis success probability.
        p1: Alternative-hypothesis success probability. Must
            differ from p0 (otherwise the test is degenerate).
        alpha: Operator-chosen Type-I error bound. Probability of
            rejecting H_0 when H_0 is true.
        beta:  Operator-chosen Type-II error bound. Probability of
            accepting H_0 when H_1 is true.

    The test is one-sided in the natural way: if p1 > p0, larger
    means support H_1 (and crossing log B means "reject H_0 in
    favour of H_1, i.e. accept p = p1"); if p1 < p0, smaller
    means support H_1.
    """

    def __init__(
        self,
        *,
        p0: float,
        p1: float,
        alpha: float = 0.05,
        beta: float = 0.05,
    ) -> None:
        if not (0 < p0 < 1):
            raise ValueError(f"p0 must be in (0, 1); got {p0}")
        if not (0 < p1 < 1):
            raise ValueError(f"p1 must be in (0, 1); got {p1}")
        if abs(p1 - p0) < 1e-12:
            raise ValueError(f"p0 and p1 must differ; got both ≈ {p0}")
        if not (0 < alpha < 1) or not (0 < beta < 1):
            raise ValueError(
                f"alpha and beta must be in (0, 1); got α={alpha}, β={beta}"
            )
        if alpha + beta >= 1:
            raise ValueError(
                f"alpha + beta must be < 1; got α+β = {alpha + beta}"
            )
        self.p0 = float(p0)
        self.p1 = float(p1)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._log_A = float(np.log(beta / (1 - alpha)))
        self._log_B = float(np.log((1 - beta) / alpha))
        self._log_p1_over_p0 = float(np.log(p1 / p0))
        self._log_q1_over_q0 = float(np.log((1 - p1) / (1 - p0)))
        self._n = 0
        self._n_successes = 0

    @property
    def n(self) -> int:
        return self._n

    @property
    def n_successes(self) -> int:
        return self._n_successes

    @property
    def log_likelihood_ratio(self) -> float:
        return (
            self._n_successes * self._log_p1_over_p0
            + (self._n - self._n_successes) * self._log_q1_over_q0
        )

    @property
    def log_A(self) -> float:
        """Lower stopping boundary (accept H_0 if log Λ ≤ log A)."""
        return self._log_A

    @property
    def log_B(self) -> float:
        """Upper stopping boundary (reject H_0 if log Λ ≥ log B)."""
        return self._log_B

    def observe(self, x: int) -> SPRTState:
        """Record one Bernoulli observation x ∈ {0, 1}; return the
        current state including any decision."""
        if x not in (0, 1):
            raise ValueError(f"observation must be 0 or 1; got {x!r}")
        self._n += 1
        if x == 1:
            self._n_successes += 1
        return self.state()

    def state(self) -> SPRTState:
        llr = self.log_likelihood_ratio
        if llr <= self._log_A:
            decision = SPRTDecision.ACCEPT_H0
        elif llr >= self._log_B:
            decision = SPRTDecision.REJECT_H0
        else:
            decision = SPRTDecision.CONTINUE
        return SPRTState(
            n=self._n,
            n_successes=self._n_successes,
            log_likelihood_ratio=float(llr),
            decision=decision,
        )

    def reset(self) -> None:
        """Clear all observations; restart the test."""
        self._n = 0
        self._n_successes = 0

    def run_until_decision(
        self,
        observations: np.ndarray,
        *,
        max_n: Optional[int] = None,
    ) -> SPRTState:
        """Convenience: feed a stream of pre-sampled observations
        through `observe` until the test decides or ``max_n`` /
        end-of-stream is reached. Returns the final state."""
        observations = np.asarray(observations, dtype=int)
        if max_n is None:
            max_n = len(observations)
        else:
            max_n = min(max_n, len(observations))
        for i in range(max_n):
            state = self.observe(int(observations[i]))
            if state.decision != SPRTDecision.CONTINUE:
                return state
        return self.state()
