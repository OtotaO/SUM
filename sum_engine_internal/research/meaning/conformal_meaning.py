"""Distribution-free upper bounds on *expected meaning-loss*.

The slider contract certifies a *lower* bound on a preservation rate
(``risk_control.certify_rate``: "≥ X % of facts survive"). Meaning-loss
asks the dual, one-sided question:

    "With confidence ≥ 1 - δ, what is the *smallest* α such that the
     expected meaning-loss (under a named proxy) ≤ α?"

That is an *upper* bound on the mean of bounded [0, 1] observations —
and it is the exact dual of the rate kernel, because for losses
``L_i ∈ [0, 1]`` the preservations ``1 - L_i`` are also in [0, 1], so

    upper-bound on E[L]  =  1 - (lower-bound on E[1 - L]).

So this module *reuses* the adversarially-hardened
``hoeffding_lower_bound`` (which already rejects NaN/inf and clamps to
[0, 1]) rather than re-deriving a concentration inequality. The result
is a Risk-Controlling Prediction (Bates et al., *JACM* 2021,
"Distribution-Free, Risk-Controlling Prediction Sets") expressed over a
meaning-space loss instead of a prediction-set size.

Why this is the historic piece. SUM already owns conformal rate bounds,
JCS canonicalisation, and Ed25519/JWS receipts. Compose them over a
meaning-loss proxy and you get an artifact that does not exist anywhere
in the literature: a **signed, same-commit-replayable certificate over a
meaning-space (not token-space) loss**. The certificate does not claim
to have measured meaning — it bounds a *named proxy* for meaning-loss,
marginally (on average over the corpus), under exchangeability. Those
three caveats are not fine print; they are the contract, and they ride
inside the receipt.

Honest boundary (identical to the rate kernel's):
  - **proxy, not meaning** — the bound is conditional on the scorer.
  - **marginal, not conditional** — it bounds the *average* loss over
    the corpus, not any single document's loss. Per-document control is
    provably not free (conditional conformal).
  - **exchangeability** — validity rests on the calibration corpus being
    exchangeable with deployment. State the corpus envelope alongside
    the bound; the number is meaningless without it.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from sum_engine_internal.research.conformal.risk_control import (
    clopper_pearson_lower_bound,
    hoeffding_lower_bound,
)


@dataclass(frozen=True, slots=True)
class MeaningRiskGuarantee:
    """A finite-sample, distribution-free *upper* bound on expected
    meaning-loss under a named proxy.

    Reads as: "with confidence ≥ ``confidence``, the expected
    meaning-loss (under ``scorer_name`` v``scorer_version``) over data
    exchangeable with the calibration corpus is ≤ ``risk_upper_bound``."
    """
    risk_upper_bound: float    # the certified ceiling on E[loss]
    point_estimate: float      # observed mean loss on the calibration sample
    n: int                     # sample size
    delta: float               # miscoverage allowance (confidence = 1 - delta)
    method: str                # "hoeffding" | "clopper_pearson"
    scorer_name: str           # which proxy was certified
    scorer_version: str        # its pinned version

    @property
    def confidence(self) -> float:
        return 1.0 - self.delta

    @property
    def slack(self) -> float:
        """Gap between the certified ceiling and the observed mean — the
        price of finite-sample, distribution-free rigour."""
        return self.risk_upper_bound - self.point_estimate

    def controls(self, alpha: float) -> bool:
        """Does this certificate control expected meaning-loss at level
        ``alpha``? True iff the certified ceiling sits at or below the
        target risk."""
        return self.risk_upper_bound <= alpha


def certify_meaning_risk(
    losses: Sequence[float],
    *,
    scorer_name: str,
    scorer_version: str,
    delta: float = 0.05,
    method: Literal["auto", "hoeffding", "clopper_pearson"] = "hoeffding",
) -> MeaningRiskGuarantee:
    """Certify a distribution-free upper bound on expected meaning-loss.

    ``losses`` are per-pair meaning-loss values in [0, 1] (typically
    ``meaning_loss.score_pairs(...)``). The bound is the dual of the rate
    kernel: ``risk_upper_bound = 1 - lower_bound(1 - losses)``.

    ``method``:
      - ``"hoeffding"`` (default) — any losses in [0, 1]; always valid,
        conservative.
      - ``"clopper_pearson"`` — only when every loss is exactly 0 or 1
        (a binary preserved/lost view); exact and tighter.
      - ``"auto"`` — Clopper–Pearson when the sample is binary, else
        Hoeffding.

    The returned guarantee carries the scorer identity so it can be
    written straight into a ``sum.meaning_risk_receipt.v1`` payload, and
    the function is deterministic in ``losses`` — re-running it on the
    same committed losses reproduces the same bound byte-for-byte (the
    receipt's replay property).
    """
    arr = np.asarray(losses, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"losses must be 1-D; got shape {arr.shape}")
    n = arr.size
    if n < 1:
        raise ValueError("losses must be non-empty")
    # Mirror the rate kernel's hardening: reject non-finite BEFORE the
    # range check (NaN evades < / > comparisons and would silently
    # poison the bound — the LCB=1.0-from-garbage failure mode).
    if not np.all(np.isfinite(arr)):
        raise ValueError("losses must all be finite (no NaN/inf)")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("losses must lie in [0, 1]")

    is_binary = bool(np.all(np.isin(arr, (0.0, 1.0))))
    chosen = method
    if method == "auto":
        chosen = "clopper_pearson" if is_binary else "hoeffding"

    preservations = 1.0 - arr
    if chosen == "clopper_pearson":
        if not is_binary:
            raise ValueError(
                "clopper_pearson requires binary (0/1) losses; "
                "use 'hoeffding' for fractional [0, 1] values"
            )
        # successes = preserved = count of zero-loss pairs.
        successes = int(round(float(preservations.sum())))
        preservation_lb = clopper_pearson_lower_bound(successes, n, delta)
    elif chosen == "hoeffding":
        preservation_lb = hoeffding_lower_bound(preservations, delta)
    else:
        raise ValueError(f"unknown method {method!r}")

    risk_ub = max(0.0, min(1.0, 1.0 - preservation_lb))
    return MeaningRiskGuarantee(
        risk_upper_bound=risk_ub,
        point_estimate=float(arr.mean()),
        n=n,
        delta=float(delta),
        method=chosen,
        scorer_name=scorer_name,
        scorer_version=scorer_version,
    )


def empirical_risk_coverage(
    true_loss_rate: float,
    n: int,
    delta: float,
    method: Literal["hoeffding", "clopper_pearson"] = "hoeffding",
    n_trials: int = 2000,
    seed: int = 0,
) -> float:
    """Fraction of trials in which the certified upper bound does NOT
    fall below ``true_loss_rate``. A valid (1-δ) upper bound must achieve
    coverage ≥ 1-δ. This is the empirical check of the provable dual
    guarantee — the meaning-loss analogue of
    ``risk_control.empirical_bound_coverage``.

    The data-generating model draws each pair's loss as Bernoulli at
    ``true_loss_rate`` (the binary preserved/lost view), so this
    exercises both methods on a common ground truth.
    """
    if not (0.0 <= true_loss_rate <= 1.0):
        raise ValueError("true_loss_rate must be in [0, 1]")
    rng = np.random.RandomState(seed)
    covered = 0
    for _ in range(n_trials):
        losses = (rng.uniform(size=n) < true_loss_rate).astype(np.float64)
        preservations = 1.0 - losses
        if method == "clopper_pearson":
            preservation_lb = clopper_pearson_lower_bound(
                int(round(float(preservations.sum()))), n, delta
            )
        else:
            preservation_lb = hoeffding_lower_bound(preservations, delta)
        risk_ub = 1.0 - preservation_lb
        if risk_ub >= true_loss_rate:
            covered += 1
    return covered / n_trials
