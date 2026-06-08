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

Why this composes into something new. SUM already owns conformal rate
bounds, JCS canonicalisation, and Ed25519/JWS receipts. Compose them
over a meaning-loss proxy and you get a **signed, same-commit-replayable
certificate that bounds a named proxy for meaning-loss** — computed in
checkable text space (not from model internals), distribution-free, and
marginal. The certificate does not claim to have measured meaning — it
bounds a *named proxy*, marginally (on average over the corpus), under
exchangeability. Those three caveats are not fine print; they are the
contract, and they ride inside the receipt. (We are not aware of a prior
artifact combining a distribution-free meaning-loss-proxy bound with a
replayable signed receipt; the novelty is the composition, not a claim
to measure meaning.)

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
    empirical_bernstein_lower_bound,
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
    method: Literal["auto", "hoeffding", "clopper_pearson", "empirical_bernstein"] = "hoeffding",
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
    elif chosen == "empirical_bernstein":
        preservation_lb = empirical_bernstein_lower_bound(preservations, delta)
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
    method: Literal["hoeffding", "clopper_pearson", "empirical_bernstein"] = "hoeffding",
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
    exercises every method on a common ground truth.
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
        elif method == "empirical_bernstein":
            preservation_lb = empirical_bernstein_lower_bound(preservations, delta)
        else:
            preservation_lb = hoeffding_lower_bound(preservations, delta)
        risk_ub = 1.0 - preservation_lb
        if risk_ub >= true_loss_rate:
            covered += 1
    return covered / n_trials


# ── Group-conditional risk control (Perspective Receipts substrate) ───


@dataclass(frozen=True, slots=True)
class GroupedMeaningRisk:
    """A *group-conditional* meaning-risk certificate: the marginal bound
    PLUS a separate, valid-within-its-group bound for each declared cohort
    (e.g. per language / genre / named perspective).

    This is the substrate for **Perspective Receipts** — instead of one
    average-over-everything bound, a bound that holds *within* each
    audience/cohort you pre-declare. ``groups`` maps a cohort id to its
    own :class:`MeaningRiskGuarantee`.
    """
    marginal: MeaningRiskGuarantee
    groups: dict[str, "MeaningRiskGuarantee"]
    simultaneous: bool   # True ⇒ per-group δ Bonferroni-split so ALL hold jointly

    def controls_all(self, alpha: float) -> bool:
        """True iff EVERY group's certified ceiling meets ``alpha`` — the
        honest 'controlled for every cohort, not just on average' check."""
        return all(g.controls(alpha) for g in self.groups.values())

    def weakest_group(self) -> tuple[str, "MeaningRiskGuarantee"]:
        """The cohort with the highest certified ceiling — the one the
        marginal average hides."""
        return max(self.groups.items(), key=lambda kv: kv[1].risk_upper_bound)


def certify_meaning_risk_by_group(
    losses: Sequence[float],
    group_ids: Sequence[str],
    *,
    scorer_name: str,
    scorer_version: str,
    delta: float = 0.05,
    method: Literal["auto", "hoeffding", "clopper_pearson", "empirical_bernstein"] = "hoeffding",
    simultaneous: bool = False,
) -> GroupedMeaningRisk:
    """Certify a meaning-loss bound *per declared cohort* — group-conditional
    risk control (the discrete-covariate case of conditional conformal,
    Gibbs–Cherian–Candès 2023).

    ``losses[i]`` is the per-pair meaning-loss; ``group_ids[i]`` is the
    cohort label of pair ``i``. Returns the marginal bound plus, for each
    distinct cohort, the bound certified over *only that cohort's* losses
    via the same distribution-free kernel.

    Honesty boundary — what this is and is NOT:
      - **Is:** an exact, finite-sample, distribution-free bound *within
        each declared cohort* (each cohort is its own exchangeable
        calibration set). Strictly more informative than the marginal
        bound: it surfaces the worst cohort the average hides.
      - **Pays full cost per cohort:** each group's bound has its own
        finite-sample radius, so a small cohort gets a *wide* bound — this
        is honest, not a defect; there is no free conditional coverage.
        The Gibbs et al. method *shares strength* across cohorts via
        quantile regression to pay ~O(d/n) instead of O(1/n_group); that
        strength-sharing is a future tightening, NOT implemented here.
      - **``simultaneous``:** when True, each cohort is certified at
        ``delta / G`` (G = cohort count, Bonferroni) so ALL cohort bounds
        hold *jointly* with confidence ≥ 1−δ. When False (default), each
        cohort bound holds at 1−δ *on its own* — the right reading for
        "this cohort's bound".
    """
    if len(losses) != len(group_ids):
        raise ValueError(
            f"losses ({len(losses)}) and group_ids ({len(group_ids)}) "
            f"must be the same length"
        )
    if len(losses) < 1:
        raise ValueError("losses must be non-empty")

    distinct = sorted(set(group_ids))
    group_delta = delta / len(distinct) if simultaneous else delta

    marginal = certify_meaning_risk(
        losses, scorer_name=scorer_name, scorer_version=scorer_version,
        delta=delta, method=method,
    )
    groups: dict[str, MeaningRiskGuarantee] = {}
    for g in distinct:
        g_losses = [losses[i] for i in range(len(losses)) if group_ids[i] == g]
        groups[g] = certify_meaning_risk(
            g_losses, scorer_name=scorer_name, scorer_version=scorer_version,
            delta=group_delta, method=method,
        )
    return GroupedMeaningRisk(
        marginal=marginal, groups=groups, simultaneous=simultaneous,
    )
