"""Split conformal prediction — distribution-free, finite-sample CIs.

The minimum kernel from Vovk-Gammerman-Shafer 2005 / Angelopoulos &
Bates 2023, designed for SUM's "wrap any point predictor with a
calibrated CI" use case.

Workflow:

    cal_predictions = predictor.predict(X_cal)
    cal_targets     = Y_cal
    sc = SplitConformal(alpha=0.1)
    sc.calibrate(cal_predictions, cal_targets)
    interval = sc.predict(predictor.predict(X_test_one))
    # → ConformalInterval(point=…, lower=…, upper=…, alpha=0.1)

Coverage guarantee (provable): under exchangeability of
(X_i, Y_i) across calibration + test, P(Y_test ∈ interval) ≥ 1-α
finite-sample.

Two non-conformity scoring rules ship:
  - "absolute" (default): s_i = |Y_i - f(X_i)|. Symmetric intervals.
  - "signed":   s_i = Y_i - f(X_i). Asymmetric intervals; useful
    when the prediction error has known directional bias.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class ConformalInterval:
    """A finite-sample, distribution-free prediction interval.

    Coverage guarantee: P(true value ∈ [lower, upper]) ≥ 1 - alpha,
    under exchangeability of calibration + test data.
    """
    point: float       # f(X_test) — the underlying point prediction
    lower: float       # interval lower bound
    upper: float       # interval upper bound
    alpha: float       # miscoverage rate (e.g. 0.1 → 90% coverage)
    score_type: str    # "absolute" or "signed"

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, y: float) -> bool:
        return self.lower <= y <= self.upper


class SplitConformal:
    """Split conformal predictor.

    Constructed empty; must be ``calibrate``d on a held-out
    (predictions, targets) pair before ``predict`` is called.
    Re-calibration is supported by calling ``calibrate`` again.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        score_type: Literal["absolute", "signed"] = "absolute",
    ) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        if score_type not in ("absolute", "signed"):
            raise ValueError(
                f"score_type must be 'absolute' or 'signed'; got {score_type!r}"
            )
        self.alpha = float(alpha)
        self.score_type = score_type
        self._q_lo: float | None = None
        self._q_hi: float | None = None
        self._n_cal: int = 0

    @property
    def is_calibrated(self) -> bool:
        return self._q_hi is not None

    def calibrate(
        self,
        cal_predictions: np.ndarray,
        cal_targets: np.ndarray,
    ) -> None:
        """Fit the empirical quantile of non-conformity scores.

        The (n+1)/n correction in the quantile index is the
        finite-sample adjustment that makes the coverage guarantee
        non-asymptotic — see Angelopoulos & Bates 2023 Theorem 1.
        """
        cal_predictions = np.asarray(cal_predictions, dtype=np.float64)
        cal_targets = np.asarray(cal_targets, dtype=np.float64)
        if cal_predictions.shape != cal_targets.shape:
            raise ValueError(
                f"calibration shape mismatch: predictions {cal_predictions.shape} "
                f"!= targets {cal_targets.shape}"
            )
        if cal_predictions.ndim != 1:
            raise ValueError(
                f"calibration arrays must be 1-D; got shape {cal_predictions.shape}"
            )
        n = len(cal_predictions)
        if n < 1:
            raise ValueError("calibration set must be non-empty")

        residuals = cal_targets - cal_predictions
        self._n_cal = n

        if self.score_type == "absolute":
            scores = np.abs(residuals)
            # ⌈(n+1)(1-α)⌉/n quantile — the Angelopoulos-Bates
            # finite-sample correction
            q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
            q = float(np.quantile(scores, q_level, method="higher"))
            self._q_lo = q
            self._q_hi = q
        else:
            # signed: separate lower/upper quantiles for asymmetric intervals
            q_lo_level = max(np.floor((n + 1) * (self.alpha / 2)) / n, 0.0)
            q_hi_level = min(np.ceil((n + 1) * (1 - self.alpha / 2)) / n, 1.0)
            self._q_lo = float(np.quantile(residuals, q_lo_level, method="lower"))
            self._q_hi = float(np.quantile(residuals, q_hi_level, method="higher"))

    def predict(self, point_prediction: float) -> ConformalInterval:
        """Return a calibrated interval around a single point
        prediction. Calibrate first."""
        if not self.is_calibrated:
            raise RuntimeError(
                "SplitConformal.predict called before calibrate; call "
                "calibrate(cal_predictions, cal_targets) first."
            )
        if self.score_type == "absolute":
            assert self._q_hi is not None
            return ConformalInterval(
                point=float(point_prediction),
                lower=float(point_prediction - self._q_hi),
                upper=float(point_prediction + self._q_hi),
                alpha=self.alpha,
                score_type=self.score_type,
            )
        # signed
        assert self._q_lo is not None and self._q_hi is not None
        return ConformalInterval(
            point=float(point_prediction),
            lower=float(point_prediction + self._q_lo),
            upper=float(point_prediction + self._q_hi),
            alpha=self.alpha,
            score_type=self.score_type,
        )

    def predict_batch(self, point_predictions: np.ndarray) -> list[ConformalInterval]:
        return [self.predict(p) for p in np.asarray(point_predictions, dtype=np.float64)]


# -- Diagnostics --------------------------------------------------------


def empirical_coverage(
    intervals: list[ConformalInterval],
    targets: np.ndarray,
) -> float:
    """Fraction of intervals that contain their target. Should
    track 1 - alpha for a properly-calibrated SplitConformal under
    exchangeability."""
    targets = np.asarray(targets, dtype=np.float64)
    if len(intervals) != len(targets):
        raise ValueError(
            f"length mismatch: {len(intervals)} intervals vs {len(targets)} targets"
        )
    if not intervals:
        return 0.0
    hits = sum(1 for iv, y in zip(intervals, targets) if iv.contains(float(y)))
    return hits / len(intervals)


def average_interval_width(intervals: list[ConformalInterval]) -> float:
    """Mean interval width — the second-order quality metric
    after coverage. Smaller is sharper."""
    if not intervals:
        return 0.0
    return float(np.mean([iv.width for iv in intervals]))
