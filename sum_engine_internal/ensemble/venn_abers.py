"""
Venn-Abers Calibration — Distribution-Free Probability Intervals

Produces a provably-valid interval [p0, p1] for the probability that a
prediction is correct, given a calibration set of (score, label) pairs.

Unlike bare scalars from a softmax or sigmoid, Venn-Abers outputs carry a
distribution-free coverage guarantee: the true probability is contained in
the interval with validity under exchangeability of calibration and test
data (Vovk & Petej, 2014). This is the standards-aligned honesty upgrade
for SUM's confidence surface, derived from Polytaxis v0.1 §2 (conformal-
prediction discipline for every surfaced claim).

Algorithm: Inductive Venn-Abers (Vovk-Petej 2014).
    - Combine calibration data with (test_score, 0), fit isotonic → p0
    - Combine calibration data with (test_score, 1), fit isotonic → p1
    - Return [min(p0, p1), max(p0, p1)]

Isotonic regression is implemented via the Pool Adjacent Violators
Algorithm (PAVA), pure-Python, no scipy/sklearn dependency.

Integration status: standalone module; production wiring into
ConfidenceCalibrator is a separate iteration.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ConfidenceInterval:
    """Venn-Abers prediction: the true probability lies in [lower, upper].

    Width of the interval is the uncertainty signal; a narrow interval on a
    large calibration set is a confident calibrated prediction, while a wide
    interval flags that the calibration data does not support a sharp answer.
    """

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.lower <= 1.0):
            raise ValueError(f"lower out of [0,1]: {self.lower}")
        if not (0.0 <= self.upper <= 1.0):
            raise ValueError(f"upper out of [0,1]: {self.upper}")
        if self.lower > self.upper:
            raise ValueError(
                f"lower ({self.lower}) must be <= upper ({self.upper})"
            )

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.lower + self.upper)

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True)
class _Block:
    low: float
    high: float
    count: int
    total: float

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0


def _pava_fit(
    scores: Sequence[float], labels: Sequence[int]
) -> list[_Block]:
    """Pool Adjacent Violators isotonic regression.

    Returns a monotone non-decreasing sequence of blocks. Each block covers
    a contiguous range of the sorted score axis; its mean is the fitted
    probability on that range.
    """
    n = len(scores)
    if n == 0:
        return []
    if n != len(labels):
        raise ValueError("scores and labels must have same length")
    for lbl in labels:
        if lbl not in (0, 1):
            raise ValueError(f"labels must be 0 or 1, got {lbl}")

    order = sorted(range(n), key=lambda i: scores[i])
    sorted_scores = [scores[i] for i in order]
    sorted_labels = [labels[i] for i in order]

    blocks: list[_Block] = [
        _Block(
            low=sorted_scores[i],
            high=sorted_scores[i],
            count=1,
            total=float(sorted_labels[i]),
        )
        for i in range(n)
    ]

    i = 0
    while i + 1 < len(blocks):
        left = blocks[i]
        right = blocks[i + 1]
        if left.mean > right.mean:
            merged = _Block(
                low=left.low,
                high=right.high,
                count=left.count + right.count,
                total=left.total + right.total,
            )
            blocks[i] = merged
            del blocks[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1

    return blocks


def _pava_predict(blocks: list[_Block], score: float) -> float:
    """Evaluate the fitted step function at a new score.

    - Below the fit range: returns the first block's mean (extrapolation).
    - Above the fit range: returns the last block's mean.
    - Within a block: returns that block's mean.
    - Tied at a boundary: returns the mean of all blocks containing the score.
    """
    if not blocks:
        return 0.5
    if score < blocks[0].low:
        return blocks[0].mean
    if score > blocks[-1].high:
        return blocks[-1].mean

    matching = [b for b in blocks if b.low <= score <= b.high]
    if matching:
        return sum(b.mean for b in matching) / len(matching)

    for i in range(len(blocks) - 1):
        if blocks[i].high < score < blocks[i + 1].low:
            return 0.5 * (blocks[i].mean + blocks[i + 1].mean)
    return 0.5


class VennAbersCalibrator:
    """Inductive Venn-Abers distribution-free probability intervals.

    Construct with a fixed calibration set of (score, label) pairs where
    label ∈ {0, 1}. Call predict_interval(score) to get a ConfidenceInterval
    for a new test score. Cost per prediction: two PAVA fits on the
    calibration set of size n+1 → O(n) amortised.

    Semantics:
        - score  : any real number the upstream scorer emits (typically [0,1])
        - label  : 1 if the prediction was correct on that calibration example,
                   0 otherwise
        - predict_interval(s) : [p0, p1] — the true P(correct | s) lies in the
                   interval under exchangeability of cal/test data

    Empty calibration set returns the non-informative [0, 1] interval,
    signalling lack of evidence rather than raising.
    """

    def __init__(
        self,
        calibration_scores: Sequence[float],
        calibration_labels: Sequence[int],
    ) -> None:
        if len(calibration_scores) != len(calibration_labels):
            raise ValueError("scores and labels must have same length")
        for lbl in calibration_labels:
            if lbl not in (0, 1):
                raise ValueError(f"labels must be 0 or 1, got {lbl}")
        self._scores: list[float] = list(calibration_scores)
        self._labels: list[int] = list(calibration_labels)

    @property
    def n_calibration(self) -> int:
        return len(self._scores)

    def predict_interval(self, score: float) -> ConfidenceInterval:
        if not self._scores:
            return ConfidenceInterval(lower=0.0, upper=1.0)

        scores_0 = self._scores + [score]
        labels_0 = self._labels + [0]
        blocks_0 = _pava_fit(scores_0, labels_0)
        p_0 = _pava_predict(blocks_0, score)

        scores_1 = self._scores + [score]
        labels_1 = self._labels + [1]
        blocks_1 = _pava_fit(scores_1, labels_1)
        p_1 = _pava_predict(blocks_1, score)

        p_0_clamped = max(0.0, min(1.0, p_0))
        p_1_clamped = max(0.0, min(1.0, p_1))

        return ConfidenceInterval(
            lower=min(p_0_clamped, p_1_clamped),
            upper=max(p_0_clamped, p_1_clamped),
        )
