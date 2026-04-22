"""Tests for sum_engine_internal.ensemble.venn_abers."""
from __future__ import annotations

import pytest

from sum_engine_internal.ensemble.venn_abers import (
    ConfidenceInterval,
    VennAbersCalibrator,
    _pava_fit,
    _pava_predict,
)


# ─── ConfidenceInterval ───────────────────────────────────────────────

def test_interval_rejects_inverted() -> None:
    with pytest.raises(ValueError):
        ConfidenceInterval(lower=0.8, upper=0.2)


def test_interval_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        ConfidenceInterval(lower=-0.1, upper=0.5)
    with pytest.raises(ValueError):
        ConfidenceInterval(lower=0.5, upper=1.5)


def test_interval_midpoint_and_width() -> None:
    i = ConfidenceInterval(lower=0.2, upper=0.6)
    assert i.midpoint == pytest.approx(0.4)
    assert i.width == pytest.approx(0.4)


# ─── PAVA ─────────────────────────────────────────────────────────────

def test_pava_empty() -> None:
    assert _pava_fit([], []) == []


def test_pava_length_mismatch() -> None:
    with pytest.raises(ValueError):
        _pava_fit([0.1, 0.2], [0])


def test_pava_rejects_invalid_label() -> None:
    with pytest.raises(ValueError):
        _pava_fit([0.1], [2])


def test_pava_already_monotone_preserved() -> None:
    blocks = _pava_fit([0.1, 0.3, 0.7, 0.9], [0, 0, 1, 1])
    assert len(blocks) == 4
    assert [b.mean for b in blocks] == [0.0, 0.0, 1.0, 1.0]


def test_pava_merges_violation() -> None:
    blocks = _pava_fit([0.1, 0.3, 0.5, 0.7], [0, 1, 0, 1])
    assert len(blocks) == 3
    means = [b.mean for b in blocks]
    assert means[0] == 0.0
    assert means[1] == pytest.approx(0.5)
    assert means[2] == 1.0


def test_pava_predict_below_range() -> None:
    blocks = _pava_fit([0.3, 0.7], [0, 1])
    assert _pava_predict(blocks, 0.1) == 0.0


def test_pava_predict_above_range() -> None:
    blocks = _pava_fit([0.3, 0.7], [0, 1])
    assert _pava_predict(blocks, 0.9) == 1.0


def test_pava_predict_empty() -> None:
    assert _pava_predict([], 0.5) == 0.5


# ─── VennAbersCalibrator ──────────────────────────────────────────────

def test_calibrator_empty_returns_non_informative() -> None:
    cal = VennAbersCalibrator([], [])
    i = cal.predict_interval(0.5)
    assert i.lower == 0.0
    assert i.upper == 1.0


def test_calibrator_length_mismatch() -> None:
    with pytest.raises(ValueError):
        VennAbersCalibrator([0.1, 0.2], [0])


def test_calibrator_invalid_label() -> None:
    with pytest.raises(ValueError):
        VennAbersCalibrator([0.1], [2])


def test_calibrator_interval_in_unit_range() -> None:
    scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    labels = [0, 0, 1, 1, 1]
    cal = VennAbersCalibrator(scores, labels)
    i = cal.predict_interval(0.6)
    assert 0.0 <= i.lower <= i.upper <= 1.0


def test_calibrator_interval_orientation() -> None:
    """p0 is always <= p1 by monotonicity of inductive Venn-Abers."""
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 1, 0, 1, 1, 1, 1]
    cal = VennAbersCalibrator(scores, labels)
    for s in [0.15, 0.45, 0.65, 0.85]:
        i = cal.predict_interval(s)
        assert i.lower <= i.upper


def test_calibrator_larger_data_tighter_interval() -> None:
    """More calibration data → narrower Venn-Abers interval at a fixed score."""
    def gen(n: int) -> tuple[list[float], list[int]]:
        scores = [i / (n - 1) for i in range(n)]
        labels = [1 if s > 0.5 else 0 for s in scores]
        return scores, labels

    small = VennAbersCalibrator(*gen(11))
    large = VennAbersCalibrator(*gen(101))
    assert large.predict_interval(0.7).width <= small.predict_interval(0.7).width


def test_calibrator_n_calibration() -> None:
    cal = VennAbersCalibrator([0.1, 0.5, 0.9], [0, 1, 1])
    assert cal.n_calibration == 3
