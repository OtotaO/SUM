"""Tests for the exchangeability advisory (`research.meaning.exchangeability`).

Uses synthetic numpy vectors so the diagnostic logic is pinned without a
judge model: two batches from the SAME distribution must be non-significant
(consistent with exchangeability); a clearly-shifted batch must be
significant (evidence against exchangeability). Determinism (fixed seed) and
the float-free report are pinned too.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from sum_engine_internal.research.meaning.exchangeability import (  # noqa: E402
    advisory_report,
    assess_exchangeability,
)


def _assess(cal, dep, **kw):
    return assess_exchangeability(
        cal, dep, calibration_corpus_id="c", judge="stub-embed",
        n_permutations=500, **kw,
    )


def test_same_distribution_is_not_distinguishable():
    rng = np.random.default_rng(1)
    cal = rng.normal(size=(40, 16))
    dep = rng.normal(size=(40, 16))  # same distribution
    a = _assess(cal, dep)
    assert a.p_value >= 0.05
    assert not a.distinguishable
    assert "no-shift-detected" in a.verdict


def test_shifted_distribution_is_distinguishable():
    rng = np.random.default_rng(2)
    cal = rng.normal(loc=0.0, size=(40, 16))
    dep = rng.normal(loc=3.0, size=(40, 16))  # clearly shifted mean
    a = _assess(cal, dep)
    assert a.p_value < 0.05
    assert a.distinguishable
    assert "shift-detected" in a.verdict


def test_advisory_is_never_gating_language():
    rng = np.random.default_rng(3)
    a = _assess(rng.normal(size=(20, 8)), rng.normal(size=(20, 8)))
    assert "ADVISORY" in a.scope and "never gating" in a.scope
    # honesty: a non-significant result must NOT claim to prove the null
    assert "not" in a.verdict.lower() or "consistent" in a.verdict.lower()


def test_deterministic_with_fixed_seed():
    rng = np.random.default_rng(4)
    cal, dep = rng.normal(size=(30, 8)), rng.normal(loc=1.0, size=(30, 8))
    a1 = _assess(cal, dep, seed=123)
    a2 = _assess(cal, dep, seed=123)
    assert a1.mmd2 == a2.mmd2 and a1.p_value == a2.p_value


def test_report_is_float_free_and_well_formed():
    rng = np.random.default_rng(5)
    a = _assess(rng.normal(size=(20, 8)), rng.normal(loc=2.0, size=(20, 8)))
    r = advisory_report(a)
    assert r["schema"] == "sum.exchangeability_advisory.v1"
    # every numeric rate is an integer micro-unit (float-free wire)
    for k in ("mmd2_micro", "p_value_micro", "alpha_micro", "n_permutations", "seed", "n_calibration", "n_deployment"):
        assert isinstance(r[k], int), k
    assert isinstance(r["distinguishable"], bool)
    assert "does NOT gate" in r["advisory"]


def test_dimension_mismatch_rejected():
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError):
        _assess(rng.normal(size=(10, 8)), rng.normal(size=(10, 16)))


def test_too_few_samples_rejected():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError):
        _assess(rng.normal(size=(1, 8)), rng.normal(size=(10, 8)))
