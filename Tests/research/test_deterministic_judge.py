"""Deterministic judge mode — same-machine bit-stability + probe harness.

Two tiers:
- Pure logic tests (compare/digest/fixture-shape): no torch, run everywhere.
- Model tests: skip unless the [judge] extra is installed AND the pinned
  NLI model is cached (same convention as test_local_judge.py — CI never
  downloads a model in the per-PR suite; the monthly judge-smoke canary is
  where the cross-architecture measurement actually runs).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sum_engine_internal.research.meaning.deterministic_judge import (
    EXPECTED_PATH,
    MARGIN_DECISIVE_MICRO,
    PROBE_SET_PATH,
    compare_probe_results,
    decisions_digest,
)

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- pure ----

def test_compare_gates_only_decisive_pairs():
    expected = [
        {"decision": True, "margin_micro": 300_000},   # decisive
        {"decision": False, "margin_micro": -20_000},  # near threshold
    ]
    # Near-threshold flip: reported, NOT gated.
    actual = [
        {"decision": True, "margin_micro": 290_000},
        {"decision": True, "margin_micro": 5_000},
    ]
    report = compare_probe_results(expected, actual)
    assert report["decisive_agreement"] is True
    assert report["n_near_threshold"] == 1
    assert report["near_threshold"][0]["index"] == 1
    # Decisive flip: gated.
    actual_bad = [
        {"decision": False, "margin_micro": -100_000},
        {"decision": False, "margin_micro": -20_000},
    ]
    report_bad = compare_probe_results(expected, actual_bad)
    assert report_bad["decisive_agreement"] is False
    assert report_bad["disagreements"][0]["index"] == 0


def test_compare_reports_margin_drift_without_gating():
    expected = [{"decision": True, "margin_micro": 200_000}]
    actual = [{"decision": True, "margin_micro": 199_990}]
    report = compare_probe_results(expected, actual)
    assert report["decisive_agreement"] is True
    assert report["max_abs_margin_drift_micro"] == 10


def test_compare_rejects_length_mismatch():
    with pytest.raises(ValueError):
        compare_probe_results([], [{"decision": True, "margin_micro": 0}])


def test_digest_is_over_decisions_only():
    a = [{"decision": True, "margin_micro": 1}]
    b = [{"decision": True, "margin_micro": 999_999}]
    assert decisions_digest(a) == decisions_digest(b)
    assert decisions_digest(a) != decisions_digest(
        [{"decision": False, "margin_micro": 1}]
    )


def test_committed_fixtures_are_consistent():
    probe = json.loads((REPO / PROBE_SET_PATH).read_text())
    expected = json.loads((REPO / EXPECTED_PATH).read_text())
    assert expected["threshold"] == probe["threshold"] == 0.5
    modes = expected["modes"]
    assert "float32-det" in modes, "the shipped mode must be committed"
    assert "int8-det" not in modes, (
        "int8-det is an experiment flag (measured decision-collapse, "
        "2026-07-10) and must NOT be committed/gated until a quantization "
        "path passes the probe — see deterministic_judge.py docstring"
    )
    for mode, mv in modes.items():
        results = mv["results"]
        assert len(results) == len(probe["pairs"]), mode
        assert mv["decisions_digest"] == decisions_digest(results), mode
        assert mv["environment"]["num_threads"] == 1, mode
        for r in results:
            assert type(r["decision"]) is bool
            assert type(r["margin_micro"]) is int  # float-free, like the wire


# --------------------------------------------------------------- model ----

def _cached_judge():
    pytest.importorskip("torch", reason="[judge] extra not installed")
    pytest.importorskip("transformers", reason="[judge] extra not installed")
    import os

    from sum_engine_internal.research.meaning.deterministic_judge import (
        DeterministicNLIJudge,
    )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    judge = DeterministicNLIJudge()
    try:
        judge.probability("warm up.", "warm up.")
    except Exception as e:  # noqa: BLE001 - model not cached / offline
        pytest.skip(f"pinned NLI model not cached offline: {e}")
    return judge


def test_same_process_repeatability_is_bit_exact():
    judge = _cached_judge()
    pair = (
        "The committee approved the budget on Tuesday.",
        "The budget was approved.",
    )
    first = judge.probability(*pair)
    for _ in range(3):
        assert judge.probability(*pair) == first  # bit-exact, not approx


def test_probe_replays_committed_expectations_on_this_machine():
    """On the reference machine this is exact; on any other it is the
    same decisive-agreement gate CI runs."""
    from sum_engine_internal.research.meaning.deterministic_judge import (
        run_probe,
    )

    judge = _cached_judge()
    probe = json.loads((REPO / PROBE_SET_PATH).read_text())
    expected = json.loads((REPO / EXPECTED_PATH).read_text())
    actual = run_probe(judge, probe["pairs"])
    report = compare_probe_results(
        expected["modes"]["float32-det"]["results"],
        actual,
        decisive_margin_micro=MARGIN_DECISIVE_MICRO,
    )
    assert report["decisive_agreement"], report["disagreements"]
