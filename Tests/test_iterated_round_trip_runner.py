"""Smoke test for the T1 iterated-round-trip runner.

Dry-run mode produces a stub receipt without LLM cost; this test
verifies the schema shape + classification logic without burning
credits. The real-mode run is the operator-side artifact that lands
the bench-hardening T1 receipt.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent


def test_dry_run_produces_valid_receipt(tmp_path):
    """End-to-end dry-run shape check. Receipt has schema, aggregate,
    per-document blocks, and a composition verdict."""
    out = tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.bench.runners.s25_iterated_round_trip",
            "--corpus", str(REPO / "scripts/bench/corpora/seed_v1.json"),
            "--k", "3",
            "--dry-run",
            "--out", str(out),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO), "PYTHONWARNINGS": "ignore", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"},
    )
    assert result.returncode == 0, f"runner failed: {result.stderr}"
    receipt = json.loads(out.read_text())
    assert receipt["schema"] == "sum.iterated_round_trip_drift.v1"
    assert receipt["corpus_id"] == "seed_v1"
    assert receipt["k_iterations"] == 3
    assert receipt["dry_run"] is True
    assert receipt["n_documents"] == 50
    assert "aggregate" in receipt
    assert "composition_verdict" in receipt
    assert receipt["composition_verdict"]["verdict"] in {
        "stable", "accumulating", "saturating", "noisy", "insufficient_data",
    }


def test_aggregate_has_all_k_steps(tmp_path):
    """For K=5, aggregate.by_k carries entries for k=1..5."""
    out = tmp_path / "receipt.json"
    subprocess.run(
        [
            sys.executable, "-m", "scripts.bench.runners.s25_iterated_round_trip",
            "--corpus", str(REPO / "scripts/bench/corpora/seed_v1.json"),
            "--k", "5",
            "--dry-run",
            "--out", str(out),
        ],
        cwd=REPO,
        env={"PYTHONPATH": str(REPO), "PYTHONWARNINGS": "ignore", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"},
        check=True,
        capture_output=True,
    )
    receipt = json.loads(out.read_text())
    by_k = receipt["aggregate"]["by_k"]
    for k in range(1, 6):
        assert str(k) in by_k, f"missing k={k} in aggregate"
        entry = by_k[str(k)]
        assert "median_drift_pct" in entry
        assert "p10_drift_pct" in entry
        assert "max_drift_pct" in entry
        assert "n_docs" in entry


def test_pinned_model_required_in_real_mode(tmp_path):
    """Without --dry-run AND without --model AND without SUM_TRANSFORM_MODEL
    env var, the runner exits 2 with a clear pointer at the cascade doc."""
    out = tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.bench.runners.s25_iterated_round_trip",
            "--corpus", str(REPO / "scripts/bench/corpora/seed_v1.json"),
            "--k", "2",
            "--out", str(out),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO), "PYTHONWARNINGS": "ignore", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"},
    )
    assert result.returncode == 2
    assert "SUM_TRANSFORM_MODEL" in result.stderr or "model" in result.stderr.lower()


def test_classifier_marks_stable_on_flat_drift():
    """Classifier: when all K medians are within ε of K=1, verdict is stable."""
    from scripts.bench.runners.s25_iterated_round_trip import classify_composition

    by_k = {
        "1": {"median_drift_pct": 2.0, "p10_drift_pct": 0.0, "max_drift_pct": 5.0, "n_docs": 50},
        "2": {"median_drift_pct": 2.3, "p10_drift_pct": 0.0, "max_drift_pct": 5.0, "n_docs": 50},
        "3": {"median_drift_pct": 2.1, "p10_drift_pct": 0.0, "max_drift_pct": 5.0, "n_docs": 50},
    }
    v = classify_composition(by_k, k=3, epsilon_pp=1.0)
    assert v["verdict"] == "stable"


def test_classifier_marks_accumulating_on_growing_drift():
    """Classifier: monotone growing drift verdicts 'accumulating'."""
    from scripts.bench.runners.s25_iterated_round_trip import classify_composition

    by_k = {
        "1": {"median_drift_pct": 5.0,  "p10_drift_pct": 0.0, "max_drift_pct": 10.0, "n_docs": 50},
        "2": {"median_drift_pct": 12.0, "p10_drift_pct": 0.0, "max_drift_pct": 20.0, "n_docs": 50},
        "3": {"median_drift_pct": 25.0, "p10_drift_pct": 0.0, "max_drift_pct": 40.0, "n_docs": 50},
        "4": {"median_drift_pct": 40.0, "p10_drift_pct": 0.0, "max_drift_pct": 60.0, "n_docs": 50},
    }
    v = classify_composition(by_k, k=4, epsilon_pp=1.0)
    assert v["verdict"] == "accumulating"
    assert "monotonically" in v["rationale"]
