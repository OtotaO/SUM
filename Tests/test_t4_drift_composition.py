"""T4 — drift_pct composition-law audit.

Pins: the runner ingests T1 iterated-round-trip receipts and emits a
`sum.drift_metric_composition.v1` receipt naming the best-fitting
composition law per corpus + a DKW worst-case bound on
composition-invariance.

Source: docs/BENCH_HARDENING_FROM_QCVV.md task T4.
License: Apache License 2.0
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from scripts.bench.runners.t4_drift_composition import (
    SCHEMA,
    _dkw_epsilon,
    _fit_additive,
    _fit_fixed_point,
    _fit_multiplicative_survival,
    _fit_saturating,
    _percentile,
    _ssr,
    analyse_receipt,
    build_receipt,
)


def _write_synthetic_t1(tmp_path: Path, corpus_id: str, drift_series_per_doc: list[list[float]], K: int) -> Path:
    per_doc = []
    for i, series in enumerate(drift_series_per_doc):
        per_doc.append({
            "doc_id": f"doc_{i+1:03d}",
            "n_truth_axioms": 1,
            "iterations": [
                {"k": k + 1, "drift_pct": d * 100.0, "exact_match_recall": 1.0 - d, "n_observed": 1, "n_missing": 0, "n_extra": 0}
                for k, d in enumerate(series[:K])
            ],
        })
    payload = {
        "schema": "sum.iterated_round_trip_drift.v1",
        "corpus_id": corpus_id,
        "n_documents": len(per_doc),
        "k_iterations": K,
        "per_document": per_doc,
    }
    p = tmp_path / f"s25_iterated_K10_{corpus_id}_2026-05-22.json"
    p.write_text(json.dumps(payload))
    return p


def test_dkw_epsilon_matches_formula() -> None:
    assert _dkw_epsilon(50, 0.05) == pytest.approx(math.sqrt(math.log(2 / 0.05) / 100))
    assert _dkw_epsilon(0) == math.inf


def test_percentile_linear_interpolation() -> None:
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert _percentile([0.0, 0.0, 1.0], 50) == 0.0
    assert _percentile([], 50) != _percentile([], 50)  # NaN


def test_ssr_zero_on_perfect_fit() -> None:
    assert _ssr([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_fit_laws_form() -> None:
    # additive: K=3, drift_1=0.1 → [0.1, 0.2, 0.3]
    assert _fit_additive(0.1, 3) == pytest.approx([0.1, 0.2, 0.3])
    # multiplicative-survival: K=2, drift_1=0.1 → [0.1, 1 - 0.81 = 0.19]
    assert _fit_multiplicative_survival(0.1, 2) == pytest.approx([0.1, 0.19])
    # fixed-point: K=4, drift_1=0.125 → [0.125, 0.125, 0.125, 0.125]
    assert _fit_fixed_point(0.125, 4) == [0.125] * 4


def test_saturating_grid_recovers_flat_series() -> None:
    # A perfectly flat series should be fit by saturating with tau → 0
    # and drift_inf = the observed value: predicted ≈ observed at every K.
    flat = [0.125] * 10
    predicted, drift_inf, tau, ssr = _fit_saturating(flat)
    assert ssr < 1e-3, f"expected near-zero SSR on flat series, got {ssr}"
    assert drift_inf == pytest.approx(0.125, abs=0.01)


def test_analyse_receipt_flat_series_picks_fixed_point(tmp_path: Path) -> None:
    """All docs report drift=0.125 flat across K=1..10. The best law
    must be 'fixed_point' (tie-break preference) and the composition-
    invariance verdict must be 'composition_invariant_within_dkw_95'."""
    series = [[0.125] * 10 for _ in range(16)]
    receipt = _write_synthetic_t1(tmp_path, "synthetic_flat", series, K=10)
    out = analyse_receipt(receipt)
    assert out["corpus_id"] == "synthetic_flat"
    assert out["best_law_by_ssr"] == "fixed_point"
    assert out["composition_invariance"]["verdict"] == "composition_invariant_within_dkw_95"
    assert out["composition_invariance"]["max_abs_delta_median_vs_K1"] == 0.0
    assert out["median_drift_by_K"] == pytest.approx([0.125] * 10)


def test_analyse_receipt_growing_series_picks_growth_law(tmp_path: Path) -> None:
    """A multiplicative-survival series should not pick fixed_point."""
    # drift_K = 1 - 0.9^K → noticeably growing
    series = [[1.0 - 0.9 ** k for k in range(1, 11)] for _ in range(20)]
    receipt = _write_synthetic_t1(tmp_path, "synthetic_growing", series, K=10)
    out = analyse_receipt(receipt)
    assert out["best_law_by_ssr"] != "fixed_point", (
        f"growing series should not be best-fit by fixed_point; got "
        f"{out['best_law_by_ssr']} with median {out['median_drift_by_K']}"
    )
    # multiplicative_survival should fit a 1 - 0.9^k series with zero SSR.
    laws = out["laws_fitted"]
    assert laws["multiplicative_survival"]["sum_squared_residuals"] < 1e-9


def test_corpus_id_falls_back_to_path_stem(tmp_path: Path) -> None:
    """When the upstream T1 receipt has corpus_id=None, the runner
    must derive the corpus_id from the receipt filename."""
    p = tmp_path / "s25_iterated_K10_seed_long_paragraphs_2026-05-21.json"
    p.write_text(json.dumps({
        "schema": "sum.iterated_round_trip_drift.v1",
        "corpus_id": None,
        "k_iterations": 1,
        "per_document": [{
            "doc_id": "d1", "n_truth_axioms": 1,
            "iterations": [{"k": 1, "drift_pct": 0.0, "exact_match_recall": 1.0, "n_observed": 1, "n_missing": 0, "n_extra": 0}],
        }],
    }))
    out = analyse_receipt(p)
    assert out["corpus_id"] == "seed_long_paragraphs"


def test_build_receipt_cross_corpus_summary(tmp_path: Path) -> None:
    p1 = _write_synthetic_t1(tmp_path, "c1", [[0.0] * 10] * 30, K=10)
    p2 = _write_synthetic_t1(tmp_path, "c2", [[0.125] * 10] * 16, K=10)
    out = build_receipt([p1, p2])
    assert out["schema"] == SCHEMA
    s = out["cross_corpus_summary"]
    assert s["n_corpora"] == 2
    assert s["all_composition_invariant_dkw_95"] is True
    assert s["best_law_distribution"] == {"fixed_point": 2}
    assert s["max_observed_delta_vs_K1"] == 0.0


def test_real_receipts_yield_composition_invariant_verdict() -> None:
    """Smoke test against the actual T1 receipts on main. If any of
    the three corpora regresses from composition-invariant, this test
    fires — that is a §2.5 load-bearing-claim regression."""
    paths = [
        Path("fixtures/bench_receipts/s25_iterated_K10_seed_v1_2026-05-21.json"),
        Path("fixtures/bench_receipts/s25_iterated_K10_seed_v2_2026-05-21.json"),
        Path("fixtures/bench_receipts/s25_iterated_K10_seed_long_paragraphs_2026-05-21.json"),
    ]
    for p in paths:
        if not p.exists():
            pytest.skip(f"T1 receipt not present: {p}")
    out = build_receipt(paths)
    assert out["cross_corpus_summary"]["all_composition_invariant_dkw_95"] is True
    assert out["cross_corpus_summary"]["best_law_distribution"] == {"fixed_point": 3}
