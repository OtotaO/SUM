"""Descriptive T4 analysis must expose paired regressions and preserve history."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.bench.runners.t4_drift_composition import (
    SCHEMA,
    _fit_additive,
    _fit_fixed_point,
    _fit_multiplicative_survival,
    _fit_saturating,
    _ssr,
    analyse_receipt,
    build_receipt,
    main,
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
    """A constant median curve is descriptive, including with small n."""
    series = [[0.125] * 10 for _ in range(16)]
    receipt = _write_synthetic_t1(tmp_path, "synthetic_flat", series, K=10)
    out = analyse_receipt(receipt)
    assert out["corpus_id"] == "synthetic_flat"
    assert out["best_law_by_ssr"] == "fixed_point"
    assert out["median_stability"]["verdict"] == "observed_medians_unchanged"
    assert out["median_stability"]["max_abs_delta_median_vs_K1"] == 0.0
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
    assert s["all_observed_medians_unchanged"] is True
    assert s["best_law_distribution"] == {"fixed_point": 2}
    assert s["max_observed_delta_vs_K1"] == 0.0


def test_stable_medians_expose_paired_regressions(tmp_path: Path) -> None:
    # Equal medians and means, but one document loses all captured facts.
    receipt = _write_synthetic_t1(tmp_path, "paired", [[0, 1], [1, 0], [0, 0]], K=2)
    out = analyse_receipt(receipt)
    assert out["median_stability"]["verdict"] == "observed_medians_unchanged"
    paired = out["paired_endpoint_changes"]
    assert paired["n_worsened"] == 1
    assert paired["n_improved"] == 1
    assert paired["n_unchanged"] == 1
    assert paired["mean_delta_drift"] == 0
    assert paired["max_delta_drift"] == 1
    assert paired["min_delta_drift"] == -1
    assert paired["inference"].startswith("descriptive_only")
    assert "composition_invariance" not in out
    assert "dkw_per_K_95" not in out


def test_missing_endpoint_is_excluded_and_disclosed(tmp_path: Path) -> None:
    receipt = _write_synthetic_t1(tmp_path, "incomplete", [[0, 0.5], [0.8]], K=2)
    out = analyse_receipt(receipt)
    paired = out["paired_endpoint_changes"]
    assert paired["n_complete_pairs"] == 1
    assert paired["n_incomplete_pairs"] == 1
    assert paired["incomplete_documents"][0]["doc_id"] == "doc_002"
    assert paired["n_worsened"] == 1
    assert paired["mean_delta_drift"] == 0.5
    assert out["descriptive_per_K"]["1"]["n_observations"] == 2
    assert out["descriptive_per_K"]["2"]["n_observations"] == 1
    assert out["hellinger_doc_frequency"]["n_complete_pairs"] == 1


def test_pairing_uses_declared_endpoint_not_row_order(tmp_path: Path) -> None:
    receipt = _write_synthetic_t1(tmp_path, "order", [[0.1, 0.3, 0.8]], K=3)
    payload = json.loads(receipt.read_text())
    payload["per_document"][0]["iterations"].reverse()
    receipt.write_text(json.dumps(payload))
    pair = analyse_receipt(receipt)["paired_endpoint_changes"]["per_document"][0]
    assert pair["drift_K1"] == 0.1
    assert pair["drift_Kmax"] == 0.8
    assert pair["delta_drift"] == pytest.approx(0.7)


def test_source_digest_binds_exact_bytes(tmp_path: Path) -> None:
    receipt = _write_synthetic_t1(tmp_path, "digest", [[0, 0]], K=2)
    first = analyse_receipt(receipt)
    assert first["t1_receipt_sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()
    receipt.write_text(receipt.read_text() + "\n")
    assert analyse_receipt(receipt)["t1_receipt_sha256"] != first["t1_receipt_sha256"]


@pytest.mark.parametrize("drift", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_drift_rejected(tmp_path: Path, drift: float) -> None:
    receipt = _write_synthetic_t1(tmp_path, "invalid", [[drift]], K=1)
    with pytest.raises(ValueError, match="drift_pct"):
        analyse_receipt(receipt)


def test_duplicate_iteration_rejected(tmp_path: Path) -> None:
    receipt = _write_synthetic_t1(tmp_path, "duplicate", [[0, 0.5]], K=2)
    payload = json.loads(receipt.read_text())
    payload["per_document"][0]["iterations"][1]["k"] = 1
    receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unique"):
        analyse_receipt(receipt)


def test_missing_whole_iteration_rejected(tmp_path: Path) -> None:
    receipt = _write_synthetic_t1(tmp_path, "missing", [[0]], K=2)
    with pytest.raises(ValueError, match="each declared iteration"):
        analyse_receipt(receipt)


def test_empty_corpus_and_input_list_rejected(tmp_path: Path) -> None:
    receipt = _write_synthetic_t1(tmp_path, "empty", [], K=2)
    with pytest.raises(ValueError, match="at least one document"):
        analyse_receipt(receipt)
    with pytest.raises(ValueError, match="at least one T1 receipt"):
        build_receipt([])


def test_real_receipts_report_worsening_documents_without_inference() -> None:
    paths = [
        Path("fixtures/bench_receipts/s25_iterated_K10_seed_v1_2026-05-21.json"),
        Path("fixtures/bench_receipts/s25_iterated_K10_seed_v2_2026-05-21.json"),
        Path("fixtures/bench_receipts/s25_iterated_K10_seed_long_paragraphs_2026-05-21.json"),
    ]
    out = build_receipt(paths)
    assert out["schema"] == "sum.drift_metric_composition.v2"
    summary = out["cross_corpus_summary"]
    assert summary["all_observed_medians_unchanged"] is True
    assert summary["best_law_distribution"] == {"fixed_point": 3}
    assert out["method"]["population_inference"] == "not_performed"
    assert out["method"]["equivalence_test"] == "not_performed"
    long = out["per_corpus"][2]["paired_endpoint_changes"]
    assert long["n_complete_pairs"] == 16
    assert long["n_worsened"] == 5
    assert long["max_drift_K1"] == pytest.approx(0.428571)
    assert long["max_drift_Kmax"] == 0.5
    seed_v2 = out["per_corpus"][1]["paired_endpoint_changes"]
    assert seed_v2["max_delta_drift"] == 1
    assert seed_v2["min_delta_drift"] == -1
    # The old artifact is retained as evidence, not silently regenerated.
    historical = out["supersedes_interpretation"]
    assert hashlib.sha256(Path(historical["artifact"]).read_bytes()).hexdigest() == historical["artifact_sha256"]
    json.dumps(out, allow_nan=False)


def test_cli_refuses_to_overwrite_historical_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _write_synthetic_t1(tmp_path, "cli", [[0]], K=1)
    historical = tmp_path / "historical.json"
    original = b'{"schema":"sum.drift_metric_composition.v1","evidence":"preserve"}'
    historical.write_bytes(original)
    monkeypatch.setattr("sys.argv", ["t4", "--receipts", str(source), "--out", str(historical)])
    assert main() == 2
    assert historical.read_bytes() == original


def test_zero_observed_count_does_not_fabricate_coefficient(tmp_path: Path) -> None:
    receipt = _write_synthetic_t1(tmp_path, "zero", [[0, 1]], K=2)
    payload = json.loads(receipt.read_text())
    payload["per_document"][0]["iterations"][1]["n_observed"] = 0
    receipt.write_text(json.dumps(payload))
    coefficient = analyse_receipt(receipt)["hellinger_doc_frequency"]
    assert coefficient["fidelity_KK"] is None
    assert coefficient["compositional_residual"] is None
