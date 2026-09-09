"""T4: descriptive composition audit over existing T1 measurements.

Fit candidate curves to the observed median drift at each iteration and
report paired per-document changes from K=1 to the declared K_max. These
are descriptions of the supplied receipts, not population inference,
equivalence tests, or evidence that source meaning survives composition.
The initial extracted axiom set, not independently annotated source facts,
is the reference. No model calls or new human assessments are performed.

New outputs use sum.drift_metric_composition.v2. Historical v1 artifacts
remain unchanged; their DKW composition-invariance interpretation is
superseded by docs/DRIFT_METRIC_COMPOSITION.md.

License: Apache License 2.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "sum.drift_metric_composition.v2"

DEFAULT_RECEIPTS: tuple[Path, ...] = (
    Path("fixtures/bench_receipts/s25_iterated_K10_seed_v1_2026-05-21.json"),
    Path("fixtures/bench_receipts/s25_iterated_K10_seed_v2_2026-05-21.json"),
    Path("fixtures/bench_receipts/s25_iterated_K10_seed_long_paragraphs_2026-05-21.json"),
)


def _ssr(observed: list[float], predicted: list[float]) -> float:
    """Sum of squared residuals."""
    return sum((o - p) ** 2 for o, p in zip(observed, predicted))


def _fit_additive(drift_1: float, K: int) -> list[float]:
    return [k * drift_1 for k in range(1, K + 1)]


def _fit_multiplicative_survival(drift_1: float, K: int) -> list[float]:
    survival = 1.0 - drift_1
    return [1.0 - (survival ** k) for k in range(1, K + 1)]


def _fit_saturating(observed: list[float]) -> tuple[list[float], float, float, float]:
    """Brute-force grid fit drift_K = drift_inf * (1 - exp(-K/tau)).

    Returns (predicted, drift_inf, tau, ssr). Grid is coarse on purpose:
    we are reporting goodness-of-fit, not optimising a model in
    production. If the saturating law is the right one, the residual
    will be small even at this grid resolution.
    """
    K = len(observed)
    best = (float("inf"), 0.0, 1e-6)  # ssr, drift_inf, tau
    # drift_inf grid: 0 to 1.5 in 0.005 steps (5% past the observed max)
    upper = min(1.5, max(observed) * 1.5 + 0.01)
    inf_grid = [round(0.001 * i, 4) for i in range(0, int(upper * 1000) + 1, 5)]
    # tau grid: 1e-6 (~ instant saturation) up to 10*K
    tau_grid = [1e-6, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0]
    for drift_inf in inf_grid:
        for tau in tau_grid:
            predicted = [drift_inf * (1.0 - math.exp(-k / tau)) for k in range(1, K + 1)]
            s = _ssr(observed, predicted)
            if s < best[0]:
                best = (s, drift_inf, tau)
    _, drift_inf, tau = best
    predicted = [drift_inf * (1.0 - math.exp(-k / tau)) for k in range(1, K + 1)]
    return predicted, drift_inf, tau, best[0]


def _fit_fixed_point(drift_1: float, K: int) -> list[float]:
    return [drift_1] * K


def _aggregate_per_K(per_document: list[dict[str, Any]], K: int) -> dict[int, list[float]]:
    """Pivot: {K: [drift_pct values across documents]}. Returns
    fractions in [0, 1] (divided by 100 from the receipt's percent
    representation)."""
    out: dict[int, list[float]] = {k: [] for k in range(1, K + 1)}
    for doc in per_document:
        for it in doc["iterations"]:
            k = int(it["k"])
            if 1 <= k <= K:
                out[k].append(float(it["drift_pct"]) / 100.0)
    return out


def _paired_endpoint_changes(per_document: list[dict[str, Any]], K: int) -> dict[str, Any]:
    """Pair the same document at K=1 and declared K_max; never impute missing rows."""
    rows = []
    incomplete = []
    for index, doc in enumerate(per_document):
        iterations = {int(it["k"]): it for it in doc["iterations"]}
        doc_id = doc.get("doc_id", f"row_{index + 1}")
        if 1 not in iterations or K not in iterations:
            incomplete.append({"document_index": index, "doc_id": doc_id})
            continue
        first, last = iterations[1], iterations[K]
        start, end = float(first["drift_pct"]) / 100, float(last["drift_pct"]) / 100
        rows.append({
            "document_index": index,
            "doc_id": doc_id,
            "drift_K1": start,
            "drift_Kmax": end,
            "delta_drift": end - start,
            "n_extra_K1": first.get("n_extra"),
            "n_extra_Kmax": last.get("n_extra"),
        })
    deltas = [r["delta_drift"] for r in rows]
    return {
        "K_start": 1,
        "K_end": K,
        "units": "drift fraction in [0, 1]; delta in [-1, 1]",
        "n_complete_pairs": len(rows),
        "n_incomplete_pairs": len(incomplete),
        "incomplete_documents": incomplete,
        "n_worsened": sum(d > 0 for d in deltas),
        "n_improved": sum(d < 0 for d in deltas),
        "n_unchanged": sum(d == 0 for d in deltas),
        "mean_delta_drift": statistics.fmean(deltas) if deltas else None,
        "median_delta_drift": statistics.median(deltas) if deltas else None,
        "min_delta_drift": min(deltas) if deltas else None,
        "max_delta_drift": max(deltas) if deltas else None,
        "max_drift_K1": max(r["drift_K1"] for r in rows) if rows else None,
        "max_drift_Kmax": max(r["drift_Kmax"] for r in rows) if rows else None,
        "per_document": rows,
        "inference": "descriptive_only; no population inference or equivalence test",
    }


def _hellinger_axiom_distribution(per_document: list[dict[str, Any]], K: int) -> dict[str, Any]:
    """Squared Bhattacharyya coefficient over document count shares.

    This is not a distribution over axiom identities. Iterated transitions
    do not have the tensor-product structure needed for an F1**K law.
    That expression is retained only as a descriptive candidate curve.
    """
    complete = [
        d for d in per_document
        if {1, K}.issubset({int(it["k"]) for it in d["iterations"]})
    ]
    truth_total = sum(d.get("n_truth_axioms", 0) for d in complete)
    result = {
        "K_max": K,
        "n_complete_pairs": len(complete),
        "fidelity_K1": None,
        "fidelity_KK": None,
        "compositional_predicted_F1_pow_K": None,
        "compositional_residual": None,
        "notes": (
            "Document-frequency approximation over complete endpoint pairs; axiom identities "
            "were not retained. F1**K is a candidate curve, not a derived composition law "
            "for these transitions. No hypothesis test or independent evidence is supplied."
        ),
    }
    if truth_total == 0:
        return result
    p = [d.get("n_truth_axioms", 0) / truth_total for d in complete]

    def coefficient_at(k: int) -> float | None:
        observed = [
            next(it for it in d["iterations"] if int(it["k"]) == k)["n_observed"]
            for d in complete
        ]
        total = sum(observed)
        if total == 0:
            return None
        return sum(math.sqrt(pi * count / total) for pi, count in zip(p, observed)) ** 2

    first, last = coefficient_at(1), coefficient_at(K)
    result["fidelity_K1"] = round(first, 6) if first is not None else None
    result["fidelity_KK"] = round(last, 6) if last is not None else None
    if first is not None and last is not None:
        result["compositional_predicted_F1_pow_K"] = round(first ** K, 6)
        result["compositional_residual"] = round(abs(last - first ** K), 6)
    return result


_DATE_SUFFIX = __import__("re").compile(r"_\d{4}-\d{2}-\d{2}$")


def _infer_corpus_id(receipt_path: Path, payload: dict[str, Any]) -> str:
    """Some T1 receipts carry corpus_id=None (an upstream bug at the
    time of writing); fall back to the path stem with the
    s25_iterated_K10_ / _YYYY-MM-DD noise stripped."""
    cid = payload.get("corpus_id")
    if cid:
        return cid
    stem = receipt_path.stem
    if stem.startswith("s25_iterated_K10_"):
        stem = stem[len("s25_iterated_K10_"):]
    return _DATE_SUFFIX.sub("", stem)


def analyse_receipt(receipt_path: Path) -> dict[str, Any]:
    source_bytes = receipt_path.read_bytes()
    payload = json.loads(source_bytes)
    corpus_id = _infer_corpus_id(receipt_path, payload)
    per_doc = payload["per_document"]
    if not per_doc:
        raise ValueError("T1 receipt must contain at least one document")
    K = int(payload.get("k_iterations") or max(
        (int(it["k"]) for d in per_doc for it in d["iterations"]), default=0
    ))
    if K < 1:
        raise ValueError("k_iterations must be positive")
    for doc in per_doc:
        seen = set()
        for it in doc["iterations"]:
            k = int(it["k"])
            drift = float(it["drift_pct"])
            if k in seen or k < 1 or k > K:
                raise ValueError("iteration indices must be unique and within 1..k_iterations")
            if not math.isfinite(drift) or not 0 <= drift <= 100:
                raise ValueError("drift_pct must be finite and within [0, 100]")
            seen.add(k)
    n_docs = len(per_doc)

    drift_by_k = _aggregate_per_K(per_doc, K)
    if any(not values for values in drift_by_k.values()):
        raise ValueError("each declared iteration must contain at least one observation")
    median_by_K = [statistics.median(drift_by_k[k]) for k in range(1, K + 1)]
    mean_by_K = [statistics.fmean(drift_by_k[k]) if drift_by_k[k] else float("nan") for k in range(1, K + 1)]

    drift_1 = median_by_K[0]
    observed = median_by_K

    # Fits — all three candidate laws + the fixed-point characterisation.
    pred_add = _fit_additive(drift_1, K)
    pred_mult = _fit_multiplicative_survival(drift_1, K)
    pred_sat, drift_inf, tau, _ = _fit_saturating(observed)
    pred_fp = _fit_fixed_point(drift_1, K)

    ssr_add = _ssr(observed, pred_add)
    ssr_mult = _ssr(observed, pred_mult)
    ssr_sat = _ssr(observed, pred_sat)
    ssr_fp = _ssr(observed, pred_fp)

    laws = {
        "additive": {
            "form": "drift_K = K * drift_1",
            "free_parameters": 0,
            "drift_1": round(drift_1, 6),
            "predicted_by_K": [round(p, 6) for p in pred_add],
            "sum_squared_residuals": round(ssr_add, 9),
        },
        "multiplicative_survival": {
            "form": "drift_K = 1 - (1 - drift_1)^K",
            "free_parameters": 0,
            "drift_1": round(drift_1, 6),
            "predicted_by_K": [round(p, 6) for p in pred_mult],
            "sum_squared_residuals": round(ssr_mult, 9),
        },
        "saturating": {
            "form": "drift_K = drift_inf * (1 - exp(-K/tau))",
            "free_parameters": 2,
            "drift_inf_fit": round(drift_inf, 6),
            "tau_fit": tau,
            "predicted_by_K": [round(p, 6) for p in pred_sat],
            "sum_squared_residuals": round(ssr_sat, 9),
        },
        "fixed_point": {
            "form": "median_drift_K = median_drift_1 (descriptive constant curve)",
            "free_parameters": 0,
            "drift_1": round(drift_1, 6),
            "predicted_by_K": [round(p, 6) for p in pred_fp],
            "sum_squared_residuals": round(ssr_fp, 9),
        },
    }

    # Deterministic tie-break for descriptive median-curve fits only.
    # A constant median does not imply a fixed point for individual documents.
    preference = {"fixed_point": 0, "saturating": 1, "multiplicative_survival": 2, "additive": 3}
    best_law = min(
        laws.items(),
        key=lambda kv: (round(kv[1]["sum_squared_residuals"], 12), preference[kv[0]]),
    )[0]

    max_delta = max(abs(median - drift_1) for median in median_by_K)
    median_stability = {
        "max_abs_delta_median_vs_K1": max_delta,
        "verdict": "observed_medians_unchanged" if max_delta == 0 else "observed_medians_changed",
        "inference": "descriptive_only; stable medians can hide worsening individual documents",
    }
    per_k = {
        str(k): {
            "n_observations": len(drift_by_k[k]),
            "empirical_median_drift": median_by_K[k - 1],
            "empirical_mean_drift": mean_by_K[k - 1],
            "empirical_max_drift": max(drift_by_k[k]),
        }
        for k in range(1, K + 1)
    }
    paired = _paired_endpoint_changes(per_doc, K)
    hellinger = _hellinger_axiom_distribution(per_doc, K)

    return {
        "corpus_id": corpus_id,
        "n_documents": n_docs,
        "K": K,
        "median_drift_by_K": [round(v, 6) for v in median_by_K],
        "mean_drift_by_K": [round(v, 6) for v in mean_by_K],
        "laws_fitted": laws,
        "best_law_by_ssr": best_law,
        "descriptive_per_K": per_k,
        "median_stability": median_stability,
        "paired_endpoint_changes": paired,
        "hellinger_doc_frequency": hellinger,
        "t1_receipt_path": str(receipt_path),
        "t1_receipt_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


def build_receipt(receipt_paths: list[Path]) -> dict[str, Any]:
    if not receipt_paths:
        raise ValueError("at least one T1 receipt is required")
    per_corpus = [analyse_receipt(p) for p in receipt_paths]

    cross_corpus_summary = {
        "n_corpora": len(per_corpus),
        "all_observed_medians_unchanged": all(
            c["median_stability"]["verdict"] == "observed_medians_unchanged"
            for c in per_corpus
        ),
        "n_documents_worsened_at_endpoint": sum(
            c["paired_endpoint_changes"]["n_worsened"] for c in per_corpus
        ),
        "inference": "descriptive_only; no population inference or equivalence test",
        "best_law_distribution": dict(Counter(c["best_law_by_ssr"] for c in per_corpus)),
        "max_observed_delta_vs_K1": max(
            c["median_stability"]["max_abs_delta_median_vs_K1"] for c in per_corpus
        ),
    }

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "supersedes_interpretation": {
            "schema": "sum.drift_metric_composition.v1",
            "artifact": "fixtures/bench_receipts/drift_composition_2026-05-22.json",
            "artifact_sha256": "cfc53e67fa5aa19f2068a7ccdb1a70fb00789f4d9502a6d9a072b542fa9acad1",
            "status": "historical_bytes_preserved; composition_invariance_inference_retired",
            "erratum": "docs/DRIFT_METRIC_COMPOSITION.md#historical-erratum-2026-09-09",
        },
        "per_corpus": per_corpus,
        "cross_corpus_summary": cross_corpus_summary,
        "definition": {
            "drift_pct": "T1 records 100 * (1 - exact_match_recall); T4 divides by 100 to report fractions in [0, 1].",
            "reference_scope": "Initial extracted axiom set, not independent source annotations; upstream omissions and full meaning are not measured.",
            "source": "scripts/bench/runners/s25_iterated_round_trip.py line ~258.",
            "K_iterations": "extract -> generate -> re-extract repeated K times per document; receipt records per-K drift.",
        },
        "method": {
            "fitted_laws": ["additive", "multiplicative_survival", "saturating", "fixed_point"],
            "fit_objective": "sum_squared_residuals against median-per-K observed drift",
            "saturating_grid": {
                "drift_inf_step": 0.005,
                "tau_grid": [1e-6, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0],
            },
            "paired_analysis": "Within-document K_max minus K1 drift on complete endpoint pairs; missing pairs reported and excluded.",
            "population_inference": "not_performed",
            "equivalence_test": "not_performed",
            "scope": "Post-processing supplied T1 receipts; no new generation, scorer replay, or independent human assessment.",
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(
        prog="t4_drift_composition",
        description="T4 — fit drift_pct composition laws over T1 receipts.",
    )
    p.add_argument(
        "--receipts",
        type=Path,
        nargs="+",
        default=list(DEFAULT_RECEIPTS),
        help="T1 iterated-round-trip receipts (default: the three landed 2026-05-21).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(f"fixtures/bench_receipts/drift_composition_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.json"),
        help="Output receipt path.",
    )
    p.add_argument("--pretty", action="store_true", help="Indent the JSON output.")
    args = p.parse_args()

    missing = [r for r in args.receipts if not r.exists()]
    if missing:
        for r in missing:
            print(f"t4_drift_composition: receipt not found: {r}")
        return 2

    if args.out.exists():
        try:
            existing = json.loads(args.out.read_bytes())
        except (ValueError, UnicodeDecodeError):
            existing = None
        if not isinstance(existing, dict) or existing.get("schema") != SCHEMA:
            print(f"t4_drift_composition: refusing to overwrite a historical or unrecognized artifact: {args.out}")
            return 2

    receipt = build_receipt(args.receipts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2 if args.pretty else None, sort_keys=True, allow_nan=False))
    summary = receipt["cross_corpus_summary"]
    print(
        f"drift-composition receipt: {args.out}\n"
        f"  corpora analysed: {summary['n_corpora']}\n"
        f"  best-law distribution: {summary['best_law_distribution']}\n"
        f"  all observed medians unchanged: {summary['all_observed_medians_unchanged']}\n"
        f"  documents with increased endpoint drift: {summary['n_documents_worsened_at_endpoint']}\n"
        f"  inference: descriptive only; no equivalence test\n"
        f"  max observed median-drift delta vs K=1: {summary['max_observed_delta_vs_K1']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
