"""T4 — Compositional metric audit for `drift_pct`.

Bench-hardening worktrail T4 (docs/BENCH_HARDENING_FROM_QCVV.md): post-
process the T1 iterated-round-trip receipts to characterise how
`drift_pct` composes under K-step iteration.

Three candidate composition laws from the spec:

  (1) additive               : drift_K = K * drift_1
  (2) multiplicative-survival: drift_K = 1 - (1 - drift_1)^K
  (3) saturating             : drift_K = drift_inf * (1 - exp(-K/tau))

For each corpus we fit each law (the first two are parameter-free given
drift_1; the third has two free parameters drift_inf, tau and is fit by
brute-force grid search to avoid a scipy dependency) and report sum-of-
squared-residuals per law. The "winning" law is whichever has the
smallest SSR.

A fourth row is reported: **fixed-point** `drift_K = drift_1` (no
composition effect). For data that is K-invariant within sampling noise
this is the right characterisation, not any of (1)/(2)/(3).

The composition bound is DKW (Dvoretzky-Kiefer-Wolfowitz):

    epsilon(n, delta) = sqrt(ln(2/delta) / (2n))

…applied to the empirical CDF of per-document drift values at each K.
With 95% confidence (delta=0.05), the true CDF lies within epsilon of
the empirical CDF uniformly across all thresholds. We report
worst_case_drift_at_K_95 = (inf {x : Fhat_K(x) >= 0.05}) - epsilon, the
lower-tail bound mirrored from T3.

Schema: ``sum.drift_metric_composition.v1`` at
``fixtures/bench_receipts/drift_composition_<YYYY-MM-DD>.json``.

Cost: pure post-processing. No LLM calls.

Source: docs/BENCH_HARDENING_FROM_QCVV.md task T4 + docs/DRIFT_METRIC_
COMPOSITION.md (the prose distillation of this runner's findings).
License: Apache License 2.0
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "sum.drift_metric_composition.v1"

DEFAULT_RECEIPTS: tuple[Path, ...] = (
    Path("fixtures/bench_receipts/s25_iterated_K10_seed_v1_2026-05-21.json"),
    Path("fixtures/bench_receipts/s25_iterated_K10_seed_v2_2026-05-21.json"),
    Path("fixtures/bench_receipts/s25_iterated_K10_seed_long_paragraphs_2026-05-21.json"),
)


def _dkw_epsilon(n: int, delta: float = 0.05) -> float:
    """Dvoretzky-Kiefer-Wolfowitz bound: with prob >= 1 - delta the true
    CDF F lies within epsilon of the empirical Fhat at every x."""
    if n <= 0:
        return float("inf")
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


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


def _hellinger_axiom_distribution(per_document: list[dict[str, Any]]) -> dict[str, Any]:
    """Hellinger fidelity over the empirical axiom-key distribution at
    K=1 vs K=K_max, treated as a sparse categorical.

    For each doc, we have the set of axiom keys at K=k (we infer them
    from n_observed/n_missing). The full axiom-key strings are not in
    the per-document record (T1 stripped them for compactness), so this
    is a *frequency-by-doc* approximation: we treat each doc's
    contribution to the corpus-wide axiom count as a categorical
    sample.

    Hellinger(p, q) = (1/sqrt(2)) * sqrt(sum_i (sqrt(p_i) - sqrt(q_i))^2)

    For our purposes the categorical is over documents: p_i = fraction
    of total truth axioms contributed by doc i; q_i = fraction of total
    re-extracted axioms at K=K_max contributed by doc i. Hellinger
    fidelity F(p, q) = 1 - Hellinger(p, q)^2 (so F = 1 means identical
    distributions). The compositional claim: F(p, q_K) should be ~ F(p,
    q_1)^K under independent stagewise noise.
    """
    if not per_document:
        return {"fidelity_K1": None, "fidelity_KK": None, "compositional_predicted": None}
    K_max = max(len(d["iterations"]) for d in per_document)
    # Build p (truth) and q_K (re-extracted at K=k) histograms over documents.
    truth_total = sum(d.get("n_truth_axioms", 0) for d in per_document)
    if truth_total == 0:
        return {"fidelity_K1": None, "fidelity_KK": None, "compositional_predicted": None}
    p = [d.get("n_truth_axioms", 0) / truth_total for d in per_document]

    def q_at(k: int) -> list[float]:
        observed = [
            next((it for it in d["iterations"] if int(it["k"]) == k), {}).get("n_observed", 0)
            for d in per_document
        ]
        tot = sum(observed)
        if tot == 0:
            return [0.0] * len(per_document)
        return [n / tot for n in observed]

    def fidelity(p: list[float], q: list[float]) -> float:
        # Bhattacharyya coefficient (= 1 - H^2 / 1, with H_max^2 = 1)
        bc = sum(math.sqrt(pi * qi) for pi, qi in zip(p, q))
        return bc * bc  # squared Bhattacharyya is the "F" we report

    q1 = q_at(1)
    qK = q_at(K_max)
    F1 = fidelity(p, q1)
    FK = fidelity(p, qK)
    F_predicted = F1 ** K_max  # under the multiplicative-survival law
    return {
        "K_max": K_max,
        "fidelity_K1": round(F1, 6),
        "fidelity_KK": round(FK, 6),
        "compositional_predicted_F1_pow_K": round(F_predicted, 6),
        "compositional_residual": round(abs(FK - F_predicted), 6),
        "notes": (
            "Document-frequency approximation; the per-axiom-key "
            "categorical was not preserved in T1 receipts. Treat as an "
            "indicator, not the canonical Hellinger metric."
        ),
    }


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
    payload = json.loads(receipt_path.read_text())
    corpus_id = _infer_corpus_id(receipt_path, payload)
    per_doc = payload["per_document"]
    K = int(payload.get("k_iterations", max(len(d["iterations"]) for d in per_doc)))
    n_docs = len(per_doc)

    drift_by_k = _aggregate_per_K(per_doc, K)
    median_by_K = [statistics.median(drift_by_k[k]) if drift_by_k[k] else float("nan") for k in range(1, K + 1)]
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
            "form": "drift_K = drift_1 (no composition effect)",
            "free_parameters": 0,
            "drift_1": round(drift_1, 6),
            "predicted_by_K": [round(p, 6) for p in pred_fp],
            "sum_squared_residuals": round(ssr_fp, 9),
        },
    }

    # Tie-breaking: when SSRs are within 1e-12 of each other, prefer
    # fixed_point > saturating > multiplicative_survival > additive.
    # This encodes the load-bearing finding: a flat drift series IS a
    # fixed point under composition, and saying so is more honest than
    # picking whichever growth law happens to also predict zero when
    # drift_1 = 0.
    preference = {"fixed_point": 0, "saturating": 1, "multiplicative_survival": 2, "additive": 3}
    best_law = min(
        laws.items(),
        key=lambda kv: (round(kv[1]["sum_squared_residuals"], 12), preference[kv[0]]),
    )[0]

    # DKW worst-case 95% lower bound on the per-K drift CDF.
    dkw_per_K = {}
    for k in range(1, K + 1):
        vals = drift_by_k[k]
        n = len(vals)
        eps = _dkw_epsilon(n, delta=0.05)
        # P5 of the empirical CDF (5th percentile of drift), minus DKW slack.
        p5 = _percentile(vals, 5.0)
        worst_case_drift_lower_95 = max(0.0, p5 - eps)
        dkw_per_K[str(k)] = {
            "n_observations": n,
            "epsilon_dkw_95": round(eps, 6),
            "empirical_p5_drift": round(p5, 6),
            "worst_case_drift_lower_95": round(worst_case_drift_lower_95, 6),
            "empirical_median_drift": round(median_by_K[k - 1], 6),
            "empirical_max_drift": round(max(vals), 6) if vals else None,
        }

    # Composition-invariance bound: |drift_K - drift_1| supremum vs DKW slack.
    delta_K1 = [abs(median_by_K[k - 1] - drift_1) for k in range(1, K + 1)]
    max_delta = max(delta_K1)
    n_min = min(len(drift_by_k[k]) for k in range(1, K + 1))
    eps_corpus = _dkw_epsilon(n_min, delta=0.05)
    composition_invariance = {
        "max_abs_delta_median_vs_K1": round(max_delta, 6),
        "dkw_epsilon_95_n_min": round(eps_corpus, 6),
        "n_min_per_K": n_min,
        "verdict": (
            "composition_invariant_within_dkw_95"
            if max_delta <= eps_corpus
            else "composition_drift_exceeds_dkw_95"
        ),
        "rationale": (
            f"sup_K |median_drift_K - median_drift_1| = {max_delta:.4f}; "
            f"DKW 95%% bound at n={n_min} is {eps_corpus:.4f}. "
            + (
                "Drift differences across K are within DKW worst-case noise — drift_pct is empirically composition-invariant on this corpus."
                if max_delta <= eps_corpus
                else "Drift differences across K exceed DKW worst-case noise — there is a real composition effect on this corpus."
            )
        ),
    }

    hellinger = _hellinger_axiom_distribution(per_doc)

    return {
        "corpus_id": corpus_id,
        "n_documents": n_docs,
        "K": K,
        "median_drift_by_K": [round(v, 6) for v in median_by_K],
        "mean_drift_by_K": [round(v, 6) for v in mean_by_K],
        "laws_fitted": laws,
        "best_law_by_ssr": best_law,
        "dkw_per_K_95": dkw_per_K,
        "composition_invariance": composition_invariance,
        "hellinger_doc_frequency": hellinger,
        "t1_receipt_path": str(receipt_path),
    }


def build_receipt(receipt_paths: list[Path]) -> dict[str, Any]:
    per_corpus = [analyse_receipt(p) for p in receipt_paths]

    cross_corpus_summary = {
        "n_corpora": len(per_corpus),
        "all_composition_invariant_dkw_95": all(
            c["composition_invariance"]["verdict"] == "composition_invariant_within_dkw_95"
            for c in per_corpus
        ),
        "best_law_distribution": dict(Counter(c["best_law_by_ssr"] for c in per_corpus)),
        "max_observed_delta_vs_K1": max(
            c["composition_invariance"]["max_abs_delta_median_vs_K1"] for c in per_corpus
        ),
    }

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "per_corpus": per_corpus,
        "cross_corpus_summary": cross_corpus_summary,
        "definition": {
            "drift_pct": "1 - exact_match_recall(axioms_predicted, axioms_truth), as a fraction in [0, 1].",
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
            "dkw_bound": "epsilon(n, delta=0.05) = sqrt(ln(2/0.05) / (2n)), two-sided uniform on the empirical CDF",
            "composition_invariance_test": "sup_K |median_drift_K - median_drift_1| vs DKW epsilon at n_min",
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

    receipt = build_receipt(args.receipts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2 if args.pretty else None, sort_keys=True))
    summary = receipt["cross_corpus_summary"]
    print(
        f"drift-composition receipt: {args.out}\n"
        f"  corpora analysed: {summary['n_corpora']}\n"
        f"  best-law distribution: {summary['best_law_distribution']}\n"
        f"  all composition-invariant within DKW 95%: {summary['all_composition_invariant_dkw_95']}\n"
        f"  max observed median-drift delta vs K=1: {summary['max_observed_delta_vs_K1']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
