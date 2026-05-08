"""
Cross-family aggregator for §4.10.1. Reads N per-model
`sum.recursive_compression_walk.v1` receipts (one per LLM family)
and produces a joint cross-family report:

  - per-(model, corpus) median fp recall and median fp n_axioms
  - per-doc cross-family agreement: at each threshold τ, how many
    models successfully identify a SUM (recall ≥ τ)?
  - aggregate: is the SUM finding model-stable, or model-dependent?

Mirrors `sheaf_path2_cross_corpus_aggregate.py` in shape — pure
function over already-captured receipts, no LLM calls. Schema:
`sum.recursive_compression_cross_family.v1`.

## Joint classification

Across the model set, for each (corpus, doc, threshold τ):

  - `SUM_AGREES_ALL_MODELS`     — every model identifies a SUM
                                  satisfying the threshold; this is
                                  the strongest claim.
  - `SUM_AGREES_MAJORITY`       — most but not all models identify
                                  a SUM at this threshold.
  - `SUM_MODEL_DEPENDENT`       — < half the models identify a SUM;
                                  the doc's compressibility under
                                  this threshold is sensitive to LLM
                                  family.
  - `SUM_NEVER_REACHED`         — no model identifies a SUM at this
                                  threshold (the doc is "hard" — its
                                  axiom-set does not survive any LLM's
                                  iterative re-render at this τ).

Aggregate finding across the corpus:

  - `RECURSIVE_COMPRESSION_MODEL_STABLE` — most cells reach
    SUM_AGREES_ALL_MODELS or SUM_AGREES_MAJORITY at τ ≥ 0.5.
  - `RECURSIVE_COMPRESSION_MODEL_DEPENDENT` — many cells diverge.
"""
from __future__ import annotations

import scripts.research._deterministic_blas  # noqa: F401, E402

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402


# ─── Receipt loading ─────────────────────────────────────────────────


def _load_receipt(path: Path) -> dict[str, Any]:
    r = json.loads(path.read_text())
    if r.get("schema") != "sum.recursive_compression_walk.v1":
        raise ValueError(
            f"{path} has schema {r.get('schema')!r}, expected "
            f"sum.recursive_compression_walk.v1."
        )
    if r.get("compressor") != "llm":
        raise ValueError(
            f"{path} compressor is {r.get('compressor')!r}, expected "
            f"'llm'. Cross-family aggregation only meaningful on the "
            f"LLM arm."
        )
    return r


# ─── Per-(corpus, doc, threshold) cross-family agreement ─────────────


def _classify_agreement(
    n_models_with_sum: int, n_models_total: int,
) -> str:
    if n_models_total == 0:
        return "SUM_NEVER_REACHED"
    if n_models_with_sum == n_models_total:
        return "SUM_AGREES_ALL_MODELS"
    if n_models_with_sum >= (n_models_total + 1) // 2:
        return "SUM_AGREES_MAJORITY"
    if n_models_with_sum >= 1:
        return "SUM_MODEL_DEPENDENT"
    return "SUM_NEVER_REACHED"


def _aggregate_corpus(
    corpus_name: str, per_model_data: dict[str, dict[str, Any]],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    """Aggregate one corpus across N models."""
    # Collect per-doc per-model data
    all_doc_ids: set[str] = set()
    for model, data in per_model_data.items():
        for doc_id in data["per_doc_walks"].keys():
            all_doc_ids.add(doc_id)
    sorted_doc_ids = sorted(all_doc_ids)

    # Per-doc agreement at each threshold
    per_doc_agreement: dict[str, dict[str, dict[str, Any]]] = {}
    for doc_id in sorted_doc_ids:
        per_doc_agreement[doc_id] = {}
        for tau in thresholds:
            tau_key = f"tau_{tau:.2f}"
            n_total = 0
            n_with_sum = 0
            per_model_sum: dict[str, dict[str, Any] | None] = {}
            for model, data in per_model_data.items():
                doc_sums = data["aggregate"]["per_doc_sums"].get(doc_id, {})
                sum_at_tau = doc_sums.get(tau_key)
                per_model_sum[model] = sum_at_tau
                n_total += 1
                if sum_at_tau is not None:
                    n_with_sum += 1
            per_doc_agreement[doc_id][tau_key] = {
                "agreement": _classify_agreement(n_with_sum, n_total),
                "n_models_with_sum": n_with_sum,
                "n_models_total": n_total,
                "per_model_sum": per_model_sum,
            }

    # Per-model summary on this corpus
    per_model_summary: dict[str, dict[str, Any]] = {}
    for model, data in per_model_data.items():
        s = data["aggregate"]["summary"]
        per_model_summary[model] = {
            "median_fixed_point_step": s["median_fixed_point_step"],
            "median_fixed_point_n_axioms": s["median_fixed_point_n_axioms"],
            "median_fixed_point_recall_vs_original": s[
                "median_fixed_point_recall_vs_original"
            ],
            "n_docs_collapsed_to_empty": s["n_docs_collapsed_to_empty"],
        }

    # Joint corpus-level finding: agreement-class distribution at τ=0.5
    # and τ=0.9
    joint_by_tau: dict[str, dict[str, Any]] = {}
    for tau in thresholds:
        tau_key = f"tau_{tau:.2f}"
        agreement_counts: dict[str, int] = {}
        for doc_id in sorted_doc_ids:
            cls = per_doc_agreement[doc_id][tau_key]["agreement"]
            agreement_counts[cls] = agreement_counts.get(cls, 0) + 1
        # Joint label
        n_docs = len(sorted_doc_ids)
        n_agrees_all = agreement_counts.get("SUM_AGREES_ALL_MODELS", 0)
        n_agrees_majority = agreement_counts.get("SUM_AGREES_MAJORITY", 0)
        n_model_dependent = agreement_counts.get("SUM_MODEL_DEPENDENT", 0)
        n_never = agreement_counts.get("SUM_NEVER_REACHED", 0)
        if n_docs == 0:
            label = "NO_DOCS"
        elif (n_agrees_all + n_agrees_majority) / n_docs >= 0.66:
            label = "RECURSIVE_COMPRESSION_MODEL_STABLE"
        elif n_model_dependent / n_docs >= 0.34:
            label = "RECURSIVE_COMPRESSION_MODEL_DEPENDENT"
        else:
            label = "MIXED"
        joint_by_tau[tau_key] = {
            "label": label,
            "agreement_counts": agreement_counts,
            "n_docs": n_docs,
        }

    return {
        "corpus": corpus_name,
        "models": sorted(per_model_data.keys()),
        "per_model_summary": per_model_summary,
        "per_doc_agreement": per_doc_agreement,
        "joint_by_tau": joint_by_tau,
    }


# ─── Main ────────────────────────────────────────────────────────────


def aggregate(receipt_paths: list[Path],
              thresholds: tuple[float, ...] = (0.5, 0.7, 0.9, 0.99)) -> dict[str, Any]:
    # Group receipts by (corpus, model). Each receipt covers N corpora,
    # we explode by corpus.
    by_corpus: dict[str, dict[str, dict[str, Any]]] = {}
    all_models: set[str] = set()
    for p in receipt_paths:
        r = _load_receipt(p)
        model = r["llm_model"]
        all_models.add(model)
        for corpus_name, corpus_data in r["by_corpus"].items():
            if model in by_corpus.setdefault(corpus_name, {}):
                raise ValueError(
                    f"Duplicate (corpus={corpus_name}, model={model}) across "
                    f"receipts. Each (corpus, model) cell should appear in "
                    f"exactly one receipt."
                )
            by_corpus[corpus_name][model] = corpus_data

    print("=" * 72)
    print(f"Cross-family recursive-walk aggregation")
    print(f"  models: {sorted(all_models)}")
    print(f"  corpora: {sorted(by_corpus.keys())}")
    print(f"  thresholds: {thresholds}")
    print("=" * 72)

    per_corpus_aggregates: dict[str, dict[str, Any]] = {}
    for corpus_name, per_model in sorted(by_corpus.items()):
        agg = _aggregate_corpus(corpus_name, per_model, thresholds)
        per_corpus_aggregates[corpus_name] = agg
        print(f"\n──── {corpus_name} ────")
        print(f"  per-model median fp recall:")
        for model, summary in agg["per_model_summary"].items():
            recall = summary["median_fixed_point_recall_vs_original"]
            n_axioms = summary["median_fixed_point_n_axioms"]
            print(f"    {model:42s} recall={recall:>5.3f}  fp_n={n_axioms}")
        print(f"  joint label by tau:")
        for tau_key, joint in agg["joint_by_tau"].items():
            print(f"    {tau_key}: {joint['label']}  counts={joint['agreement_counts']}")

    # Overall joint finding: aggregate the per-corpus joint labels at τ=0.5
    overall_labels = [
        agg["joint_by_tau"]["tau_0.50"]["label"]
        for agg in per_corpus_aggregates.values()
    ]
    if all(l == "RECURSIVE_COMPRESSION_MODEL_STABLE" for l in overall_labels):
        overall = "RECURSIVE_COMPRESSION_MODEL_STABLE_ACROSS_CORPORA"
    elif all(l == "RECURSIVE_COMPRESSION_MODEL_DEPENDENT" for l in overall_labels):
        overall = "RECURSIVE_COMPRESSION_MODEL_DEPENDENT_ACROSS_CORPORA"
    else:
        overall = "MIXED_ACROSS_CORPORA"
    print(f"\n  overall finding (across corpora at τ=0.50): {overall}")

    out: dict[str, Any] = {
        "schema": "sum.recursive_compression_cross_family.v1",
        "models": sorted(all_models),
        "corpora": sorted(by_corpus.keys()),
        "thresholds": list(thresholds),
        "per_corpus_aggregates": per_corpus_aggregates,
        "overall_finding": overall,
        "method_notes": (
            "Aggregator reads N sum.recursive_compression_walk.v1 "
            "receipts (one per LLM model, each covering 1+ corpora) "
            "and produces a per-(corpus, doc, τ) agreement classification: "
            "SUM_AGREES_ALL_MODELS / SUM_AGREES_MAJORITY / "
            "SUM_MODEL_DEPENDENT / SUM_NEVER_REACHED. Aggregate per-corpus "
            "label is RECURSIVE_COMPRESSION_MODEL_STABLE if ≥66% of docs "
            "reach majority-or-better agreement at τ=0.50; "
            "RECURSIVE_COMPRESSION_MODEL_DEPENDENT if ≥34% are "
            "model-dependent. The aggregator is a pure function over "
            "the input receipts; no LLM calls."
        ),
    }
    from scripts.research.sheaf_v3_2_validation import (
        compute_bench_digest, quantize_for_digest,
    )
    quantized = quantize_for_digest(out)
    out["bench_digest"] = compute_bench_digest(quantized)
    print(f"\n  bench_digest: {out['bench_digest']}")
    return out


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receipts", nargs="+", required=True,
        help="Per-model recursive walk receipts to aggregate "
             "(sum.recursive_compression_walk.v1, compressor=llm).",
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.5, 0.7, 0.9, 0.99],
    )
    args = parser.parse_args()

    receipt_paths = [Path(p) for p in args.receipts]
    for p in receipt_paths:
        if not p.exists():
            raise SystemExit(f"receipt not found: {p}")

    out = aggregate(receipt_paths, tuple(args.thresholds))

    from scripts.research._receipt_paths import resolve_receipt_path
    repo_root = Path(__file__).resolve().parents[2]
    receipts_dir = repo_root / "fixtures" / "bench_receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = resolve_receipt_path(
        receipts_dir, "recursive_compression_cross_family",
    )
    receipt_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\n→ wrote {receipt_path.relative_to(repo_root)}")
    return out


if __name__ == "__main__":
    main()
