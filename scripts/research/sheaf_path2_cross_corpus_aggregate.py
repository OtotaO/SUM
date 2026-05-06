"""
Cross-corpus aggregator for the §4.7.4 extension. Reads N
``sum.sheaf_path2_multi_llm_compare.v1`` receipts (one per corpus)
and produces a single aggregate receipt that classifies the
cross-corpus joint finding.

The §4.7.3 result was *one corpus × six LLM lineages*. This bench
extends to *N corpora × M models* and asks: does the per-corpus
joint finding (e.g. ``STRUCTURAL_GAP_NO_MODEL_BEATS``) reproduce
across stylistically distinct corpora? If it does, the structural
claim is corpus-independent. If it doesn't, the §4.7.3 finding
turns out to be corpus-specific and the §4.7.x narrative needs
honest revision.

## Output classification

Across corpora:
  - All corpora's joint findings agree AND no model BEATS in any
    cell → ``STRUCTURAL_GAP_INVARIANT_TO_CORPUS``
  - All corpora's joint findings agree but some model BEATS in
    some cell → ``HYBRID_BEATS_INVARIANT_TO_CORPUS``
  - Joint findings differ across corpora → ``CROSS_CORPUS_VERDICTS_DIVERGE``
    (with a per-corpus breakdown)

A finer-grained per-(corpus, model) matrix is always emitted so a
reader can see exactly where the divergence is.

## Determinism

Phase 2 of each per-corpus receipt is byte-stable given each
corpus's snapshots. The aggregator is a pure function over the
input receipts; no LLM calls. Output schema:
``sum.sheaf_path2_cross_corpus_compare.v1``.
"""
from __future__ import annotations

import scripts.research._deterministic_blas  # noqa: F401, E402

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402


_BEATS = "HYBRID_BEATS_BASELINE_ON_REAL_LLM"
_TIES = "HYBRID_TIES_BASELINE_ON_REAL_LLM"
_LOSES = "HYBRID_LOSES_TO_BASELINE_ON_REAL_LLM"


def _load_receipt(path: Path) -> dict[str, Any]:
    with open(path) as f:
        r = json.load(f)
    if r.get("schema") != "sum.sheaf_path2_multi_llm_compare.v1":
        raise ValueError(
            f"{path} has schema {r.get('schema')!r}, expected "
            f"sum.sheaf_path2_multi_llm_compare.v1. The aggregator "
            f"reads compare receipts only, not raw v3 bench receipts."
        )
    return r


def _classify_cross_corpus(per_corpus: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Produce a cross-corpus joint finding and a per-(corpus, model) matrix.

    The matrix is the load-bearing artifact for the §4.7.4 narrative —
    it shows exactly which corpus × model cells produce which verdict.
    """
    # Per-corpus joint findings
    joint_findings = {c: r["joint_finding"] for c, r in per_corpus.items()}

    # Per-(corpus, model) verdict matrix
    matrix: dict[str, dict[str, str]] = {}
    deltas: dict[str, dict[str, float]] = {}
    for corpus, r in per_corpus.items():
        matrix[corpus] = dict(r["per_model_verdict"])
        deltas[corpus] = dict(r["per_model_delta_borda_vs_b2"])

    # Aggregate verdict labels across all (corpus, model) cells
    all_cells = [v for verdicts in matrix.values() for v in verdicts.values()]
    cell_set = set(all_cells)
    n_beats_cells = sum(1 for v in all_cells if v == _BEATS)
    n_loses_cells = sum(1 for v in all_cells if v == _LOSES)
    n_ties_cells = sum(1 for v in all_cells if v == _TIES)

    # Cross-corpus joint finding
    distinct_jfs = set(joint_findings.values())
    if len(distinct_jfs) == 1 and cell_set <= {_TIES, _LOSES}:
        joint = "STRUCTURAL_GAP_INVARIANT_TO_CORPUS"
    elif len(distinct_jfs) == 1 and cell_set == {_BEATS}:
        joint = "HYBRID_BEATS_INVARIANT_TO_CORPUS"
    else:
        joint = "CROSS_CORPUS_VERDICTS_DIVERGE"

    summary = {
        "per_corpus_joint_finding": joint_findings,
        "per_corpus_model_verdict_matrix": matrix,
        "per_corpus_model_delta_matrix": deltas,
        "n_cells_total": len(all_cells),
        "n_cells_beats": n_beats_cells,
        "n_cells_ties": n_ties_cells,
        "n_cells_loses": n_loses_cells,
        "distinct_joint_findings": sorted(distinct_jfs),
    }
    return joint, summary


def aggregate(receipt_paths: list[Path]) -> dict[str, Any]:
    per_corpus: dict[str, dict[str, Any]] = {}
    for p in receipt_paths:
        r = _load_receipt(p)
        corpus = r["corpus"]
        if corpus in per_corpus:
            raise ValueError(
                f"Duplicate corpus {corpus!r} across receipts: "
                f"{p} clashes with the earlier load."
            )
        per_corpus[corpus] = r

    joint, summary = _classify_cross_corpus(per_corpus)

    print("=" * 72)
    print(f"Cross-corpus aggregation across {len(per_corpus)} corpora")
    print("=" * 72)
    for corpus, r in per_corpus.items():
        print(f"  {corpus:30s}  {r['joint_finding']}")
    print(f"\n  joint cross-corpus finding: {joint}")
    print(f"  cells: {summary['n_cells_beats']} BEATS / "
          f"{summary['n_cells_ties']} TIES / {summary['n_cells_loses']} LOSES")
    print(f"  per-(corpus, model) matrix:")
    for corpus, verdicts in summary["per_corpus_model_verdict_matrix"].items():
        print(f"    {corpus}:")
        for model, verdict in sorted(verdicts.items()):
            d = summary["per_corpus_model_delta_matrix"][corpus][model]
            short = verdict.replace("HYBRID_", "").replace("_BASELINE_ON_REAL_LLM", "")
            print(f"      {model:42s}  Δ={d:+.4f}  {short}")

    out: dict[str, Any] = {
        "schema": "sum.sheaf_path2_cross_corpus_compare.v1",
        "corpora": sorted(per_corpus.keys()),
        "per_corpus_receipts": {c: r for c, r in per_corpus.items()},
        "per_corpus_joint_finding": summary["per_corpus_joint_finding"],
        "per_corpus_model_verdict_matrix": summary["per_corpus_model_verdict_matrix"],
        "per_corpus_model_delta_matrix": summary["per_corpus_model_delta_matrix"],
        "n_corpora": len(per_corpus),
        "n_cells_total": summary["n_cells_total"],
        "n_cells_beats": summary["n_cells_beats"],
        "n_cells_ties": summary["n_cells_ties"],
        "n_cells_loses": summary["n_cells_loses"],
        "joint_finding": joint,
        "method_notes": (
            "Aggregator reads N sum.sheaf_path2_multi_llm_compare.v1 "
            "receipts (one per corpus) and classifies the cross-corpus "
            "joint finding. STRUCTURAL_GAP_INVARIANT_TO_CORPUS requires "
            "every (corpus, model) cell to TIE or LOSE. "
            "HYBRID_BEATS_INVARIANT_TO_CORPUS requires every cell to "
            "BEAT. Anything else is CROSS_CORPUS_VERDICTS_DIVERGE — the "
            "§4.7.3 structural-gap claim turns out to be corpus-"
            "specific. The per-(corpus, model) matrix is the load-"
            "bearing artifact for §4.7.4."
        ),
    }
    return out


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--receipts", nargs="+", required=True,
        help="Paths to per-corpus compare receipts "
             "(sum.sheaf_path2_multi_llm_compare.v1).",
    )
    args = parser.parse_args()

    receipt_paths = [Path(p) for p in args.receipts]
    for p in receipt_paths:
        if not p.exists():
            raise SystemExit(f"receipt not found: {p}")

    report = aggregate(receipt_paths)

    from scripts.research._receipt_paths import resolve_receipt_path
    repo_root = Path(__file__).resolve().parents[2]
    receipts_dir = repo_root / "fixtures" / "bench_receipts"
    out = resolve_receipt_path(receipts_dir, "path2_cross_corpus_compare")
    receipts_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n→ wrote {out.relative_to(repo_root)}")
    return report


if __name__ == "__main__":
    main()
