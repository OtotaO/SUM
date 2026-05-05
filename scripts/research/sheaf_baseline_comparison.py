"""
Baseline comparison bench for the v3.x sheaf-Laplacian arc.

Two trivial reproducible baselines, scored on the SAME (clean, perturbed)
triple-set pairs the v3.x detectors consume:

  B1 ENTITY_PRESENCE_DEFICIT:
      score = 1.0 - (|source_entities ∩ rendered_entities| / |source_entities|)
      A perturbed render that drops/swaps entities scores higher than clean.

  B2 JACCARD_DISTANCE:
      score = 1.0 - |source_entities ∩ rendered_entities| /
                    |source_entities ∪ rendered_entities|
      Symmetric variant of B1; penalises spurious entities too.

Both are pure set ops on entity sets. No floating-point, no LAPACK, no
randomness in the scoring step. AUC reproduces exactly across runs.

These are the *minimum-defensible reproducible baselines*. Stronger
LM-based baselines (sequence log-probability under a pinned LM,
MiniCheck-FT5 entailment) are deferred to v0.2 — they introduce
model-pinning machinery orthogonal to the v0.1 scope.

The point: if the v3.x detectors don't beat trivial set comparison on
the same perturbed-triple inputs, the detector's value claim is dead.
This bench surfaces that question with a number.

Output: fixtures/bench_receipts/baseline_comparison_<DATE>.json
        schema: sum.sheaf_baseline_comparison.v1
"""
from __future__ import annotations

import datetime as _dt
import json
import random
from pathlib import Path
from typing import Any, Callable

# Reuse the v3 ROC bench's perturbation harness + corpus loader so the
# baselines see byte-identical inputs to v2.2 / v3 / v3.1 / v3.2.
from scripts.research.sheaf_v3_roc_bench import (
    extract_corpus_triples,
    partition_trust,
    perturb_a1_on_target,
    perturb_a2_on_target,
    perturb_a4_drop_target,
    roc_auc,
)
from scripts.research.sheaf_v3_2_validation import (
    compute_bench_digest,
    quantize_for_digest,
)

REPO = Path(__file__).resolve().parents[2]
RECEIPTS_DIR = REPO / "fixtures" / "bench_receipts"

Triple = tuple[str, str, str]
BaselineFn = Callable[[list[Triple], list[Triple]], float]


def entity_set(triples: list[Triple]) -> set[str]:
    """Heads ∪ tails. Predicates excluded — baselines are entity-shape only."""
    out: set[str] = set()
    for h, _p, t in triples:
        out.add(h)
        out.add(t)
    return out


def score_b1_entity_presence_deficit(
    source_triples: list[Triple], rendered_triples: list[Triple]
) -> float:
    """1.0 − recall of source entities in rendered triples.
    Range [0.0, 1.0]. Empty source returns 0.0 (degenerate; no signal).
    """
    src = entity_set(source_triples)
    if not src:
        return 0.0
    rnd = entity_set(rendered_triples)
    return 1.0 - len(src & rnd) / len(src)


def score_b2_jaccard_distance(
    source_triples: list[Triple], rendered_triples: list[Triple]
) -> float:
    """1.0 − Jaccard(source_entities, rendered_entities).
    Symmetric: penalises both missing and spurious entities.
    Empty union returns 0.0.
    """
    src = entity_set(source_triples)
    rnd = entity_set(rendered_triples)
    union = src | rnd
    if not union:
        return 0.0
    return 1.0 - len(src & rnd) / len(union)


BASELINES: dict[str, BaselineFn] = {
    "b1_entity_presence_deficit": score_b1_entity_presence_deficit,
    "b2_jaccard_distance": score_b2_jaccard_distance,
}


def _perturb_one(
    cls: str,
    triples: list[Triple],
    target: Triple,
    all_entities: list[str],
    all_relations: list[str],
    rng_seed: int,
) -> list[Triple] | None:
    rng = random.Random(rng_seed)
    if cls == "A1":
        return perturb_a1_on_target(triples, target, all_entities, rng)
    if cls == "A2":
        return perturb_a2_on_target(triples, target, all_relations, rng)
    if cls == "A4":
        return perturb_a4_drop_target(triples, target)
    raise ValueError(f"unknown perturbation class: {cls}")


def run_baseline_comparison() -> dict[str, Any]:
    """Mirrors the v3 ROC bench's loop structure so per-cell AUCs are
    directly comparable to §4.4 / §4.7 of the preprint."""
    print("=" * 72)
    print("Baseline comparison bench — B1 / B2 vs v3.x detectors")
    print("=" * 72)

    print("\n[1] Sieve-extracting triples from seed_long_paragraphs…")
    corpus = extract_corpus_triples()
    print(f"    {len(corpus)} docs with non-empty extractions")

    all_entities = sorted({e for _, ts in corpus for h, _, t in ts for e in (h, t)})
    all_relations = sorted({p for _, ts in corpus for _, p, _ in ts})
    print(f"    union vocab: {len(all_entities)} entities, "
          f"{len(all_relations)} relations")

    print("\n[2] Per-doc trust partition + targeted perturbations + scoring…")
    cells: dict[tuple[str, str, str], list[tuple[float, int]]] = {}

    n_with_partition = 0
    for doc_idx, (doc_id, source) in enumerate(corpus):
        if len(source) < 2:
            continue
        trusted, untrusted = partition_trust(source, doc_idx)
        if not trusted or not untrusted:
            continue
        n_with_partition += 1

        for cls in ("A1", "A2", "A4"):
            for target_label, target_set in (
                ("trusted", trusted), ("untrusted", untrusted)
            ):
                for tgt_idx, target in enumerate(target_set):
                    perturbed = _perturb_one(
                        cls, source, target, all_entities, all_relations,
                        rng_seed=doc_idx * 1000 + tgt_idx,
                    )
                    if perturbed is None or perturbed == source:
                        continue
                    for baseline_id, scorer in BASELINES.items():
                        clean_score = scorer(source, source)
                        perturbed_score = scorer(source, perturbed)
                        key = (baseline_id, cls, target_label)
                        cells.setdefault(key, []).extend([
                            (clean_score, 0),
                            (perturbed_score, 1),
                        ])

    print(f"    docs with partition: {n_with_partition} "
          f"(of {len(corpus)} total)")

    print("\n[3] Per-cell AUC:")
    per_cell_auc: dict[str, float] = {}
    for (baseline_id, cls, target), pairs in sorted(cells.items()):
        scores = [s for s, _ in pairs]
        labels = [l for _, l in pairs]
        auc = roc_auc(scores, labels)
        cell_key = f"{baseline_id}|{cls}|{target}"
        per_cell_auc[cell_key] = auc
        print(f"    {cell_key:55s} = {auc:.3f}")

    print("\n[4] Trusted-mean AUC by baseline (across A1+A2+A4 trusted cells):")
    trusted_mean: dict[str, float] = {}
    for baseline_id in BASELINES:
        trusted_aucs = [
            auc for key, auc in per_cell_auc.items()
            if key.startswith(f"{baseline_id}|") and key.endswith("|trusted")
        ]
        trusted_mean[baseline_id] = (
            sum(trusted_aucs) / len(trusted_aucs) if trusted_aucs else 0.0
        )
        print(f"    {baseline_id:35s} = {trusted_mean[baseline_id]:.3f}")

    report = {
        "schema": "sum.sheaf_baseline_comparison.v1",
        "corpus": "seed_long_paragraphs",
        "n_docs_total": len(corpus),
        "n_docs_with_partition": n_with_partition,
        "baselines": sorted(BASELINES.keys()),
        "perturbation_classes": ["A1", "A2", "A4"],
        "per_cell_auc": per_cell_auc,
        "trusted_mean_auc_by_baseline": trusted_mean,
        "method_notes": (
            "B1 = 1 - recall of source entities in rendered triples. "
            "B2 = 1 - jaccard(source_entities, rendered_entities). "
            "Both pure set ops on entity sets; predicates excluded. "
            "Inputs are the same (clean, perturbed) triple-set pairs the "
            "v3.x detectors consume (same partition_trust + perturb_a* "
            "harness from sheaf_v3_roc_bench)."
        ),
    }

    quantized = quantize_for_digest(report)
    report["bench_digest"] = compute_bench_digest(quantized)
    print(f'\n  "bench_digest": "{report["bench_digest"]}"')

    return report


def main() -> dict[str, Any]:
    report = run_baseline_comparison()
    from scripts.research._receipt_paths import resolve_receipt_path
    out = resolve_receipt_path(RECEIPTS_DIR, "baseline_comparison")
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n→ wrote {out.relative_to(REPO)}")
    return report


if __name__ == "__main__":
    main()
