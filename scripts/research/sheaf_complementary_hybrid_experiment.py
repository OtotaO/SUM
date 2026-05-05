"""
Carmack-recovery final test: Borda fusion of (v3.2 + per-triple) and B2.

Structural prediction: the two detectors are complementary.
  - B2 catches A1/A4 (entity-set changes) at AUC 1.000, blind to A2 at 0.500.
  - v3.2+per-triple catches A2 at ~0.67, partial on A1/A4.

Borda fusion should beat B2 alone because v3.2+per-triple contributes
UNIQUE signal on A2 where B2 has none. A2's per-cell AUC under fusion
should equal the per-triple channel's A2 AUC (B2 contributes only noise
there); A1/A4 should stay at or near 1.000 (B2 dominates the rank order).

If trusted-mean(borda) ≥ B2 + 0.03, the v3.x arc has a real composition
win: cryptographically-anchored sheaf-Laplacian scoring extends trivial
entity-set baselines to cover predicate-violation perturbations.

Output: fixtures/bench_receipts/complementary_hybrid_<DATE>.json
        schema: sum.sheaf_complementary_hybrid.v1
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from sum_engine_internal.research.sheaf_laplacian_v2 import (
    combined_detector_score,
    train_restriction_maps,
)
from sum_engine_internal.research.sheaf_laplacian_v3 import weights_from_receipts
from scripts.research.sheaf_v3_roc_bench import (
    _build_doc_sheaf,
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
    score_v32_combined,
)
from scripts.research.sheaf_baseline_comparison import score_b2_jaccard_distance
from scripts.research.sheaf_per_triple_integration_experiment import (
    score_v32_with_per_triple,
)
from scripts.research.sheaf_hybrid_comparison import borda_fuse

REPO = Path(__file__).resolve().parents[2]
RECEIPTS_DIR = REPO / "fixtures" / "bench_receipts"

GAMMA = 0.1


def run_experiment() -> dict[str, Any]:
    rng = random.Random(0)
    print("=" * 72)
    print("Complementary hybrid — Borda(v3.2+per_triple, B2)")
    print("=" * 72)

    print("\n[1] Sieve-extracting triples…")
    corpus = extract_corpus_triples()
    all_triples = [t for _, ts in corpus for t in ts]
    all_entities = sorted({e for h, _, t in all_triples for e in (h, t)})
    all_relations = sorted({r for _, r, _ in all_triples})

    print("\n[2] Training v2.1 sheaf…")
    trained, embeddings, _ = train_restriction_maps(
        all_triples, stalk_dim=8, epochs=200, learning_rate=0.005,
        margin=0.5, n_negatives_per_positive=3, seed=0,
    )

    print("\n[3] Auto-calibrating λ…")
    per_edge_means: list[float] = []
    for doc_id, source in corpus:
        try:
            doc_sheaf, doc_emb = _build_doc_sheaf(source, trained, embeddings)
            clean = combined_detector_score(doc_sheaf, doc_emb, source)
            per_edge_means.append(clean["v_laplacian"] / max(len(source), 1))
        except (ValueError, KeyError):
            pass
    lambda_auto = float(np.mean(per_edge_means)) if per_edge_means else 0.05
    print(f"    λ_auto = {lambda_auto:.4f}")

    print(f"\n[4] Per-doc scoring (γ={GAMMA})…")
    cells: dict[tuple[str, str, str], list[tuple[float, int]]] = {}
    n_with_partition = 0
    for doc_id, source in corpus:
        if len(source) < 4:
            continue
        stable_seed = int.from_bytes(
            hashlib.sha256(doc_id.encode()).digest()[:4], "big",
        )
        trusted, untrusted = partition_trust(source, seed=stable_seed)
        if not trusted or not untrusted:
            continue
        try:
            doc_sheaf, doc_emb = _build_doc_sheaf(source, trained, embeddings)
        except (ValueError, KeyError):
            continue
        weights = weights_from_receipts(doc_sheaf, trusted_edges=trusted)

        clean_v32_pt = score_v32_with_per_triple(
            doc_sheaf, doc_emb, source, weights, lambda_auto, GAMMA,
            global_sheaf=trained, global_embeddings=embeddings,
        )
        clean_b2 = 0.0

        target_t = rng.choice(trusted)
        target_u = rng.choice(untrusted)
        for target_label, target in (("trusted", target_t), ("untrusted", target_u)):
            for cls, perturb_fn in [
                ("A1", lambda r=rng, tg=target: perturb_a1_on_target(
                    source, tg, all_entities, r)),
                ("A2", lambda r=rng, tg=target: perturb_a2_on_target(
                    source, tg, all_relations, r)),
                ("A4", lambda tg=target: perturb_a4_drop_target(source, tg)),
            ]:
                perturbed = perturb_fn()
                if perturbed == source:
                    continue
                try:
                    p_v32_pt = score_v32_with_per_triple(
                        doc_sheaf, doc_emb, perturbed, weights, lambda_auto, GAMMA,
                        global_sheaf=trained, global_embeddings=embeddings,
                    )
                except Exception:  # noqa: BLE001
                    continue
                p_b2 = score_b2_jaccard_distance(source, perturbed)
                cells.setdefault(("v32_plus_per_triple", cls, target_label), []).extend([
                    (clean_v32_pt, 0), (p_v32_pt, 1),
                ])
                cells.setdefault(("b2_jaccard", cls, target_label), []).extend([
                    (clean_b2, 0), (p_b2, 1),
                ])
        n_with_partition += 1

    print(f"    docs with partition: {n_with_partition}")

    # Borda fuse v3.2+per_triple with B2 per cell
    print("\n[5] Borda fusion of (v3.2+per_triple) and B2 per cell…")
    hybrid_per_cell_auc: dict[str, float] = {}
    for (cls, tgt) in {(c, t) for (_, c, t) in cells}:
        v32_pairs = cells[("v32_plus_per_triple", cls, tgt)]
        b2_pairs = cells[("b2_jaccard", cls, tgt)]
        assert [l for _, l in v32_pairs] == [l for _, l in b2_pairs]
        v32_scores = [s for s, _ in v32_pairs]
        b2_scores = [s for s, _ in b2_pairs]
        labels = [l for _, l in v32_pairs]
        fused = borda_fuse(v32_scores, b2_scores)
        hybrid_per_cell_auc[f"borda_v32pt_b2|{cls}|{tgt}"] = roc_auc(fused, labels)

    component_per_cell_auc: dict[str, float] = {}
    for (det, cls, tgt), pairs in cells.items():
        component_per_cell_auc[f"{det}|{cls}|{tgt}"] = roc_auc(
            [s for s, _ in pairs], [l for _, l in pairs],
        )

    all_per_cell = {**component_per_cell_auc, **hybrid_per_cell_auc}
    print("\n[6] Per-cell AUC:")
    for k in sorted(all_per_cell):
        print(f"    {k:55s} = {all_per_cell[k]:.3f}")

    print("\n[7] Trusted-mean AUC across A1+A2+A4 trusted cells:")
    trusted_means: dict[str, float] = {}
    for det in ("v32_plus_per_triple", "b2_jaccard", "borda_v32pt_b2"):
        aucs = [auc for k, auc in all_per_cell.items()
                if k.startswith(f"{det}|") and k.endswith("|trusted")]
        trusted_means[det] = sum(aucs) / len(aucs) if aucs else 0.0
        print(f"    {det:30s} = {trusted_means[det]:.3f}")

    delta = trusted_means["borda_v32pt_b2"] - trusted_means["b2_jaccard"]
    if delta >= 0.03:
        verdict = "HYBRID_BEATS_BASELINE"
    elif delta >= -0.02:
        verdict = "HYBRID_TIES_BASELINE"
    else:
        verdict = "HYBRID_LOSES_TO_BASELINE"
    print(f"\n[8] Δ(borda - b2) = {delta:+.3f} → {verdict}")

    report = {
        "schema": "sum.sheaf_complementary_hybrid.v1",
        "corpus": "seed_long_paragraphs",
        "n_docs_total": len(corpus),
        "n_docs_with_partition": n_with_partition,
        "lambda_auto": lambda_auto,
        "gamma_used": GAMMA,
        "alpha": 1.0,
        "beta": 1.0,
        "detectors": ["v32_plus_per_triple", "b2_jaccard", "borda_v32pt_b2"],
        "perturbation_classes": ["A1", "A2", "A4"],
        "per_cell_auc": all_per_cell,
        "trusted_mean_auc_by_detector": trusted_means,
        "delta_borda_vs_b2_trusted_mean": delta,
        "verdict": verdict,
        "method_notes": (
            "Borda rank-fusion of (v3.2 + per-rendered-triple channel) with "
            "B2 jaccard. Structural hypothesis: detectors complementary — "
            "B2 catches entity-set-changing perturbations (A1, A4); per-triple "
            "channel catches predicate-flip A2 (entity-set-preserving). "
            "Fusion should add v3.2's A2 signal without disrupting B2's "
            "perfect A1/A4 because Borda is rank-based and B2's perturbed "
            "scores are already at the top of every A1/A4 pool."
        ),
    }
    quantized = quantize_for_digest(report)
    report["bench_digest"] = compute_bench_digest(quantized)
    print(f'\n  "bench_digest": "{report["bench_digest"]}"')
    return report


def main() -> dict[str, Any]:
    report = run_experiment()
    today = _dt.date.today().isoformat()
    out = RECEIPTS_DIR / f"complementary_hybrid_{today}.json"
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n→ wrote {out.relative_to(REPO)}")
    return report


if __name__ == "__main__":
    main()
