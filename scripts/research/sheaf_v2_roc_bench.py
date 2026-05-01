"""ROC benchmark for v2.x detector across A1/A2/A3/A4 perturbations.

Path 1 of the publishable-artifact plan: re-use the existing seed
corpora (no LLM API spend, no Worker round-trip) and measure
per-class ROC AUC on synthetic perturbations of real sieve-extracted
triple sets.

Corpus: scripts/bench/corpora/seed_long_paragraphs.json (16 multi-
fact paragraphs from the §2.5 frontier-LLM benchmark work).

Methodology:

  1. Load each long paragraph; sieve-extract its triples → source_D.
  2. Train ONE v2.1 sheaf on the UNION of all source_D triples
     (transductive setup; trained vocabulary covers every entity
     and relation that will appear in any test perturbation).
  3. For each doc D, generate 5 renders:
       - clean: source_D
       - A1 entity-swap: replace one subject with a different
         in-vocab entity
       - A2 predicate-flip: replace one predicate with a different
         in-vocab predicate
       - A3 off-graph fabrication: append a triple with a
         FABRICATED relation (out-of-vocab; structural detection)
       - A4 triple-drop: remove one triple from source_D
  4. For each render, score with v2.x:
       - combined_detector_score (uses source_D as the sheaf-
         restriction graph; render becomes the cochain support)
       - score_rendered_triples_v2 (per-rendered-triple V; oov
         signal)
       Score per-class:
       - A1, A2, A4: max in-vocab V_triple from
         score_rendered_triples_v2 (detection signal: max V
         distinguishes perturbed from clean)
       - A3: oov count from score_rendered_triples_v2 (detection
         signal: any OOV is the structural catch)
       - "Overall" detector: alpha * v_combined +
         beta * max_in_vocab_v + gamma * n_oov
  5. Per-class ROC: clean baseline (label=0) vs perturbed (label=1)
     across all docs. Compute AUC.

Output: a JSON record + console summary.

This script is a measurement, not a unit test. Re-run via:
  PYTHONPATH=. python scripts/research/sheaf_v2_roc_bench.py

CPU-only; runtime ~1-2 minutes for the seed_long_paragraphs corpus.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Repo root on sys.path so sum_engine_internal imports work
import sys
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from sum_engine_internal.research.sheaf_laplacian_v2 import (
    KnowledgeSheafV2,
    train_restriction_maps,
    score_rendered_triple_v2,
    score_rendered_triples_v2,
    combined_detector_score,
)
from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve


CORPUS_PATH = REPO / "scripts" / "bench" / "corpora" / "seed_long_paragraphs.json"


# ── Sieve extraction over the corpus ──────────────────────────────────


def extract_corpus_triples() -> list[tuple[str, list[tuple[str, str, str]]]]:
    """Sieve-extract triples for each doc; return [(doc_id, triples), ...]."""
    with open(CORPUS_PATH) as f:
        data = json.load(f)
    docs = data["documents"]
    sieve = DeterministicSieve()
    out: list[tuple[str, list[tuple[str, str, str]]]] = []
    for d in docs:
        triples = list(sieve.extract_triplets(d["text"]))
        if not triples:
            continue
        out.append((d["id"], triples))
    return out


# ── Perturbation generators ───────────────────────────────────────────


def generate_a1_entity_swap(
    source_triples: list[tuple[str, str, str]],
    all_entities: list[str],
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    """Swap one subject with a different in-vocab entity."""
    if len(source_triples) == 0:
        return []
    idx = rng.randrange(len(source_triples))
    h, r, t = source_triples[idx]
    candidates = [e for e in all_entities if e != h and e != t]
    if not candidates:
        return list(source_triples)
    h_new = rng.choice(candidates)
    out = list(source_triples)
    out[idx] = (h_new, r, t)
    return out


def generate_a2_predicate_flip(
    source_triples: list[tuple[str, str, str]],
    all_relations: list[str],
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    """Flip one triple's predicate to a different in-vocab predicate."""
    if len(source_triples) == 0 or len(all_relations) < 2:
        return list(source_triples)
    idx = rng.randrange(len(source_triples))
    h, r, t = source_triples[idx]
    candidates = [p for p in all_relations if p != r]
    p_new = rng.choice(candidates)
    out = list(source_triples)
    out[idx] = (h, p_new, t)
    return out


def generate_a3_off_graph_fabrication(
    source_triples: list[tuple[str, str, str]],
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    """Append a triple with an out-of-vocabulary relation."""
    out = list(source_triples)
    fabricated = (
        f"FABRICATED_ENTITY_{rng.randint(0, 999)}",
        f"FABRICATED_RELATION_{rng.randint(0, 999)}",
        f"FABRICATED_OBJECT_{rng.randint(0, 999)}",
    )
    out.append(fabricated)
    return out


def generate_a4_triple_drop(
    source_triples: list[tuple[str, str, str]],
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    """Drop one triple."""
    if len(source_triples) <= 1:
        return list(source_triples)
    idx = rng.randrange(len(source_triples))
    return [t for i, t in enumerate(source_triples) if i != idx]


# ── ROC AUC ───────────────────────────────────────────────────────────


def roc_auc(scores: list[float], labels: list[int]) -> float:
    """Standard binary ROC AUC via the rank-sum identity.

    scores: detector scores (higher = more perturbed-suspect).
    labels: 0 = clean, 1 = perturbed.

    Returns AUC ∈ [0, 1]. 0.5 = random; 1.0 = perfect; <0.5 = signal
    is in the wrong direction.
    """
    n = len(scores)
    if n == 0:
        return 0.5
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    # Mann-Whitney U / (|pos| * |neg|) = AUC
    u = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                u += 1.0
            elif sp == sn:
                u += 0.5
    return u / (len(pos) * len(neg))


# ── Main bench ─────────────────────────────────────────────────────────


def main() -> dict[str, Any]:
    rng = random.Random(0)

    print("=" * 72)
    print("v2.x ROC bench — Path 1 (synthetic perturbations on real corpus)")
    print("=" * 72)

    print("\n[1] Sieve-extracting triples per doc from seed_long_paragraphs…")
    corpus = extract_corpus_triples()
    print(f"    {len(corpus)} docs with non-empty extractions")
    total_triples = sum(len(t) for _, t in corpus)
    print(f"    {total_triples} total source triples")

    # Build the union vocabulary (transductive: train sees every
    # entity/relation that will appear in any in-vocab perturbation).
    all_triples = [t for _, ts in corpus for t in ts]
    all_entities = sorted({e for h, _, t in all_triples for e in (h, t)})
    all_relations = sorted({r for _, r, _ in all_triples})
    print(f"    union vocab: {len(all_entities)} entities, {len(all_relations)} relations")

    print("\n[2] Training v2.1 sheaf on union vocabulary…")
    trained, embeddings, history = train_restriction_maps(
        all_triples,
        stalk_dim=8,
        epochs=200,
        learning_rate=0.005,
        margin=0.5,
        n_negatives_per_positive=3,
        seed=0,
    )
    print(f"    training loss: first-half mean = {np.mean(history[:100]):.4f}, "
          f"second-half mean = {np.mean(history[100:]):.4f}")

    # Per-class score collection.
    # We collect (score, label) pairs across all docs for each class.
    # ── λ auto-calibration ──────────────────────────────────────────
    # First sweep: compute per-doc clean Laplacian-per-edge mean, then
    # pick λ such that 1 missing entity ≈ 1 average per-edge Laplacian
    # contribution. This is the principled calibration that scales
    # with corpus size; the v2.2 default λ=0.05 was tuned on a toy
    # 4-fact graph and does NOT transfer to corpora where the
    # Laplacian magnitude is 10×-50× larger.
    print("\n[2.5] Auto-calibrating λ from corpus Laplacian statistics…")
    per_edge_means: list[float] = []
    for doc_id, source in corpus:
        if len(source) == 0:
            continue
        try:
            doc_sheaf = KnowledgeSheafV2.from_triples(source, stalk_dim=8)
            doc_emb = np.zeros((len(doc_sheaf.vertices), 8), dtype=np.float64)
            for i, v in enumerate(doc_sheaf.vertices):
                if v in trained.vertex_index:
                    doc_emb[i] = embeddings[trained.vertex_index[v]]
            clean = combined_detector_score(doc_sheaf, doc_emb, source)
            per_edge_means.append(clean["v_laplacian"] / max(len(source), 1))
        except (ValueError, KeyError):
            pass
    lambda_auto = float(np.mean(per_edge_means)) if per_edge_means else 0.05
    print(f"    per-edge mean Laplacian across {len(per_edge_means)} docs: "
          f"{lambda_auto:.4f}")
    print(f"    auto-calibrated λ = {lambda_auto:.4f} "
          f"(default v2.2 λ=0.05 was {0.05/lambda_auto:.0%} of this; "
          f"too small for this corpus scale).")

    print("\n[3] Generating perturbations + scoring…")
    class_scores: dict[str, list[tuple[float, int]]] = {
        "A1_entity_swap": [],
        "A2_predicate_flip": [],
        "A3_off_graph_fabrication": [],
        "A4_triple_drop": [],
    }

    for doc_id, source in corpus:
        # Score the clean render once per doc
        clean_combined = combined_detector_score(trained, embeddings, source)
        clean_per_triple = score_rendered_triples_v2(trained, embeddings, source)
        # Per-class detection score for clean (label = 0)
        clean_max_v = clean_per_triple["max_in_vocab_v"] or 0.0
        clean_oov = clean_per_triple["n_oov"]

        # A1: detection signal = max in-vocab V on render
        a1 = generate_a1_entity_swap(source, all_entities, rng)
        a1_per_triple = score_rendered_triples_v2(trained, embeddings, a1)
        a1_max_v = a1_per_triple["max_in_vocab_v"] or 0.0
        class_scores["A1_entity_swap"].append((clean_max_v, 0))
        class_scores["A1_entity_swap"].append((a1_max_v, 1))

        # A2: same — max in-vocab V signals predicate-flip
        a2 = generate_a2_predicate_flip(source, all_relations, rng)
        a2_per_triple = score_rendered_triples_v2(trained, embeddings, a2)
        a2_max_v = a2_per_triple["max_in_vocab_v"] or 0.0
        class_scores["A2_predicate_flip"].append((clean_max_v, 0))
        class_scores["A2_predicate_flip"].append((a2_max_v, 1))

        # A3: detection signal = oov count
        a3 = generate_a3_off_graph_fabrication(source, rng)
        a3_per_triple = score_rendered_triples_v2(trained, embeddings, a3)
        a3_oov = a3_per_triple["n_oov"]
        class_scores["A3_off_graph_fabrication"].append((float(clean_oov), 0))
        class_scores["A3_off_graph_fabrication"].append((float(a3_oov), 1))

        # A4: detection signal = combined (Laplacian + presence-deficit)
        # Note: combined_detector_score uses sheaf as the SOURCE-graph
        # restriction structure. For per-doc evaluation we'd need a
        # per-doc sheaf. Approximate: compute deficit on the trained
        # sheaf's vertex set using the source's vertices as expected.
        # Cleaner approach: build a per-doc sheaf for combined detector.
        # For Path-1 expediency, use a per-doc sheaf via from_triples.
        try:
            doc_sheaf = KnowledgeSheafV2.from_triples(source, stalk_dim=8)
            doc_emb = np.zeros((len(doc_sheaf.vertices), 8), dtype=np.float64)
            for i, v in enumerate(doc_sheaf.vertices):
                if v in trained.vertex_index:
                    doc_emb[i] = embeddings[trained.vertex_index[v]]
            clean_score = combined_detector_score(
                doc_sheaf, doc_emb, source, presence_weight=lambda_auto,
            )
            a4 = generate_a4_triple_drop(source, rng)
            a4_score = combined_detector_score(
                doc_sheaf, doc_emb, a4, presence_weight=lambda_auto,
            )
            class_scores["A4_triple_drop"].append((clean_score["v_combined"], 0))
            class_scores["A4_triple_drop"].append((a4_score["v_combined"], 1))
        except (ValueError, KeyError):
            # Skip docs where the per-doc sheaf can't be built (e.g.,
            # restriction-map shape mismatch under sub-vocabulary).
            pass

    # Compute AUC per class.
    print("\n[4] Per-class ROC AUC:")
    print(f"    {'class':<28} {'n_pairs':>8} {'AUC':>8}")
    print("    " + "-" * 46)
    aucs: dict[str, float] = {}
    for cls, pairs in class_scores.items():
        scores = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]
        auc = roc_auc(scores, labels)
        aucs[cls] = auc
        marker = "✓" if auc >= 0.75 else ("~" if auc >= 0.5 else "✗")
        print(f"    {cls:<28} {len(pairs):>8} {auc:>8.3f}  {marker}")

    overall = float(np.mean(list(aucs.values())))
    print(f"\n    Overall (mean across classes): {overall:.3f}")

    print("\n[5] Honest verdict:")
    if overall >= 0.75:
        print(f"    ✓ Overall AUC {overall:.3f} ≥ 0.75 (P1 target met).")
    else:
        print(f"    ⚠ Overall AUC {overall:.3f} < 0.75 (P1 target unmet).")
        below = [c for c, a in aucs.items() if a < 0.75]
        print(f"    Classes below 0.75: {below}")

    return {
        "schema": "sum.sheaf_v2_roc_bench.v1",
        "corpus": "seed_long_paragraphs",
        "n_docs": len(corpus),
        "n_source_triples": total_triples,
        "vocab_size_entities": len(all_entities),
        "vocab_size_relations": len(all_relations),
        "stalk_dim": 8,
        "training_epochs": 200,
        "lambda_auto_calibrated": lambda_auto,
        "lambda_default_v2_2": 0.05,
        "per_class_auc": aucs,
        "overall_auc": overall,
        "p1_target": 0.75,
        "p1_met": overall >= 0.75,
    }


if __name__ == "__main__":
    result = main()
    print("\n[6] Receipt JSON:")
    print(json.dumps(result, indent=2))
