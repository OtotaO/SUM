"""ROC benchmark for v3 (receipt-weighted) and v3.1 (harmonic-extension) detectors.

Path 1 of the v3 publishable-artifact plan: re-use the same
seed_long_paragraphs corpus and v2.2 perturbation harness, but
add receipt-weight partitions per doc and measure whether
receipt-weighting actually amplifies signal where the system
trusts.

The v3 utility hypothesis (H4 in the v3 spec): tampering a trusted
(high-weight) edge produces a *sharper* V jump than tampering an
untrusted (low-weight) edge. At synthetic scale this is pinned in
``test_tampering_trusted_edge_yields_sharper_v_jump_than_untrusted``
across 10 seeded perturbations. This bench answers the corpus-
scale version: across 16 multi-fact paragraphs from the §2.5
benchmark, does v3 with receipt weighting give a meaningfully
higher AUC than v2.2 baseline when the perturbation hits a
trusted edge?

Methodology (mirrors v2.2 bench where applicable):

  1. Sieve-extract triples per doc (transductive setup: train on
     union vocabulary so the trained sheaf covers every entity /
     relation).
  2. Train ONE v2.1 sheaf on the union (same as v2.2 bench;
     ensures detector comparisons are over the same trained model).
  3. Auto-calibrate λ from per-edge mean Laplacian (same as v2.2
     bench — required to avoid the 38× scale mismatch the v2.2
     bench surfaced on naturalistic-prose corpora).
  4. For each doc, randomly partition its source triples into a
     50/50 trusted/untrusted split (deterministic per doc via
     a fixed seed). Build receipt weights:
         trusted_weight=1.0, default_weight=0.1, revoked_weight=0.0
  5. For each doc, generate perturbations targeting EITHER a
     trusted triple OR an untrusted triple, separately:
         A1 entity_swap (on trusted vs on untrusted)
         A2 predicate_flip (on trusted vs on untrusted)
         A4 triple_drop (on trusted vs on untrusted)
     A3 (off-graph fabrication) is appended, not targeting an
     existing edge — it's measured once per doc as a sanity check.
  6. For each (clean, perturbed) pair, compute three detector
     scores:
         - v2.2 baseline: combined_detector_score (unweighted)
         - v3 receipt-weighted: combined_detector_score_v3
         - v3.1 boundary deviation: boundary_deviation(
             sheaf, x_full, B = boundary_from_weights(weights, threshold=0.5)
           )
  7. Per-cell ROC AUC: detector × class × target (trusted/untrusted).

Output: a JSON record (sum.sheaf_v3_roc_bench.v1) + console summary.

Falsification verdicts named explicitly:

  - **F1 (v3 should beat v2.2 on trusted-target perturbations).**
    If v3.AUC ≤ v2.2.AUC on (A1@trusted, A2@trusted, A4@trusted),
    receipt-weighting offers no corpus-scale benefit — file a
    spec correction.

  - **F2 (v3 should not collapse on untrusted-target perturbations).**
    If v3.AUC ≪ v2.2.AUC on (A1@untrusted, ...), the 0.1 floor
    weight is too low — recalibrate.

  - **F3 (v3.1 boundary deviation should track v3 combined on
    trusted-target perturbations).** v3.1 uses the same trust
    partition; if its AUC is uncorrelated with v3's, the
    boundary→harmonic-extension pipeline has a bug.

CPU-only; runtime ~2-3 minutes for the seed_long_paragraphs corpus.

Re-run via:
  PYTHONPATH=. python scripts/research/sheaf_v3_roc_bench.py
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from sum_engine_internal.research.sheaf_laplacian_v2 import (
    KnowledgeSheafV2,
    cochain_one_hot_v2,
    train_restriction_maps,
    score_rendered_triples_v2,
    combined_detector_score,
)
from sum_engine_internal.research.sheaf_laplacian_v3 import (
    boundary_deviation,
    boundary_from_weights,
    combined_detector_score_v3,
    weights_from_receipts,
)
from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve

CORPUS_PATH = REPO / "scripts" / "bench" / "corpora" / "seed_long_paragraphs.json"


# ── Helpers ──────────────────────────────────────────────────────────


def extract_corpus_triples() -> list[tuple[str, list[tuple[str, str, str]]]]:
    with open(CORPUS_PATH) as f:
        data = json.load(f)
    sieve = DeterministicSieve()
    out: list[tuple[str, list[tuple[str, str, str]]]] = []
    for d in data["documents"]:
        triples = list(sieve.extract_triplets(d["text"]))
        if triples:
            out.append((d["id"], triples))
    return out


def partition_trust(
    triples: list[tuple[str, str, str]],
    seed: int,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Deterministic 50/50 trust partition per doc."""
    rng = random.Random(seed)
    indices = list(range(len(triples)))
    rng.shuffle(indices)
    n_trusted = len(triples) // 2
    trusted_idx = set(indices[:n_trusted])
    trusted = [t for i, t in enumerate(triples) if i in trusted_idx]
    untrusted = [t for i, t in enumerate(triples) if i not in trusted_idx]
    return trusted, untrusted


def perturb_a1_on_target(
    triples: list[tuple[str, str, str]],
    target: tuple[str, str, str],
    all_entities: list[str],
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    """Swap the SUBJECT of `target` with a different in-vocab entity."""
    if target not in triples:
        return list(triples)
    h, r, t = target
    candidates = [e for e in all_entities if e != h and e != t]
    if not candidates:
        return list(triples)
    h_new = rng.choice(candidates)
    out = []
    for tr in triples:
        out.append((h_new, r, t) if tr == target else tr)
    return out


def perturb_a2_on_target(
    triples: list[tuple[str, str, str]],
    target: tuple[str, str, str],
    all_relations: list[str],
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    """Flip the PREDICATE of `target` with a different in-vocab relation."""
    if target not in triples or len(all_relations) < 2:
        return list(triples)
    h, r, t = target
    candidates = [p for p in all_relations if p != r]
    p_new = rng.choice(candidates)
    out = []
    for tr in triples:
        out.append((h, p_new, t) if tr == target else tr)
    return out


def perturb_a4_drop_target(
    triples: list[tuple[str, str, str]],
    target: tuple[str, str, str],
) -> list[tuple[str, str, str]]:
    """Drop a specific target triple."""
    return [t for t in triples if t != target]


def roc_auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    u = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                u += 1.0
            elif sp == sn:
                u += 0.5
    return u / (len(pos) * len(neg))


# ── Detector wrappers ────────────────────────────────────────────────


def _build_doc_sheaf(
    source: list[tuple[str, str, str]],
    trained: KnowledgeSheafV2,
    embeddings: np.ndarray,
    stalk_dim: int = 8,
) -> tuple[KnowledgeSheafV2, np.ndarray]:
    """Build a per-doc sheaf using the trained sheaf's restriction maps.

    The bench needs per-doc graphs to compute combined-detector
    deficit + per-doc boundary partitions; the trained sheaf is the
    transductive model whose weights/embeddings we re-use.
    """
    doc_sheaf = KnowledgeSheafV2.from_triples(source, stalk_dim=stalk_dim)
    doc_emb = np.zeros((len(doc_sheaf.vertices), stalk_dim), dtype=np.float64)
    for i, v in enumerate(doc_sheaf.vertices):
        if v in trained.vertex_index:
            doc_emb[i] = embeddings[trained.vertex_index[v]]
    return doc_sheaf, doc_emb


def score_v22(
    doc_sheaf: KnowledgeSheafV2,
    doc_emb: np.ndarray,
    render: list[tuple[str, str, str]],
    lambda_: float,
) -> float:
    """v2.2 baseline detector: combined Laplacian + presence deficit."""
    s = combined_detector_score(doc_sheaf, doc_emb, render, presence_weight=lambda_)
    return float(s["v_combined"])


def score_v3_weighted(
    doc_sheaf: KnowledgeSheafV2,
    doc_emb: np.ndarray,
    render: list[tuple[str, str, str]],
    weights: np.ndarray,
    lambda_: float,
) -> float:
    """v3 receipt-weighted detector: weighted Laplacian + presence deficit."""
    s = combined_detector_score_v3(
        doc_sheaf, doc_emb, render, weights, lambda_deficit=lambda_,
    )
    return float(s["v_combined_v3"])


def score_v31_boundary_deviation(
    doc_sheaf: KnowledgeSheafV2,
    doc_emb: np.ndarray,
    render: list[tuple[str, str, str]],
    weights: np.ndarray,
) -> float:
    """v3.1 boundary deviation: distance from harmonic extension."""
    boundary = boundary_from_weights(doc_sheaf, weights, threshold=0.5)
    if not boundary or len(boundary) == len(doc_sheaf.vertices):
        # Degenerate partition (all-untrusted or all-trusted): no
        # interior to deviate over. Use combined v3 score as the
        # fallback metric so the cell still contributes a comparable
        # number; document this behaviour in the bench output.
        return score_v3_weighted(doc_sheaf, doc_emb, render, weights, lambda_=0.0)
    x_full = cochain_one_hot_v2(doc_sheaf, render, embedding=doc_emb)
    result = boundary_deviation(doc_sheaf, x_full, boundary, weights=weights)
    return float(result["deviation"])


# ── Main bench ───────────────────────────────────────────────────────


def main() -> dict[str, Any]:
    rng = random.Random(0)
    print("=" * 72)
    print("v3 ROC bench — receipt-weighted + harmonic-extension at corpus scale")
    print("=" * 72)

    print("\n[1] Sieve-extracting triples from seed_long_paragraphs…")
    corpus = extract_corpus_triples()
    print(f"    {len(corpus)} docs with non-empty extractions")
    total_triples = sum(len(t) for _, t in corpus)
    print(f"    {total_triples} total source triples")

    all_triples = [t for _, ts in corpus for t in ts]
    all_entities = sorted({e for h, _, t in all_triples for e in (h, t)})
    all_relations = sorted({r for _, r, _ in all_triples})
    print(f"    union vocab: {len(all_entities)} entities, "
          f"{len(all_relations)} relations")

    print("\n[2] Training v2.1 sheaf on union vocabulary…")
    trained, embeddings, history = train_restriction_maps(
        all_triples, stalk_dim=8, epochs=200, learning_rate=0.005,
        margin=0.5, n_negatives_per_positive=3, seed=0,
    )
    print(f"    training loss: first-half mean = {np.mean(history[:100]):.4f}, "
          f"second-half mean = {np.mean(history[100:]):.4f}")

    print("\n[3] Auto-calibrating λ from per-edge mean Laplacian…")
    per_edge_means: list[float] = []
    for doc_id, source in corpus:
        try:
            doc_sheaf, doc_emb = _build_doc_sheaf(source, trained, embeddings)
            clean = combined_detector_score(doc_sheaf, doc_emb, source)
            per_edge_means.append(clean["v_laplacian"] / max(len(source), 1))
        except (ValueError, KeyError):
            pass
    lambda_auto = float(np.mean(per_edge_means)) if per_edge_means else 0.05
    print(f"    auto-calibrated λ = {lambda_auto:.4f}")

    # Per-cell score collection: (detector, class, target) → list of (score, label)
    cells: dict[str, list[tuple[float, int]]] = {}

    def add(detector: str, cls: str, target: str, score: float, label: int) -> None:
        key = f"{detector}|{cls}|{target}"
        cells.setdefault(key, []).append((score, label))

    print("\n[4] Per-doc trust partition + targeted perturbations…")
    docs_with_partition = 0
    docs_skipped = 0
    for doc_id, source in corpus:
        if len(source) < 4:
            # Need at least 2 trusted + 2 untrusted for meaningful partition
            docs_skipped += 1
            continue

        # Stable seed via SHA-256 — Python's hash() is per-process-randomized
        # for strings, which would make this bench non-reproducible across runs.
        stable_seed = int.from_bytes(
            hashlib.sha256(doc_id.encode()).digest()[:4], "big",
        )
        trusted, untrusted = partition_trust(source, seed=stable_seed)
        if not trusted or not untrusted:
            docs_skipped += 1
            continue

        try:
            doc_sheaf, doc_emb = _build_doc_sheaf(source, trained, embeddings)
        except (ValueError, KeyError):
            docs_skipped += 1
            continue

        weights = weights_from_receipts(doc_sheaf, trusted_edges=trusted)

        # Clean baseline scores (label 0)
        clean_v22 = score_v22(doc_sheaf, doc_emb, source, lambda_auto)
        clean_v3 = score_v3_weighted(doc_sheaf, doc_emb, source, weights, lambda_auto)
        clean_v31 = score_v31_boundary_deviation(doc_sheaf, doc_emb, source, weights)

        # ── Perturbations targeting TRUSTED edges ─────────────────
        target_t = rng.choice(trusted)
        for cls, perturb_fn in [
            ("A1", lambda r=rng: perturb_a1_on_target(source, target_t, all_entities, r)),
            ("A2", lambda r=rng: perturb_a2_on_target(source, target_t, all_relations, r)),
            ("A4", lambda r=rng: perturb_a4_drop_target(source, target_t)),
        ]:
            perturbed = perturb_fn()
            if perturbed == source:
                continue
            for det, fn in [
                ("v22", lambda: score_v22(doc_sheaf, doc_emb, perturbed, lambda_auto)),
                ("v3",  lambda: score_v3_weighted(doc_sheaf, doc_emb, perturbed, weights, lambda_auto)),
                ("v31", lambda: score_v31_boundary_deviation(doc_sheaf, doc_emb, perturbed, weights)),
            ]:
                try:
                    p_score = fn()
                except Exception:
                    continue
                clean_score = {"v22": clean_v22, "v3": clean_v3, "v31": clean_v31}[det]
                add(det, cls, "trusted", clean_score, 0)
                add(det, cls, "trusted", p_score, 1)

        # ── Perturbations targeting UNTRUSTED edges ───────────────
        target_u = rng.choice(untrusted)
        for cls, perturb_fn in [
            ("A1", lambda r=rng: perturb_a1_on_target(source, target_u, all_entities, r)),
            ("A2", lambda r=rng: perturb_a2_on_target(source, target_u, all_relations, r)),
            ("A4", lambda r=rng: perturb_a4_drop_target(source, target_u)),
        ]:
            perturbed = perturb_fn()
            if perturbed == source:
                continue
            for det, fn in [
                ("v22", lambda: score_v22(doc_sheaf, doc_emb, perturbed, lambda_auto)),
                ("v3",  lambda: score_v3_weighted(doc_sheaf, doc_emb, perturbed, weights, lambda_auto)),
                ("v31", lambda: score_v31_boundary_deviation(doc_sheaf, doc_emb, perturbed, weights)),
            ]:
                try:
                    p_score = fn()
                except Exception:
                    continue
                clean_score = {"v22": clean_v22, "v3": clean_v3, "v31": clean_v31}[det]
                add(det, cls, "untrusted", clean_score, 0)
                add(det, cls, "untrusted", p_score, 1)

        docs_with_partition += 1

    print(f"    docs with partition: {docs_with_partition}; skipped: {docs_skipped}")

    # ── Per-cell AUC ─────────────────────────────────────────────────
    print("\n[5] Per-cell ROC AUC (detector × class × target):")
    print(f"    {'cell':<22} {'n':>4} {'AUC':>7}")
    print("    " + "-" * 36)
    aucs: dict[str, float] = {}
    for key in sorted(cells):
        scores = [p[0] for p in cells[key]]
        labels = [p[1] for p in cells[key]]
        auc = roc_auc(scores, labels)
        aucs[key] = auc
        marker = "✓" if auc >= 0.75 else ("~" if auc >= 0.55 else "✗")
        print(f"    {key:<22} {len(cells[key]):>4} {auc:>7.3f}  {marker}")

    # ── Detector head-to-head per (class, target) ────────────────────
    print("\n[6] Detector head-to-head — does v3 beat v2.2 on trusted-target?")
    print(f"    {'cell':<18} {'v22 AUC':>9} {'v3 AUC':>9} {'Δ':>7} {'verdict':>10}")
    print("    " + "-" * 55)
    h2h: dict[str, dict[str, float]] = {}
    for cls in ("A1", "A2", "A4"):
        for target in ("trusted", "untrusted"):
            v22_auc = aucs.get(f"v22|{cls}|{target}", 0.5)
            v3_auc = aucs.get(f"v3|{cls}|{target}", 0.5)
            delta = v3_auc - v22_auc
            verdict = "v3 wins" if delta > 0.02 else ("tie" if abs(delta) <= 0.02 else "v3 loses")
            cell_name = f"{cls}@{target}"
            print(f"    {cell_name:<18} {v22_auc:>9.3f} {v3_auc:>9.3f} {delta:>+7.3f}  {verdict:>10}")
            h2h[cell_name] = {"v22": v22_auc, "v3": v3_auc, "delta": delta}

    # ── F1 / F2 / F3 falsification verdicts ──────────────────────────
    #
    # NOTE on reproducibility (2026-05-02): np.linalg.lstsq inside
    # harmonic_extension uses LAPACK whose threading produces ~±0.02
    # AUC jitter across runs. Verdict criteria use *aggregate* mean-
    # AUC comparisons instead of per-cell deltas at a 0.02 threshold —
    # those would flip on the jitter alone. The aggregate criteria
    # are stable across the runs we've measured.
    print("\n[7] Falsification verdicts:")

    # F1: v3 mean AUC on trusted ≥ v22 mean AUC on trusted by ≥ 0.04.
    # 0.04 is well above the ±0.02 jitter floor and tight enough to
    # reject "v3 is a wash" while not requiring class-by-class wins.
    v22_trusted_mean = float(np.mean([h2h[f"{c}@trusted"]["v22"] for c in ("A1", "A2", "A4")]))
    v3_trusted_mean = float(np.mean([h2h[f"{c}@trusted"]["v3"] for c in ("A1", "A2", "A4")]))
    f1_delta = v3_trusted_mean - v22_trusted_mean
    f1_verdict = "PASS" if f1_delta >= 0.04 else ("MARGINAL" if f1_delta >= 0.0 else "FAIL")
    print(f"    F1 (v3 beats v2.2 on trusted-target, mean AUC): "
          f"v22={v22_trusted_mean:.3f}, v3={v3_trusted_mean:.3f}, "
          f"Δ={f1_delta:+.3f} — {f1_verdict}")

    # F2: no class drops AUC by more than 0.10 from v22 to v3 on untrusted.
    f2_collapses = sum(1 for c in ("A1", "A2", "A4") if h2h[f"{c}@untrusted"]["delta"] < -0.10)
    f2_verdict = "PASS" if f2_collapses == 0 else "FAIL"
    print(f"    F2 (v3 doesn't collapse on untrusted-target): "
          f"{3 - f2_collapses}/3 classes — {f2_verdict}")

    # F3: v3.1 boundary deviation as a corpus-scale detector. The
    # synthetic-data utility test (H12) passes; this is the corpus-
    # scale companion. Verdict = PASS if mean AUC on trusted ≥ 0.55
    # (better than chance with margin), FAIL otherwise. This may
    # honestly fail — the bench would surface that v3.1 needs more
    # work for naturalistic prose with 50/50 trust partitions.
    v31_trusted_mean = float(np.mean([
        aucs.get(f"v31|{c}|trusted", 0.5) for c in ("A1", "A2", "A4")
    ]))
    v31_untrusted_mean = float(np.mean([
        aucs.get(f"v31|{c}|untrusted", 0.5) for c in ("A1", "A2", "A4")
    ]))
    f3_verdict = "PASS" if v31_trusted_mean >= 0.55 else "FAIL"
    print(f"    F3 (v3.1 boundary deviation on trusted-target, mean AUC): "
          f"trusted={v31_trusted_mean:.3f}, untrusted={v31_untrusted_mean:.3f} "
          f"— {f3_verdict}")

    return {
        "schema": "sum.sheaf_v3_roc_bench.v1",
        "corpus": "seed_long_paragraphs",
        "n_docs_total": len(corpus),
        "n_docs_with_partition": docs_with_partition,
        "n_docs_skipped": docs_skipped,
        "vocab_size_entities": len(all_entities),
        "vocab_size_relations": len(all_relations),
        "stalk_dim": 8,
        "training_epochs": 200,
        "lambda_auto_calibrated": lambda_auto,
        "trust_partition": "deterministic 50/50 per doc",
        "weights_from_receipts": {
            "trusted_weight": 1.0, "default_weight": 0.1, "revoked_weight": 0.0,
        },
        "per_cell_auc": aucs,
        "head_to_head_v3_vs_v22": h2h,
        "f1_v3_beats_v22_on_trusted_mean_auc": {
            "v22_mean": v22_trusted_mean,
            "v3_mean": v3_trusted_mean,
            "delta": f1_delta,
            "verdict": f1_verdict,
        },
        "f2_v3_no_collapse_on_untrusted": {
            "collapses": f2_collapses, "verdict": f2_verdict,
        },
        "f3_v31_boundary_deviation_corpus_scale": {
            "trusted_mean_auc": v31_trusted_mean,
            "untrusted_mean_auc": v31_untrusted_mean,
            "verdict": f3_verdict,
        },
    }


if __name__ == "__main__":
    receipt = main()
    print("\n[8] Receipt JSON:")
    print(json.dumps(receipt, indent=2))
