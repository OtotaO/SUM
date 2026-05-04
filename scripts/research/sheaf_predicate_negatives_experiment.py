"""
Carmack-recovery option 2: add predicate-perturbation negatives to the
contrastive training loop, retrain, re-bench v3.2.

Hypothesis: A2 = 0.500 across all v3.x detectors is a training-distribution
gap, not a structural detector limitation. The current LCWA sampler does
tail-perturbation only — (h, r, t) → (h, r, t'). The trained restriction
maps F_h, F_t never see predicate-flip negatives during training, so they
have no specific reason to score (h, r', t) as high V.

Adding (h, r, t) → (h, r', t) negatives at training time should raise A2
AUC for v3.x detectors. Baselines (B1/B2) don't train, so they stay at
A2 = 0.500. If the experiment works, v3.x has at least one column where
it strictly beats baselines.

Methodology: copy the v2 training loop locally (no production change),
add the predicate-perturbation sampler, retrain, run the v3.2 bench
loop on the new sheaf.

Output: fixtures/bench_receipts/predicate_negatives_experiment_<DATE>.json
        schema: sum.sheaf_predicate_negatives_experiment.v1
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
    KnowledgeSheafV2,
    combined_detector_score,
)
from sum_engine_internal.research.sheaf_laplacian_v3 import (
    weights_from_receipts,
)
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
from scripts.research.sheaf_baseline_comparison import (
    score_b2_jaccard_distance,
)

REPO = Path(__file__).resolve().parents[2]
RECEIPTS_DIR = REPO / "fixtures" / "bench_receipts"

Triple = tuple[str, str, str]


def _sample_mixed_negatives(
    triples: list[Triple],
    n_tail_negatives: int,
    n_predicate_negatives: int,
    rng: np.random.Generator,
) -> list[Triple]:
    """For each positive (h, r, t), produce:
       - n_tail_negatives via tail-perturb: (h, r, t')
       - n_predicate_negatives via predicate-flip: (h, r', t)
    Returns list of length len(triples) * (n_tail + n_predicate),
    flattened in (positive_index, neg_kind, neg_idx) order so that the
    training loop's slice-pairing still works.
    """
    pos_set = set(triples)
    all_entities = sorted({e for h, _, t in triples for e in (h, t)})
    all_relations = sorted({r for _, r, _ in triples})
    if len(all_entities) < 2 and len(all_relations) < 2:
        return []
    out: list[Triple] = []
    for (h, r, t) in triples:
        # Tail negatives
        tails: list[Triple] = []
        for _ in range(n_tail_negatives):
            for _attempt in range(10):
                t_prime = all_entities[int(rng.integers(0, len(all_entities)))]
                if t_prime != t and (h, r, t_prime) not in pos_set:
                    tails.append((h, r, t_prime))
                    break
        # Predicate negatives
        preds: list[Triple] = []
        if len(all_relations) >= 2:
            for _ in range(n_predicate_negatives):
                for _attempt in range(10):
                    r_prime = all_relations[int(rng.integers(0, len(all_relations)))]
                    if r_prime != r and (h, r_prime, t) not in pos_set:
                        preds.append((h, r_prime, t))
                        break
        out.extend(tails + preds)
    return out


def train_with_predicate_negatives(
    triples: list[Triple],
    stalk_dim: int = 8,
    epochs: int = 200,
    learning_rate: float = 0.005,
    margin: float = 0.5,
    n_tail_negatives: int = 3,
    n_predicate_negatives: int = 3,
    seed: int = 0,
) -> tuple[KnowledgeSheafV2, np.ndarray]:
    """Local copy of v2 training loop with predicate-flip negatives added.
    Mirrors sum_engine_internal.research.sheaf_laplacian_v2.train_restriction_maps
    SGD step exactly; only the negative sampler differs.
    """
    rng = np.random.default_rng(seed)
    sheaf = KnowledgeSheafV2.from_triples(triples, stalk_dim=stalk_dim)
    n_v = len(sheaf.vertices)
    d = stalk_dim
    if d >= n_v:
        embeddings = np.eye(n_v, d, dtype=np.float64)
    else:
        embeddings = rng.standard_normal((n_v, d))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9

    F_h = sheaf.F_h.copy()
    F_t = sheaf.F_t.copy()
    n_neg_per_pos = n_tail_negatives + n_predicate_negatives

    for epoch in range(epochs):
        negatives = _sample_mixed_negatives(
            list(triples), n_tail_negatives, n_predicate_negatives, rng,
        )
        for i, (h, r, t) in enumerate(triples):
            r_idx = sheaf.relation_index[r]
            h_idx = sheaf.vertex_index[h]
            t_idx = sheaf.vertex_index[t]
            res_pos = F_h[r_idx] @ embeddings[h_idx] - F_t[r_idx] @ embeddings[t_idx]
            v_pos = float(np.dot(res_pos, res_pos))

            neg_slice = negatives[i * n_neg_per_pos : (i + 1) * n_neg_per_pos]
            for (h_n, r_n, t_n) in neg_slice:
                rn_idx = sheaf.relation_index[r_n]
                hn_idx = sheaf.vertex_index[h_n]
                tn_idx = sheaf.vertex_index[t_n]
                res_neg = F_h[rn_idx] @ embeddings[hn_idx] - F_t[rn_idx] @ embeddings[tn_idx]
                v_neg = float(np.dot(res_neg, res_neg))
                gap = v_pos + margin - v_neg
                if gap > 0:
                    # SGD updates — same form as v2 training
                    grad_F_h_pos = 2 * np.outer(res_pos, embeddings[h_idx])
                    grad_F_t_pos = -2 * np.outer(res_pos, embeddings[t_idx])
                    grad_emb_h_pos = 2 * (F_h[r_idx].T @ res_pos)
                    grad_emb_t_pos = -2 * (F_t[r_idx].T @ res_pos)

                    grad_F_h_neg = 2 * np.outer(res_neg, embeddings[hn_idx])
                    grad_F_t_neg = -2 * np.outer(res_neg, embeddings[tn_idx])
                    grad_emb_h_neg = 2 * (F_h[rn_idx].T @ res_neg)
                    grad_emb_t_neg = -2 * (F_t[rn_idx].T @ res_neg)

                    F_h[r_idx] -= learning_rate * grad_F_h_pos
                    F_t[r_idx] -= learning_rate * grad_F_t_pos
                    embeddings[h_idx] -= learning_rate * grad_emb_h_pos
                    embeddings[t_idx] -= learning_rate * grad_emb_t_pos

                    F_h[rn_idx] += learning_rate * grad_F_h_neg
                    F_t[rn_idx] += learning_rate * grad_F_t_neg
                    embeddings[hn_idx] += learning_rate * grad_emb_h_neg
                    embeddings[tn_idx] += learning_rate * grad_emb_t_neg

    trained = KnowledgeSheafV2(
        vertices=sheaf.vertices,
        edges=sheaf.edges,
        relations=sheaf.relations,
        vertex_index=sheaf.vertex_index,
        relation_index=sheaf.relation_index,
        edge_relation=sheaf.edge_relation,
        stalk_dim=sheaf.stalk_dim,
        F_h=F_h,
        F_t=F_t,
    )
    return trained, embeddings


def run_experiment() -> dict[str, Any]:
    rng = random.Random(0)
    print("=" * 72)
    print("Predicate-perturbation training experiment — does A2 unlock?")
    print("=" * 72)

    print("\n[1] Sieve-extracting triples…")
    corpus = extract_corpus_triples()
    all_triples = [t for _, ts in corpus for t in ts]
    all_entities = sorted({e for h, _, t in all_triples for e in (h, t)})
    all_relations = sorted({r for _, r, _ in all_triples})
    print(f"    {len(corpus)} docs, {len(all_triples)} triples, "
          f"{len(all_entities)} entities, {len(all_relations)} relations")

    print("\n[2] Training with mixed negatives (3 tail + 3 predicate per pos)…")
    print("    (longer runtime than baseline — 6 negs/pos vs 3)…")
    trained, embeddings = train_with_predicate_negatives(
        all_triples,
        stalk_dim=8, epochs=200, learning_rate=0.005, margin=0.5,
        n_tail_negatives=3, n_predicate_negatives=3, seed=0,
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

    GAMMA = 0.1
    print(f"\n[4] Per-doc scoring (γ={GAMMA}, mirrors v3.2 main)…")
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

        clean_v32 = score_v32_combined(
            doc_sheaf, doc_emb, source, weights, lambda_auto, GAMMA,
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
                    p_v32 = score_v32_combined(
                        doc_sheaf, doc_emb, perturbed, weights, lambda_auto, GAMMA,
                    )
                except Exception:  # noqa: BLE001
                    continue
                p_b2 = score_b2_jaccard_distance(source, perturbed)
                cells.setdefault(("v32_g0.1_pred_neg", cls, target_label), []).extend([
                    (clean_v32, 0), (p_v32, 1),
                ])
                cells.setdefault(("b2_jaccard", cls, target_label), []).extend([
                    (clean_b2, 0), (p_b2, 1),
                ])
        n_with_partition += 1

    print(f"    docs with partition: {n_with_partition}")

    print("\n[5] Per-cell AUC:")
    per_cell_auc: dict[str, float] = {}
    for (det, cls, tgt), pairs in sorted(cells.items()):
        scores = [s for s, _ in pairs]
        labels = [l for _, l in pairs]
        cell = f"{det}|{cls}|{tgt}"
        per_cell_auc[cell] = roc_auc(scores, labels)
        print(f"    {cell:55s} = {per_cell_auc[cell]:.3f}")

    print("\n[6] Trusted-mean AUC:")
    trusted_means: dict[str, float] = {}
    for det in ("v32_g0.1_pred_neg", "b2_jaccard"):
        aucs = [auc for k, auc in per_cell_auc.items()
                if k.startswith(f"{det}|") and k.endswith("|trusted")]
        trusted_means[det] = sum(aucs) / len(aucs) if aucs else 0.0
        print(f"    {det:25s} = {trusted_means[det]:.3f}")

    print("\n[7] A2 lift comparison vs published v3.2 numbers:")
    a2_t_new = per_cell_auc.get("v32_g0.1_pred_neg|A2|trusted", 0.0)
    a2_u_new = per_cell_auc.get("v32_g0.1_pred_neg|A2|untrusted", 0.0)
    print(f"    v3.2 A2 trusted   (published): 0.500   →  with pred-neg: {a2_t_new:.3f}")
    print(f"    v3.2 A2 untrusted (published): 0.500   →  with pred-neg: {a2_u_new:.3f}")
    print(f"    B2 A2 trusted   (always):      0.500")
    print(f"    B2 A2 untrusted (always):      0.500")

    if a2_t_new >= 0.70 or a2_u_new >= 0.70:
        verdict = "A2_RECOVERED"
    elif a2_t_new >= 0.60 or a2_u_new >= 0.60:
        verdict = "A2_PARTIAL"
    else:
        verdict = "A2_STILL_CHANCE"
    print(f"\n[8] Verdict: {verdict}")

    report = {
        "schema": "sum.sheaf_predicate_negatives_experiment.v1",
        "corpus": "seed_long_paragraphs",
        "n_docs_total": len(corpus),
        "n_docs_with_partition": n_with_partition,
        "training": {
            "n_tail_negatives": 3,
            "n_predicate_negatives": 3,
            "stalk_dim": 8,
            "epochs": 200,
            "learning_rate": 0.005,
            "margin": 0.5,
            "seed": 0,
        },
        "lambda_auto": lambda_auto,
        "gamma_used": GAMMA,
        "per_cell_auc": per_cell_auc,
        "trusted_mean_auc_by_detector": trusted_means,
        "a2_trusted_with_pred_neg": a2_t_new,
        "a2_untrusted_with_pred_neg": a2_u_new,
        "a2_published_v3_2": 0.500,
        "verdict": verdict,
        "method_notes": (
            "Local copy of v2 train_restriction_maps with mixed negatives "
            "(3 tail + 3 predicate per positive). Same hyperparams "
            "otherwise (stalk_dim=8, epochs=200, lr=0.005, margin=0.5, "
            "seed=0). Bench loop matches sheaf_v3_2_validation.main: "
            "one trusted + one untrusted target per doc, stable_seed "
            "partition. B2 jaccard runs on the SAME (clean, perturbed) "
            "pairs as a control."
        ),
    }
    quantized = quantize_for_digest(report)
    report["bench_digest"] = compute_bench_digest(quantized)
    print(f'\n  "bench_digest": "{report["bench_digest"]}"')
    return report


def main() -> dict[str, Any]:
    report = run_experiment()
    today = _dt.date.today().isoformat()
    out = RECEIPTS_DIR / f"predicate_negatives_experiment_{today}.json"
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n→ wrote {out.relative_to(REPO)}")
    return report


if __name__ == "__main__":
    main()
