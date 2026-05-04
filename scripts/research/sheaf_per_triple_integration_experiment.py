"""
Carmack-recovery option 2.5: integrate the per-rendered-triple V channel
back into v3.x scoring. This is the channel v2.2 §4.3 ROC bench used to
hit A1/A2/A3 = 1.000 — it scores each rendered triple individually under
the trained sheaf, surfacing predicate-flips, off-graph, and entity-swap
violations directly.

The v3.x cochain-on-source-graph channel is structurally blind to A2
(predicate flip preserves entity set → cochain unchanged → V unchanged).
The per-rendered-triple channel scores each (h, r, t) in the render
against F_h(r), F_t(r), so an A2 perturbation (h, r → r', t) lands at
F_h(r'), F_t(r') with high V_triple.

Combined v3.2 + per-triple score (one possible composition):

    V_combined =  v_laplacian_w
               + γ_dev · deviation_w
               + λ_def · v_deficit
               + α     · max_in_vocab_v_triple
               + β     · n_oov

where the last two terms are the §3.5 per-rendered-triple channel
aggregates. α, β chosen at moderate scale to match v_laplacian_w
magnitude (~10).

Output: fixtures/bench_receipts/per_triple_integration_<DATE>.json
        schema: sum.sheaf_per_triple_integration.v1
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
    score_rendered_triples_v2,
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
from scripts.research.sheaf_baseline_comparison import (
    score_b2_jaccard_distance,
)

REPO = Path(__file__).resolve().parents[2]
RECEIPTS_DIR = REPO / "fixtures" / "bench_receipts"

GAMMA = 0.1
ALPHA = 1.0   # weight on max_in_vocab_v_triple
BETA = 1.0    # weight on n_oov


def score_v32_with_per_triple(
    doc_sheaf,
    doc_emb,
    render_triples,
    weights,
    lambda_: float,
    gamma: float,
    alpha: float = ALPHA,
    beta: float = BETA,
    global_sheaf=None,
    global_embeddings=None,
) -> float:
    """v3.2 cochain V + α·max_per_triple_V + β·n_oov.

    The per-triple V is computed against the *global* trained sheaf so
    OOV detection is informative (the per-doc sheaf has the same vocab
    as the source — every source triple is in-vocab by construction).
    """
    cochain_score = score_v32_combined(
        doc_sheaf, doc_emb, render_triples, weights, lambda_, gamma,
    )
    if global_sheaf is None or global_embeddings is None:
        per_triple_summary = score_rendered_triples_v2(
            doc_sheaf, doc_emb, list(render_triples),
        )
    else:
        per_triple_summary = score_rendered_triples_v2(
            global_sheaf, global_embeddings, list(render_triples),
        )
    max_v = per_triple_summary["max_in_vocab_v"]
    n_oov = per_triple_summary["n_oov"]
    per_triple_term = (max_v if max_v is not None else 0.0)
    return float(cochain_score + alpha * per_triple_term + beta * n_oov)


def run_experiment() -> dict[str, Any]:
    rng = random.Random(0)
    print("=" * 72)
    print("Per-rendered-triple V integration experiment")
    print("Hypothesis: adding §3.5 per-triple channel to v3.2 unlocks A2")
    print("=" * 72)

    print("\n[1] Sieve-extracting triples…")
    corpus = extract_corpus_triples()
    all_triples = [t for _, ts in corpus for t in ts]
    all_entities = sorted({e for h, _, t in all_triples for e in (h, t)})
    all_relations = sorted({r for _, r, _ in all_triples})
    print(f"    {len(corpus)} docs, {len(all_triples)} triples")

    print("\n[2] Training v2.1 sheaf (matches v3.2 main hyperparams)…")
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

    print(f"\n[4] Per-doc scoring (γ={GAMMA}, α={ALPHA}, β={BETA})…")
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

        clean_v32_only = score_v32_combined(
            doc_sheaf, doc_emb, source, weights, lambda_auto, GAMMA,
        )
        clean_v32_plus_pt = score_v32_with_per_triple(
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
                    p_v32_only = score_v32_combined(
                        doc_sheaf, doc_emb, perturbed, weights, lambda_auto, GAMMA,
                    )
                    p_v32_plus_pt = score_v32_with_per_triple(
                        doc_sheaf, doc_emb, perturbed, weights, lambda_auto, GAMMA,
                        global_sheaf=trained, global_embeddings=embeddings,
                    )
                except Exception:  # noqa: BLE001
                    continue
                p_b2 = score_b2_jaccard_distance(source, perturbed)

                cells.setdefault(("v32_g0.1_only", cls, target_label), []).extend([
                    (clean_v32_only, 0), (p_v32_only, 1),
                ])
                cells.setdefault(("v32_g0.1_plus_per_triple", cls, target_label), []).extend([
                    (clean_v32_plus_pt, 0), (p_v32_plus_pt, 1),
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
    for det in ("v32_g0.1_only", "v32_g0.1_plus_per_triple", "b2_jaccard"):
        aucs = [auc for k, auc in per_cell_auc.items()
                if k.startswith(f"{det}|") and k.endswith("|trusted")]
        trusted_means[det] = sum(aucs) / len(aucs) if aucs else 0.0
        print(f"    {det:35s} = {trusted_means[det]:.3f}")

    delta_vs_baseline = (
        trusted_means["v32_g0.1_plus_per_triple"] - trusted_means["b2_jaccard"]
    )
    if delta_vs_baseline >= 0.03:
        verdict = "DETECTOR_BEATS_BASELINE"
    elif delta_vs_baseline >= -0.02:
        verdict = "DETECTOR_TIES_BASELINE"
    else:
        verdict = "DETECTOR_LOSES_TO_BASELINE"
    print(f"\n[7] Δ(v3.2+per_triple − b2_jaccard) = {delta_vs_baseline:+.3f}"
          f" → {verdict}")

    report = {
        "schema": "sum.sheaf_per_triple_integration.v1",
        "corpus": "seed_long_paragraphs",
        "n_docs_total": len(corpus),
        "n_docs_with_partition": n_with_partition,
        "lambda_auto": lambda_auto,
        "gamma_used": GAMMA,
        "alpha": ALPHA,
        "beta": BETA,
        "detectors": ["v32_g0.1_only", "v32_g0.1_plus_per_triple", "b2_jaccard"],
        "perturbation_classes": ["A1", "A2", "A4"],
        "per_cell_auc": per_cell_auc,
        "trusted_mean_auc_by_detector": trusted_means,
        "delta_v32_plus_per_triple_vs_b2": delta_vs_baseline,
        "verdict": verdict,
        "method_notes": (
            "v32_g0.1_plus_per_triple = v3.2(γ=0.1) cochain-on-source-graph "
            "score + α · max_in_vocab_v_triple + β · n_oov. The per-triple "
            "channel from §3.5 is computed against the GLOBAL trained sheaf "
            "(not per-doc sheaf) so OOV detection is informative. α=β=1.0 "
            "chosen as moderate scale; production tuning is per-corpus. "
            "The cochain channel alone misses A2 by structural necessity "
            "(predicate-flip preserves entity set → cochain unchanged); "
            "per-triple channel scores (h, r, t) directly under F_h(r), "
            "F_t(r), so r → r' produces high V_triple and recovers A2."
        ),
    }
    quantized = quantize_for_digest(report)
    report["bench_digest"] = compute_bench_digest(quantized)
    print(f'\n  "bench_digest": "{report["bench_digest"]}"')
    return report


def main() -> dict[str, Any]:
    report = run_experiment()
    today = _dt.date.today().isoformat()
    out = RECEIPTS_DIR / f"per_triple_integration_{today}.json"
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n→ wrote {out.relative_to(REPO)}")
    return report


if __name__ == "__main__":
    main()
