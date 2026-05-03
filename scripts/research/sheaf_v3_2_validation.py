"""v3.2 validation bench — does the F3-fix actually surface signal?

PR #124's v3 corpus ROC bench gave F3 FAIL on v3.1's standalone
``deviation`` field (trusted-mean AUC ≈ 0.50). PR #125's 8-cell
diagnostic settled this as a structural blind spot when the per-doc
graph has ``L_IB = 0``. v3.2 (this PR) is a strict generalization
of v3 that adds the deviation as a complementary term:

    v_combined_v32 = v_laplacian_w + γ · deviation_w + λ · v_deficit

This bench tests v3.2 at multiple ``γ`` values on the same corpus
+ partitions + perturbations as PR #124, and asks two questions:

  F4. Does v3.2 with γ ≥ 0 achieve trusted-mean AUC ≥ 0.55 (the
      same threshold the F3 verdict used)?
  F5. Does v3.2 with γ > 0 achieve trusted-mean AUC ≥ v3's, or
      at least not regress by more than 0.02? Pure subsumption
      (γ = 0) trivially ties with v3, so the question is whether
      γ > 0 *helps* or merely *doesn't hurt*.

Schema: ``sum.sheaf_v3_2_validation.v1``. Carries the same
``bench_digest`` substrate as the F3 diagnostic — JCS-canonical
SHA-256 over a quantized payload (AUCs to 3 decimals, scalar floats
to 4) — so the receipt is reproducibility-checkable across runs.

This bench imports v3 bench's machinery rather than re-implementing
to keep the partition / perturbation / training axes byte-identical.
That's the F4 fairness contract: every detector sees the same
inputs.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from sum_engine_internal.research.sheaf_laplacian_v2 import (
    cochain_one_hot_v2,
    combined_detector_score,
    train_restriction_maps,
)
from sum_engine_internal.research.sheaf_laplacian_v3 import (
    boundary_deviation,
    boundary_from_weights,
    weights_from_receipts,
)
from sum_engine_internal.research.sheaf_laplacian_v32 import (
    combined_detector_score_v32,
)
from sum_engine_internal.infrastructure.jcs import canonicalize as jcs_canonicalize

from scripts.research.sheaf_v3_roc_bench import (  # type: ignore[import-not-found]
    _build_doc_sheaf,
    extract_corpus_triples,
    partition_trust,
    perturb_a1_on_target,
    perturb_a2_on_target,
    perturb_a4_drop_target,
    roc_auc,
    score_v22,
    score_v3_weighted,
    score_v31_boundary_deviation,
)

# γ values to scan. 0.0 is the subsumption point (== v3); 1.0 is the
# default v3.2; auto means calibrate to mean(v_laplacian_w_clean) /
# mean(deviation_w_clean) on this corpus so the two terms contribute
# comparably.
GAMMA_GRID = (0.0, 0.1, 1.0, "auto")

# γ_auto is computed from clean-render statistics; honest fall-back
# is 1.0 if either of the means turns out zero.
GAMMA_AUTO_FALLBACK = 1.0


def score_v32_combined(
    doc_sheaf,
    doc_emb,
    render,
    weights,
    lambda_: float,
    gamma: float,
) -> float:
    """v3.2 combined detector at a specified γ."""
    boundary = boundary_from_weights(doc_sheaf, weights, threshold=0.5)
    s = combined_detector_score_v32(
        doc_sheaf, doc_emb, render, weights,
        lambda_deficit=lambda_, gamma_deviation=gamma,
        boundary_indices=boundary,
    )
    return float(s["v_combined_v32"])


def calibrate_gamma_auto(
    corpus: list[tuple[str, list[tuple[str, str, str]]]],
    trained,
    embeddings: np.ndarray,
) -> float:
    """γ_auto = mean(v_laplacian_w_clean) / mean(deviation_w_clean).

    Computed across docs that produce non-degenerate partitions. The
    intuition: if v_laplacian_w is typically O(10) and deviation_w
    is typically O(1), γ ≈ 10 makes the two terms contribute on
    similar scales. If the means come out zero (e.g. all-degenerate
    corpus), fall back to GAMMA_AUTO_FALLBACK.
    """
    laplacian_means: list[float] = []
    deviation_means: list[float] = []
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
        boundary = boundary_from_weights(doc_sheaf, weights, threshold=0.5)
        if not boundary or len(boundary) == len(doc_sheaf.vertices):
            continue
        clean_score = combined_detector_score_v32(
            doc_sheaf, doc_emb, source, weights,
            gamma_deviation=1.0, boundary_indices=boundary,
        )
        laplacian_means.append(float(clean_score["v_laplacian_w"]))
        deviation_means.append(float(clean_score["deviation_w"]))

    if not laplacian_means or not deviation_means:
        return GAMMA_AUTO_FALLBACK
    mean_lap = float(np.mean(laplacian_means))
    mean_dev = float(np.mean(deviation_means))
    if mean_dev <= 0:
        return GAMMA_AUTO_FALLBACK
    return mean_lap / mean_dev


def quantize_for_digest(report: dict[str, Any]) -> dict[str, Any]:
    """Quantize per-cell AUCs to 3 decimals, scalar floats to 4.

    Same convention as the F3 diagnostic. LAPACK threading inside
    np.linalg.lstsq introduces ~±0.02 AUC jitter across runs;
    quantization absorbs this so the digest is a reproducibility
    witness, not a noise canary.
    """
    def quantize(x: Any, decimals: int) -> Any:
        if isinstance(x, float):
            return round(x, decimals)
        if isinstance(x, dict):
            return {k: quantize(v, decimals) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [quantize(v, decimals) for v in x]
        return x

    out = dict(report)
    if "per_cell_auc_by_gamma" in out:
        out["per_cell_auc_by_gamma"] = quantize(out["per_cell_auc_by_gamma"], 3)
    if "verdicts_by_gamma" in out:
        out["verdicts_by_gamma"] = quantize(out["verdicts_by_gamma"], 4)
    if "v3_baseline" in out:
        out["v3_baseline"] = quantize(out["v3_baseline"], 4)
    if "lambda_auto_calibrated" in out:
        out["lambda_auto_calibrated"] = round(out["lambda_auto_calibrated"], 4)
    if "gamma_auto" in out:
        out["gamma_auto"] = round(out["gamma_auto"], 4)
    return out


def compute_bench_digest(report: dict[str, Any]) -> str:
    """JCS-canonical SHA-256 over the quantized payload (minus digest)."""
    payload = dict(report)
    payload.pop("bench_digest", None)
    quantized = quantize_for_digest(payload)
    canonical = jcs_canonicalize(quantized)
    return hashlib.sha256(canonical).hexdigest()


def main() -> dict[str, Any]:
    rng = random.Random(0)
    print("=" * 72)
    print("v3.2 validation bench — F4/F5 verdicts on the F3 STRUCTURAL FAIL closer")
    print("=" * 72)

    print("\n[1] Sieve-extracting triples from seed_long_paragraphs…")
    corpus = extract_corpus_triples()
    print(f"    {len(corpus)} docs with non-empty extractions")

    all_triples = [t for _, ts in corpus for t in ts]
    all_entities = sorted({e for h, _, t in all_triples for e in (h, t)})
    all_relations = sorted({r for _, r, _ in all_triples})
    print(f"    union vocab: {len(all_entities)} entities, "
          f"{len(all_relations)} relations")

    print("\n[2] Training v2.1 sheaf (same hyperparams as PR #124)…")
    trained, embeddings, _history = train_restriction_maps(
        all_triples, stalk_dim=8, epochs=200, learning_rate=0.005,
        margin=0.5, n_negatives_per_positive=3, seed=0,
    )

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
    print(f"    λ_auto = {lambda_auto:.4f}")

    print("\n[4] Auto-calibrating γ from clean-render statistics…")
    gamma_auto = calibrate_gamma_auto(corpus, trained, embeddings)
    print(f"    γ_auto = {gamma_auto:.4f}")

    # Resolve the γ grid to numeric values.
    gamma_resolved: list[tuple[str, float]] = []
    for g in GAMMA_GRID:
        if g == "auto":
            gamma_resolved.append(("auto", gamma_auto))
        else:
            gamma_resolved.append((str(g), float(g)))

    # Per-(γ, detector_label, class, target) score collector
    cells: dict[str, list[tuple[float, int]]] = {}

    def add(label: str, score: float, lab: int) -> None:
        cells.setdefault(label, []).append((score, lab))

    print("\n[5] Per-doc scoring across γ values + v3 baseline…")
    docs_with_partition = 0
    docs_skipped = 0
    for doc_id, source in corpus:
        if len(source) < 4:
            docs_skipped += 1
            continue
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

        # Clean-render baselines for each scoring path
        clean_v3 = score_v3_weighted(doc_sheaf, doc_emb, source, weights, lambda_auto)
        clean_v32_by_gamma = {
            label: score_v32_combined(
                doc_sheaf, doc_emb, source, weights, lambda_auto, g_val,
            )
            for label, g_val in gamma_resolved
        }

        # Trusted + untrusted target perturbations
        target_t = rng.choice(trusted)
        target_u = rng.choice(untrusted)
        for target_label, target in (("trusted", target_t), ("untrusted", target_u)):
            for cls, perturb_fn in [
                ("A1", lambda r=rng, tg=target: perturb_a1_on_target(source, tg, all_entities, r)),
                ("A2", lambda r=rng, tg=target: perturb_a2_on_target(source, tg, all_relations, r)),
                ("A4", lambda tg=target: perturb_a4_drop_target(source, tg)),
            ]:
                perturbed = perturb_fn()
                if perturbed == source:
                    continue

                # v3 baseline (label "v3" — for direct comparison)
                p_v3 = score_v3_weighted(doc_sheaf, doc_emb, perturbed, weights, lambda_auto)
                add(f"v3|{cls}|{target_label}", clean_v3, 0)
                add(f"v3|{cls}|{target_label}", p_v3, 1)

                # v3.2 at each γ
                for label, g_val in gamma_resolved:
                    try:
                        p_v32 = score_v32_combined(
                            doc_sheaf, doc_emb, perturbed, weights, lambda_auto, g_val,
                        )
                    except Exception:  # noqa: BLE001
                        continue
                    add(f"v32_{label}|{cls}|{target_label}",
                        clean_v32_by_gamma[label], 0)
                    add(f"v32_{label}|{cls}|{target_label}",
                        p_v32, 1)

        docs_with_partition += 1

    print(f"    docs with partition: {docs_with_partition}; skipped: {docs_skipped}")

    # Per-cell AUC
    print("\n[6] Per-cell AUC:")
    print(f"    {'cell':<32} {'n':>4} {'AUC':>7}")
    print("    " + "-" * 46)
    aucs: dict[str, float] = {}
    for key in sorted(cells):
        scores = [p[0] for p in cells[key]]
        labels = [p[1] for p in cells[key]]
        auc = roc_auc(scores, labels)
        aucs[key] = auc
        marker = "✓" if auc >= 0.75 else ("~" if auc >= 0.55 else "✗")
        print(f"    {key:<32} {len(cells[key]):>4} {auc:>7.3f}  {marker}")

    # Per-γ trusted-mean / untrusted-mean
    print("\n[7] v3.2 mean AUC by γ (across A1/A2/A4):")
    print(f"    {'γ':>10} {'trusted':>10} {'untrusted':>10} {'F4':>6} {'F5 vs v3':>10}")
    print("    " + "-" * 50)
    v3_trusted_mean = float(np.mean([
        aucs.get(f"v3|{c}|trusted", 0.5) for c in ("A1", "A2", "A4")
    ]))
    v3_untrusted_mean = float(np.mean([
        aucs.get(f"v3|{c}|untrusted", 0.5) for c in ("A1", "A2", "A4")
    ]))
    print(f"    {'v3 (ref)':>10} {v3_trusted_mean:>10.3f} {v3_untrusted_mean:>10.3f}")

    verdicts: dict[str, dict[str, float | str]] = {}
    for label, g_val in gamma_resolved:
        t_mean = float(np.mean([
            aucs.get(f"v32_{label}|{c}|trusted", 0.5) for c in ("A1", "A2", "A4")
        ]))
        u_mean = float(np.mean([
            aucs.get(f"v32_{label}|{c}|untrusted", 0.5) for c in ("A1", "A2", "A4")
        ]))
        f4 = "PASS" if t_mean >= 0.55 else "FAIL"
        delta_v3 = t_mean - v3_trusted_mean
        f5 = "PASS" if delta_v3 >= -0.02 else "FAIL"
        gamma_str = f"{label}({g_val:.2f})" if label == "auto" else label
        print(f"    {gamma_str:>10} {t_mean:>10.3f} {u_mean:>10.3f} "
              f"{f4:>6} {delta_v3:>+10.3f}")
        verdicts[label] = {
            "gamma_value": float(g_val),
            "trusted_mean_auc": t_mean,
            "untrusted_mean_auc": u_mean,
            "f4_verdict": f4,
            "delta_vs_v3": delta_v3,
            "f5_verdict": f5,
        }

    report: dict[str, Any] = {
        "schema": "sum.sheaf_v3_2_validation.v1",
        "corpus": "seed_long_paragraphs",
        "n_docs_total": len(corpus),
        "n_docs_with_partition": docs_with_partition,
        "n_docs_skipped": docs_skipped,
        "vocab_size_entities": len(all_entities),
        "vocab_size_relations": len(all_relations),
        "stalk_dim": 8,
        "training_epochs": 200,
        "lambda_auto_calibrated": lambda_auto,
        "gamma_auto": gamma_auto,
        "trust_partition": "deterministic 50/50 per doc (SHA-256 seeded)",
        "weights_from_receipts": {
            "trusted_weight": 1.0, "default_weight": 0.1, "revoked_weight": 0.0,
        },
        "per_cell_auc_by_gamma": aucs,
        "v3_baseline": {
            "trusted_mean_auc": v3_trusted_mean,
            "untrusted_mean_auc": v3_untrusted_mean,
        },
        "verdicts_by_gamma": verdicts,
        # Reproducibility contract: bench_digest matches across runs ONLY
        # when invoked with PYTHONHASHSEED=0. Set-iteration order in the
        # sieve and KnowledgeSheafV2.from_triples is hash-randomized
        # otherwise, which permutes the trained vertex ordering and
        # propagates ~±0.005 noise into per-cell AUCs. This caveat applies
        # to every bench in this repo that goes through the sieve +
        # training pipeline (v3 corpus ROC bench, F3 diagnostic, this
        # bench). Future PR: make the substrate hash-seed-independent
        # by sorting at every set→list conversion in the pipeline.
        "reproducibility_requires": "PYTHONHASHSEED=0",
    }
    report["bench_digest"] = compute_bench_digest(report)
    return report


if __name__ == "__main__":
    receipt = main()
    print("\n[8] Receipt JSON:")
    print(json.dumps(receipt, indent=2))
