"""
Hybrid detector experiment — does v3.2 + B2 jaccard, fused via Borda
rank-addition, beat either component alone?

NOTE on cross-run reproducibility: the bench_digest of THIS bench
(unlike v3.2 validation, per_triple_integration, and complementary_hybrid)
is INTERMITTENT across runs on the same machine — `np.linalg.lstsq`
inside the v3.1 boundary-deviation path has multi-threaded LAPACK
non-determinism at the per-pair score level. Quantization to 3 decimals
on AUC absorbs the jitter on most benches, but Borda fusion of ONLY the
cochain V channel (no per-rendered-triple V magnitude to break ties)
makes the fused-cell AUC sensitive to rank-assignment ordering when
two scores are within ~1 ULP of each other. The substantive finding —
Borda(v3.2_only, B2) LOSES to B2 alone by Δ ≈ −0.025 — is invariant.
The pinned test (`test_hybrid_comparison_loss_finding_holds`) asserts
verdict label + Δ-in-range, not byte-digest. v0.2 follow-up: add a
secondary sort key to `borda_fuse` for stable tie-breaking, OR widen
quantization to 2 decimals on AUC for cochain-only fusion benches.

Carmack-style decision experiment. Three outcomes:

  Hybrid trusted-mean ≥ B2 + 0.03 → detector adds value via composition;
                                     reshape preprint around hybrid.
  Hybrid within ±0.02 of B2       → detector adds nothing material;
                                     substrate-first reframe justified.
  Hybrid < B2 − 0.03              → detector anti-correlated with B2;
                                     substrate-first reframe mandatory.

Methodology mirrors scripts/research/sheaf_v3_2_validation.main exactly
(one trusted target + one untrusted target per doc, same partition seed,
same training hyperparams) so v3.2's published trusted-mean reproduces
inside this bench. B2 jaccard is then scored on the SAME (clean, perturbed)
triple-set pairs for apples-to-apples comparison.

Borda fusion: per (class, target) cell, rank both detectors' scores
ascending; sum ranks per pair; take that as the hybrid score. Magnitude-
invariant and parameter-free.

Output: fixtures/bench_receipts/hybrid_comparison_<DATE>.json
        schema: sum.sheaf_hybrid_comparison.v1
"""
from __future__ import annotations

# MUST come before any numpy/scipy import — sets BLAS thread vars
# at process startup so bench_digest is byte-stable across fresh
# Python processes. See `_deterministic_blas` for the rationale.
import scripts.research._deterministic_blas  # noqa: F401

import datetime as _dt
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from sum_engine_internal.research.sheaf_laplacian_v2 import train_restriction_maps
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
    calibrate_gamma_auto,
    compute_bench_digest,
    quantize_for_digest,
    score_v32_combined,
)
from scripts.research.sheaf_baseline_comparison import (
    score_b2_jaccard_distance,
)
from sum_engine_internal.research.sheaf_laplacian_v2 import (
    combined_detector_score,
)

REPO = Path(__file__).resolve().parents[2]
RECEIPTS_DIR = REPO / "fixtures" / "bench_receipts"

# Match v3.2 main: γ=0.1 is the F5-PASS regime.
GAMMA = 0.1


def borda_fuse(scores_a: list[float], scores_b: list[float]) -> list[float]:
    """Per-pool rank fusion. Returns rank_a + rank_b per index.
    Higher fused = higher in both detectors' rankings."""
    assert len(scores_a) == len(scores_b)
    n = len(scores_a)
    rank_a = _ranks(scores_a)
    rank_b = _ranks(scores_b)
    return [float(rank_a[i] + rank_b[i]) for i in range(n)]


_RANK_TIE_PRECISION = 6   # decimals; absorb LAPACK jitter while
                          # preserving scoring-function signal
                          # (sheaf-Laplacian residuals: O(0.1)-O(10);
                          # signal lives in tenths/hundredths)


def _ranks(xs: list[float]) -> list[float]:
    """Average-ranked. Ties get mean rank.

    Sort key + tie detection are both quantized to _RANK_TIE_PRECISION
    decimals (1e-6 absolute). LAPACK threading jitter inside
    `np.linalg.lstsq` (used in v3.1's harmonic-extension pathway) can
    produce score differences up to ~1e-7 magnitude on the same inputs
    across runs; the 6-decimal quantization absorbs that range while
    leaving the load-bearing signal (in the tenths/hundredths) intact.

    The initial v0.2 fix (9-decimal quantization) was too tight — some
    runs still produced a divergent digest because the LAPACK jitter
    occasionally exceeded 1e-9. Tightening to 6 decimals is the
    correct trade-off: well below scoring-function signal precision,
    well above LAPACK noise.

    The secondary sort key is the original index, giving a stable
    order among tied items independent of input ordering.
    """
    n = len(xs)
    qxs = [round(x, _RANK_TIE_PRECISION) for x in xs]
    indexed = sorted(range(n), key=lambda i: (qxs[i], i))
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and qxs[indexed[j + 1]] == qxs[indexed[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-indexed
        for k in range(i, j + 1):
            out[indexed[k]] = avg
        i = j + 1
    return out


def run_hybrid_comparison() -> dict[str, Any]:
    rng = random.Random(0)
    print("=" * 72)
    print("Hybrid detector experiment — v3.2 vs B2 vs Borda(v3.2, B2)")
    print("=" * 72)

    print("\n[1] Sieve-extracting triples from seed_long_paragraphs…")
    corpus = extract_corpus_triples()
    print(f"    {len(corpus)} docs")

    all_triples = [t for _, ts in corpus for t in ts]
    all_entities = sorted({e for h, _, t in all_triples for e in (h, t)})
    all_relations = sorted({r for _, r, _ in all_triples})

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

    print("\n[4] Per-doc scoring (v3.2-main loop structure)…")
    # cells[(detector, class, target_label)] = list[(score, label)]
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

        # Clean-render scores. Quantize at storage time to absorb
        # LAPACK threading jitter (np.linalg.lstsq inside v3.1 path
        # can produce ~1e-7-magnitude variance on identical inputs).
        # 6-decimal absolute precision is well below scoring-function
        # signal precision (sheaf-Laplacian residuals: O(0.1)-O(10);
        # signal in tenths/hundredths). Without this, downstream Borda
        # rank-fusion on cochain-only scores is sensitive to ULP-jitter
        # rank shuffles even with quantized sort keys.
        clean_v32 = round(score_v32_combined(
            doc_sheaf, doc_emb, source, weights, lambda_auto, GAMMA,
        ), 6)
        clean_b2 = round(
            score_b2_jaccard_distance(source, source), 6,  # = 0.0
        )

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
                    p_v32 = round(score_v32_combined(
                        doc_sheaf, doc_emb, perturbed, weights, lambda_auto, GAMMA,
                    ), 6)
                except Exception:  # noqa: BLE001
                    continue
                p_b2 = round(score_b2_jaccard_distance(source, perturbed), 6)

                cells.setdefault(("v32_g0.1", cls, target_label), []).extend([
                    (clean_v32, 0), (p_v32, 1),
                ])
                cells.setdefault(("b2_jaccard", cls, target_label), []).extend([
                    (clean_b2, 0), (p_b2, 1),
                ])

        n_with_partition += 1

    print(f"    docs with partition: {n_with_partition}")

    # Build Borda hybrid AT THE PER-CELL LEVEL — needs per-pair scores.
    print("\n[5] Borda fusion of v3.2(γ=0.1) + B2 jaccard per cell…")
    hybrid_per_cell_auc: dict[str, float] = {}
    for (cls, target_label) in {(c, t) for (_, c, t) in cells}:
        v32_pairs = cells[("v32_g0.1", cls, target_label)]
        b2_pairs = cells[("b2_jaccard", cls, target_label)]
        # Sanity: same number of pairs, same labels in same order
        assert [l for _, l in v32_pairs] == [l for _, l in b2_pairs], (
            f"label mismatch at {cls}|{target_label}"
        )
        v32_scores = [s for s, _ in v32_pairs]
        b2_scores = [s for s, _ in b2_pairs]
        labels = [l for _, l in v32_pairs]
        fused = borda_fuse(v32_scores, b2_scores)
        auc = roc_auc(fused, labels)
        cell_key = f"borda_v32_b2|{cls}|{target_label}"
        hybrid_per_cell_auc[cell_key] = auc

    # Per-cell AUC for the components
    component_per_cell_auc: dict[str, float] = {}
    for (det, cls, target_label), pairs in cells.items():
        scores = [s for s, _ in pairs]
        labels = [l for _, l in pairs]
        cell_key = f"{det}|{cls}|{target_label}"
        component_per_cell_auc[cell_key] = roc_auc(scores, labels)

    print("\n[6] Per-cell AUC:")
    all_per_cell = {**component_per_cell_auc, **hybrid_per_cell_auc}
    for key in sorted(all_per_cell):
        print(f"    {key:50s} = {all_per_cell[key]:.3f}")

    # Trusted-mean per detector
    print("\n[7] Trusted-mean AUC across A1+A2+A4 trusted cells:")
    trusted_means: dict[str, float] = {}
    for det in ("v32_g0.1", "b2_jaccard", "borda_v32_b2"):
        trusted_aucs = [
            auc for key, auc in all_per_cell.items()
            if key.startswith(f"{det}|") and key.endswith("|trusted")
        ]
        trusted_means[det] = (
            sum(trusted_aucs) / len(trusted_aucs) if trusted_aucs else 0.0
        )
        print(f"    {det:25s} = {trusted_means[det]:.3f}")

    # Decision report
    delta_borda_vs_b2 = trusted_means["borda_v32_b2"] - trusted_means["b2_jaccard"]
    if delta_borda_vs_b2 >= 0.03:
        verdict = "BORDA_BEATS_B2"
    elif delta_borda_vs_b2 >= -0.02:
        verdict = "BORDA_TIES_B2"
    else:
        verdict = "BORDA_LOSES_TO_B2"
    print(f"\n[8] Δ(borda - b2) = {delta_borda_vs_b2:+.3f} → {verdict}")

    report = {
        "schema": "sum.sheaf_hybrid_comparison.v1",
        "corpus": "seed_long_paragraphs",
        "n_docs_total": len(corpus),
        "n_docs_with_partition": n_with_partition,
        "lambda_auto": lambda_auto,
        "gamma_used": GAMMA,
        "detectors": ["v32_g0.1", "b2_jaccard", "borda_v32_b2"],
        "perturbation_classes": ["A1", "A2", "A4"],
        "per_cell_auc": all_per_cell,
        "trusted_mean_auc_by_detector": trusted_means,
        "delta_borda_vs_b2_trusted_mean": delta_borda_vs_b2,
        "verdict": verdict,
        "method_notes": (
            "Loop structure mirrors sheaf_v3_2_validation.main: one trusted "
            "and one untrusted target per doc, same stable_seed partition, "
            "same training hyperparams. v3.2 and B2 are scored on the SAME "
            "(clean, perturbed) pairs (sanity-asserted by label-order match). "
            "Borda fusion: average-ranked scores per (class, target) pool, "
            "then summed; magnitude-invariant and parameter-free."
        ),
    }

    quantized = quantize_for_digest(report)
    report["bench_digest"] = compute_bench_digest(quantized)
    print(f'\n  "bench_digest": "{report["bench_digest"]}"')

    return report


def main() -> dict[str, Any]:
    report = run_hybrid_comparison()
    from scripts.research._receipt_paths import resolve_receipt_path
    out = resolve_receipt_path(RECEIPTS_DIR, "hybrid_comparison")
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n→ wrote {out.relative_to(REPO)}")
    return report


if __name__ == "__main__":
    main()
