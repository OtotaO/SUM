"""
Pin the bench_digest values produced by the Sprint 7.5 recovery
experiments. If any of these tests fail, EITHER the v3 ROC bench's
perturbation harness changed, the corpus changed, the training math
changed, or the scoring composition changed. Investigate before
updating the pinned constants.

All four marked slow because each bench retrains the v2.1 sheaf
(~2-3 min CPU per run). Run with:

    python3 -m pytest Tests/research/test_recovery_experiment_digests.py -q

To skip in dev:

    python3 -m pytest -m "not slow" -q
"""
from __future__ import annotations

import pytest


# Marked slow because each retrains the sheaf (~2-3 min CPU).
pytestmark = pytest.mark.slow


def test_hybrid_comparison_loss_finding_holds():
    """Borda(v3.2_only, B2) — first negative result; locks the
    'baseline rank-fusion of cochain-channel-only v3.2 LOSES to B2'
    finding.

    SHAPE PIN (verdict + Δ-in-range), NOT byte-digest. The v0.2
    latent-fix arc tried two quantization layers (rank-key
    quantization in `_ranks`, then per-pair-score quantization at
    storage time) but neither fully absorbs the LAPACK-jitter
    sensitivity in this specific bench. Same-machine reruns
    produce two stable digests (`a7965803…` and `7fac833a…`)
    differing by quantization-bucket flips at small numbers of
    cells.

    Why this bench is intrinsically more sensitive than its
    siblings:

      - hybrid_comparison fuses ONLY cochain V (v3.2 score with
        γ=0.1) with B2 jaccard. There's no per-rendered-triple V
        magnitude to dominate the cell-AUC computation.
      - LAPACK threading inside `np.linalg.lstsq` (used in v3.1's
        harmonic-extension pathway) produces score variance up to
        ~1e-6 magnitude. With cochain-only fusion, this lands at
        quantization-bucket boundaries on some cells.
      - complementary_hybrid (which DOES include per-triple V) is
        stable — the per-triple magnitude breaks LAPACK ties
        cleanly.

    The substantive finding is invariant: Borda(v3.2_only, B2)
    LOSES to B2 alone by Δ ≈ −0.025 trusted-mean. The TWO
    candidate digest outcomes are equally valid
    `BORDA_LOSES_TO_B2` realisations.

    v0.3+ candidate fixes:
      - Replace `np.linalg.lstsq` with a deterministic SVD-based
        pinv in `harmonic_extension` (eliminates threading
        non-determinism; v3 refactor)
      - Set OPENBLAS_NUM_THREADS=1 at process entry (works but
        environment-dependent; doesn't compose with multi-process
        benches)
      - Compute cell AUCs at higher precision and quantize at the
        AUC level rather than score level

    For now (v0.2): shape-pin. The two-layer quantization (in
    `_ranks` AND at score-storage) is retained because it
    statistically improves stability even though it doesn't make
    the digest unconditional.
    """
    from scripts.research.sheaf_hybrid_comparison import run_hybrid_comparison
    report = run_hybrid_comparison()
    # Layer 1: verdict label (substantive finding)
    assert report["verdict"] == "BORDA_LOSES_TO_B2", (
        f"hybrid_comparison verdict drift: got {report['verdict']!r}. "
        "The substantive finding — Borda(v3.2_only, B2) loses to B2 "
        "alone — is load-bearing for §4.7.1's STOP-THE-LINE narrative."
    )
    # Layer 2: loss-margin range
    delta = report["delta_borda_vs_b2_trusted_mean"]
    assert -0.10 <= delta <= -0.02, (
        f"delta_borda_vs_b2 drift: got {delta:.4f}, expected in "
        "[-0.10, -0.02]. The loss should be a clear margin."
    )
    # Layer 3: schema check on bench_digest
    assert isinstance(report["bench_digest"], str)
    assert len(report["bench_digest"]) == 64
    int(report["bench_digest"], 16)


def test_predicate_negatives_experiment_digest_pinned():
    """Option 2 — predicate-perturbation training. Locks the load-bearing
    STRUCTURAL FINDING that A2 stayed at 0.500 even with predicate
    negatives, which surfaced the cochain-channel structural blindness
    to entity-set-preserving perturbations.

    Three-layer pin (most-specific to most-general), upgraded
    2026-05-05 from shape-only to byte-digest after the v0.2 refactor
    that replaced the local v2-training-loop copy with a call to
    production `train_restriction_maps(...,
    n_predicate_negatives_per_positive=3)`. The pre-refactor bench
    used a local copy whose SGD trajectory was Python-version-sensitive
    (operator/Modal Python 3.10 matched; CI Python 3.12 diverged).
    Single training-loop source eliminates that cross-version drift.

      1. Byte-digest. Verified 5× in fresh procs.
      2. Verdict label `A2_STILL_CHANCE`.
      3. A2 cells at exactly 0.500 (cochain blindness).
    """
    from scripts.research.sheaf_predicate_negatives_experiment import run_experiment
    PINNED = "ddf41484b1eba2f1cf5927d6e9691a922e5843be703fedac83e8afee001f59c3"
    report = run_experiment()
    # Layer 1: byte-digest
    assert report["bench_digest"] == PINNED, (
        f"predicate_negatives digest drift: got {report['bench_digest']}, "
        f"expected {PINNED}. Post-refactor (production train_restriction_maps "
        f"with n_predicate_negatives_per_positive=3), this digest should be "
        f"cross-Python-version stable. If only the digest drifted but "
        f"verdict={report.get('verdict')!r} and A2-cells-at-0.500 still "
        f"hold, the substantive finding is intact — investigate corpus / "
        f"production training math."
    )
    # Layer 2: verdict label (substantive finding)
    assert report["verdict"] == "A2_STILL_CHANCE", (
        f"predicate-negatives verdict drift: got {report['verdict']!r}. "
        "The structural finding (A2 stays at chance even with predicate "
        "negatives in training) is the load-bearing claim — if this "
        "verdict label changes, the cochain-blindness diagnosis from "
        "§3.4.5 of docs/SHEAF_HALLUCINATION_DETECTOR.md may have shifted."
    )
    # Layer 3: A2 cells at exactly chance
    a2_t = report["per_cell_auc"].get("v32_g0.1_pred_neg|A2|trusted")
    a2_u = report["per_cell_auc"].get("v32_g0.1_pred_neg|A2|untrusted")
    assert a2_t == 0.5, f"A2 trusted should be at chance; got {a2_t}"
    assert a2_u == 0.5, f"A2 untrusted should be at chance; got {a2_u}"


def test_per_triple_integration_digest_pinned():
    """Option 2.5 — per-rendered-triple V channel integration. Locks the
    finding that adding the §3.5 per-triple channel lifts A2 from 0.500
    to 0.671 (trusted) but trusted-mean still loses to B2 alone."""
    from scripts.research.sheaf_per_triple_integration_experiment import run_experiment
    PINNED = "7025436f3c010e681bfbd06a04730d017e031df2b376e8e2bb5b404df81fd4fa"
    report = run_experiment()
    assert report["bench_digest"] == PINNED, (
        f"per_triple_integration digest drift: got {report['bench_digest']}. "
        "If this digest changes, the A2-lift-via-per-triple finding may "
        "have shifted — re-investigate score_v32_with_per_triple."
    )


def test_complementary_hybrid_digest_pinned():
    """LOAD-BEARING: complementary Borda(v3.2+per_triple, B2). This is
    the published WIN — trusted-mean 0.876 vs B2's 0.833, Δ=+0.043,
    HYBRID_BEATS_BASELINE. Pin this digest tightly: if it changes, the
    detector's competitive claim against trivial baselines shifts."""
    from scripts.research.sheaf_complementary_hybrid_experiment import run_experiment
    PINNED = "dc6e0260f14042fa0b6151a6ca6b443bb0910eabb996f6876f854633969343ce"
    report = run_experiment()
    assert report["bench_digest"] == PINNED, (
        f"complementary_hybrid digest drift: got {report['bench_digest']}. "
        "This is the load-bearing WIN digest. Investigate before updating: "
        "did the perturbation harness change? scoring math? Borda fusion?"
    )
    # Also assert the verdict label, not just the digest, so failures
    # point at the substantive claim.
    assert report["verdict"] == "HYBRID_BEATS_BASELINE", (
        f"verdict drift: got {report['verdict']}. The hybrid is supposed "
        "to beat B2 by ≥ 0.03 trusted-mean. If this fails, the published "
        "preprint's central detector claim is invalidated."
    )
