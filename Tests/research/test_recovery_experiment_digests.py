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

    NOTE: pinned by behavior-shape (verdict label + Δ in the loss
    range), NOT by byte-digest. The bench's Borda fusion combines
    only the cochain V channel (no per-rendered-triple V magnitude
    to break ties), so LAPACK-jitter at the per-pair score level
    can shift rank assignments in tied cells, which can move the
    fused-cell AUC across the 3-decimal quantization boundary,
    which can change the bench_digest. The substantive finding —
    Borda(v3.2_only, B2) LOSES to B2 alone by a clear margin
    (Δ ≈ −0.025) — is invariant: B2 catches A1/A4 at 1.000 alone,
    and adding the cochain channel to the rank fusion can only
    add noise (the cochain channel is at chance or anti-correlated
    on those classes). The complementary-hybrid pin
    (`test_complementary_hybrid_digest_pinned`) IS byte-digest
    pinned because the per-rendered-triple V channel adds magnitude
    that dominates LAPACK jitter and breaks ties cleanly.

    Operator-environment digest from one canonical run:
        a7965803ccf2e703d80364dc21b3ac410491db9768cdfcf91bfefd29356c2003
    Some same-machine runs produce
        7fac833a23a8d5be3acf2e3b88d5f117ddb2283e37bf7c0b1daff8a7283bcb97
    instead — both are equally valid "BORDA_LOSES_TO_B2" outcomes
    differing by a 1-ULP rank shuffle.

    v0.2 follow-up: either (a) post-hoc tie-breaking by an explicit
    secondary sort key in `borda_fuse`, or (b) increase quantization
    to 2 decimals on AUC for benches without per-triple-channel
    magnitude.
    """
    from scripts.research.sheaf_hybrid_comparison import run_hybrid_comparison
    report = run_hybrid_comparison()
    assert report["verdict"] == "BORDA_LOSES_TO_B2", (
        f"hybrid_comparison verdict drift: got {report['verdict']}. "
        "The substantive finding — Borda(v3.2_only, B2) loses to B2 "
        "alone — is the load-bearing claim. If this verdict label "
        "changes, the §4.7.1 STOP-THE-LINE narrative may have shifted."
    )
    delta = report["delta_borda_vs_b2_trusted_mean"]
    assert -0.10 <= delta <= -0.02, (
        f"delta_borda_vs_b2 drift: got {delta:.4f}, expected in [-0.10, -0.02]. "
        "The loss should be a clear margin, not a near-tie."
    )
    # bench_digest still required to be present + 64-hex (schema check)
    assert isinstance(report["bench_digest"], str)
    assert len(report["bench_digest"]) == 64
    int(report["bench_digest"], 16)


def test_predicate_negatives_experiment_structural_finding_holds():
    """Option 2 — predicate-perturbation training. Locks the load-bearing
    STRUCTURAL FINDING that A2 stayed at 0.500 even with predicate
    negatives, which surfaced the cochain-channel structural blindness
    to entity-set-preserving perturbations.

    NOTE: pinned by behavior-shape (verdict label + A2 cells at 0.500),
    NOT by byte-digest. The bench uses a local copy of the v2 training
    loop (`train_with_predicate_negatives`) and that copy is
    Python-version-sensitive: the AUC quantization layer absorbs LAPACK
    jitter on the same Python version, but cross-version (Python 3.10
    operator-side vs Python 3.12 CI-side) the trained-weight bits diverge
    enough to shift AUC quantization buckets and therefore the digest.
    The substantive finding is invariant — A2 stays at chance regardless
    of Python version; the digest is environment-specific.

    Operator-environment digest (Python 3.10 / numpy 1.x):
        aa34b6e8640621da07823f985ddf35196a85047a64f942493854e09b75c866e7
    CI-environment digest (Python 3.12 / numpy 2.x): differs but
        verdict + A2-at-chance hold identically. Cross-Python-version
        digest reproducibility is a v0.2 follow-up (would require
        upstreaming the predicate-negative sampler into the production
        train_restriction_maps so the bench uses a single training-loop
        source).
    """
    from scripts.research.sheaf_predicate_negatives_experiment import run_experiment
    report = run_experiment()
    assert report["verdict"] == "A2_STILL_CHANCE", (
        f"predicate-negatives verdict drift: got {report['verdict']}. "
        "The structural finding (A2 stays at chance even with predicate "
        "negatives in training) is the load-bearing claim — if this "
        "verdict label changes, the cochain-blindness diagnosis from "
        "§3.4.5 of docs/SHEAF_HALLUCINATION_DETECTOR.md may have shifted."
    )
    a2_t = report["per_cell_auc"].get("v32_g0.1_pred_neg|A2|trusted")
    a2_u = report["per_cell_auc"].get("v32_g0.1_pred_neg|A2|untrusted")
    assert a2_t == 0.5, f"A2 trusted should be at chance; got {a2_t}"
    assert a2_u == 0.5, f"A2 untrusted should be at chance; got {a2_u}"
    # bench_digest still required to be present + 64-hex (schema check)
    assert isinstance(report["bench_digest"], str)
    assert len(report["bench_digest"]) == 64
    int(report["bench_digest"], 16)


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
