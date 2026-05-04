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


def test_hybrid_comparison_digest_pinned():
    """Borda(v3.2_only, B2) — first negative result; locks the
    'baseline rank-fusion of cochain-channel-only v3.2 LOSES to B2'
    finding."""
    from scripts.research.sheaf_hybrid_comparison import run_hybrid_comparison
    PINNED = "a7965803ccf2e703d80364dc21b3ac410491db9768cdfcf91bfefd29356c2003"
    report = run_hybrid_comparison()
    assert report["bench_digest"] == PINNED, (
        f"hybrid_comparison digest drift: got {report['bench_digest']}. "
        "Borda(v3.2_only, B2) loses to B2 alone in the published numbers; "
        "if this digest changes, the loss-margin claim shifts."
    )


def test_predicate_negatives_experiment_digest_pinned():
    """Option 2 — predicate-perturbation training. Locks the structural
    finding that A2 stayed at 0.500 even with predicate negatives, which
    surfaced the cochain-channel structural blindness to entity-set-
    preserving perturbations."""
    from scripts.research.sheaf_predicate_negatives_experiment import run_experiment
    PINNED = "aa34b6e8640621da07823f985ddf35196a85047a64f942493854e09b75c866e7"
    report = run_experiment()
    assert report["bench_digest"] == PINNED, (
        f"predicate_negatives digest drift: got {report['bench_digest']}. "
        "If this digest changes, the structural-blindness finding "
        "(A2 stays at 0.500 even with predicate negatives in training) "
        "may have shifted — re-investigate the cochain construction."
    )


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
