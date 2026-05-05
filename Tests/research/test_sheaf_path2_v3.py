"""
Pin the Path 2 v3 bench's bench_digest. Tests the deterministic Phase 2
(score-from-cached-snapshot) of the real-LLM-rendered adversarial bench.

The Phase 1 (LLM capture) run is gated on `OPENAI_API_KEY` and is
NOT exercised by this test — the snapshot at
`fixtures/bench_renders/path2_seed_long_paragraphs.json` is the
checked-in canonical artifact, captured once against
gpt-4o-mini-2024-07-18 and committed to the repo. Re-running Phase 2
against this snapshot reproduces the digest byte-identically.

The bench's substantive verdict (`HYBRID_TIES_BASELINE_ON_REAL_LLM`)
is the load-bearing finding the §4.7.x narrative depends on: the
synthetic-bench WIN does NOT generalise to real LLM hallucinations.
The pin asserts both the digest AND the verdict label.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / "fixtures" / "bench_renders" / "path2_seed_long_paragraphs.json"

_DETERMINISTIC_BLAS_ENV = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@pytest.mark.skipif(not SNAPSHOT.exists(), reason="snapshot missing — run Phase 1 with OPENAI_API_KEY")
def test_path2_v3_bench_digest_pinned():
    """Run the Path 2 bench in a fresh subprocess (deterministic-BLAS env
    set), parse the resulting receipt, assert digest + verdict.

    Subprocess invocation is required because pytest imports numpy via
    the Hypothesis plugin BEFORE test functions run; in-process env-var
    setdefault is a no-op in the pytest process.
    """
    PINNED_DIGEST = "7b364fc6ae23ce4ea24c69cf7b299b10402237f6f0c4364b18fbcb1dbcc4b75e"

    proc = subprocess.run(
        [sys.executable, "-m", "scripts.research.sheaf_path2_v3_bench"],
        cwd=str(REPO),
        env=_DETERMINISTIC_BLAS_ENV,
        capture_output=True,
        text=True,
        check=True,
    )
    out_path = None
    for line in proc.stdout.splitlines():
        if line.lstrip().startswith("→ wrote "):
            out_path = REPO / line.split("→ wrote ", 1)[1].strip()
            break
    assert out_path is not None, (
        f"Bench did not print '→ wrote' line. stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    report = json.loads(out_path.read_text())

    assert report["bench_digest"] == PINNED_DIGEST, (
        f"path2_v3 digest drift: got {report['bench_digest']}, "
        f"expected {PINNED_DIGEST}. Phase 2 is supposed to be byte-stable "
        f"given the cached snapshot. Investigate: did the snapshot file "
        f"change? Did the deterministic sieve change? Did the scoring "
        f"composition change?"
    )

    # Substantive verdict: hybrid does NOT beat baseline on real LLM
    # perturbations. The synthetic-bench HYBRID_BEATS_BASELINE WIN
    # (Δ=+0.043) does NOT generalise — on real LLM hallucinations the
    # hybrid LOSES by a small margin (Δ≈−0.02).
    #
    # The verdict-threshold classification (−0.02 boundary between
    # TIES and LOSES) is somewhat arbitrary; the load-bearing claim
    # is "hybrid does not BEAT baseline on real LLM" — accept any
    # verdict that's TIES or LOSES, fail on BEATS.
    assert report["verdict"] in (
        "HYBRID_TIES_BASELINE_ON_REAL_LLM",
        "HYBRID_LOSES_TO_BASELINE_ON_REAL_LLM",
    ), (
        f"Path 2 verdict drift: got {report['verdict']}. The §4.7.x "
        "real-LLM narrative depends on the hybrid not BEATING the "
        "baseline on real perturbations (the synthetic-vs-real gap "
        "is the load-bearing finding). If this label flips to "
        "HYBRID_BEATS_BASELINE_ON_REAL_LLM, the synthetic-bench WIN "
        "would have generalised — that would be a substantive shift "
        "in the §4.7.x narrative worth investigating before update."
    )
    # Pin the actual margin range — should be slightly negative.
    delta = report["delta_borda_vs_b2_mean"]
    assert -0.05 <= delta < 0.03, (
        f"Δ(borda - b2) out of expected range: {delta:.4f}, "
        "expected in [-0.05, 0.03). The hybrid's edge-vs-B2 should "
        "remain small on real LLM perturbations regardless of the "
        "specific TIES/LOSES classification."
    )

    # Sanity check on detector mean AUCs: both should be in the
    # weak-but-non-trivial range (0.55-0.75) on real LLM. The
    # synthetic-vs-real gap (B2 0.833 synthetic vs ~0.66 real) is
    # the structural finding — pin the real-LLM range loosely.
    b2_mean = report["mean_auc_by_detector"].get("b2_jaccard")
    hybrid_mean = report["mean_auc_by_detector"].get("borda_v32pt_b2")
    assert 0.5 <= b2_mean <= 0.85, (
        f"B2 mean AUC out of expected range on real LLM: {b2_mean:.3f}"
    )
    assert 0.5 <= hybrid_mean <= 0.85, (
        f"Hybrid mean AUC out of expected range on real LLM: {hybrid_mean:.3f}"
    )
