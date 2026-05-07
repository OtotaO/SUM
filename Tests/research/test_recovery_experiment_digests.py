"""
Pin the bench_digest values produced by the Sprint 7.5 recovery
experiments.

Two-tier pinning architecture (added v0.3 to address the
hypothesis/pytest-imports-numpy-first determinism gap):

  - **Receipt-pin tests** (default; always run): read the on-disk
    receipt at `fixtures/bench_receipts/<bench>_<DATE>.json` and
    assert the digest matches the published value. The receipt is
    the published artifact; this test catches receipt tampering /
    accidental overwrites. Fast — pure JSON parse.

  - **Bench-rerun tests** (marked slow; opt-in): execute the bench
    in a clean subprocess with deterministic-BLAS env vars set,
    parse the bench_digest from stdout, assert it matches the
    pinned value. Catches drift in the bench logic itself. The
    subprocess isolation is required because pytest imports numpy
    via the Hypothesis plugin BEFORE this test's
    `scripts.research._deterministic_blas` import can take effect;
    running the bench in-process makes the env-var setdefault a
    no-op and the bench digest goes intermittent.

Run all (receipt-pins fast, bench-reruns ~5-10 min):
    python3 -m pytest Tests/research/test_recovery_experiment_digests.py -q

Skip the slow bench-rerun layer (default in dev):
    python3 -m pytest Tests/research/test_recovery_experiment_digests.py -m "not slow" -q

If any digest test fails, EITHER the v3 ROC bench's perturbation
harness changed, the corpus changed, the training math changed, the
scoring composition changed, or the receipt-on-disk was modified.
Investigate before updating the pinned constants.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RECEIPTS = REPO / "fixtures" / "bench_receipts"


# All thread vars passed via env to bench subprocesses. Set at
# subprocess invocation rather than at this module's import because
# pytest itself imports numpy via the Hypothesis plugin BEFORE any
# test function runs, which makes a module-level setdefault a no-op
# in the pytest process.
_DETERMINISTIC_BLAS_ENV = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",   # Apple Accelerate
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _run_bench_in_subproc(module: str) -> dict:
    """Run a bench module in a clean subprocess with deterministic-BLAS
    env vars set at process startup, parse its bench_digest line, and
    parse the on-disk receipt the bench wrote (idempotent overwrite of
    the canonical date-stamped file). Returns the parsed receipt dict.

    Fresh-process invocation is required: pytest has already imported
    numpy by the time test functions run, so module-level
    `os.environ.setdefault` for BLAS thread vars is a no-op in the
    pytest process. Subprocess gets a clean numpy-import-time env.
    """
    proc = subprocess.run(
        [sys.executable, "-m", module],
        cwd=str(REPO),
        env=_DETERMINISTIC_BLAS_ENV,
        capture_output=True,
        text=True,
        check=True,
    )
    # The bench writes to fixtures/bench_receipts/<schema_prefix>_<DATE>.json
    # via the resolve_receipt_path helper (overwrites the canonical
    # existing receipt when one exists, else creates today's). Find by
    # parsing the "→ wrote ..." line.
    out_path = None
    for line in proc.stdout.splitlines():
        if line.lstrip().startswith("→ wrote "):
            out_path = REPO / line.split("→ wrote ", 1)[1].strip()
            break
    if out_path is None:
        raise AssertionError(
            f"Bench {module} did not print a '→ wrote' line. stdout:\n"
            f"{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return json.loads(out_path.read_text())


def test_hybrid_comparison_digest_pinned():
    """Borda(v3.2_only, B2) — first negative result; locks the
    'baseline rank-fusion of cochain-channel-only v3.2 LOSES to B2'
    finding.

    Three-layer pin (most-specific to most-general):

      1. Byte-digest. Verified 10× in fresh procs unconditionally
         after the v0.3 deterministic-BLAS fix landed
         (`scripts/research/_deterministic_blas.py` sets
         `VECLIB_MAXIMUM_THREADS=1` and friends at process startup,
         eliminating Apple Accelerate / OpenBLAS thread-pool-size
         variance that previously caused two-outcome non-
         determinism).
      2. Verdict label `BORDA_LOSES_TO_B2`. If the digest drifts
         but the verdict label still holds, the load-bearing
         finding is intact.
      3. Loss-margin range: Δ ∈ [−0.10, −0.02].

    Pre-v0.3 (Sprint 7.5 latent-fix arc):
      - Rank-key quantization in `_ranks` to 9 decimals → too tight
      - Tightened to 6 decimals → still intermittent (~38%/~62%
        across two stable outcomes a7965803… and 7fac833a…)
      - Per-pair score storage quantization → still intermittent
      - Diagnosis converged to: thread-pool-size variance at
        numpy-import time on Apple Accelerate (process-level
        non-determinism, not in-process arithmetic). VECLIB
        thread var fixed at 1 → 10/10 stable.

    The two earlier-stable outcomes (a7965803… and 7fac833a…) are
    documented as historical for traceability of the latent-fix
    arc.
    """
    # Two architecture-stable digests, both documented in the
    # docstring above as historical outcomes:
    #   - `a7965803…`  Apple Accelerate (Apple Silicon) ;
    #                  OpenBLAS x86_64 (CI Linux runners and Modal Py 3.10/3.12)
    #   - `7fac833a…`  OpenBLAS arm64 (e.g. miniforge / conda-forge numpy on
    #                  Apple Silicon, where the BLAS backend is
    #                  OpenBLAS-aarch64 rather than Apple Accelerate)
    # Both stem from the same code path; the difference is BLAS kernel
    # numerics under different CPU SIMD instruction sets. PR #154's
    # `VECLIB_MAXIMUM_THREADS=1` fixes thread-pool variance for Apple
    # Accelerate but cannot reconcile cross-architecture floating-point
    # differences (arm64 NEON vs x86_64 AVX2 in OpenBLAS) that compound
    # ~1 ULP per lstsq across 200 SGD epochs and occasionally cross the
    # 3-decimal AUC quantization boundary.
    #
    # Both digests produce the substantively-identical verdict
    # (`BORDA_LOSES_TO_B2`) and overlapping loss-margin range. The
    # substantive layers below (Layer 2 verdict, Layer 3 delta range)
    # are the load-bearing assertions; the digest layer pins which
    # of the two BLAS-architecture outcomes was produced, no more.
    ACCEPTED_DIGESTS = (
        "a7965803ccf2e703d80364dc21b3ac410491db9768cdfcf91bfefd29356c2003",
        "7fac833a23a8d5be3acf2e3b88d5f117ddb2283e37bf7c0b1daff8a7283bcb97",
    )
    report = _run_bench_in_subproc("scripts.research.sheaf_hybrid_comparison")
    # Layer 1: byte-digest must match one of the two documented
    # cross-architecture stable outcomes.
    assert report["bench_digest"] in ACCEPTED_DIGESTS, (
        f"hybrid_comparison digest drift: got {report['bench_digest']}, "
        f"expected one of {ACCEPTED_DIGESTS}. Post-v0.3 deterministic-BLAS "
        f"fix, this should be byte-stable per (arch, BLAS) cell. If a "
        f"third digest appears, that means a third numerics regime — "
        f"new architecture, BLAS rebuild, numpy upgrade, or a regression "
        f"in the bench's determinism. Investigate before re-pinning. The "
        f"substantive finding remains intact if "
        f"verdict={report.get('verdict')!r} and "
        f"Δ={report.get('delta_borda_vs_b2_trusted_mean'):.4f} are still "
        f"in the loss range — see Layer 2 / Layer 3 below."
    )
    # Layer 2: verdict label (substantive finding)
    assert report["verdict"] == "BORDA_LOSES_TO_B2", (
        f"hybrid_comparison verdict drift: got {report['verdict']!r}. "
        "The substantive finding — Borda(v3.2_only, B2) loses to B2 "
        "alone — is load-bearing for §4.7.1's STOP-THE-LINE narrative."
    )
    # Layer 3: loss-margin range
    delta = report["delta_borda_vs_b2_trusted_mean"]
    assert -0.10 <= delta <= -0.02, (
        f"delta_borda_vs_b2 drift: got {delta:.4f}, expected in "
        "[-0.10, -0.02]. The loss should be a clear margin."
    )


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
    PINNED = "ddf41484b1eba2f1cf5927d6e9691a922e5843be703fedac83e8afee001f59c3"
    report = _run_bench_in_subproc("scripts.research.sheaf_predicate_negatives_experiment")
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
    PINNED = "7025436f3c010e681bfbd06a04730d017e031df2b376e8e2bb5b404df81fd4fa"
    report = _run_bench_in_subproc("scripts.research.sheaf_per_triple_integration_experiment")
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
    PINNED = "dc6e0260f14042fa0b6151a6ca6b443bb0910eabb996f6876f854633969343ce"
    report = _run_bench_in_subproc("scripts.research.sheaf_complementary_hybrid_experiment")
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
