"""
Pin the multi-LLM compare's Phase-2 path. Two tests:

  - n=1 (gpt-4o-mini only) — always runs once the OpenAI snapshot is
    committed (PR #156). Asserts the multi-LLM wrapper is a no-op on
    the scoring path (per-model digest matches the Path 2 v3 pin)
    and that n=1 reports SINGLE_MODEL_<verdict> rather than dressing
    itself up as a cross-model structural finding.

  - n=2 (gpt-4o-mini + claude-haiku-4-5) — runs once the Anthropic
    snapshot is committed (Phase 1 was operator-gated; landed in
    PR #158). Pins both per-model digests, the joint finding
    (STRUCTURAL_GAP_ALL_MODELS_LOSE), and the Δ-spread bounds. The
    structural-gap finding is the load-bearing claim that the
    §4.7.2 synthetic-vs-real gap is not gpt-4o-mini-specific —
    it generalises to a different LLM family (Anthropic Claude).

Both Phase-2 paths run from the cached snapshots in CI without API
keys. Tests are subprocess-based so deterministic-BLAS env vars
land before numpy import (pytest's Hypothesis plugin imports numpy
before test functions run).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT_OPENAI = REPO / "fixtures" / "bench_renders" / "path2_seed_long_paragraphs.json"
SNAPSHOT_ANTHROPIC = (
    REPO / "fixtures" / "bench_renders" / "path2_claude-haiku-4-5-20251001.json"
)
# Backwards-compat alias for the n=1 test below.
SNAPSHOT = SNAPSHOT_OPENAI

_DETERMINISTIC_BLAS_ENV = {
    **os.environ,
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

PINNED_PATH2_V3_DIGEST = (
    "7b364fc6ae23ce4ea24c69cf7b299b10402237f6f0c4364b18fbcb1dbcc4b75e"
)
PINNED_PATH2_V3_CLAUDE_DIGEST = (
    "d0f9f175662216d50dbfd1ec23d90eb8b4774bb95d220e2f951399e8ed52f6f7"
)


@pytest.mark.skipif(
    not SNAPSHOT.exists(), reason="snapshot missing — run Phase 1 with OPENAI_API_KEY",
)
def test_multi_llm_compare_single_model_gpt4o_mini():
    """Run the compare bench against only the gpt-4o-mini snapshot
    (Anthropic capture is operator-gated; CI cannot exercise it)."""
    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.research.sheaf_path2_multi_llm_compare",
            "--models", "gpt-4o-mini-2024-07-18",
        ],
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
        f"Compare bench did not print '→ wrote' line. stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    report = json.loads(out_path.read_text())

    assert report["schema"] == "sum.sheaf_path2_multi_llm_compare.v1"
    assert report["n_models"] == 1
    assert report["models"] == ["gpt-4o-mini-2024-07-18"]

    # The wrapper must not perturb the scoring path: the embedded
    # per-model report's bench_digest should match the Path 2 v3 pin.
    per_model = report["per_model_reports"]["gpt-4o-mini-2024-07-18"]
    assert per_model["bench_digest"] == PINNED_PATH2_V3_DIGEST, (
        f"Multi-LLM wrapper perturbed scoring: per-model digest "
        f"{per_model['bench_digest']} differs from Path 2 v3 pin "
        f"{PINNED_PATH2_V3_DIGEST}. The compare wrapper should be a "
        f"no-op on the scoring path."
    )

    # Verdict must not flip to BEATS on the cached snapshot — the
    # synthetic-vs-real gap is the load-bearing finding.
    assert per_model["verdict"] in (
        "HYBRID_TIES_BASELINE_ON_REAL_LLM",
        "HYBRID_LOSES_TO_BASELINE_ON_REAL_LLM",
    )

    # n=1 must NOT be classified as a cross-model structural finding.
    # SINGLE_MODEL_* is the honest label for n=1.
    joint = report["joint_finding"]
    assert joint.startswith("SINGLE_MODEL_"), (
        f"n=1 joint finding should be SINGLE_MODEL_*, got {joint!r}. "
        f"Cross-model structural classifications require ≥2 models — "
        f"reporting otherwise overstates the evidence."
    )

    # Δ-spread is mechanically zero for n=1.
    assert report["delta_spread"] == 0.0


@pytest.mark.skipif(
    not (SNAPSHOT_OPENAI.exists() and SNAPSHOT_ANTHROPIC.exists()),
    reason="n=2 case requires both gpt-4o-mini and claude-haiku snapshots",
)
def test_multi_llm_compare_two_models_structural_gap():
    """Run the compare bench against both committed snapshots. The
    load-bearing claim is that the §4.7.2 synthetic-vs-real gap is
    NOT gpt-4o-mini-specific — both LLM families have the hybrid
    LOSING to the B2 baseline."""
    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.research.sheaf_path2_multi_llm_compare",
            "--models", "gpt-4o-mini-2024-07-18", "claude-haiku-4-5-20251001",
        ],
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
        f"Compare bench did not print '→ wrote' line. stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    report = json.loads(out_path.read_text())

    assert report["schema"] == "sum.sheaf_path2_multi_llm_compare.v1"
    assert report["n_models"] == 2
    assert sorted(report["models"]) == [
        "claude-haiku-4-5-20251001", "gpt-4o-mini-2024-07-18",
    ]

    # Per-model digests — both Phase 2 paths are byte-stable given
    # the cached snapshots. Drift on either means the scorer changed.
    openai_report = report["per_model_reports"]["gpt-4o-mini-2024-07-18"]
    claude_report = report["per_model_reports"]["claude-haiku-4-5-20251001"]
    assert openai_report["bench_digest"] == PINNED_PATH2_V3_DIGEST, (
        f"gpt-4o-mini digest drift: got {openai_report['bench_digest']}"
    )
    assert claude_report["bench_digest"] == PINNED_PATH2_V3_CLAUDE_DIGEST, (
        f"claude-haiku-4-5 digest drift: got {claude_report['bench_digest']}"
    )

    # Joint finding: BOTH models LOSE to baseline. This is the
    # cross-family corroboration of the §4.7.2 synthetic-vs-real gap.
    # If this label flips to MIXED_VERDICTS_MODEL_DEPENDENT or
    # HYBRID_BEATS_*, the structural claim from §4.7.2 weakens and
    # the narrative needs rewriting.
    joint = report["joint_finding"]
    assert joint == "STRUCTURAL_GAP_ALL_MODELS_LOSE", (
        f"Joint finding drift: got {joint!r}. Both models should "
        f"have hybrid LOSING to baseline on real-LLM perturbations. "
        f"Per-model verdicts: {report['per_model_verdict']}, "
        f"deltas: {report['per_model_delta_borda_vs_b2']}. If this "
        f"label flipped, the §4.7.2 cross-family corroboration is "
        f"no longer load-bearing — investigate whether the scoring "
        f"composition or one of the snapshots changed."
    )

    # Δ-spread between models is small (~0.0065). Pin a generous
    # bound — a sudden 10× spread would mean one snapshot drifted
    # significantly from the other and the structural claim weakens.
    spread = report["delta_spread"]
    assert spread <= 0.05, (
        f"Δ-spread between models too large: {spread:.4f}. The "
        f"structural-gap claim depends on per-model deltas being "
        f"roughly in agreement (both LOSING by similar margins)."
    )
