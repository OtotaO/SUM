"""
Pin the multi-LLM compare's Phase-2 path through the gpt-4o-mini
snapshot. The Anthropic snapshot is operator-gated (Phase 1 needs
ANTHROPIC_API_KEY); this test exercises only the deterministic
single-model path so CI does not depend on a second LLM family
having been captured.

The load-bearing assertions are:

  - per-model verdict for gpt-4o-mini matches the Path 2 v3 finding
    (HYBRID does NOT beat the B2 baseline on real-LLM perturbations)
  - per-model bench_digest is byte-identical to the pin in
    Tests/research/test_sheaf_path2_v3.py — proves the multi-LLM
    wrapper is a no-op on the scoring path
  - joint_finding for n=1 is reported as SINGLE_MODEL_<verdict>,
    not dressed up as a cross-model structural finding
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

PINNED_PATH2_V3_DIGEST = (
    "7b364fc6ae23ce4ea24c69cf7b299b10402237f6f0c4364b18fbcb1dbcc4b75e"
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
