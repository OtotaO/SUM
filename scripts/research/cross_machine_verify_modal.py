"""
Cross-machine bench_digest verification on Modal x86_64.

Builds a Modal Image pinned to a specific commit SHA, installs sum-engine
with the research extras, runs both load-bearing benches (v3.2 validation
+ complementary hybrid), and returns each digest plus environment data
(platform, numpy version, LAPACK provider).

Compares against the operator-side digests to determine §4.8 outcome:
  A. Both digests match → cross-machine reproducibility holds across LAPACK
  B. Digests differ but per-cell AUCs match within quantization → narrow
     digest claim to "byte-stable within identical LAPACK environments"
  C. AUCs themselves diverge → narrow further to single-machine

Cost: container cold-start ~3-5 min (one-time per image hash); function
exec <30s each. Total ~$0.001 in Modal credits per full run.

Operator preconditions:
  - `pip install modal` (confirmed)
  - `modal token new` configured (confirmed: profile=ototao)
  - PINNED_SHA must be reachable on origin/<branch>; this script's sibling
    branch arxiv/v0.1-cross-machine carries it.

Run with:
  modal run scripts/research/cross_machine_verify_modal.py
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

REPO_URL = "https://github.com/OtotaO/SUM.git"


def _local_repo_paths() -> tuple[Path, Path]:
    """Local-only — fails inside Modal container. Used only from
    local_entrypoint where __file__ has the repo's full path layout."""
    repo = Path(__file__).resolve().parents[2]
    return repo, repo / "fixtures" / "bench_receipts"

# Pinned to main HEAD post-merge of the Sprint 7+7.5 stack
# (was "37351e2" pre-merge; the squash-merge SHAs of #142, #146, #144,
# #145, #147, #148 are now on main; b5fe92b is the post-#148 HEAD).
# Update if you re-run after additional commits land.
PINNED_SHA = "b5fe92b"

EXPECTED_DIGESTS = {
    "v3_2_validation": (
        "b4d26c01d4962fa30f67c00313bbce8982ca16e3a97df34819747876ee14ed5a"
    ),
    "complementary_hybrid": (
        "dc6e0260f14042fa0b6151a6ca6b443bb0910eabb996f6876f854633969343ce"
    ),
}

app = modal.App("sum-cross-machine-verify")

# Image build: clone repo at pinned SHA, install with research extras.
# scripts/ is excluded from the wheel dist so we use editable install
# from the cloned tree (see CLAUDE.md packages.find rule).
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential")
    .run_commands(
        f"git clone {REPO_URL} /repo",
        # Post-merge: PINNED_SHA reachable from origin/main; no
        # branch-specific fetch required.
        f"cd /repo && git checkout {PINNED_SHA}",
        # research = numpy + scipy; sieve = spacy (DeterministicSieve
        # is required by extract_corpus_triples).
        "cd /repo && pip install -e '.[research,sieve]'",
        # spaCy model download — DeterministicSieve loads en_core_web_sm
        # at instantiation. ~12 MB, amortized into the image hash.
        "python -m spacy download en_core_web_sm",
    )
)


def _capture_env() -> dict[str, str]:
    """Return platform / numpy / LAPACK metadata for the receipt."""
    import io
    import platform
    import sys

    import numpy as np

    # numpy.show_config() prints to stdout; capture it
    buf = io.StringIO()
    saved = sys.stdout
    try:
        sys.stdout = buf
        np.show_config()
    finally:
        sys.stdout = saved

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "numpy_show_config": buf.getvalue(),
    }


@app.function(image=image, timeout=600)
def verify_v32_validation_digest() -> dict[str, object]:
    """Run v3.2 validation bench and return digest + environment."""
    import sys
    sys.path.insert(0, "/repo")
    from scripts.research.sheaf_v3_2_validation import main as run_v32_main
    report = run_v32_main()
    env = _capture_env()
    return {
        "bench_name": "v3_2_validation",
        "bench_digest": report["bench_digest"],
        "trusted_mean_auc_by_gamma": {
            label: float(report["per_cell_auc_by_gamma"].get(
                f"v32_{label}|A1|trusted", 0.0
            )) for label in ("0.0", "0.1", "1.0", "auto")
        },
        "n_docs_with_partition": int(report["n_docs_with_partition"]),
        "lambda_auto": float(report["lambda_auto_calibrated"]),
        **env,
    }


@app.function(image=image, timeout=600)
def verify_complementary_hybrid_digest() -> dict[str, object]:
    """Run complementary hybrid bench (the load-bearing WIN) and return
    digest + environment."""
    import sys
    sys.path.insert(0, "/repo")
    from scripts.research.sheaf_complementary_hybrid_experiment import (
        run_experiment as run_hybrid,
    )
    report = run_hybrid()
    env = _capture_env()
    return {
        "bench_name": "complementary_hybrid",
        "bench_digest": report["bench_digest"],
        "verdict": report["verdict"],
        "trusted_mean_auc_by_detector": {
            k: float(v) for k, v in report["trusted_mean_auc_by_detector"].items()
        },
        "delta_borda_vs_b2_trusted_mean": float(
            report["delta_borda_vs_b2_trusted_mean"]
        ),
        "n_docs_with_partition": int(report["n_docs_with_partition"]),
        **env,
    }


@app.local_entrypoint()
def main():
    print("=" * 72)
    print("SUM cross-machine bench_digest verification on Modal x86_64")
    print(f"  pinned SHA: {PINNED_SHA}")
    print("=" * 72)

    # Run both functions remotely.
    print("\n[1] Running v3.2 validation bench on Modal…")
    v32 = verify_v32_validation_digest.remote()
    print(f"    bench_digest = {v32['bench_digest']}")

    print("\n[2] Running complementary hybrid bench on Modal…")
    hyb = verify_complementary_hybrid_digest.remote()
    print(f"    bench_digest = {hyb['bench_digest']}")

    # Compare against expected (operator-side) digests.
    print("\n[3] Comparison vs operator-side digests:")
    rows = [
        ("v3_2_validation", v32["bench_digest"], EXPECTED_DIGESTS["v3_2_validation"]),
        ("complementary_hybrid", hyb["bench_digest"], EXPECTED_DIGESTS["complementary_hybrid"]),
    ]
    outcomes: dict[str, str] = {}
    for name, got, expected in rows:
        match = got == expected
        outcomes[name] = "MATCH" if match else "DIGEST_DIFFER"
        print(f"    {name:30s} {outcomes[name]}")
        print(f"      operator: {expected}")
        print(f"      modal:    {got}")

    # Build receipt.
    receipt = {
        "schema": "sum.cross_machine_verification.v1",
        "pinned_sha": PINNED_SHA,
        "operator_environment": {
            "platform": "Apple Silicon (operator-side; documented separately)",
            "lapack_provider": "Apple Accelerate (assumed; per CLAUDE.md profile)",
        },
        "modal_environment": {
            "platform": v32["platform"],
            "machine": v32["machine"],
            "python_version": v32["python_version"],
            "numpy_version": v32["numpy_version"],
            "numpy_show_config": v32["numpy_show_config"],
        },
        "v3_2_validation": {
            "operator_digest": EXPECTED_DIGESTS["v3_2_validation"],
            "modal_digest": v32["bench_digest"],
            "outcome": outcomes["v3_2_validation"],
            "modal_trusted_mean_auc_by_gamma": v32["trusted_mean_auc_by_gamma"],
            "modal_n_docs_with_partition": v32["n_docs_with_partition"],
            "modal_lambda_auto": v32["lambda_auto"],
        },
        "complementary_hybrid": {
            "operator_digest": EXPECTED_DIGESTS["complementary_hybrid"],
            "modal_digest": hyb["bench_digest"],
            "outcome": outcomes["complementary_hybrid"],
            "modal_verdict": hyb["verdict"],
            "modal_trusted_mean_auc_by_detector": hyb["trusted_mean_auc_by_detector"],
            "modal_delta_borda_vs_b2_trusted_mean": hyb["delta_borda_vs_b2_trusted_mean"],
            "modal_n_docs_with_partition": hyb["n_docs_with_partition"],
        },
    }

    # Honest §4.8 outcome label
    if all(o == "MATCH" for o in outcomes.values()):
        section_4_8_outcome = "BRANCH_A_DIGESTS_MATCH"
    else:
        # Need to compare AUCs to distinguish B vs C; for now, label as B
        # (digest differs) and let operator judge whether AUCs match within
        # quantization on inspection of the receipt.
        section_4_8_outcome = "BRANCH_B_OR_C_DIGEST_DIFFERS_INVESTIGATE_AUCS"
    receipt["section_4_8_outcome"] = section_4_8_outcome
    print(f"\n[4] §4.8 outcome label: {section_4_8_outcome}")

    # Write receipt
    import datetime as _dt
    _, receipts_dir = _local_repo_paths()
    today = _dt.date.today().isoformat()
    out = receipts_dir / f"cross_machine_verification_{today}.json"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"\n→ wrote {out}")
