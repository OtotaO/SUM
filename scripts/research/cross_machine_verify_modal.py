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

# Pinned to main HEAD post-merge of the v0.3 deterministic-BLAS fix.
# SHA progression for traceability:
#   "37351e2" pre-merge (orig PR #144 branch tip)
#   "b5fe92b" post Sprint 7.5 merges (#142/#146/#144/#145/#147/#148)
#   "5715c40" post #150 (latent-fix arc — predicate_negatives refactor +
#             borda_fuse tie-break + receipt-path helper + Py 3.12 env)
#   "0f2f079" post #154 (v0.3 deterministic-BLAS fix — VECLIB/OPENBLAS
#             thread-count env vars set at module-import time;
#             closes the last shape-pinned recovery digest.
#             hybrid_comparison now byte-digest pinned).
# Update if you re-run after additional commits land.
PINNED_SHA = "0f2f079"

EXPECTED_DIGESTS = {
    "v3_2_validation": (
        "b4d26c01d4962fa30f67c00313bbce8982ca16e3a97df34819747876ee14ed5a"
    ),
    "complementary_hybrid": (
        "dc6e0260f14042fa0b6151a6ca6b443bb0910eabb996f6876f854633969343ce"
    ),
    # 2026-05-05: post-Issue-3 refactor digest. Pinned in
    # Tests/research/test_recovery_experiment_digests.py.
    "predicate_negatives": (
        "ddf41484b1eba2f1cf5927d6e9691a922e5843be703fedac83e8afee001f59c3"
    ),
    # v0.3 deterministic-BLAS fix: hybrid_comparison was previously
    # shape-pinned because BLAS thread-pool variance at numpy-import
    # time produced two stable outcomes across fresh procs. With the
    # `_deterministic_blas` helper imported before numpy in the bench
    # script, the digest is now byte-stable. Pinned in
    # Tests/research/test_recovery_experiment_digests.py.
    "hybrid_comparison": (
        "a7965803ccf2e703d80364dc21b3ac410491db9768cdfcf91bfefd29356c2003"
    ),
}

app = modal.App("sum-cross-machine-verify")

# Image build helper: clone repo at pinned SHA, install with research
# extras. scripts/ is excluded from the wheel dist so we use editable
# install from the cloned tree (see CLAUDE.md packages.find rule).
def _build_image(python_version: str):
    return (
        modal.Image.debian_slim(python_version=python_version)
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


# Two LAPACK environments tested:
#   image_310 — Python 3.10 + numpy 1.25 + OpenBLAS-via-PyPI (was the
#               original Modal target; already verified MATCH against
#               operator's Apple Accelerate).
#   image_312 — Python 3.12 + numpy 2.x + OpenBLAS-via-PyPI (new in
#               this v0.2 latent-fix; verifies cross-Python-version
#               digest stability for the post-Issue-3-refactor
#               predicate_negatives bench, AND adds a third LAPACK
#               environment to §4.8's published cross-machine claim).
image_310 = _build_image("3.10")
image_312 = _build_image("3.12")


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


def _run_v32_validation() -> dict[str, object]:
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


def _run_complementary_hybrid() -> dict[str, object]:
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


def _run_hybrid_comparison() -> dict[str, object]:
    """v0.3 deterministic-BLAS-fix addition: verify hybrid_comparison
    bench_digest is now byte-stable across fresh procs (was shape-pinned
    pre-v0.3 because cochain-only Borda fusion's per-cell AUC was
    sensitive to BLAS thread-pool-size variance at numpy-import time)."""
    import sys
    sys.path.insert(0, "/repo")
    from scripts.research.sheaf_hybrid_comparison import run_hybrid_comparison
    report = run_hybrid_comparison()
    env = _capture_env()
    return {
        "bench_name": "hybrid_comparison",
        "bench_digest": report["bench_digest"],
        "verdict": report["verdict"],
        "delta_borda_vs_b2_trusted_mean": float(
            report["delta_borda_vs_b2_trusted_mean"]
        ),
        "n_docs_with_partition": int(report["n_docs_with_partition"]),
        **env,
    }


def _run_predicate_negatives() -> dict[str, object]:
    """v0.2 latent-fix addition: verify the post-refactor
    predicate_negatives bench digest is cross-Python-version stable
    (was Python-version-sensitive when bench used a local v2-training
    copy; now uses production train_restriction_maps with new
    n_predicate_negatives_per_positive parameter)."""
    import sys
    sys.path.insert(0, "/repo")
    from scripts.research.sheaf_predicate_negatives_experiment import run_experiment
    report = run_experiment()
    env = _capture_env()
    a2_t = report["per_cell_auc"].get("v32_g0.1_pred_neg|A2|trusted")
    a2_u = report["per_cell_auc"].get("v32_g0.1_pred_neg|A2|untrusted")
    return {
        "bench_name": "predicate_negatives",
        "bench_digest": report["bench_digest"],
        "verdict": report["verdict"],
        "a2_trusted": a2_t,
        "a2_untrusted": a2_u,
        "n_docs_with_partition": int(report["n_docs_with_partition"]),
        **env,
    }


# Python 3.10 (matches operator) — was the existing single-env target.
@app.function(image=image_310, timeout=600)
def verify_v32_validation_digest_310() -> dict[str, object]:
    return _run_v32_validation()


@app.function(image=image_310, timeout=600)
def verify_complementary_hybrid_digest_310() -> dict[str, object]:
    return _run_complementary_hybrid()


@app.function(image=image_310, timeout=600)
def verify_predicate_negatives_digest_310() -> dict[str, object]:
    return _run_predicate_negatives()


@app.function(image=image_310, timeout=600)
def verify_hybrid_comparison_digest_310() -> dict[str, object]:
    return _run_hybrid_comparison()


# Python 3.12 (numpy 2.x; tests cross-version + cross-LAPACK-version).
@app.function(image=image_312, timeout=600)
def verify_v32_validation_digest_312() -> dict[str, object]:
    return _run_v32_validation()


@app.function(image=image_312, timeout=600)
def verify_complementary_hybrid_digest_312() -> dict[str, object]:
    return _run_complementary_hybrid()


@app.function(image=image_312, timeout=600)
def verify_predicate_negatives_digest_312() -> dict[str, object]:
    return _run_predicate_negatives()


@app.function(image=image_312, timeout=600)
def verify_hybrid_comparison_digest_312() -> dict[str, object]:
    return _run_hybrid_comparison()


@app.local_entrypoint()
def main():
    print("=" * 72)
    print("SUM cross-machine bench_digest verification on Modal x86_64")
    print(f"  pinned SHA: {PINNED_SHA}")
    print("  environments: Python 3.10 + 3.12 (both Debian slim, OpenBLAS)")
    print("=" * 72)

    # Run all 8 invocations: 4 benches × 2 Python versions.
    print("\n[1] Python 3.10 — v3.2 validation…")
    v32_310 = verify_v32_validation_digest_310.remote()
    print(f"    bench_digest = {v32_310['bench_digest']}")
    print("\n[2] Python 3.10 — complementary hybrid…")
    hyb_310 = verify_complementary_hybrid_digest_310.remote()
    print(f"    bench_digest = {hyb_310['bench_digest']}")
    print("\n[3] Python 3.10 — predicate negatives…")
    pn_310 = verify_predicate_negatives_digest_310.remote()
    print(f"    bench_digest = {pn_310['bench_digest']}")
    print("\n[4] Python 3.10 — hybrid comparison (v0.3 addition)…")
    hc_310 = verify_hybrid_comparison_digest_310.remote()
    print(f"    bench_digest = {hc_310['bench_digest']}")
    print("\n[5] Python 3.12 — v3.2 validation…")
    v32_312 = verify_v32_validation_digest_312.remote()
    print(f"    bench_digest = {v32_312['bench_digest']}")
    print("\n[6] Python 3.12 — complementary hybrid…")
    hyb_312 = verify_complementary_hybrid_digest_312.remote()
    print(f"    bench_digest = {hyb_312['bench_digest']}")
    print("\n[7] Python 3.12 — predicate negatives…")
    pn_312 = verify_predicate_negatives_digest_312.remote()
    print(f"    bench_digest = {pn_312['bench_digest']}")
    print("\n[8] Python 3.12 — hybrid comparison (v0.3 addition)…")
    hc_312 = verify_hybrid_comparison_digest_312.remote()
    print(f"    bench_digest = {hc_312['bench_digest']}")

    # Compare across operator + Modal-310 + Modal-312.
    print("\n[9] Cross-environment comparison:")
    rows = [
        ("v3_2_validation",       v32_310["bench_digest"], v32_312["bench_digest"], EXPECTED_DIGESTS["v3_2_validation"]),
        ("complementary_hybrid",  hyb_310["bench_digest"], hyb_312["bench_digest"], EXPECTED_DIGESTS["complementary_hybrid"]),
        ("predicate_negatives",   pn_310["bench_digest"],  pn_312["bench_digest"],  EXPECTED_DIGESTS["predicate_negatives"]),
        ("hybrid_comparison",     hc_310["bench_digest"],  hc_312["bench_digest"],  EXPECTED_DIGESTS["hybrid_comparison"]),
    ]
    outcomes: dict[str, str] = {}
    for name, m310, m312, op in rows:
        match_310 = m310 == op
        match_312 = m312 == op
        match_inter = m310 == m312
        if match_310 and match_312:
            outcomes[name] = "ALL_MATCH"
        elif match_inter and not match_310:
            outcomes[name] = "MODAL_INTERNAL_MATCH_BUT_DIFFER_FROM_OPERATOR"
        elif match_310 and not match_312:
            outcomes[name] = "OPERATOR_AND_310_MATCH_BUT_312_DIFFERS"
        elif match_312 and not match_310:
            outcomes[name] = "OPERATOR_AND_312_MATCH_BUT_310_DIFFERS"
        else:
            outcomes[name] = "ALL_THREE_DIFFER"
        print(f"    {name:30s} {outcomes[name]}")
        print(f"      operator:        {op}")
        print(f"      modal py3.10:    {m310}")
        print(f"      modal py3.12:    {m312}")

    # Build receipt with all three environments + per-bench outcomes.
    receipt = {
        "schema": "sum.cross_machine_verification.v1",
        "pinned_sha": PINNED_SHA,
        "operator_environment": {
            "platform": "Apple Silicon (operator-side; documented separately)",
            "lapack_provider": "Apple Accelerate (assumed; per CLAUDE.md profile)",
        },
        "modal_python_3_10_environment": {
            "platform": v32_310["platform"],
            "machine": v32_310["machine"],
            "python_version": v32_310["python_version"],
            "numpy_version": v32_310["numpy_version"],
            "numpy_show_config": v32_310["numpy_show_config"],
        },
        "modal_python_3_12_environment": {
            "platform": v32_312["platform"],
            "machine": v32_312["machine"],
            "python_version": v32_312["python_version"],
            "numpy_version": v32_312["numpy_version"],
            "numpy_show_config": v32_312["numpy_show_config"],
        },
        "v3_2_validation": {
            "operator_digest": EXPECTED_DIGESTS["v3_2_validation"],
            "modal_digest_310": v32_310["bench_digest"],
            "modal_digest_312": v32_312["bench_digest"],
            "outcome": outcomes["v3_2_validation"],
            "modal_310_trusted_mean_auc_by_gamma": v32_310["trusted_mean_auc_by_gamma"],
            "modal_312_trusted_mean_auc_by_gamma": v32_312["trusted_mean_auc_by_gamma"],
            "modal_310_lambda_auto": v32_310["lambda_auto"],
            "modal_312_lambda_auto": v32_312["lambda_auto"],
        },
        "complementary_hybrid": {
            "operator_digest": EXPECTED_DIGESTS["complementary_hybrid"],
            "modal_digest_310": hyb_310["bench_digest"],
            "modal_digest_312": hyb_312["bench_digest"],
            "outcome": outcomes["complementary_hybrid"],
            "modal_310_verdict": hyb_310["verdict"],
            "modal_312_verdict": hyb_312["verdict"],
            "modal_310_trusted_mean_auc_by_detector": hyb_310["trusted_mean_auc_by_detector"],
            "modal_312_trusted_mean_auc_by_detector": hyb_312["trusted_mean_auc_by_detector"],
            "modal_310_delta_borda_vs_b2_trusted_mean": hyb_310["delta_borda_vs_b2_trusted_mean"],
            "modal_312_delta_borda_vs_b2_trusted_mean": hyb_312["delta_borda_vs_b2_trusted_mean"],
        },
        "predicate_negatives": {
            "operator_digest": EXPECTED_DIGESTS["predicate_negatives"],
            "modal_digest_310": pn_310["bench_digest"],
            "modal_digest_312": pn_312["bench_digest"],
            "outcome": outcomes["predicate_negatives"],
            "modal_310_verdict": pn_310["verdict"],
            "modal_312_verdict": pn_312["verdict"],
            "modal_310_a2_trusted": pn_310["a2_trusted"],
            "modal_312_a2_trusted": pn_312["a2_trusted"],
            "modal_310_a2_untrusted": pn_310["a2_untrusted"],
            "modal_312_a2_untrusted": pn_312["a2_untrusted"],
        },
        "hybrid_comparison": {
            "operator_digest": EXPECTED_DIGESTS["hybrid_comparison"],
            "modal_digest_310": hc_310["bench_digest"],
            "modal_digest_312": hc_312["bench_digest"],
            "outcome": outcomes["hybrid_comparison"],
            "modal_310_verdict": hc_310["verdict"],
            "modal_312_verdict": hc_312["verdict"],
            "modal_310_delta_borda_vs_b2_trusted_mean": hc_310["delta_borda_vs_b2_trusted_mean"],
            "modal_312_delta_borda_vs_b2_trusted_mean": hc_312["delta_borda_vs_b2_trusted_mean"],
        },
    }
    # Aggregate the per-receipt outcomes for compatibility with §4.8 prose
    # which speaks of "BRANCH_A" naming.
    v32 = v32_310     # for compat with old receipt-write code below
    hyb = hyb_310

    # Honest §4.8 outcome label
    if all(o == "ALL_MATCH" for o in outcomes.values()):
        section_4_8_outcome = "BRANCH_A_THREE_ENVIRONMENTS_DIGESTS_MATCH"
    elif all("MATCH" in o for o in outcomes.values()):
        # Some non-ALL_MATCH partial matches — still informative
        section_4_8_outcome = "BRANCH_A_PARTIAL_PER_BENCH_INVESTIGATE_OUTCOMES"
    else:
        section_4_8_outcome = "BRANCH_B_OR_C_DIGEST_DIFFERS_INVESTIGATE_OUTCOMES"
    receipt["section_4_8_outcome"] = section_4_8_outcome
    print(f"\n[10] §4.8 outcome label: {section_4_8_outcome}")

    # Write receipt
    _, receipts_dir = _local_repo_paths()
    receipts_dir.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(receipts_dir.parent.parent))
    from scripts.research._receipt_paths import resolve_receipt_path
    out = resolve_receipt_path(receipts_dir, "cross_machine_verification")
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"\n→ wrote {out}")
