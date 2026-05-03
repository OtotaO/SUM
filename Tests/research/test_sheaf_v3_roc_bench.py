"""Smoke test: the v3 ROC bench runs end-to-end and produces a
well-formed receipt with the expected verdict structure.

Does NOT pin specific AUC numbers — there's ~±0.02 LAPACK jitter
in `np.linalg.lstsq` inside `harmonic_extension` that causes per-
cell AUC drift. The aggregate verdicts (F1/F2/F3) are stable.

This test runs in ~60-90 seconds locally. CI should run it once;
local devs can skip via ``-m 'not slow'`` if the bench file gets
heavy.
"""
from __future__ import annotations

import pytest

# Skip whole file if [research] extras aren't installed
np = pytest.importorskip("numpy")


pytestmark = pytest.mark.slow


def test_v3_roc_bench_runs_end_to_end_with_expected_verdict_shape():
    """The bench script should run without exception, return a
    receipt dict with the documented schema, and surface F1/F2/F3
    verdict keys with sensible string values.

    We don't pin per-cell AUC numbers (LAPACK jitter), but we DO
    pin:
      - schema name
      - presence of every documented top-level field
      - F2 verdict is PASS (the most stable verdict — v3 doesn't
        collapse on untrusted, robust across runs)
      - F3 verdict is in {PASS, FAIL} (it's currently FAIL on
        seed_long_paragraphs; if it ever PASSes, the corpus-scale
        utility of v3.1 boundary deviation has been earned and
        the spec doc's §3.4.2 should be updated)
    """
    from scripts.research.sheaf_v3_roc_bench import main

    receipt = main()

    # Schema + top-level keys
    assert receipt["schema"] == "sum.sheaf_v3_roc_bench.v1"
    expected_keys = {
        "corpus", "n_docs_total", "n_docs_with_partition",
        "vocab_size_entities", "vocab_size_relations", "stalk_dim",
        "training_epochs", "lambda_auto_calibrated",
        "trust_partition", "weights_from_receipts",
        "per_cell_auc", "head_to_head_v3_vs_v22",
        "f1_v3_beats_v22_on_trusted_mean_auc",
        "f2_v3_no_collapse_on_untrusted",
        "f3_v31_boundary_deviation_corpus_scale",
    }
    missing = expected_keys - set(receipt)
    assert not missing, f"receipt missing keys: {missing}"

    # F2 PASS pin: this is the most robust empirical claim.
    # If v3 ever collapses by > 0.10 AUC on untrusted-target
    # perturbations, we want to know immediately — that breaks
    # v3's utility floor.
    assert receipt["f2_v3_no_collapse_on_untrusted"]["verdict"] == "PASS", (
        f"F2 must PASS — v3 must not collapse on untrusted-target "
        f"perturbations. Got: {receipt['f2_v3_no_collapse_on_untrusted']}"
    )

    # F3 pin (currently FAIL): if F3 ever PASSes, that's good news
    # but also means the spec doc §3.4.2 description of v3.1's
    # corpus-scale failure is stale and should be updated.
    f3_verdict = receipt["f3_v31_boundary_deviation_corpus_scale"]["verdict"]
    assert f3_verdict in ("PASS", "FAIL"), (
        f"F3 verdict must be PASS or FAIL; got {f3_verdict!r}"
    )
    if f3_verdict == "PASS":
        # If v3.1 starts working at corpus scale, that's a major
        # finding — make sure docs/SHEAF_HALLUCINATION_DETECTOR.md
        # §3.4.2 (which currently says "F3 FAIL") gets updated.
        # We don't fail the test on PASS, but flag it loudly.
        import warnings
        warnings.warn(
            "F3 PASSed (v3.1 boundary deviation now works at "
            "corpus scale). Update docs/SHEAF_HALLUCINATION_"
            "DETECTOR.md §3.4.2 to reflect this — the doc "
            "currently says FAIL and that's now stale.",
            stacklevel=1,
        )

    # Per-cell shape: 3 detectors × 3 classes × 2 targets = 18 cells
    cells = receipt["per_cell_auc"]
    expected_cell_count = 3 * 3 * 2
    assert len(cells) == expected_cell_count, (
        f"expected {expected_cell_count} per-cell AUC values, "
        f"got {len(cells)}"
    )

    # Every AUC ∈ [0, 1]
    for cell, auc in cells.items():
        assert 0.0 <= auc <= 1.0, f"AUC for {cell} out of [0, 1]: {auc}"

    # Headline anchored claim: A4 triple-drop catches strongly under
    # both detectors. Any combined v22+v3 mean AUC on A4 below 0.7
    # would mean the detector is broken — pin this as a sanity floor.
    a4_aucs = [
        cells.get("v22|A4|trusted", 0.5),
        cells.get("v22|A4|untrusted", 0.5),
        cells.get("v3|A4|trusted", 0.5),
        cells.get("v3|A4|untrusted", 0.5),
    ]
    a4_mean = float(np.mean(a4_aucs))
    assert a4_mean >= 0.7, (
        f"A4 triple-drop mean AUC across v22+v3 must be ≥ 0.7 "
        f"(current measurements ~0.9); got {a4_mean:.3f}. "
        f"If this fires, the trained sheaf is no longer detecting "
        f"missing triples — major regression."
    )
