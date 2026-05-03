"""Smoke + shape pin for the F3 diagnostic harness.

Pins:
  - Diagnostic runs end-to-end without exception.
  - Returns a DiagnosticReport with the documented schema.
  - All 8 cells present.
  - At least one cell produces non-degenerate diagnostics (boundary
    size > 0 AND interior size > 0).
  - load_bearing_hypothesis is in the documented enum.

Does NOT pin specific AUC numbers — same LAPACK jitter as PR #124.
"""
from __future__ import annotations

import pytest

# Skip whole file if [research] extras aren't installed
np = pytest.importorskip("numpy")


pytestmark = pytest.mark.slow


def test_v3_1_f3_diagnostic_runs_with_expected_shape() -> None:
    """End-to-end smoke + shape pin.

    Pins:
      - main() runs without exception.
      - DiagnosticReport schema is sum.sheaf_v3_1_f3_diagnostic.v1.
      - All 8 cells present (2×2×2 over graph_size × cochain × partition).
      - load_bearing_hypothesis ∈ {A, B, C, none, multiple}.
      - bench_digest is a 64-char lowercase hex SHA-256.
      - At least one non-degenerate cell (boundary_size > 0 AND
        interior_size > 0) exists — otherwise the harness measured
        nothing and the report is uninformative.

    Does NOT pin specific AUC numbers — LAPACK threading inside
    np.linalg.lstsq introduces ~±0.02 jitter (same caveat as
    PR #124's bench).
    """
    import re
    from scripts.research.sheaf_v3_1_f3_diagnostic import (
        CellConfig,
        DiagnosticReport,
        main,
    )

    report = main()

    assert isinstance(report, DiagnosticReport)
    assert report.schema == "sum.sheaf_v3_1_f3_diagnostic.v1"
    assert report.corpus == "seed_long_paragraphs"
    assert len(report.cells) == 8, (
        f"expected 8 cells (2×2×2); got {len(report.cells)}"
    )

    # Cell ids cover the full 2×2×2 product
    expected_ids = {
        f"{g}|{c}|{p}"
        for g in ("per_doc", "aggregated_4_docs")
        for c in ("one_hot_default", "trained_embedding")
        for p in ("random_50_50", "concentrated_by_doc")
    }
    actual_ids = {cell.config_cell_id for cell in report.cells}
    assert actual_ids == expected_ids, (
        f"cell ids don't cover the 2×2×2 product; "
        f"missing {expected_ids - actual_ids}, "
        f"extra {actual_ids - expected_ids}"
    )

    # Verdict is a valid enum value
    assert report.load_bearing_hypothesis in ("A", "B", "C", "none", "multiple")

    # Digest is a well-formed SHA-256 hex string
    assert re.fullmatch(r"[0-9a-f]{64}", report.bench_digest), (
        f"bench_digest must be 64-char lowercase hex; got {report.bench_digest!r}"
    )

    # At least one non-degenerate cell — otherwise the harness
    # measured nothing useful.
    non_degenerate = [
        cell for cell in report.cells
        if cell.diagnostics.mean_boundary_size is not None
        and cell.diagnostics.mean_interior_size is not None
        and cell.diagnostics.mean_boundary_size > 0
        and cell.diagnostics.mean_interior_size > 0
    ]
    assert non_degenerate, (
        "no cell produced a non-degenerate boundary/interior — "
        "the F3 diagnostic measured nothing. Either every cell's "
        "partition produced fully-trusted or fully-untrusted graphs "
        "(check partition logic) or the trust threshold filter is "
        "stripping every vertex."
    )


def test_v3_1_f3_diagnostic_digest_is_quantization_stable() -> None:
    """Two consecutive runs must produce the same digest.

    The digest's reason for being is reproducibility under LAPACK
    jitter. If quantization (3 decimals on AUCs, 4 on diagnostic
    floats) doesn't absorb the jitter, the digest is a noise
    canary, not a reproducibility witness — and one of the three
    use cases (cross-runtime portability, signable artifact,
    reproducibility canary) collapses.

    If this test ever flakes, EITHER (a) tighten quantization
    further, OR (b) accept that the diagnostic isn't bit-stable
    and update the bench_digest docstring to reflect that — don't
    pretend the digest reproduces when it doesn't.
    """
    from scripts.research.sheaf_v3_1_f3_diagnostic import main

    r1 = main()
    r2 = main()
    assert r1.bench_digest == r2.bench_digest, (
        f"digest drift across in-process re-runs:\n"
        f"  run 1: {r1.bench_digest}\n"
        f"  run 2: {r2.bench_digest}\n"
        f"Quantization (AUCs to 3 decimals; diagnostics to 4) was "
        f"insufficient to absorb LAPACK jitter. Either tighten "
        f"quantization or revise bench_digest's contract."
    )
