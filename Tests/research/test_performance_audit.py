"""
Pin the performance-audit receipt at the level of *measured shape*,
not exact wall-clock numbers. Timing benches are inherently noisy;
the substantive claim is the scaling exponents and the bottleneck
identity, not the microsecond counts.

This test reads the on-disk receipt at
`fixtures/bench_receipts/performance_characterisation_2026-05-07.json`
and asserts:

  - schema version
  - bottleneck operation is `merge` at the largest measured size
  - scaling exponents land in the expected ranges:
      ingest k ≈ 0   (constant per-triple)
      entail k ≈ 1.0 (linear in N)
      merge  k in [1.3, 1.7]  (sub-quadratic empirical)
      encode k in [1.7, 2.1]  (near-quadratic empirical)
  - recursive-compression feasibility envelope produces estimates for
    every named target (small_book through modest_library)
  - the bench_digest field is present and well-formed (64 hex chars)

The test does NOT re-run the audit (timing tests in CI would be
flaky); it pins the committed receipt's substantive shape. Re-running
the audit and committing a fresh receipt will require updating the
expected ranges if the fits genuinely shift — which is itself a
useful signal that something architectural changed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RECEIPT = (
    REPO / "fixtures" / "bench_receipts"
    / "performance_characterisation_2026-05-07.json"
)


@pytest.mark.skipif(
    not RECEIPT.exists(),
    reason="performance audit receipt not committed; run "
           "scripts.bench.performance_audit to produce it",
)
def test_performance_audit_receipt_shape():
    """Pin the receipt's substantive shape, not the timing details."""
    report = json.loads(RECEIPT.read_text())

    assert report["schema"] == "sum.performance_characterisation.v1"
    assert isinstance(report.get("bench_digest"), str)
    assert re.fullmatch(r"[0-9a-f]{64}", report["bench_digest"]), (
        f"bench_digest is not a 64-char hex string: "
        f"{report['bench_digest']!r}"
    )

    # Bottleneck identity is the load-bearing claim.
    bottleneck = report["bottleneck_at_largest_size"]
    assert bottleneck["operation"] == "merge", (
        f"Bottleneck drift: expected `merge` at the largest measured size, "
        f"got {bottleneck['operation']!r}. PROOF_BOUNDARY §4 asserts merge "
        f"is the substrate's scaling bottleneck; if a different op is "
        f"slower at N=10000 the substrate has changed shape."
    )
    assert bottleneck["corpus_size"] == 10000

    # Scaling exponents — pinned to the regime observed empirically.
    fits = report["scaling_fits"]
    assert -0.1 < fits["ingest"]["k"] < 0.1, (
        f"ingest scaling k drift: expected ~0 (constant per-triple), "
        f"got {fits['ingest']['k']}. If non-constant, the get-or-mint "
        f"path has acquired N-dependent work."
    )
    assert 0.85 < fits["entail"]["k"] < 1.15, (
        f"entail scaling k drift: expected ~1.0 (linear in N via "
        f"O(bit-length) state-integer modulo), got {fits['entail']['k']}."
    )
    assert 1.3 < fits["merge"]["k"] < 1.75, (
        f"merge scaling k drift: expected 1.3-1.7 (sub-quadratic "
        f"empirical via Zig-accelerated GCD), got {fits['merge']['k']}. "
        f"The asymptotic is O(N²) for naive big-int LCM; significantly "
        f"higher k means the Zig path has dropped out or the GCD impl "
        f"has regressed."
    )
    assert 1.7 < fits["encode"]["k"] < 2.15, (
        f"encode scaling k drift: expected 1.7-2.1 (near-quadratic from "
        f"O(N) iter × O(N) bit-length), got {fits['encode']['k']}."
    )

    # Recursive-compression feasibility envelope: every named target
    # should produce a numeric per-iter estimate.
    feasibility = report["recursive_compression_feasibility"]
    estimates = feasibility["library_scale_estimates"]
    expected_targets = {
        "small_book", "medium_book", "large_book",
        "small_library", "modest_library",
    }
    seen = {est["target"] for est in estimates}
    assert seen == expected_targets, (
        f"feasibility envelope target set drift: expected "
        f"{expected_targets}, got {seen}."
    )
    for est in estimates:
        if not est.get("extrapolated_beyond_measured"):
            # Measured-range targets must have positive estimates.
            assert est["per_iter_est_s"] > 0, (
                f"{est['target']} per-iter estimate is non-positive: {est}"
            )

    # Zig engine status — the receipt should record whether Zig was
    # active during measurement; whichever value is fine, but the field
    # must be present.
    zig = report["zig_engine"]
    assert "available" in zig, (
        f"zig_engine record missing 'available' field: {zig}"
    )
