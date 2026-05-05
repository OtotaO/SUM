"""
Shared helper: resolve the canonical output path for a research-bench
receipt, avoiding manifest drift on re-runs.

Problem: research bench scripts have historically written a date-stamped
receipt (e.g. `<schema_prefix>_2026-05-04.json`) every run. Re-running
the bench on a later date creates a new file (e.g. `..._2026-05-05.json`)
which then drifts `meta/repo_manifest.json`'s stable-fields check and
trips CI. The substantive content is byte-identical when the bench
itself reproduces (which all the load-bearing benches do), so creating
multiple dated copies is pure noise.

Resolution: by default, `resolve_receipt_path` returns the SOLE existing
date-stamped receipt for the bench's schema if exactly one is present
(the canonical historical filename, which the manifest knows about).
If zero exist, today's date is used (genuinely-new measurement).
If two or more exist, the lexicographically-latest is used (preserves
the convention of "newest is canonical" while preventing further proliferation).

This keeps the date-stamped naming convention for traceability of when
a measurement was first taken, while making re-runs idempotent at the
filesystem level.

Operators who genuinely want a new dated receipt can pass `force_new=True`
or specify `today_iso=` explicitly.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path


def resolve_receipt_path(
    receipts_dir: Path,
    schema_prefix: str,
    *,
    force_new: bool = False,
    today_iso: str | None = None,
) -> Path:
    """Return the canonical output path for a bench's receipt.

    Args:
        receipts_dir: directory holding bench receipts (e.g.
            `fixtures/bench_receipts/`).
        schema_prefix: filename stem matching the bench, without the
            date suffix (e.g. `"complementary_hybrid"`,
            `"baseline_comparison"`).
        force_new: if True, always return today's-dated path,
            ignoring any existing receipts. Use for genuinely-new
            measurements.
        today_iso: override today's date (testing).

    Behavior:
        - 1 existing match → that file (re-runs idempotent).
        - 0 existing matches → today's-dated new file.
        - 2+ existing matches → lexicographically-latest (the "newest"
          historical receipt), with a warning printed to stderr.
    """
    today = today_iso or _dt.date.today().isoformat()
    new_path = receipts_dir / f"{schema_prefix}_{today}.json"
    if force_new:
        return new_path
    existing = sorted(receipts_dir.glob(f"{schema_prefix}_*.json"))
    if len(existing) == 0:
        return new_path
    if len(existing) == 1:
        return existing[0]
    # 2+ existing — use latest, but flag it
    import sys
    print(
        f"[resolve_receipt_path] {len(existing)} existing receipts for "
        f"{schema_prefix!r}; using {existing[-1].name}. Consider archiving "
        f"older copies to fixtures/bench_receipts/archive/.",
        file=sys.stderr,
    )
    return existing[-1]
