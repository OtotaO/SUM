"""Phase E.2 bench harness — measures per-axis drift across a corpus.

For each (corpus_doc, axis, axis_position) cell:
    1. Extract source triples from doc.
    2. Render via slider_renderer.render at the axis position
       (other axes at default 0.5 / density=1.0).
    3. Re-extract triples from the rendered tome.
    4. Compute drift_per_axis per docs/SLIDER_CONTRACT.md.
    5. Emit one JSONL row per cell.

Aggregate output: median + p75 + p90 drift per (axis, position) over
the full corpus. The thresholds in docs/SLIDER_CONTRACT.md are derived
from these distributions.

SCAFFOLD STATE: type signatures + JSON schema only. Full bench loop
ships in EXECUTE state alongside the renderer logic.

Author: ototao
License: Apache License 2.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

# Per docs/SLIDER_CONTRACT.md the bin centres for 5 bins are
# {0.1, 0.3, 0.5, 0.7, 0.9}. The bench probes every position.
AXIS_POSITIONS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)

AXES: tuple[str, ...] = ("density", "length", "formality", "audience", "perspective")


@dataclass(frozen=True)
class BenchCell:
    """One (doc, axis, position) measurement."""

    schema: str = "sum.slider_drift_bench.v1"
    doc_id: str = ""
    axis: str = ""
    position: float = 0.0
    drift_value: float = float("nan")
    drift_threshold: float = float("nan")
    classification: str = ""           # "ok" | "warn" | "fail"
    fact_preservation: float = float("nan")  # |source ∩ reextracted| / |source|
    n_source_triples: int = 0
    n_reextracted_triples: int = 0
    wall_clock_ms: int = 0
    cache_status: str = ""
    error: str = ""                    # populated on failure; otherwise empty


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="slider_drift_bench",
        description=(
            "Phase E.2 — measures per-axis drift over a corpus "
            "via the slider renderer. Emits NDJSON of BenchCell rows."
        ),
    )
    p.add_argument(
        "--corpus", type=Path, required=True,
        help="JSON file with documents to process (same shape as scripts/bench/corpora/*.json).",
    )
    p.add_argument(
        "--out", type=Path, default=Path("slider_drift_bench.jsonl"),
        help="NDJSON output path. Default: ./slider_drift_bench.jsonl.",
    )
    p.add_argument(
        "--positions", type=float, nargs="+", default=list(AXIS_POSITIONS),
        help="Slider positions to probe per axis. Default: 0.1 0.3 0.5 0.7 0.9.",
    )
    p.add_argument(
        "--axes", nargs="+", choices=AXES, default=list(AXES),
        help="Axes to probe. Default: all five.",
    )
    p.add_argument(
        "--max-docs", type=int, default=0,
        help="Limit corpus size for smoke runs. 0 = no limit.",
    )
    return p.parse_args()


async def _bench_one_cell(
    doc_id: str,
    doc_text: str,
    axis: str,
    position: float,
) -> BenchCell:
    """Run one cell. STATE 4 fills the body; STATE 2 returns a stub
    error so the harness's structure is exercised end-to-end."""
    return BenchCell(
        doc_id=doc_id,
        axis=axis,
        position=position,
        error="STATE 4 — _bench_one_cell awaiting render() implementation.",
    )


async def main_async(args: argparse.Namespace) -> int:
    if not args.corpus.exists():
        print(f"slider_drift_bench: corpus {args.corpus} not found", file=sys.stderr)
        return 2

    corpus = json.loads(args.corpus.read_text())
    docs = corpus["documents"] if isinstance(corpus, dict) and "documents" in corpus else corpus
    if args.max_docs:
        docs = docs[: args.max_docs]

    cells: list[BenchCell] = []
    for doc in docs:
        for axis in args.axes:
            for position in args.positions:
                cell = await _bench_one_cell(doc["id"], doc["text"], axis, position)
                cells.append(cell)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for c in cells:
            f.write(json.dumps(asdict(c)) + "\n")

    # Aggregate per (axis, position) — only over cells without errors.
    print(f"# slider_drift_bench: {len(cells)} cells written to {args.out}", file=sys.stderr)
    by_cell: dict[tuple[str, float], list[float]] = {}
    for c in cells:
        if c.error or c.drift_value != c.drift_value:  # NaN
            continue
        by_cell.setdefault((c.axis, c.position), []).append(c.drift_value)
    print("# axis,position,n,median,p75,p90", file=sys.stderr)
    for (axis, pos), values in sorted(by_cell.items()):
        if not values:
            continue
        s = sorted(values)
        median = statistics.median(s)
        p75 = s[int(len(s) * 0.75)]
        p90 = s[int(len(s) * 0.90)] if len(s) >= 10 else s[-1]
        print(f"# {axis},{pos},{len(s)},{median:.4f},{p75:.4f},{p90:.4f}", file=sys.stderr)

    # Stop-the-line: any failure-error row → exit 1 so CI surfaces.
    failures = [c for c in cells if c.error]
    if failures:
        print(
            f"slider_drift_bench: {len(failures)}/{len(cells)} cells errored "
            f"(STATE 2 — expected until render() lands).",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    return asyncio.run(main_async(cli()))


if __name__ == "__main__":
    sys.exit(main())
