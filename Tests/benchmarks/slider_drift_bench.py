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
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from sum_engine_internal.ensemble.live_llm_adapter import (
    LiveLLMAdapter,
    OpenAIChatClient,
)
from sum_engine_internal.ensemble.slider_renderer import (
    Triple,
    cache_key,
    render,
)
from sum_engine_internal.ensemble.tome_sliders import TomeSliders

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


def _build_sliders(axis: str, position: float) -> TomeSliders:
    """Construct a TomeSliders with `axis` at `position` and the other
    axes at their neutral defaults. Density is special: the bench
    parameterises density at the requested position; other axes default
    to 1.0 density (full coverage) so drift on the LLM axes isn't
    confounded by triple-set thinning."""
    if axis == "density":
        return TomeSliders(density=position)
    base = {"density": 1.0, "length": 0.5, "formality": 0.5, "audience": 0.5, "perspective": 0.5}
    base[axis] = position
    return TomeSliders(**base)


async def _bench_one_cell(
    extractor,
    llm_client,
    doc_id: str,
    doc_text: str,
    axis: str,
    position: float,
) -> BenchCell:
    """Extract triples from `doc_text`, render at the given (axis,
    position), measure drift. One LLM call per cell unless the axis is
    `density` (canonical path, no LLM). Errors are captured into the
    cell rather than raised so one failure doesn't kill the run."""
    t_start = time.monotonic()
    try:
        source_triples: list[Triple] = await extractor(doc_text)
        if not source_triples:
            return BenchCell(
                doc_id=doc_id, axis=axis, position=position,
                wall_clock_ms=int((time.monotonic() - t_start) * 1000),
                error="extractor returned no triples",
            )

        sliders = _build_sliders(axis, position)
        result = await render(source_triples, sliders, llm_client, extractor)

        # The drift entry for the axis under test.
        axis_drift = next((d for d in result.drift if d.axis.value == axis), None)
        if axis_drift is None:
            return BenchCell(
                doc_id=doc_id, axis=axis, position=position,
                wall_clock_ms=int((time.monotonic() - t_start) * 1000),
                error=f"no drift entry for axis {axis}",
            )

        n_source = len(source_triples)
        n_reextracted = len(result.triples_used)
        # Fact-preservation per SLIDER_CONTRACT.md §"Fact preservation invariant".
        source_keys = {f"{s}||{p}||{o}" for (s, p, o) in source_triples}
        used_keys = {f"{s}||{p}||{o}" for (s, p, o) in result.triples_used}
        kept = len(source_keys & used_keys)
        fact_pres = kept / n_source if n_source else 0.0

        return BenchCell(
            doc_id=doc_id,
            axis=axis,
            position=position,
            drift_value=axis_drift.value,
            drift_threshold=axis_drift.threshold,
            classification=axis_drift.classification,
            fact_preservation=fact_pres,
            n_source_triples=n_source,
            n_reextracted_triples=n_reextracted,
            wall_clock_ms=result.wall_clock_ms,
            cache_status=result.cache_status.value,
        )
    except Exception as e:  # noqa: BLE001 — bench captures all failures into the row
        return BenchCell(
            doc_id=doc_id, axis=axis, position=position,
            wall_clock_ms=int((time.monotonic() - t_start) * 1000),
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
        )


async def main_async(args: argparse.Namespace) -> int:
    if not args.corpus.exists():
        print(f"slider_drift_bench: corpus {args.corpus} not found", file=sys.stderr)
        return 2

    corpus = json.loads(args.corpus.read_text())
    docs = corpus["documents"] if isinstance(corpus, dict) and "documents" in corpus else corpus
    if args.max_docs:
        docs = docs[: args.max_docs]

    # The bench needs both an extractor (for source + re-extracted triples)
    # and an LLM chat client (for the render path). Both come from
    # LiveLLMAdapter, which requires OPENAI_API_KEY in the environment.
    adapter = LiveLLMAdapter()
    llm_client = OpenAIChatClient(adapter)
    extractor = adapter.extract_triplets

    cells: list[BenchCell] = []
    for doc in docs:
        for axis in args.axes:
            for position in args.positions:
                cell = await _bench_one_cell(
                    extractor, llm_client, doc["id"], doc["text"], axis, position,
                )
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
