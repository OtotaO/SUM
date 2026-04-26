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
    fact_preservation,
    fact_preservation_normalized,
    order_preservation,
    render,
    semantic_fact_preservation,
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
    # Three-layer fact preservation per docs/SLIDER_V02_RESEARCH.md.
    # All three reported so future readers can see what each layer
    # contributes and where extraction noise vs real loss live.
    fact_preservation_strict: float = float("nan")     # exact (s,p,o) key match (regression check on extractor stability)
    fact_preservation_normalized: float = float("nan") # A3: after predicate / entity normalization
    fact_preservation_semantic: float = float("nan")   # A1: cosine similarity ≥ τ on triple-as-text embeddings
    order_preservation: float = float("nan")           # MontageLie defense (pairwise order, NaN if <2 preserved)
    # v0.3 — constrained-decoding self-attestation signal.
    claim_reextract_jaccard: float = float("nan")      # |claimed ∩ reextracted| / |claimed ∪ reextracted|
    n_source_triples: int = 0
    n_reextracted_triples: int = 0
    n_claimed_triples: int = 0          # 0 on legacy chat-only path; populated on structured path
    tome_word_count: int = 0           # output length, for length-axis recalibration
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
    p.add_argument(
        "--concurrency", type=int, default=16,
        help=(
            "Number of cells to process in parallel. Default 16 — safe "
            "for OpenAI tier 1+ (500 RPM gpt-4o-mini). Lower if you see "
            "RateLimitError; raise on higher tiers for faster bench."
        ),
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
    embed_fn,
    doc_id: str,
    source_triples: list[Triple],
    axis: str,
    position: float,
) -> BenchCell:
    """Render at the given (axis, position) against pre-extracted
    source_triples, measure drift. Two LLM calls per cell on the LLM
    path (render + re-extraction inside render); zero on the density
    canonical path. Errors are captured into the cell rather than
    raised so one failure doesn't kill the run.

    Source extraction is hoisted to main_async — once per doc rather
    than once per cell — eliminating (n_axes × n_positions − 1)
    duplicate calls per doc.
    """
    t_start = time.monotonic()
    try:
        if not source_triples:
            return BenchCell(
                doc_id=doc_id, axis=axis, position=position,
                wall_clock_ms=int((time.monotonic() - t_start) * 1000),
                error="extractor returned no triples (source step)",
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
        # Three-layer fact preservation. Strict is the regression-check
        # on extractor stability; normalized (A3) catches preposition /
        # auxiliary-verb / article drift; semantic (A1) catches the
        # remaining synonym / paraphrase cases via embedding similarity.
        fact_strict = fact_preservation(source_triples, result.reextracted_triples)
        fact_normalized = fact_preservation_normalized(
            source_triples, result.reextracted_triples,
        )
        fact_semantic = await semantic_fact_preservation(
            source_triples, result.reextracted_triples, embed_fn,
        )
        order_pres = order_preservation(source_triples, result.reextracted_triples)

        # v0.3: claim-vs-reextract agreement. Jaccard over normalized-key
        # sets (A3 normalization keeps surface-form drift from making the
        # LLM's self-attestation look worse than it is). NaN when the
        # render didn't go through the structured path (legacy clients).
        claim_jaccard = float("nan")
        if result.claimed_triples or result.reextracted_triples:
            from sum_engine_internal.ensemble.slider_renderer import _normalize_triple
            claimed_keys = {f"{a}||{b}||{c}" for (a, b, c) in
                            (_normalize_triple(t) for t in result.claimed_triples)}
            reext_keys = {f"{a}||{b}||{c}" for (a, b, c) in
                          (_normalize_triple(t) for t in result.reextracted_triples)}
            if claimed_keys or reext_keys:
                inter = len(claimed_keys & reext_keys)
                union = len(claimed_keys | reext_keys)
                claim_jaccard = inter / union if union else float("nan")

        return BenchCell(
            doc_id=doc_id,
            axis=axis,
            position=position,
            drift_value=axis_drift.value,
            drift_threshold=axis_drift.threshold,
            classification=axis_drift.classification,
            fact_preservation_strict=fact_strict,
            fact_preservation_normalized=fact_normalized,
            fact_preservation_semantic=fact_semantic,
            order_preservation=order_pres,
            claim_reextract_jaccard=claim_jaccard,
            n_source_triples=n_source,
            n_reextracted_triples=len(result.reextracted_triples),
            n_claimed_triples=len(result.claimed_triples),
            tome_word_count=len(result.tome.split()),
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

    # The bench needs an extractor (for source + re-extracted triples),
    # an LLM chat client (for the render path), and an embedder (for
    # the A1 semantic fact-preservation layer). All come from
    # LiveLLMAdapter, which requires OPENAI_API_KEY in the environment.
    adapter = LiveLLMAdapter()
    llm_client = OpenAIChatClient(adapter)
    extractor = adapter.extract_triplets
    embed_fn = adapter.get_embedding

    # Phase 1: extract source triples for every doc once, in parallel.
    # Without this hoist the bench re-extracts the same source 25× per
    # doc (once per axis × position). Eliminates ~125 redundant LLM
    # calls per 8-doc / 25-position run.
    sem = asyncio.Semaphore(args.concurrency)

    async def _gated(coro):
        async with sem:
            return await coro

    t_extract_start = time.monotonic()
    src_results = await asyncio.gather(
        *[_gated(extractor(doc["text"])) for doc in docs],
        return_exceptions=True,
    )
    src_by_doc: dict[str, list[Triple]] = {}
    src_errors: dict[str, str] = {}
    for doc, res in zip(docs, src_results):
        if isinstance(res, Exception):
            src_errors[doc["id"]] = f"{type(res).__name__}: {res}"
            src_by_doc[doc["id"]] = []
        else:
            src_by_doc[doc["id"]] = list(res)
    print(
        f"# slider_drift_bench: extracted source for {len(docs)} docs in "
        f"{time.monotonic() - t_extract_start:.1f}s "
        f"(concurrency={args.concurrency})",
        file=sys.stderr,
    )

    # Phase 2: build all (doc, axis, position) cells; run with the
    # same semaphore so per-cell render+re-extract calls share the
    # rate-limit budget with the source-extraction phase above.
    t_cells_start = time.monotonic()
    cells: list[BenchCell] = []
    tasks = []
    for doc in docs:
        # If source extraction failed for this doc, emit one error row
        # per cell synchronously rather than running render against an
        # empty triple set.
        if doc["id"] in src_errors:
            for axis in args.axes:
                for position in args.positions:
                    cells.append(BenchCell(
                        doc_id=doc["id"], axis=axis, position=position,
                        error=f"source-extract-failed: {src_errors[doc['id']]}",
                    ))
            continue
        src = src_by_doc[doc["id"]]
        for axis in args.axes:
            for position in args.positions:
                tasks.append(_gated(_bench_one_cell(
                    extractor, llm_client, embed_fn,
                    doc["id"], src, axis, position,
                )))

    completed = 0
    total = len(tasks)
    for fut in asyncio.as_completed(tasks):
        cell = await fut
        cells.append(cell)
        completed += 1
        if completed % 25 == 0 or completed == total:
            print(
                f"# slider_drift_bench: {completed}/{total} cells done "
                f"({time.monotonic() - t_cells_start:.1f}s elapsed)",
                file=sys.stderr,
            )

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

    # Per-axis fact preservation (three layers + claim/reextract agreement).
    print(
        "# fact_preservation by axis,position — strict / normalized / semantic / order / claim_jaccard",
        file=sys.stderr,
    )
    by_fact: dict[tuple[str, float], list[tuple[float, float, float, float, float]]] = {}
    for c in cells:
        if c.error:
            continue
        by_fact.setdefault((c.axis, c.position), []).append((
            c.fact_preservation_strict,
            c.fact_preservation_normalized,
            c.fact_preservation_semantic,
            c.order_preservation,
            c.claim_reextract_jaccard,
        ))
    for (axis, pos), tuples in sorted(by_fact.items()):
        # Filter NaN per column when computing medians.
        cols = list(zip(*tuples))
        med_strs = []
        for col in cols:
            valid = [v for v in col if v == v]  # drop NaN
            med_strs.append(f"{statistics.median(valid):.3f}" if valid else "n/a")
        print(
            f"# {axis},{pos}  strict={med_strs[0]}  norm={med_strs[1]}  "
            f"semantic={med_strs[2]}  order={med_strs[3]}  claim_jaccard={med_strs[4]}",
            file=sys.stderr,
        )

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
