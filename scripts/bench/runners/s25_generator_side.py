"""§2.5 generator-side intervention runner — produces receipts under
sum.s25_canonical_first_generator.v1 / sum.s25_constrained_extractor.v1
/ sum.s25_combined.v1.

Three ablations against the same seed_v1 corpus the L0–L3 canonicalisation
receipt used:

    Ablation 1 — canonical-first generator only.
        Replaces ``LiveLLMAdapter.generate_text``'s system prompt with one
        that requires surfacing each source claim verbatim before
        elaborating. Extractor is the unconstrained baseline.

    Ablation 2 — constrained extractor only.
        Baseline generator. Extractor uses a per-doc Pydantic schema
        with ``Literal`` enums pinned to the source-axiom vocabulary
        (subject ∈ source_subjects, predicate ∈ source_predicates ∪
        DEFAULT_CANONICAL_PREDICATES, object ∈ source_objects).

    Ablation 3 — combined.
        Both interventions stacked.

Each ablation produces per-doc:
    - n_source_axioms
    - n_reconstructed_axioms
    - drift_pct (per-doc symmetric-difference / max)
    - exact_match_recall (matched / |source|)
    - missing_claims, extra_claims (full sets)

And aggregate:
    - drift_pct mean / median
    - exact_match_recall mean / p10
    - n_docs at full recall (1.0)
    - delta vs L0 baseline from the canonicalisation receipt

The runner is offline-testable in --dry-run mode: it skips API calls
and produces a stubbed receipt with synthetic data, useful for verifying
schema serialisation and per-doc field shapes before any spend.

Usage:
    # Smoke test (2 docs, ~$0.005)
    OPENAI_API_KEY=... python -m scripts.bench.runners.s25_generator_side \\
        --ablation combined --max-docs 2 --out /tmp/smoke.json

    # Full 50-doc, all three ablations (~$0.20 total)
    OPENAI_API_KEY=... python -m scripts.bench.runners.s25_generator_side \\
        --ablation all --out /tmp/s25_generator_side.json

    # Offline dry-run (no API, no spend)
    python -m scripts.bench.runners.s25_generator_side \\
        --ablation combined --dry-run --out /tmp/dryrun.json

Pinned model: gpt-4o-mini-2024-07-18 (matches the L0 baseline run so
deltas are clean).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from sum_engine_internal.ensemble.s25_interventions import (
    CANONICAL_FIRST_SYS_PROMPT,
    DEFAULT_CANONICAL_PREDICATES,
    RECEIPT_SCHEMA_CANONICAL_FIRST,
    RECEIPT_SCHEMA_CONSTRAINED_EXTRACTOR,
    RECEIPT_SCHEMA_COMBINED,
    build_canonical_first_user_prompt,
    build_constrained_extraction_schema,
    vocabulary_summary,
)


Triple = Tuple[str, str, str]


# ---------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------


def load_seed_v1(corpus_path: Path) -> list[dict]:
    """Load seed_v1 corpus. Schema: ``{id, documents: [{id, text, gold_triples}, ...]}``."""
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "documents" in data:
        return data["documents"]
    if isinstance(data, list):
        return data
    raise SystemExit(
        f"unexpected corpus shape at {corpus_path}: "
        f"expected list or dict with 'documents'"
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def per_doc_metrics(
    source: Sequence[Triple], reconstructed: Sequence[Triple]
) -> dict:
    src = {tuple(t) for t in source}
    rec = {tuple(t) for t in reconstructed}
    if not src and not rec:
        return {
            "drift_pct": 0.0,
            "exact_match_recall": 1.0,
            "matched": 0,
            "n_source": 0,
            "n_reconstructed": 0,
            "missing": [],
            "extra": [],
        }
    matched = src & rec
    sym = src.symmetric_difference(rec)
    denom = max(len(src), len(rec))
    drift = (100.0 * len(sym) / denom) if denom else 0.0
    recall = (len(matched) / len(src)) if src else 0.0
    return {
        "drift_pct": round(drift, 4),
        "exact_match_recall": round(recall, 4),
        "matched": len(matched),
        "n_source": len(src),
        "n_reconstructed": len(rec),
        "missing": sorted([list(t) for t in (src - rec)]),
        "extra": sorted([list(t) for t in (rec - src)]),
    }


def aggregate(per_doc: list[dict]) -> dict:
    """Aggregate per-doc records.

    Timed-out docs (``error_class == "timeout"``) are excluded from
    recall/drift means so a single hanging API call doesn't poison
    the headline number. They are counted separately in
    ``n_docs_timed_out``. The receipt-reader can see at a glance
    whether the run completed cleanly or had operator-visible
    errors during execution.
    """
    measured = [d for d in per_doc if d.get("error_class") != "timeout"]
    timed_out = [d for d in per_doc if d.get("error_class") == "timeout"]

    drifts = [d["drift_pct"] for d in measured]
    recalls = [d["exact_match_recall"] for d in measured]
    n_full = sum(1 for r in recalls if r >= 0.999)
    n_measured = len(measured)
    return {
        "n_docs": len(per_doc),
        "n_docs_measured": n_measured,
        "n_docs_timed_out": len(timed_out),
        "timed_out_doc_ids": [d["doc_id"] for d in timed_out],
        "drift_pct_mean": round(statistics.mean(drifts), 4) if drifts else 0.0,
        "drift_pct_median": round(statistics.median(drifts), 4) if drifts else 0.0,
        "exact_match_recall_mean": round(statistics.mean(recalls), 4) if recalls else 0.0,
        "exact_match_recall_p10": round(_pct(recalls, 10), 4) if recalls else 0.0,
        "n_docs_full_recall": n_full,
        "fraction_full_recall": round(n_full / n_measured, 4) if n_measured else 0.0,
    }


def _pct(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round((pct / 100.0) * (len(s) - 1)))
    return s[k]


# ---------------------------------------------------------------------
# Per-call timeout discipline
# ---------------------------------------------------------------------
#
# The seed_long capstone surfaced a real failure mode: an OpenAI
# structured-output call hung for 14+ minutes with the python process
# alive but no CPU progress (CHANGELOG entry under §2.5 capstone).
# The OpenAI SDK has its own request timeout but the empirical fail
# was outside that envelope — likely a stuck websocket on the
# structured-output response stream.
#
# Defence-in-depth: every LLM call goes through ``_with_call_timeout``
# which wraps the coroutine in ``asyncio.wait_for``. On timeout, raise
# a tagged ``S25CallTimeoutError`` that ``run_doc`` catches and
# converts to a per-doc skip — the surrounding ablation continues, the
# receipt records ``error_class: "timeout"`` for that doc, and the
# aggregate excludes timed-out docs from recall/drift means while
# reporting them in a separate ``n_docs_timed_out`` field.

DEFAULT_CALL_TIMEOUT_S: float = 60.0


class S25CallTimeoutError(Exception):
    """Raised when a per-call timeout fires on an LLM API call."""

    def __init__(self, what: str, timeout_s: float):
        self.what = what
        self.timeout_s = timeout_s
        super().__init__(f"{what} timed out after {timeout_s:.1f}s")


async def _with_call_timeout(coro, timeout_s: float, what: str):
    """Wrap an LLM-call coroutine with an asyncio.wait_for hard cap.

    On timeout, raises ``S25CallTimeoutError`` (re-raised cleanly so
    the caller can convert to a per-doc skip).
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except asyncio.TimeoutError:
        raise S25CallTimeoutError(what, timeout_s) from None


# ---------------------------------------------------------------------
# Async LLM ablation runners
# ---------------------------------------------------------------------


async def _baseline_extract(
    adapter, text: str, *, call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S
) -> list[Triple]:
    """Baseline (unconstrained) extractor — same shape as
    LiveLLMAdapter.extract_triplets but called inline so the runner
    owns the experiment. Vendor-agnostic via ``adapter``."""
    from sum_engine_internal.ensemble.live_llm_adapter import ExtractionResponse

    sys_prompt = (
        "Extract all distinct factual claims from the text as "
        "subject-predicate-object triplets. Lowercase. snake_case "
        "for multi-word predicates. At most 64 triplets."
    )
    parsed = await _with_dispatch_timeout(
        adapter.parse_structured(
            system=sys_prompt,
            user=text,
            schema=ExtractionResponse,
            call_timeout_s=call_timeout_s,
        ),
        what="baseline_extract",
        call_timeout_s=call_timeout_s,
    )
    if parsed is None:
        return []
    return [
        (t.subject.lower().strip(), t.predicate.lower().strip(), t.object_.lower().strip())
        for t in parsed.triplets
    ]


async def _constrained_extract(
    adapter,
    text: str,
    source_axioms: Sequence[Triple],
    *,
    call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
) -> list[Triple]:
    """Constrained extractor — Pydantic schema pinned to source vocabulary."""
    Schema = build_constrained_extraction_schema(source_axioms)
    sys_prompt = (
        "Extract subject-predicate-object triplets from the text. "
        "You MUST emit only triplets whose subject, predicate, and "
        "object appear in the allowed vocabulary supplied by the "
        "schema. If no triplet's tokens fit the allowed vocabulary, "
        "return an empty list. Lowercase."
    )
    parsed = await _with_dispatch_timeout(
        adapter.parse_structured(
            system=sys_prompt,
            user=text,
            schema=Schema,
            call_timeout_s=call_timeout_s,
        ),
        what="constrained_extract",
        call_timeout_s=call_timeout_s,
    )
    if parsed is None:
        return []
    return [
        (str(t.subject).lower(), str(t.predicate).lower(), str(t.object_).lower())
        for t in parsed.triplets
    ]


async def _baseline_generate(
    adapter,
    source_axioms: Sequence[Triple],
    *,
    call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
) -> str:
    sys_prompt = (
        "You are a precise technical writer. Extrapolate the "
        "following absolute facts into a cohesive narrative. "
        "Do not invent facts."
    )
    user_prompt = "FACTS TO INCLUDE:\n" + "\n".join(
        f"{s} {p} {o}" for (s, p, o) in source_axioms
    )
    return await _with_dispatch_timeout(
        adapter.generate_text(
            system=sys_prompt,
            user=user_prompt,
            call_timeout_s=call_timeout_s,
        ),
        what="baseline_generate",
        call_timeout_s=call_timeout_s,
    )


async def _canonical_first_generate(
    adapter,
    source_axioms: Sequence[Triple],
    *,
    call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
) -> str:
    target_axioms_str = [f"{s} {p} {o}" for (s, p, o) in source_axioms]
    user_prompt = build_canonical_first_user_prompt(target_axioms_str)
    return await _with_dispatch_timeout(
        adapter.generate_text(
            system=CANONICAL_FIRST_SYS_PROMPT,
            user=user_prompt,
            call_timeout_s=call_timeout_s,
        ),
        what="canonical_first_generate",
        call_timeout_s=call_timeout_s,
    )


async def _with_dispatch_timeout(coro, *, what: str, call_timeout_s: float):
    """Bridge between the dispatcher's ``LLMCallTimeoutError`` and the
    runner's ``S25CallTimeoutError``. Both encode the same condition
    (per-call hard cap exceeded); the rebrand keeps the runner's
    receipt schema and per-doc skip path unchanged."""
    from sum_engine_internal.ensemble.llm_dispatch import LLMCallTimeoutError
    try:
        return await coro
    except LLMCallTimeoutError as e:
        raise S25CallTimeoutError(what, call_timeout_s) from e


# ---------------------------------------------------------------------
# Per-doc round-trip under a chosen ablation
# ---------------------------------------------------------------------


async def run_doc(
    adapter,
    doc: dict,
    ablation: str,
    *,
    call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
    dry_run: bool = False,
) -> dict:
    """Execute one doc round-trip under the chosen ablation.

    Per-call timeouts are enforced at every LLM call site. If any call
    raises ``S25CallTimeoutError``, this function returns a per-doc
    record tagged with ``error_class: "timeout"`` rather than letting
    the exception propagate. The surrounding ablation continues; the
    aggregate excludes timed-out docs from recall/drift means and
    counts them in ``n_docs_timed_out``.
    """
    doc_id = doc["id"]
    text = doc["text"]

    if dry_run:
        # Stubbed result with synthetic but realistic data shape.
        source = [tuple(t) for t in doc["gold_triples"]]
        return {
            "doc_id": doc_id,
            "source_axioms": [list(t) for t in source],
            "reconstructed_axioms": [list(t) for t in source],  # perfect dry-run
            "narrative_excerpt": "(dry-run; no LLM call)",
            "vocabulary_summary": vocabulary_summary(source),
            **per_doc_metrics(source, source),
        }

    try:
        # 1. Source extraction (baseline, all ablations).
        source = await _baseline_extract(
            adapter, text, call_timeout_s=call_timeout_s
        )
        if not source:
            return {
                "doc_id": doc_id,
                "source_axioms": [],
                "reconstructed_axioms": [],
                "narrative_excerpt": "(no source axioms extracted)",
                **per_doc_metrics([], []),
            }

        # 2. Generator pass.
        if ablation in ("canonical_first", "combined"):
            narrative = await _canonical_first_generate(
                adapter, source, call_timeout_s=call_timeout_s
            )
        else:  # constrained_only or baseline
            narrative = await _baseline_generate(
                adapter, source, call_timeout_s=call_timeout_s
            )

        # 3. Reconstruction extractor.
        if ablation in ("constrained_extractor", "combined"):
            reconstructed = await _constrained_extract(
                adapter, narrative, source, call_timeout_s=call_timeout_s
            )
        else:  # canonical_first or baseline
            reconstructed = await _baseline_extract(
                adapter, narrative, call_timeout_s=call_timeout_s
            )
    except S25CallTimeoutError as e:
        # Per-doc skip on timeout. Tagged so the receipt and aggregate
        # can distinguish timeouts from real recall failures.
        return {
            "doc_id": doc_id,
            "error_class": "timeout",
            "error_what": e.what,
            "error_timeout_s": e.timeout_s,
            "source_axioms": [],
            "reconstructed_axioms": [],
            "narrative_excerpt": f"(timeout: {e.what} after {e.timeout_s:.1f}s)",
        }

    metrics = per_doc_metrics(source, reconstructed)
    return {
        "doc_id": doc_id,
        "source_axioms": [list(t) for t in source],
        "reconstructed_axioms": [list(t) for t in reconstructed],
        "narrative_excerpt": narrative[:200],
        "vocabulary_summary": vocabulary_summary(source),
        **metrics,
    }


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------


_ABLATIONS = {
    "canonical_first": (RECEIPT_SCHEMA_CANONICAL_FIRST, "Intervention A: canonical-first generator prompt; baseline extractor"),
    "constrained_extractor": (RECEIPT_SCHEMA_CONSTRAINED_EXTRACTOR, "Intervention B: baseline generator; vocab-pinned extractor"),
    "combined": (RECEIPT_SCHEMA_COMBINED, "A + B: canonical-first generator + vocab-pinned extractor"),
}


async def run_ablation(
    adapter,
    corpus: list[dict],
    ablation: str,
    *,
    call_timeout_s: float,
    dry_run: bool,
) -> dict:
    schema, description = _ABLATIONS[ablation]
    per_doc: list[dict] = []
    for doc in corpus:
        result = await run_doc(
            adapter,
            doc,
            ablation,
            call_timeout_s=call_timeout_s,
            dry_run=dry_run,
        )
        per_doc.append(result)
        if result.get("error_class") == "timeout":
            print(
                f"  [{ablation}] {result['doc_id']}: "
                f"TIMEOUT ({result['error_what']} > {result['error_timeout_s']:.0f}s)",
                flush=True,
            )
        else:
            recall = result.get("exact_match_recall", 0)
            print(
                f"  [{ablation}] {result['doc_id']}: "
                f"src={result.get('n_source', 0)} "
                f"rec={result.get('n_reconstructed', 0)} "
                f"recall={recall:.2f}",
                flush=True,
            )
    return {
        "schema": schema,
        "ablation": ablation,
        "description": description,
        "model": getattr(adapter, "model", None),
        "call_timeout_s": call_timeout_s,
        "aggregate": aggregate(per_doc),
        "per_doc": per_doc,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ablation",
        choices=list(_ABLATIONS.keys()) + ["all"],
        default="combined",
    )
    parser.add_argument("--corpus", default="scripts/bench/corpora/seed_v1.json")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18",
        help="Pinned model snapshot. MUST match L0 baseline for clean delta.",
    )
    parser.add_argument("--max-docs", type=int, default=None, help="Cap doc count for smoke tests.")
    parser.add_argument(
        "--call-timeout",
        type=float,
        default=DEFAULT_CALL_TIMEOUT_S,
        help=(
            f"Per-LLM-call timeout in seconds (default {DEFAULT_CALL_TIMEOUT_S:.0f}). "
            f"Each individual extract/generate call is wrapped in asyncio.wait_for; "
            f"on timeout, the doc gets a per-doc skip with error_class='timeout' "
            f"and the surrounding ablation continues. The aggregate excludes timed-out "
            f"docs from recall/drift means and counts them in n_docs_timed_out."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls; verify scaffold only.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    corpus = load_seed_v1(Path(args.corpus))
    if args.max_docs is not None:
        corpus = corpus[: args.max_docs]
    print(f"corpus: {Path(args.corpus).name} ({len(corpus)} docs)", file=sys.stderr)

    if args.dry_run:
        adapter = None  # unused
    else:
        from sum_engine_internal.ensemble.llm_dispatch import get_adapter
        try:
            adapter = get_adapter(args.model)
        except ValueError as e:
            raise SystemExit(str(e))
        except ImportError as e:
            raise SystemExit(str(e))
        # Surface a friendly message if the matching API key is missing.
        if args.model.lower().startswith("claude-") and not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit(
                "ANTHROPIC_API_KEY not set. Required for claude-* models. "
                "Set it or use --dry-run for offline scaffold check."
            )
        if not args.model.lower().startswith("claude-") and not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(
                "OPENAI_API_KEY not set. Required for gpt-/o*-* models. "
                "Set it or use --dry-run for offline scaffold check."
            )

    ablations_to_run = (
        list(_ABLATIONS.keys()) if args.ablation == "all" else [args.ablation]
    )

    results: list[dict] = []
    for abl in ablations_to_run:
        print(f"\n=== ablation: {abl} ===", file=sys.stderr)
        result = asyncio.run(
            run_ablation(
                adapter,
                corpus,
                abl,
                call_timeout_s=args.call_timeout,
                dry_run=args.dry_run,
            )
        )
        results.append(result)
        agg = result["aggregate"]
        msg = (
            f"  → drift_mean={agg['drift_pct_mean']:.2f} "
            f"recall_mean={agg['exact_match_recall_mean']:.4f} "
            f"full_recall={agg['n_docs_full_recall']}/{agg['n_docs_measured']}"
        )
        if agg["n_docs_timed_out"]:
            msg += f" [timed_out: {agg['n_docs_timed_out']} doc(s)]"
        print(msg, file=sys.stderr)

    provider = (
        "anthropic" if args.model.lower().startswith("claude-")
        else "openai" if args.model.lower().startswith(("gpt-", "o1-", "o3-", "o4-"))
        else "unknown"
    )
    payload = {
        "schema_family": "sum.s25_generator_side.v1",
        "issued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus": Path(args.corpus).name,
        "n_docs": len(corpus),
        "model": args.model,
        "provider": provider,
        "dry_run": args.dry_run,
        "baseline_reference": {
            "receipt": "fixtures/bench_receipts/s25_canonicalization_replay_2026-04-28.json",
            "L0_drift_mean": 107.75,
            "L0_recall_mean": 0.12,
            "L0_full_recall": "6/50",
        },
        "ablations": results,
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\nreceipt written: {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
