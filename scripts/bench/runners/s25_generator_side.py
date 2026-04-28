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
    drifts = [d["drift_pct"] for d in per_doc]
    recalls = [d["exact_match_recall"] for d in per_doc]
    n_full = sum(1 for r in recalls if r >= 0.999)
    return {
        "n_docs": len(per_doc),
        "drift_pct_mean": round(statistics.mean(drifts), 4) if drifts else 0.0,
        "drift_pct_median": round(statistics.median(drifts), 4) if drifts else 0.0,
        "exact_match_recall_mean": round(statistics.mean(recalls), 4) if recalls else 0.0,
        "exact_match_recall_p10": round(_pct(recalls, 10), 4) if recalls else 0.0,
        "n_docs_full_recall": n_full,
        "fraction_full_recall": round(n_full / len(recalls), 4) if recalls else 0.0,
    }


def _pct(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = int(round((pct / 100.0) * (len(s) - 1)))
    return s[k]


# ---------------------------------------------------------------------
# Async LLM ablation runners
# ---------------------------------------------------------------------


async def _baseline_extract(client, model: str, text: str) -> list[Triple]:
    """Baseline (unconstrained) extractor — same shape as
    LiveLLMAdapter.extract_triplets but called inline so the runner
    owns the experiment."""
    from sum_engine_internal.ensemble.live_llm_adapter import ExtractionResponse

    sys_prompt = (
        "Extract all distinct factual claims from the text as "
        "subject-predicate-object triplets. Lowercase. snake_case "
        "for multi-word predicates. At most 64 triplets."
    )
    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": text},
        ],
        response_format=ExtractionResponse,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        return []
    return [
        (t.subject.lower().strip(), t.predicate.lower().strip(), t.object_.lower().strip())
        for t in parsed.triplets
    ]


async def _constrained_extract(
    client, model: str, text: str, source_axioms: Sequence[Triple]
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
    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": text},
        ],
        response_format=Schema,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        return []
    return [
        (str(t.subject).lower(), str(t.predicate).lower(), str(t.object_).lower())
        for t in parsed.triplets
    ]


async def _baseline_generate(client, model: str, source_axioms: Sequence[Triple]) -> str:
    sys_prompt = (
        "You are a precise technical writer. Extrapolate the "
        "following absolute facts into a cohesive narrative. "
        "Do not invent facts."
    )
    user_prompt = "FACTS TO INCLUDE:\n" + "\n".join(
        f"{s} {p} {o}" for (s, p, o) in source_axioms
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


async def _canonical_first_generate(
    client, model: str, source_axioms: Sequence[Triple]
) -> str:
    target_axioms_str = [f"{s} {p} {o}" for (s, p, o) in source_axioms]
    user_prompt = build_canonical_first_user_prompt(target_axioms_str)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CANONICAL_FIRST_SYS_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------
# Per-doc round-trip under a chosen ablation
# ---------------------------------------------------------------------


async def run_doc(
    client,
    model: str,
    doc: dict,
    ablation: str,
    *,
    dry_run: bool = False,
) -> dict:
    """Execute one doc round-trip under the chosen ablation."""
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

    # 1. Source extraction (baseline, all ablations).
    source = await _baseline_extract(client, model, text)
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
        narrative = await _canonical_first_generate(client, model, source)
    else:  # constrained_only or baseline
        narrative = await _baseline_generate(client, model, source)

    # 3. Reconstruction extractor.
    if ablation in ("constrained_extractor", "combined"):
        reconstructed = await _constrained_extract(client, model, narrative, source)
    else:  # canonical_first or baseline
        reconstructed = await _baseline_extract(client, model, narrative)

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
    client, model: str, corpus: list[dict], ablation: str, dry_run: bool
) -> dict:
    schema, description = _ABLATIONS[ablation]
    per_doc: list[dict] = []
    for doc in corpus:
        result = await run_doc(client, model, doc, ablation, dry_run=dry_run)
        per_doc.append(result)
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
        "model": model,
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
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls; verify scaffold only.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    corpus = load_seed_v1(Path(args.corpus))
    if args.max_docs is not None:
        corpus = corpus[: args.max_docs]
    print(f"corpus: {Path(args.corpus).name} ({len(corpus)} docs)", file=sys.stderr)

    if args.dry_run:
        client = None  # unused
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit(
                "OPENAI_API_KEY not set. Set it or use --dry-run for offline scaffold check."
            )
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)

    ablations_to_run = (
        list(_ABLATIONS.keys()) if args.ablation == "all" else [args.ablation]
    )

    results: list[dict] = []
    for abl in ablations_to_run:
        print(f"\n=== ablation: {abl} ===", file=sys.stderr)
        result = asyncio.run(
            run_ablation(client, args.model, corpus, abl, args.dry_run)
        )
        results.append(result)
        agg = result["aggregate"]
        print(
            f"  → drift_mean={agg['drift_pct_mean']:.2f} "
            f"recall_mean={agg['exact_match_recall_mean']:.4f} "
            f"full_recall={agg['n_docs_full_recall']}/{agg['n_docs']}",
            file=sys.stderr,
        )

    payload = {
        "schema_family": "sum.s25_generator_side.v1",
        "issued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus": Path(args.corpus).name,
        "n_docs": len(corpus),
        "model": args.model,
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
