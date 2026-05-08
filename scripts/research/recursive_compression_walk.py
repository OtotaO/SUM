"""
Recursive-compression walk: corpus → axioms → render → re-extract →
… → fixed point. The "SUM" of a corpus under a given compressor is
the (axiom-set, fixed-point-step) pair where iterated compression no
longer changes the axiom signature, or — if no fixed point is
reached within a budget — the point at which round-trip recall vs the
original axiom-set crosses a degradation threshold τ.

Two compressors are measured:

  1. **Deterministic canonical tome** — `AutoregressiveTomeGenerator
     .generate_canonical`. Renders each axiom as a bare-lemma string
     ("The alice like cat."). The sieve cannot re-parse this because
     the verb is uninflected ("like" instead of "likes"). Empirical
     consequence: the deterministic walk collapses to ∅ at step 1
     for every input. Documenting this collapse is the load-bearing
     finding for the deterministic arm — it surfaces the
     sieve↔canonical-tome asymmetry that is invisible at the
     state-integer level.

  2. **LLM-mediated grammatical render** — calls the live LLM via
     `llm_dispatch.get_adapter(model)` to render the axiom-set as a
     coherent paragraph with proper noun forms and verb conjugation.
     The sieve can re-extract from this prose. The walk produces a
     real recursive-compression curve. Capture-once-replay-forever:
     the per-step LLM responses are cached in the snapshot file so
     re-running Phase 2 (sieve re-extract + recall scoring) is
     byte-deterministic against the cached snapshot.

The "SUM" identification:

  - For the deterministic arm: the SUM is ∅ at step 1. The walk
    reveals the asymmetry; nothing more.
  - For the LLM arm: the SUM is the smallest axiom-set $A_k$ such
    that recall_vs_original($A_k$) ≥ τ AND
    sigfile($A_k$) ∈ {sigfile($A_{k-1}$), sigfile($A_{k-2}$), …}
    (fixed point or cycle). For thresholds τ ∈ {0.5, 0.7, 0.9}, the
    SUM may be reached at different steps.

Schema: `sum.recursive_compression_walk.v1`
"""
from __future__ import annotations

import scripts.research._deterministic_blas  # noqa: F401, E402

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Awaitable, Callable  # noqa: E402

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra  # noqa: E402
from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve  # noqa: E402
from sum_engine_internal.ensemble.tome_generator import (  # noqa: E402
    AutoregressiveTomeGenerator,
)

REPO = Path(__file__).resolve().parents[2]
RECEIPTS_DIR = REPO / "fixtures" / "bench_receipts"
RENDERS_DIR = REPO / "fixtures" / "bench_renders"

# Capture-once-replay-forever path for LLM-mediated walks.
SNAPSHOT_DIR = RENDERS_DIR
DEFAULT_LLM_MODEL = "gpt-4o-mini-2024-07-18"
LLM_TIMEOUT_S = 60.0
LLM_RETRIES = 2
LLM_RETRY_BACKOFF_S = 5.0

# Walk budget.
DEFAULT_MAX_STEPS = 8
DEFAULT_RECALL_THRESHOLDS = (0.5, 0.7, 0.9, 0.99)


# ─── Deterministic canonical render ──────────────────────────────────


def _deterministic_render(triples: list[tuple[str, str, str]]) -> str:
    """Render via canonical tome (lemmatized; will not round-trip)."""
    if not triples:
        return ""
    algebra = GodelStateAlgebra()
    state = algebra.encode_chunk_state(triples)
    gen = AutoregressiveTomeGenerator(algebra)
    return gen.generate_canonical(state, title="Iteration")


# ─── LLM-mediated render ─────────────────────────────────────────────


_LLM_RENDER_SYSTEM = (
    "You are a precise technical writer. Render the supplied facts as "
    "grammatical English prose using proper noun forms, definite/indefinite "
    "articles where natural, and inflected verbs (third-person singular, "
    "tense agreement). Preserve every fact exactly: do NOT invent, drop, "
    "or paraphrase any subject, predicate, or object. The output must be "
    "extractable by a syntactic SVO sieve — keep one fact per simple "
    "declarative sentence; no compound clauses, no relative clauses, no "
    "passive voice, no negation, no hedging."
)


def _format_facts_block(triples: list[tuple[str, str, str]]) -> str:
    return "\n".join(f"- {s} {p} {o}" for s, p, o in triples)


async def _llm_render(
    adapter: Any, triples: list[tuple[str, str, str]],
) -> str:
    facts = _format_facts_block(triples)
    user = f"FACTS TO INCLUDE:\n{facts}\n\nRender as simple declarative sentences, one fact per sentence."
    last_err: Exception | None = None
    from sum_engine_internal.ensemble.llm_dispatch import LLMCallTimeoutError

    for attempt in range(1 + LLM_RETRIES):
        try:
            return await adapter.generate_text(
                system=_LLM_RENDER_SYSTEM,
                user=user,
                call_timeout_s=LLM_TIMEOUT_S,
            )
        except LLMCallTimeoutError as e:
            last_err = e
            if attempt < LLM_RETRIES:
                await asyncio.sleep(LLM_RETRY_BACKOFF_S)
                continue
    assert last_err is not None
    raise last_err


# ─── Walk core ───────────────────────────────────────────────────────


def _signature(triples: list[tuple[str, str, str]]) -> tuple[tuple[str, str, str], ...]:
    """Order-independent canonical signature for fixed-point detection."""
    return tuple(sorted(set(triples)))


async def _walk_one_doc(
    text: str,
    render_fn: Callable[[list[tuple[str, str, str]]], Awaitable[str] | str],
    max_steps: int,
    sieve: DeterministicSieve,
    snapshot_writeback: dict | None = None,
    snapshot_key: str | None = None,
) -> dict[str, Any]:
    """Walk a single doc through `max_steps` iterations of compression.

    `render_fn` may be sync or async. If `snapshot_writeback` is given
    along with `snapshot_key`, the rendered prose at each step is
    captured into snapshot_writeback[snapshot_key][step] for
    capture-once-replay-forever.
    """
    A_0 = list(sieve.extract_triplets(text))
    A_0_set = set(A_0)
    n0 = len(A_0_set)

    history: list[dict[str, Any]] = [{
        "step": 0,
        "n_axioms": n0,
        "recall_vs_original": 1.0,
        "fixed_point": False,
        "note": "step-0 baseline (sieve from prose)",
    }]
    if n0 == 0:
        return {"steps": history, "fixed_point_step": 0,
                "fixed_point_axioms": [], "n_axioms_original": 0}

    A_curr = A_0
    sigs_seen = {_signature(A_0)}
    fixed_step: int | None = None

    for step in range(1, max_steps + 1):
        if not A_curr:
            history.append({
                "step": step, "n_axioms": 0,
                "recall_vs_original": 0.0, "fixed_point": True,
                "note": "collapsed_to_empty",
            })
            fixed_step = step
            break

        result = render_fn(A_curr)
        prose = await result if asyncio.iscoroutine(result) else result

        if snapshot_writeback is not None and snapshot_key is not None:
            snapshot_writeback.setdefault(snapshot_key, {})[str(step)] = prose

        A_next = list(sieve.extract_triplets(prose))
        A_next_set = set(A_next)
        recall = len(A_next_set & A_0_set) / max(n0, 1) if n0 else 0.0

        sig = _signature(A_next)
        is_fixed = sig in sigs_seen
        history.append({
            "step": step,
            "n_axioms": len(A_next_set),
            "recall_vs_original": round(recall, 4),
            "fixed_point": is_fixed,
            "note": "fixed-point" if is_fixed else "advance",
        })
        if is_fixed:
            fixed_step = step
            break
        sigs_seen.add(sig)
        A_curr = A_next

    if fixed_step is None:
        # Walked to budget; treat last step as the working fixed point.
        fixed_step = len(history) - 1

    final = history[fixed_step]
    return {
        "steps": history,
        "fixed_point_step": fixed_step,
        "fixed_point_n_axioms": final["n_axioms"],
        "fixed_point_recall_vs_original": final["recall_vs_original"],
        "n_axioms_original": n0,
    }


def _identify_sums(walk: dict[str, Any], thresholds: tuple[float, ...]) -> dict[str, Any]:
    """For each threshold τ, the SUM is the smallest n_axioms reached
    along the walk that still satisfies recall_vs_original ≥ τ. If no
    step satisfies the threshold, the SUM is None for that τ."""
    out: dict[str, Any] = {}
    for tau in thresholds:
        feasible = [
            s for s in walk["steps"]
            if s.get("recall_vs_original", 0.0) >= tau
        ]
        if not feasible:
            out[f"tau_{tau:.2f}"] = None
            continue
        smallest = min(feasible, key=lambda s: s["n_axioms"])
        out[f"tau_{tau:.2f}"] = {
            "step": smallest["step"],
            "n_axioms": smallest["n_axioms"],
            "recall_vs_original": smallest["recall_vs_original"],
            "compression_ratio": (
                round(smallest["n_axioms"] / max(walk["n_axioms_original"], 1), 4)
                if walk["n_axioms_original"] else None
            ),
        }
    return out


# ─── Snapshot management for the LLM arm ─────────────────────────────


def _safe_filename(s: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "_", s.lower())


def _llm_snapshot_path(corpus: str, model: str) -> Path:
    return SNAPSHOT_DIR / f"recursive_walk_{corpus}_{_safe_filename(model)}.json"


async def _walk_corpus_llm(
    corpus_path: Path, corpus_name: str, model: str, max_steps: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    """LLM-mediated walk over every doc in a corpus. Returns
    (per_doc_walks, snapshot_dict). Snapshot is written by caller.
    """
    from sum_engine_internal.ensemble.llm_dispatch import get_adapter

    snap_path = _llm_snapshot_path(corpus_name, model)
    if snap_path.exists():
        cached = json.loads(snap_path.read_text())
        renders = cached["renders"]
        print(f"[walk-llm] using cached snapshot: {snap_path.name}")
    else:
        if not os.getenv("OPENAI_API_KEY") and "/" not in model:
            raise SystemExit(
                f"No OPENAI_API_KEY in env and model={model!r} is not "
                f"HF-routed. Set OPENAI_API_KEY (or use an HF model)."
            )
        renders = {}

    adapter = get_adapter(model) if not snap_path.exists() else None
    sieve = DeterministicSieve()
    with corpus_path.open() as f:
        corpus = json.load(f)

    async def render(triples):
        if adapter is None:
            raise RuntimeError("snapshot path; should not be called")
        return await _llm_render(adapter, triples)

    per_doc_walks: dict[str, dict[str, Any]] = {}

    if snap_path.exists():
        # Replay from cached renders
        for doc in corpus["documents"]:
            doc_id = doc["id"]
            text = doc["text"]
            doc_renders = renders.get(doc_id, {})

            # Recreate the per-step rendered prose by replaying the cached
            # data through the same walk logic, but using a render_fn that
            # reads from the cache instead of the LLM.
            step_counter = {"i": 0}

            def cached_render(_triples):
                step_counter["i"] += 1
                return doc_renders.get(str(step_counter["i"]), "")

            walk = await _walk_one_doc(
                text, cached_render, max_steps, sieve,
                snapshot_writeback=None, snapshot_key=None,
            )
            per_doc_walks[doc_id] = walk
        return per_doc_walks, renders

    # Capture path
    print(f"[walk-llm] capturing fresh snapshot at {snap_path}")
    for i, doc in enumerate(corpus["documents"], start=1):
        doc_id = doc["id"]
        text = doc["text"]
        print(f"  [{i}/{len(corpus['documents'])}] {doc_id} …")
        walk = await _walk_one_doc(
            text, render, max_steps, sieve,
            snapshot_writeback=renders, snapshot_key=doc_id,
        )
        per_doc_walks[doc_id] = walk

    snapshot_obj = {
        "schema": "sum.recursive_walk_render_snapshot.v1",
        "corpus": corpus_name,
        "model": model,
        "max_steps": max_steps,
        "renders": renders,
    }
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snap_path.write_text(json.dumps(snapshot_obj, indent=2, sort_keys=True) + "\n")
    print(f"[walk-llm] wrote {snap_path}")
    return per_doc_walks, renders


def _walk_corpus_deterministic(
    corpus_path: Path, max_steps: int,
) -> dict[str, dict[str, Any]]:
    sieve = DeterministicSieve()
    with corpus_path.open() as f:
        corpus = json.load(f)
    per_doc: dict[str, dict[str, Any]] = {}
    for doc in corpus["documents"]:
        # Sync render path; just await on a coroutine that immediately returns
        async def run(text=doc["text"]):
            return await _walk_one_doc(
                text, _deterministic_render, max_steps,
                sieve, snapshot_writeback=None, snapshot_key=None,
            )
        per_doc[doc["id"]] = asyncio.run(run())
    return per_doc


# ─── Aggregation ─────────────────────────────────────────────────────


def _aggregate(
    per_doc: dict[str, dict[str, Any]], thresholds: tuple[float, ...],
) -> dict[str, Any]:
    if not per_doc:
        return {}

    sums_by_doc = {
        doc_id: _identify_sums(walk, thresholds)
        for doc_id, walk in per_doc.items()
    }
    all_n_orig = [walk["n_axioms_original"] for walk in per_doc.values()]
    all_fp_step = [walk["fixed_point_step"] for walk in per_doc.values()]
    all_fp_recall = [walk["fixed_point_recall_vs_original"] for walk in per_doc.values()]
    all_fp_n = [walk["fixed_point_n_axioms"] for walk in per_doc.values()]

    def median(xs):
        s = sorted(xs)
        n = len(s)
        if n == 0:
            return None
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    summary = {
        "n_docs": len(per_doc),
        "median_n_axioms_original": median(all_n_orig),
        "median_fixed_point_step": median(all_fp_step),
        "median_fixed_point_n_axioms": median(all_fp_n),
        "median_fixed_point_recall_vs_original": (
            round(median(all_fp_recall), 4) if all_fp_recall else None
        ),
        "n_docs_collapsed_to_empty": sum(
            1 for w in per_doc.values() if w["fixed_point_n_axioms"] == 0
        ),
    }
    return {"per_doc_sums": sums_by_doc, "summary": summary}


# ─── Run ─────────────────────────────────────────────────────────────


def run_recursive_walk(
    corpora: list[str], compressor: str, model: str,
    max_steps: int, thresholds: tuple[float, ...],
) -> dict[str, Any]:
    print("=" * 72)
    print(f"Recursive-compression walk — compressor={compressor}")
    if compressor == "llm":
        print(f"  model: {model}")
    print(f"  corpora: {corpora}")
    print(f"  max_steps: {max_steps}, thresholds: {thresholds}")
    print("=" * 72)

    by_corpus: dict[str, dict[str, Any]] = {}
    for corpus_name in corpora:
        corpus_path = REPO / "scripts" / "bench" / "corpora" / f"{corpus_name}.json"
        if not corpus_path.exists():
            raise SystemExit(f"corpus not found: {corpus_path}")
        print(f"\n──── {corpus_name} ────")
        if compressor == "deterministic":
            per_doc = _walk_corpus_deterministic(corpus_path, max_steps)
        elif compressor == "llm":
            per_doc, _ = asyncio.run(
                _walk_corpus_llm(corpus_path, corpus_name, model, max_steps)
            )
        else:
            raise ValueError(f"unknown compressor: {compressor}")
        agg = _aggregate(per_doc, thresholds)
        by_corpus[corpus_name] = {
            "per_doc_walks": per_doc,
            "aggregate": agg,
        }
        if agg.get("summary"):
            s = agg["summary"]
            print(f"  median_fp_step={s['median_fixed_point_step']}, "
                  f"median_fp_n_axioms={s['median_fixed_point_n_axioms']}, "
                  f"median_fp_recall={s['median_fixed_point_recall_vs_original']}, "
                  f"collapsed_empty={s['n_docs_collapsed_to_empty']}/{s['n_docs']}")

    report: dict[str, Any] = {
        "schema": "sum.recursive_compression_walk.v1",
        "compressor": compressor,
        "llm_model": model if compressor == "llm" else None,
        "corpora": corpora,
        "max_steps": max_steps,
        "recall_thresholds": list(thresholds),
        "by_corpus": by_corpus,
        "method_notes": (
            "Recursive walk: A₀ = sieve(prose); A_{k+1} = sieve(render(A_k)). "
            "Two compressors:\n"
            "  - deterministic: AutoregressiveTomeGenerator.generate_canonical "
            "renders bare lemmas ('The alice like cat.'); sieve cannot "
            "re-extract uninflected verbs → walk collapses to ∅ at step 1.\n"
            "  - llm: live LLM renders axioms as grammatical prose; sieve "
            "re-extracts; walk produces a real recursive-compression curve.\n"
            "Fixed point = first step whose axiom signature has been seen "
            "before (most often the previous step). The SUM at threshold τ "
            "is the smallest n_axioms along the walk with recall_vs_original "
            "≥ τ. Capture-once-replay-forever for the llm arm (snapshot at "
            "fixtures/bench_renders/recursive_walk_<corpus>_<model>.json)."
        ),
    }
    from scripts.research.sheaf_v3_2_validation import (
        compute_bench_digest, quantize_for_digest,
    )
    quantized = quantize_for_digest(report)
    report["bench_digest"] = compute_bench_digest(quantized)
    print(f"\n  bench_digest: {report['bench_digest']}")
    return report


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpora", nargs="+",
        default=["seed_long_paragraphs", "seed_news_briefs"],
        help="Corpus identifiers (under scripts/bench/corpora/<id>.json).",
    )
    parser.add_argument(
        "--compressor", choices=("deterministic", "llm"), default="deterministic",
        help="Compression operator. 'deterministic' uses canonical-tome "
             "(collapses to ∅ at step 1; documents the asymmetry). 'llm' "
             "uses live LLM rendering (real recursive-compression curve).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_LLM_MODEL,
        help="Model id for --compressor=llm. Default gpt-4o-mini-2024-07-18.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS,
        help=f"Walk-budget per doc. Default {DEFAULT_MAX_STEPS}.",
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=list(DEFAULT_RECALL_THRESHOLDS),
        help=f"Recall thresholds for SUM identification. Default {DEFAULT_RECALL_THRESHOLDS}.",
    )
    args = parser.parse_args()

    report = run_recursive_walk(
        corpora=args.corpora,
        compressor=args.compressor,
        model=args.model,
        max_steps=args.max_steps,
        thresholds=tuple(args.thresholds),
    )

    from scripts.research._receipt_paths import resolve_receipt_path
    prefix = (
        f"recursive_compression_walk_{args.compressor}"
        if args.compressor == "deterministic"
        else f"recursive_compression_walk_{args.compressor}_{_safe_filename(args.model)}"
    )
    out = resolve_receipt_path(RECEIPTS_DIR, prefix)
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n→ wrote {out.relative_to(REPO)}")
    return report


if __name__ == "__main__":
    main()
