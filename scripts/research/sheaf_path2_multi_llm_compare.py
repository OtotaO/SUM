"""
Multi-LLM Path 2 comparison — does the synthetic-vs-real gap from
§4.7.2 hold across LLM families, or is it gpt-4o-mini-specific?

Path 2 v3 (`sheaf_path2_v3_bench`) found
`HYBRID_LOSES_TO_BASELINE_ON_REAL_LLM` (Δ=−0.021) on
gpt-4o-mini-2024-07-18 — the synthetic-bench WIN
(HYBRID_BEATS_BASELINE, Δ=+0.043) does NOT generalise to real LLM
hallucinations on that one model. The §4.7.2 narrative depends on
this gap being structural, not gpt-4o-mini-specific.

This bench captures parallel snapshots from multiple LLM families
using the SAME source corpus, SAME prompt classes, and SAME
deterministic Phase 2 scorer, then aggregates the per-model verdicts
into a joint classification. Default model set spans six
organisational lineages: OpenAI gpt-4o-mini, Anthropic Claude Haiku
4.5, Meta Llama-3.3-70B, Alibaba Qwen3.6-35B-A3B, DeepSeek V3-0324,
Google Gemma-3-27B. The four open-weights members route through
Hugging Face Inference Providers via the OpenAI-compatible router
(``HF_TOKEN`` env var); the closed pair use their respective vendor
APIs.

## Findings the joint classification can produce

  STRUCTURAL_GAP_ALL_MODELS_LOSE     — every model: hybrid LOSES
  STRUCTURAL_GAP_NO_MODEL_BEATS      — every model: TIES or LOSES
  HYBRID_BEATS_ALL_MODELS            — every model: hybrid BEATS
  HYBRID_BEATS_OR_TIES_ALL_MODELS    — every model: BEATS or TIES
  MIXED_VERDICTS_MODEL_DEPENDENT     — verdicts diverge across LLMs

The first two findings strengthen the §4.7.2 narrative — the
synthetic-vs-real gap is structural, not an artifact of one LLM.
The fourth and fifth complicate it: the gap may be specific to
gpt-4o-mini's perturbation distribution.

## Architecture

Phase 1 (one-time, requires API keys): one capture per model →
`fixtures/bench_renders/path2_<model_safe>.json`. Closed-model
snapshots use vendor APIs (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`);
open-weights snapshots route through HF (`HF_TOKEN`).

Phase 2 (deterministic): for each model snapshot, runs the same
`run_path2_v3_bench(snapshot)` from `sheaf_path2_v3_bench`. Same
DeterministicSieve re-extraction, same v3.2+per-triple+B2 Borda
fusion, same verdict thresholds (BEATS ≥ +0.03; TIES ≥ −0.02;
else LOSES). Per-model digests are byte-stable given each
snapshot.

Output: `fixtures/bench_receipts/path2_multi_llm_compare_<DATE>.json`
Schema: `sum.sheaf_path2_multi_llm_compare.v1`

## Honest scope

The default 6-model set is the working sample of "LLM family." A
robust structural claim would want a deeper sample, but six
organisational lineages (OpenAI / Anthropic / Meta / Alibaba /
DeepSeek / Google) spanning closed and open weights is a
substantively wider cross-family check than the single-model
§4.7.2 result. Earlier pairwise framing kept here for context:
two models is a thin sample of "LLM family"; 3+ families
(open-weights, GPT-class, Claude-class
at minimum). Two models is enough to disprove "model-independent"
if they disagree, and enough to weakly support it if they agree.
The receipt records this scope so downstream readers can weight
the finding correctly.
"""
from __future__ import annotations

# MUST come before any numpy/scipy import — sets BLAS thread vars
# at process startup so bench_digest is byte-stable across fresh
# Python processes. See `_deterministic_blas` for the rationale.
import scripts.research._deterministic_blas  # noqa: F401

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve

from scripts.research.sheaf_path2_v3_bench import (
    _PROMPT_CLASSES,
    _build_prompt,
    run_path2_v3_bench,
    CORPUS_PATH,
    RECEIPTS_DIR,
    RENDERS_DIR,
    PINNED_MODEL as PINNED_OPENAI_MODEL,
)

# Default cross-family pair. The OpenAI member matches the existing
# Path 2 snapshot — its digest is pinned by Tests/research/test_sheaf_path2_v3.py.
PINNED_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
# Open-weights models routed via HF Inference Providers
# (https://router.huggingface.co/v1). One representative per family,
# picked from the set the operator's HF account can route. Each
# spans a distinct training corpus + organisational lineage from
# the closed pair above.
PINNED_META_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
PINNED_QWEN_MODEL = "Qwen/Qwen3.6-35B-A3B"
PINNED_DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V3-0324"
PINNED_GOOGLE_MODEL = "google/gemma-3-27b-it"
DEFAULT_MODELS = (
    PINNED_OPENAI_MODEL,
    PINNED_ANTHROPIC_MODEL,
    PINNED_META_MODEL,
    PINNED_QWEN_MODEL,
    PINNED_DEEPSEEK_MODEL,
    PINNED_GOOGLE_MODEL,
)

# Per-call budget for capture. A single render is short (<500 tokens)
# but Anthropic's tool-free `messages.create` for narrative output
# can take several seconds at peak; 60s is generous and rarely hit.
CAPTURE_TIMEOUT_S = 180.0
# HF routed providers under load occasionally serve a slow request
# (60-120s tail). Retry once before failing so a single transient
# slow call doesn't waste the entire capture run.
CAPTURE_RETRIES_ON_TIMEOUT = 2
CAPTURE_RETRY_BACKOFF_S = 5.0

# Default corpus name. The legacy `seed_long_paragraphs` keeps every
# existing per-model digest pin valid; non-default corpora go under
# distinct snapshot/receipt paths so a cross-corpus bench can run
# alongside the n=6 bench without collision.
DEFAULT_CORPUS = "seed_long_paragraphs"


def _corpus_path(corpus: str) -> Path:
    """Resolve the corpus JSON file from a corpus identifier.

    Walks up from RENDERS_DIR (the only repo-relative path imported
    here) to the repo root, then descends to scripts/bench/corpora.
    """
    repo_root = RENDERS_DIR.parents[1]
    return repo_root / "scripts" / "bench" / "corpora" / f"{corpus}.json"


def _safe_model_filename(model: str) -> str:
    """Produce a filesystem-safe slug for the snapshot path. Lowercases,
    replaces any non-[a-z0-9._-] with '_'. Two different model ids will
    not collide as long as they differ outside the unsafe character set."""
    return re.sub(r"[^a-z0-9._-]+", "_", model.lower())


def _snapshot_path_for_model(model: str, corpus: str = DEFAULT_CORPUS) -> Path:
    """Return the per-(corpus, model) snapshot path.

    For the default corpus + gpt-4o-mini combination, returns the
    legacy path (``path2_seed_long_paragraphs.json``) so the existing
    pinned digest in Tests/research/test_sheaf_path2_v3.py keeps
    working. For the default corpus + any other model, returns the
    `path2_<model_safe>.json` path used by the n=6 bench. For a
    non-default corpus, the corpus name is incorporated into the
    filename so cross-corpus snapshots don't collide with the n=6 set.
    """
    if corpus == DEFAULT_CORPUS:
        if model == PINNED_OPENAI_MODEL:
            return RENDERS_DIR / "path2_seed_long_paragraphs.json"
        return RENDERS_DIR / f"path2_{_safe_model_filename(model)}.json"
    return RENDERS_DIR / f"path2_{corpus}_{_safe_model_filename(model)}.json"


# ─── Phase 1 — Per-model snapshot capture ────────────────────────────


async def _render_one_via_dispatch(adapter: Any, triples: list[tuple[str, str, str]],
                                   prompt_class: str) -> str:
    """Single LLM call via the vendor-agnostic dispatcher's generate_text.

    Both OpenAIAdapter and AnthropicAdapter expose generate_text(system,
    user, call_timeout_s); this keeps the capture path identical across
    families. Retries on per-call timeout up to
    ``CAPTURE_RETRIES_ON_TIMEOUT`` extra attempts so a single slow HF
    routed call doesn't kill a 64-call capture.
    """
    from sum_engine_internal.ensemble.llm_dispatch import LLMCallTimeoutError

    sys_prompt, user_prompt = _build_prompt(triples, prompt_class)
    last_err: Exception | None = None
    for attempt in range(1 + CAPTURE_RETRIES_ON_TIMEOUT):
        try:
            return await adapter.generate_text(
                system=sys_prompt,
                user=user_prompt,
                call_timeout_s=CAPTURE_TIMEOUT_S,
            )
        except LLMCallTimeoutError as e:
            last_err = e
            if attempt < CAPTURE_RETRIES_ON_TIMEOUT:
                print(f"[capture] timeout (attempt {attempt+1}/"
                      f"{1+CAPTURE_RETRIES_ON_TIMEOUT}); "
                      f"retrying after {CAPTURE_RETRY_BACKOFF_S:.1f}s")
                await asyncio.sleep(CAPTURE_RETRY_BACKOFF_S)
                continue
    assert last_err is not None
    raise last_err


async def _capture_snapshot_for_model(model: str,
                                      corpus: str = DEFAULT_CORPUS) -> dict[str, Any]:
    """Phase 1 capture for an arbitrary (corpus, model) cell.

    Requires the appropriate API key for the model's family:
      - openai (gpt-/o*-)  → OPENAI_API_KEY
      - anthropic (claude-) → ANTHROPIC_API_KEY
      - HF-namespaced (org/model) → HF_TOKEN
    """
    from sum_engine_internal.ensemble.llm_dispatch import get_adapter

    adapter = get_adapter(model)
    print(f"[capture] corpus: {corpus} model: {model}")

    sieve = DeterministicSieve()
    with open(_corpus_path(corpus)) as f:
        data = json.load(f)
    docs_raw = data["documents"]
    docs: list[tuple[str, str, list[tuple[str, str, str]]]] = []
    for d in docs_raw:
        triples = list(sieve.extract_triplets(d["text"]))
        if triples:
            docs.append((d["id"], d["text"], triples))
    print(f"[capture] {len(docs)} docs with non-empty extractions")

    snapshot: dict[str, Any] = {
        "schema": "sum.sheaf_path2_render_snapshot.v1",
        "corpus": corpus,
        "model": model,
        "prompt_classes": list(_PROMPT_CLASSES),
        "renders": {},
    }
    n_calls = 0
    for i, (doc_id, _text, source_triples) in enumerate(docs):
        snapshot["renders"][doc_id] = {
            "source_triples": [list(t) for t in source_triples],
            "by_prompt_class": {},
        }
        for prompt_class in _PROMPT_CLASSES:
            tome = await _render_one_via_dispatch(adapter, source_triples, prompt_class)
            snapshot["renders"][doc_id]["by_prompt_class"][prompt_class] = tome
            n_calls += 1
            print(f"[capture] [{i+1}/{len(docs)}] {doc_id} {prompt_class:<18} "
                  f"({len(tome)} chars)")

    snapshot["n_llm_calls"] = n_calls
    return snapshot


def _ensure_snapshot_for_model(model: str, corpus: str = DEFAULT_CORPUS,
                               force: bool = False) -> dict[str, Any]:
    """Load the per-(corpus, model) snapshot, regenerating via Phase 1 if
    missing or `force=True`. Persists JSON-canonicalised to the
    per-(corpus, model) path."""
    path = _snapshot_path_for_model(model, corpus)
    if path.exists() and not force:
        with open(path) as f:
            snap = json.load(f)
        print(f"[capture] using cached snapshot: {path.name}")
        return snap
    print(f"[capture] regenerating snapshot at {path}")
    snap = asyncio.run(_capture_snapshot_for_model(model, corpus))
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n")
    print(f"[capture] wrote {path}")
    return snap


# ─── Phase 2 — Per-model scoring + cross-model aggregation ───────────


def _aggregate_verdicts(per_model: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Classify the joint cross-model finding from per-model verdicts.

    Returns (joint_finding, summary) where summary contains the per-model
    Δ(borda − b2) and verdict labels. The joint finding is the load-
    bearing claim: does the synthetic-vs-real gap appear consistently?
    """
    verdicts = {m: r["verdict"] for m, r in per_model.items()}
    deltas = {m: r["delta_borda_vs_b2_mean"] for m, r in per_model.items()}

    BEATS = "HYBRID_BEATS_BASELINE_ON_REAL_LLM"
    TIES = "HYBRID_TIES_BASELINE_ON_REAL_LLM"
    LOSES = "HYBRID_LOSES_TO_BASELINE_ON_REAL_LLM"

    vs = set(verdicts.values())
    # n=1 has no cross-model signal; report the single-model verdict
    # without dressing it up as a structural finding.
    if len(per_model) < 2:
        joint = f"SINGLE_MODEL_{next(iter(vs))}"
    elif vs == {LOSES}:
        joint = "STRUCTURAL_GAP_ALL_MODELS_LOSE"
    elif vs == {BEATS}:
        joint = "HYBRID_BEATS_ALL_MODELS"
    elif vs <= {TIES, LOSES}:
        joint = "STRUCTURAL_GAP_NO_MODEL_BEATS"
    elif vs <= {TIES, BEATS}:
        joint = "HYBRID_BEATS_OR_TIES_ALL_MODELS"
    else:
        # Mix that includes both BEATS and LOSES — verdict is model-dependent.
        joint = "MIXED_VERDICTS_MODEL_DEPENDENT"

    summary = {
        "per_model_verdict": verdicts,
        "per_model_delta_borda_vs_b2": deltas,
        "delta_spread": max(deltas.values()) - min(deltas.values()) if deltas else 0.0,
    }
    return joint, summary


def run_multi_llm_compare(models: tuple[str, ...],
                          corpus: str = DEFAULT_CORPUS,
                          force_capture: bool = False) -> dict[str, Any]:
    """Capture (or load) snapshots for each (corpus, model) cell, score
    each via the pinned Phase 2 path, aggregate verdicts.

    Each per-model report is a `sum.sheaf_path2_v3_bench.v1` receipt
    (same shape as PR #156's). The compare receipt embeds them under
    `per_model_reports` and adds a joint classification.
    """
    print("=" * 72)
    print("Multi-LLM Path 2 comparison")
    print(f"Corpus: {corpus}")
    print(f"Models: {models}")
    print("=" * 72)

    per_model: dict[str, dict[str, Any]] = {}
    for model in models:
        print(f"\n──── {model} ────")
        snap = _ensure_snapshot_for_model(model, corpus=corpus,
                                          force=force_capture)
        report = run_path2_v3_bench(snap)
        per_model[model] = report

    joint_finding, summary = _aggregate_verdicts(per_model)

    print("\n" + "=" * 72)
    print("Cross-model summary")
    print("=" * 72)
    for model in models:
        r = per_model[model]
        print(f"  {model:42s}  Δ={r['delta_borda_vs_b2_mean']:+.4f}  "
              f"{r['verdict']}")
    print(f"\n  joint finding: {joint_finding}")
    print(f"  Δ-spread across models: {summary['delta_spread']:+.4f}")

    out: dict[str, Any] = {
        "schema": "sum.sheaf_path2_multi_llm_compare.v1",
        "corpus": corpus,
        "models": list(models),
        "per_model_reports": per_model,
        "per_model_verdict": summary["per_model_verdict"],
        "per_model_delta_borda_vs_b2": summary["per_model_delta_borda_vs_b2"],
        "delta_spread": summary["delta_spread"],
        "joint_finding": joint_finding,
        "n_models": len(models),
        "method_notes": (
            "Multi-LLM extension of Path 2 v3 bench. Each model produces "
            "an independent capture-once snapshot at "
            "fixtures/bench_renders/path2_<model_safe>.json (gpt-4o-mini "
            "uses the legacy filename for digest-pin compatibility). "
            "Phase 2 scoring is the same deterministic v3.2+per-triple + "
            "B2 Borda fusion as the single-LLM bench. Joint finding "
            "classifies whether the synthetic-vs-real gap from §4.7.2 "
            "appears consistently across LLM families or is "
            "gpt-4o-mini-specific."
        ),
        "honest_scope": (
            "Two LLM families (OpenAI gpt-4o-mini, Anthropic Claude "
            "Haiku 4.5) is a thin sample. A robust 'structural' claim "
            "would want 3+ families. Two is enough to falsify "
            "model-independence if they disagree, and enough to weakly "
            "support it if they agree."
        ),
    }
    return out


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=list(DEFAULT_MODELS),
        help="Model ids to compare. Defaults to "
             f"{DEFAULT_MODELS}.",
    )
    parser.add_argument(
        "--corpus", default=DEFAULT_CORPUS,
        help=f"Corpus identifier (file under scripts/bench/corpora/<corpus>.json). "
             f"Default: {DEFAULT_CORPUS!r}. Non-default corpora go under distinct "
             f"snapshot/receipt paths so they don't collide with the n=6 set "
             f"or its pinned digests.",
    )
    parser.add_argument(
        "--regenerate-snapshots", action="store_true",
        help="Force re-running Phase 1 for every listed model. Costs API budget.",
    )
    args = parser.parse_args()

    report = run_multi_llm_compare(
        tuple(args.models),
        corpus=args.corpus,
        force_capture=args.regenerate_snapshots,
    )

    from scripts.research._receipt_paths import resolve_receipt_path
    # Always include the corpus suffix in the receipt prefix. The bare
    # `path2_multi_llm_compare_<date>.json` filename is reserved for the
    # n=6 PR #161 historical receipt (claude-haiku-4.5 snapshot was
    # captured then); subsequent runs — including future runs on the
    # default corpus — write to corpus-suffixed paths so they don't
    # silently collide with the historical record.
    receipt_prefix = f"path2_multi_llm_compare_{args.corpus}"
    out = resolve_receipt_path(RECEIPTS_DIR, receipt_prefix)
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n→ wrote {out.relative_to(out.parents[2])}")
    return report


if __name__ == "__main__":
    main()
