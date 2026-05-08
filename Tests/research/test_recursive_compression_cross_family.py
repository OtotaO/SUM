"""
Pin the §4.10.1 cross-family aggregator.

Asserts the substantive shape, not exact wall-clock-affected
numbers:

  - schema, bench_digest format
  - five LLM families covered (OpenAI, Meta, Alibaba, DeepSeek,
    Google)
  - both corpora covered
  - the joint label at τ=0.50 is RECURSIVE_COMPRESSION_MODEL_STABLE
    on each corpus AND across corpora
  - per-model median fp recall is in the empirically-observed
    range — bounds generous to allow LLM stochasticity / spaCy
    version drift, but tight enough to flag substrate-shape
    regressions.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RECEIPT = (
    REPO / "fixtures" / "bench_receipts"
    / "recursive_compression_cross_family_2026-05-08.json"
)

EXPECTED_MODELS = {
    "gpt-4o-mini-2024-07-18",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen3.6-35B-A3B",
    "deepseek-ai/DeepSeek-V3-0324",
    "google/gemma-3-27b-it",
}

EXPECTED_CORPORA = {"seed_long_paragraphs", "seed_news_briefs"}


@pytest.mark.skipif(
    not RECEIPT.exists(),
    reason="cross-family aggregate receipt not committed",
)
def test_cross_family_aggregate_model_stable():
    report = json.loads(RECEIPT.read_text())

    assert report["schema"] == "sum.recursive_compression_cross_family.v1"
    assert re.fullmatch(r"[0-9a-f]{64}", report["bench_digest"]), (
        f"bench_digest is not a 64-char hex string: {report['bench_digest']!r}"
    )

    assert set(report["models"]) == EXPECTED_MODELS, (
        f"Cross-family receipt covers wrong model set: "
        f"got {set(report['models'])}, expected {EXPECTED_MODELS}."
    )
    assert set(report["corpora"]) == EXPECTED_CORPORA

    # Joint cross-corpus finding.
    assert report["overall_finding"] == "RECURSIVE_COMPRESSION_MODEL_STABLE_ACROSS_CORPORA", (
        f"Overall finding drift: got {report['overall_finding']!r}. The "
        f"§4.10.1 narrative depends on the SUM-identification claim "
        f"being model-stable across all five LLM lineages on both "
        f"corpora at τ=0.50. If this label flips to MIXED or "
        f"MODEL_DEPENDENT, one or more corpora has at least one doc "
        f"where SUM identification disagrees across families — "
        f"investigate which doc and why before re-pinning."
    )

    # Per-corpus joint label at τ=0.50 — both corpora should be
    # RECURSIVE_COMPRESSION_MODEL_STABLE.
    for corpus in EXPECTED_CORPORA:
        agg = report["per_corpus_aggregates"][corpus]
        joint = agg["joint_by_tau"]["tau_0.50"]
        assert joint["label"] == "RECURSIVE_COMPRESSION_MODEL_STABLE", (
            f"Per-corpus joint label drift on {corpus}: got "
            f"{joint['label']!r}. Both corpora are expected to be "
            f"MODEL_STABLE at τ=0.50."
        )

    # Per-model median fp recall ranges. Generous bounds to allow LLM
    # stochasticity but catch substrate-shape regressions.
    long_corpus = report["per_corpus_aggregates"]["seed_long_paragraphs"]
    news_corpus = report["per_corpus_aggregates"]["seed_news_briefs"]

    # On seed_long_paragraphs: recall 0.55-0.85 across all models
    for model, summary in long_corpus["per_model_summary"].items():
        recall = summary["median_fixed_point_recall_vs_original"]
        assert 0.55 <= recall <= 0.85, (
            f"{model} on seed_long_paragraphs: median fp recall "
            f"{recall} outside expected range [0.55, 0.85]."
        )

    # On seed_news_briefs: recall 0.7-1.0 across all models
    for model, summary in news_corpus["per_model_summary"].items():
        recall = summary["median_fixed_point_recall_vs_original"]
        assert 0.7 <= recall <= 1.0, (
            f"{model} on seed_news_briefs: median fp recall "
            f"{recall} outside expected range [0.7, 1.0]."
        )
