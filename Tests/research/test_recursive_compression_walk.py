"""
Pin the recursive-compression walk receipts at the level of
*substantive shape*, not exact numbers. The walk machinery is
deterministic given the cached LLM-render snapshot; this test
verifies the receipt's signature — bench_digest, schema, the
expected per-corpus medians, and the expected categorical
findings (deterministic arm produces some collapsed-to-empty
docs on news briefs; LLM arm produces no collapses on either
corpus).

Re-running the walk locally requires:
  - deterministic arm: no API key, instant
  - llm arm Phase 2 replay: no API key, reads cached snapshot
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RECEIPTS_DIR = REPO / "fixtures" / "bench_receipts"

DETERMINISTIC_RECEIPT = (
    RECEIPTS_DIR / "recursive_compression_walk_deterministic_2026-05-08.json"
)
LLM_RECEIPT = (
    RECEIPTS_DIR
    / "recursive_compression_walk_llm_gpt-4o-mini-2024-07-18_2026-05-08.json"
)


@pytest.mark.skipif(
    not DETERMINISTIC_RECEIPT.exists(),
    reason="deterministic walk receipt not committed",
)
def test_deterministic_walk_documents_sieve_canonical_asymmetry():
    """The deterministic arm uses canonical_tome, which produces
    bare-lemma prose ('The alice like cat.') that the sieve cannot
    fully re-extract. The walk converges in 2-3 median steps, with
    median fixed-point recall ≤ 0.5 — substantial information loss
    that the substrate's verifier-layer 'lossless round-trip'
    claim does not surface. Pinning the deterministic-arm result
    documents the sieve↔canonical-tome asymmetry."""
    report = json.loads(DETERMINISTIC_RECEIPT.read_text())

    assert report["schema"] == "sum.recursive_compression_walk.v1"
    assert report["compressor"] == "deterministic"
    assert re.fullmatch(r"[0-9a-f]{64}", report["bench_digest"])

    # Deterministic arm operates on both corpora.
    assert sorted(report["corpora"]) == sorted([
        "seed_long_paragraphs", "seed_news_briefs",
    ])

    # Per-corpus medians: substantial recall loss is the load-bearing
    # finding. Bound the medians generously to allow for sieve / spaCy
    # version drift but catch substrate-shape regressions.
    for corpus, max_median_recall in [
        ("seed_long_paragraphs", 0.5),
        ("seed_news_briefs", 0.5),
    ]:
        summary = report["by_corpus"][corpus]["aggregate"]["summary"]
        recall = summary["median_fixed_point_recall_vs_original"]
        assert recall <= max_median_recall, (
            f"deterministic-arm median fp recall on {corpus} is "
            f"unexpectedly high ({recall}); the asymmetry finding "
            f"depends on recall ≤ ~0.5. If this assertion fires the "
            f"canonical_tome rendering may have changed."
        )

    # News briefs should produce some collapsed-to-empty docs under
    # the deterministic compressor (their event-style prose has fewer
    # markdown-friendly subjects).
    news_summary = report["by_corpus"]["seed_news_briefs"]["aggregate"]["summary"]
    assert news_summary["n_docs_collapsed_to_empty"] >= 1, (
        "Expected ≥1 seed_news_briefs doc to collapse to ∅ under the "
        "deterministic walk — the canonical tome's lemmatized output "
        "is not sieve-extractable for many news-style prose patterns."
    )


@pytest.mark.skipif(
    not LLM_RECEIPT.exists(),
    reason="LLM walk receipt not committed; capture requires "
           "OPENAI_API_KEY for Phase 1, replay reads cached snapshot",
)
def test_llm_walk_preserves_more_recall_than_deterministic():
    """LLM-mediated grammatical render survives sieve re-extraction
    much better than the bare-lemma canonical tome. Median fp recall
    should be ≥ 0.5 on both corpora and ≥ 0.7 on seed_news_briefs."""
    report = json.loads(LLM_RECEIPT.read_text())

    assert report["schema"] == "sum.recursive_compression_walk.v1"
    assert report["compressor"] == "llm"
    assert report["llm_model"] == "gpt-4o-mini-2024-07-18"
    assert re.fullmatch(r"[0-9a-f]{64}", report["bench_digest"])
    assert sorted(report["corpora"]) == sorted([
        "seed_long_paragraphs", "seed_news_briefs",
    ])

    # Per-corpus median fixed-point recall.
    for corpus, min_median_recall in [
        ("seed_long_paragraphs", 0.5),
        ("seed_news_briefs", 0.7),
    ]:
        summary = report["by_corpus"][corpus]["aggregate"]["summary"]
        recall = summary["median_fixed_point_recall_vs_original"]
        assert recall >= min_median_recall, (
            f"LLM-arm median fp recall on {corpus} is below the pinned "
            f"floor ({recall} < {min_median_recall}). The LLM's "
            f"grammatical render should preserve substantially more of "
            f"the original axiom-set than the deterministic compressor."
        )

    # Neither corpus should have docs collapse to ∅ under the LLM
    # compressor — the LLM produces grammatical prose that the sieve
    # can extract from.
    for corpus in ("seed_long_paragraphs", "seed_news_briefs"):
        summary = report["by_corpus"][corpus]["aggregate"]["summary"]
        assert summary["n_docs_collapsed_to_empty"] == 0, (
            f"LLM-arm should not produce collapsed-to-∅ docs on "
            f"{corpus}; got {summary['n_docs_collapsed_to_empty']}."
        )

    # Recall thresholds: every threshold must be present in at least
    # one doc's per-doc-sums dict.
    assert sorted(report["recall_thresholds"]) == [0.5, 0.7, 0.9, 0.99]


@pytest.mark.skipif(
    not (DETERMINISTIC_RECEIPT.exists() and LLM_RECEIPT.exists()),
    reason="both receipts required for cross-arm comparison",
)
def test_llm_arm_dominates_deterministic_arm_on_recall():
    """Direct comparison: at the median, the LLM compressor preserves
    more recall than the deterministic compressor on the same corpus.
    This is the substantive comparative claim of the recursive-
    compression measurement."""
    det = json.loads(DETERMINISTIC_RECEIPT.read_text())
    llm = json.loads(LLM_RECEIPT.read_text())

    for corpus in ("seed_long_paragraphs", "seed_news_briefs"):
        det_recall = (
            det["by_corpus"][corpus]["aggregate"]["summary"]
            ["median_fixed_point_recall_vs_original"]
        )
        llm_recall = (
            llm["by_corpus"][corpus]["aggregate"]["summary"]
            ["median_fixed_point_recall_vs_original"]
        )
        assert llm_recall > det_recall, (
            f"LLM-arm fp recall on {corpus} ({llm_recall}) is not "
            f"higher than deterministic-arm fp recall ({det_recall}). "
            f"The substantive comparative claim — LLM grammatical "
            f"render preserves more recall than the lemmatized "
            f"canonical-tome compressor — is load-bearing for the "
            f"recursive-compression narrative."
        )
