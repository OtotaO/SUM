"""
Pin the sum.sheaf_baseline_comparison.v1 schema + digest stability +
scorer-polarity contracts.
"""
from __future__ import annotations

import pytest

from scripts.research.sheaf_baseline_comparison import (
    BASELINES,
    entity_set,
    run_baseline_comparison,
    score_b1_entity_presence_deficit,
    score_b2_jaccard_distance,
)


def test_baseline_report_schema_pinned():
    """sum.sheaf_baseline_comparison.v1 shape contract."""
    report = run_baseline_comparison()
    assert report["schema"] == "sum.sheaf_baseline_comparison.v1"
    assert report["corpus"] == "seed_long_paragraphs"
    assert isinstance(report["n_docs_total"], int)
    assert isinstance(report["n_docs_with_partition"], int)
    assert sorted(report["baselines"]) == sorted(BASELINES.keys())
    assert report["perturbation_classes"] == ["A1", "A2", "A4"]
    assert isinstance(report["per_cell_auc"], dict)
    assert isinstance(report["trusted_mean_auc_by_baseline"], dict)
    assert isinstance(report["bench_digest"], str)
    assert len(report["bench_digest"]) == 64
    int(report["bench_digest"], 16)  # is hex


def test_baseline_bench_digest_in_process_stable():
    """Run baseline comparison twice in-process; assert digests equal."""
    r1 = run_baseline_comparison()
    r2 = run_baseline_comparison()
    assert r1["bench_digest"] == r2["bench_digest"]
    assert r1["per_cell_auc"] == r2["per_cell_auc"]


def test_b1_clean_score_zero_when_all_entities_preserved():
    """B1 polarity: clean render scores 0.0 (no signal)."""
    triples = [("a", "knows", "b"), ("b", "owns", "c")]
    assert score_b1_entity_presence_deficit(triples, triples) == 0.0


def test_b1_increases_when_entity_dropped():
    """B1: dropping a triple raises the score above 0.0."""
    src = [("a", "knows", "b"), ("c", "owns", "d")]
    dropped = [("a", "knows", "b")]
    assert score_b1_entity_presence_deficit(src, dropped) > 0.0


def test_b1_empty_source_returns_zero():
    """Degenerate source: no signal."""
    assert score_b1_entity_presence_deficit([], [("a", "knows", "b")]) == 0.0


def test_b2_jaccard_clean_score_zero():
    """B2 polarity: clean render scores 0.0."""
    triples = [("a", "knows", "b"), ("b", "owns", "c")]
    assert score_b2_jaccard_distance(triples, triples) == 0.0


def test_b2_jaccard_increases_with_spurious_entity():
    """B2 (vs B1) is symmetric: also penalises rendered-only entities."""
    src = [("a", "knows", "b")]
    spurious = [("a", "knows", "b"), ("c", "owns", "d")]
    # B1 gives 0.0 (every source entity present); B2 gives > 0 (spurious added)
    assert score_b1_entity_presence_deficit(src, spurious) == 0.0
    assert score_b2_jaccard_distance(src, spurious) > 0.0


def test_entity_set_excludes_predicates():
    """Predicates are intentionally NOT in the baselines' entity model."""
    triples = [("alice", "knows", "bob")]
    assert entity_set(triples) == {"alice", "bob"}


def test_baseline_digest_is_pinned():
    """Lock the published bench_digest. If this changes, either the
    perturbation harness, the corpus, or the scoring math has shifted —
    investigate before updating this constant."""
    PINNED = "cb32c617a3c692bc03bff49d85ae20e424c46cbb9ff47f9ea02285a90fd34e3b"
    report = run_baseline_comparison()
    assert report["bench_digest"] == PINNED, (
        f"bench_digest drift: got {report['bench_digest']}, expected {PINNED}. "
        "Either the v3 ROC bench's perturbation harness changed (call sites: "
        "partition_trust, perturb_a1/a2/a4_on_target), the corpus changed, "
        "or the baseline scoring math changed. Check, then update PINNED."
    )
