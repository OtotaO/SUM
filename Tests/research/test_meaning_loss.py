"""Sanity contract for the meaning-loss proxies.

These mirror the MeaningBERT-style sanity tests (Beauchemin et al.,
2023): a meaning-loss measure must score identity at 0, disjoint text
near 1, and must not *decrease* loss when source content is deleted from
the transform. The proxies here make no claim to measure meaning — but
they MUST satisfy these minimal monotonicity/boundedness properties or
they are not even a coherent proxy.
"""
from __future__ import annotations

import pytest

from sum_engine_internal.research.meaning.meaning_loss import (
    EntailmentScorer,
    LexicalCoverageScorer,
    MeaningScorer,
    score_pairs,
)


SOURCE = (
    "The treaty was signed in Vienna in 1815. Delegates from the great "
    "powers redrew the map of Europe. The settlement held for decades."
)


# ── LexicalCoverageScorer ─────────────────────────────────────────────


def test_lexical_is_a_meaning_scorer():
    assert isinstance(LexicalCoverageScorer(), MeaningScorer)


def test_lexical_identity_is_zero_loss():
    s = LexicalCoverageScorer()
    assert s.loss(SOURCE, SOURCE) == 0.0


def test_lexical_disjoint_is_total_loss():
    s = LexicalCoverageScorer()
    loss = s.loss(SOURCE, "Quantum chromodynamics describes gluon colour charge.")
    assert loss == pytest.approx(1.0, abs=1e-9)


def test_lexical_two_empties_are_zero_loss():
    s = LexicalCoverageScorer()
    assert s.loss("", "") == 0.0


def test_lexical_bounded_in_unit_interval():
    s = LexicalCoverageScorer()
    for transform in [
        SOURCE,
        "The treaty was signed.",
        "Vienna 1815 treaty Europe settlement decades powers delegates map",
        "Totally unrelated sentence about marine biology and coral.",
        "",
    ]:
        loss = s.loss(SOURCE, transform)
        assert 0.0 <= loss <= 1.0


def test_lexical_monotone_under_deletion():
    """Deleting a source content-unit from the transform must not
    decrease the loss."""
    s = LexicalCoverageScorer()
    full = SOURCE
    dropped_one = SOURCE.replace("Vienna", "")
    dropped_two = dropped_one.replace("Europe", "")
    l_full = s.loss(SOURCE, full)
    l_one = s.loss(SOURCE, dropped_one)
    l_two = s.loss(SOURCE, dropped_two)
    assert l_full <= l_one <= l_two
    assert l_full == 0.0  # identity baseline


def test_lexical_partial_coverage_is_partial_loss():
    s = LexicalCoverageScorer()
    partial = "The treaty was signed in Vienna in 1815."
    loss = s.loss(SOURCE, partial)
    assert 0.0 < loss < 1.0


def test_lexical_is_deterministic():
    s = LexicalCoverageScorer()
    a = s.loss(SOURCE, "The treaty was signed in Vienna.")
    b = s.loss(SOURCE, "The treaty was signed in Vienna.")
    assert a == b


def test_lexical_fabrication_penalised():
    """Content in the transform absent from the source raises loss via
    the fabrication term, even when recall is perfect."""
    s = LexicalCoverageScorer()
    faithful = "Treaty signed Vienna 1815 delegates great powers map Europe settlement held decades"
    with_fab = faithful + " aliens dragons cryptocurrency"
    assert s.loss(SOURCE, with_fab) > s.loss(SOURCE, faithful)


# ── EntailmentScorer (injected judge — no model dependency) ───────────


def _keyword_judge(premise: str, hypothesis: str) -> bool:
    """A deterministic stand-in NLI judge for tests: 'premise entails
    hypothesis' iff every content word of the hypothesis appears in the
    premise. Crude, but deterministic and dependency-free — exercises
    the EntailmentScorer plumbing without a real model."""
    from sum_engine_internal.research.meaning.meaning_loss import _content_units

    p = set(_content_units(premise))
    h = set(_content_units(hypothesis))
    return h.issubset(p) and bool(h)


def test_entailment_is_a_meaning_scorer():
    s = EntailmentScorer(entails=_keyword_judge, judge_name="keyword-test")
    assert isinstance(s, MeaningScorer)


def test_entailment_name_carries_judge():
    s = EntailmentScorer(
        entails=_keyword_judge, judge_name="keyword-test", judge_version="3"
    )
    assert s.name == "bidirectional-entailment[keyword-test]"
    assert s.version == "3"


def test_entailment_identity_is_zero_loss():
    s = EntailmentScorer(entails=_keyword_judge, judge_name="keyword-test")
    assert s.loss(SOURCE, SOURCE) == pytest.approx(0.0, abs=1e-9)


def test_entailment_disjoint_is_total_loss():
    s = EntailmentScorer(entails=_keyword_judge, judge_name="keyword-test")
    loss = s.loss(SOURCE, "Gluons carry colour charge. Quarks are confined.")
    assert loss == pytest.approx(1.0, abs=1e-9)


def test_entailment_bounded():
    s = EntailmentScorer(entails=_keyword_judge, judge_name="keyword-test")
    for transform in [SOURCE, "The treaty was signed in Vienna in 1815.", ""]:
        assert 0.0 <= s.loss(SOURCE, transform) <= 1.0


def test_entailment_empty_transform_high_loss():
    s = EntailmentScorer(entails=_keyword_judge, judge_name="keyword-test")
    # Empty transform: recall 0, fidelity defined as 1.0 → loss = w_recall.
    assert s.loss(SOURCE, "") == pytest.approx(s.w_recall, abs=1e-9)


# ── score_pairs ───────────────────────────────────────────────────────


def test_score_pairs_returns_one_loss_per_pair():
    s = LexicalCoverageScorer()
    pairs = [(SOURCE, SOURCE), (SOURCE, "The treaty was signed."), (SOURCE, "")]
    losses = score_pairs(pairs, s)
    assert len(losses) == 3
    assert all(0.0 <= x <= 1.0 for x in losses)
    assert losses[0] == 0.0
