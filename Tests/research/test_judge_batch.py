"""The batched judge path must be BIT-EXACT with the scalar path.

`EntailmentScorer` takes a fast `entails_batch` hook (the EmbeddingJudge
provides it) that embeds each sentence once instead of O(m·n) times. These
tests pin that the batch path (a) is actually taken and (b) yields the
identical loss + drop-lists as the scalar `entails` callback.

The stub tests are torch-free; one `[judge]`-gated test confirms the real
EmbeddingJudge.entails_batch agrees with per-call entails.
"""
from __future__ import annotations

import pytest

from sum_engine_internal.research.meaning.meaning_loss import EntailmentScorer

_PAIRS = [
    ("a b c. d e f. g h.", "a b c. d e."),
    ("the cat sat. the dog ran.", "the cat sat."),
    ("alpha beta gamma. delta.", "alpha. delta epsilon."),
    ("one two. three four. five six.", "one two three four five six."),
    ("solo.", ""),                       # empty transform
]


def _word_subset(premise: str, hyp: str) -> bool:
    # mirror the real judge: a blank hypothesis is not entailed
    return bool(hyp.strip()) and set(hyp.lower().split()) <= set(premise.lower().split())


def _batch(premise: str, hyps):
    return [_word_subset(premise, h) for h in hyps]


def _boom(*_a):
    raise AssertionError("scalar entails() was called — the batch path was not taken")


def test_batch_path_is_bit_exact_with_scalar():
    scalar = EntailmentScorer(entails=_word_subset, judge_name="stub", judge_version="t")
    # entails raises: if loss/explain succeed, the batch path was used exclusively
    batched = EntailmentScorer(
        entails=_boom, entails_batch=_batch, judge_name="stub", judge_version="t",
    )
    for src, tr in _PAIRS:
        assert scalar.loss(src, tr) == batched.loss(src, tr), (src, tr)
        a, b = scalar.explain(src, tr), batched.explain(src, tr)
        assert a.loss == b.loss
        assert a.dropped_claims == b.dropped_claims
        assert a.unsupported_claims == b.unsupported_claims
        assert a.preserved_claims == b.preserved_claims


def test_no_batch_hook_falls_back_to_scalar():
    # without entails_batch, the scalar callback is used (no crash, same answer)
    s = EntailmentScorer(entails=_word_subset, judge_name="stub", judge_version="t")
    assert 0.0 <= s.loss(*_PAIRS[0]) <= 1.0


def test_entails_batch_handles_blanks_and_order():
    # blanks → False, decisions aligned to input order
    out = _batch("a b c", ["a b", "", "x y", "c"])
    assert out == [True, False, False, True]


def test_real_embedding_judge_batch_matches_scalar():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from sum_engine_internal.research.meaning.local_judge import EmbeddingJudge

    j = EmbeddingJudge()
    premise = "The committee approved the budget. The vote was unanimous."
    hyps = ["The budget was approved.", "", "The sky is green.", "The vote was unanimous."]
    batch = j.entails_batch(premise, hyps)
    scalar = [j.entails(premise, h) if h.strip() else False for h in hyps]
    assert batch == scalar
