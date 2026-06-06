"""The local embedding judge fixes the F18 paraphrase misranking.

F18 (docs/DOGFOOD_FINDINGS_2026-06-06.md): the lexical scorer ranked a
faithful paraphrase as MORE lossy (0.739) than a near-empty tag (0.720) —
inverted for the writer use-case. This pins the fix: a local,
deterministic, offline embedding judge restores a sensible ranking
(paraphrase < tag), at zero $.

Skipped unless transformers + torch are installed AND the model is
locally available (offline) — so CI never downloads a model. Run it after
`pip install 'sum-engine[judge]'` with the model cached.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("torch", reason="[judge] extra (torch) not installed")
pytest.importorskip("transformers", reason="[judge] extra not installed")

from sum_engine_internal.research.meaning.local_judge import (
    EmbeddingJudge,
    embedding_entailment_scorer,
    nli_entailment_scorer,
)
from sum_engine_internal.research.meaning.meaning_loss import LexicalCoverageScorer


@pytest.fixture(scope="module")
def scorer():
    """Build the embedding scorer; skip if the model can't load offline
    (not cached) — keeps CI free of model downloads."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    s = embedding_entailment_scorer(threshold=0.5)
    try:
        s.loss("warm up the model.", "warm up.")  # forces lazy load
    except Exception as e:  # noqa: BLE001 - model not cached / offline
        pytest.skip(f"embedding model not available offline: {e}")
    return s


# The F18 corpus (a real paragraph + three writer-style versions).
SOURCE = (
    "The printing press, introduced to Europe in the mid-fifteenth century, "
    "transformed how knowledge spread. Before it, books were copied by hand, "
    "slowly and expensively, so literacy stayed confined to a small clergy "
    "and aristocracy. Movable type made books cheap enough that ordinary "
    "people could own them, and within a few generations literacy rates "
    "climbed across the continent."
)
EXTRACTIVE = (
    "The printing press, introduced to Europe in the mid-fifteenth century, "
    "transformed how knowledge spread. Movable type made books cheap enough "
    "that ordinary people could own them, and within a few generations "
    "literacy rates climbed across the continent."
)
PARAPHRASE = (
    "Cheap printed books broke the clergy's monopoly on reading, lifting "
    "literacy across Europe."
)
TAG = "Printing press democratised knowledge."


def test_embedding_judge_fixes_f18_misranking(scorer):
    """The load-bearing fix: a faithful paraphrase must score LESS lossy
    than a near-empty tag — the exact inversion the lexical scorer made."""
    loss_paraphrase = scorer.loss(SOURCE, PARAPHRASE)
    loss_tag = scorer.loss(SOURCE, TAG)
    assert loss_paraphrase < loss_tag, (
        f"paraphrase {loss_paraphrase:.3f} should be < tag {loss_tag:.3f}"
    )


def test_embedding_judge_beats_lexical_on_paraphrase(scorer):
    """On the same faithful paraphrase, the embedding judge reports far
    less loss than the lexical scorer (which over-reports the reword)."""
    lexical = LexicalCoverageScorer().loss(SOURCE, PARAPHRASE)
    embedding = scorer.loss(SOURCE, PARAPHRASE)
    assert embedding < lexical
    assert embedding < 0.5  # the paraphrase is recognised as well-preserved


def test_embedding_identity_is_zero(scorer):
    assert scorer.loss(SOURCE, SOURCE) == pytest.approx(0.0, abs=1e-9)


def test_embedding_monotone_with_compression(scorer):
    """More compression → not-less loss, across the frontier."""
    losses = [
        scorer.loss(SOURCE, EXTRACTIVE),
        scorer.loss(SOURCE, PARAPHRASE),
        scorer.loss(SOURCE, TAG),
    ]
    assert losses[0] <= losses[1] <= losses[2]


def test_judge_is_deterministic(scorer):
    """Same inputs → identical loss (eval mode, no sampling) — the
    property a replayable receipt needs (machine-pinned)."""
    a = scorer.loss(SOURCE, PARAPHRASE)
    b = scorer.loss(SOURCE, PARAPHRASE)
    assert a == b


def test_scorer_name_records_the_judge(scorer):
    assert scorer.name == "bidirectional-entailment[minilm-cosine-0.5]"


def test_empty_hypothesis_not_entailed():
    judge = EmbeddingJudge()
    # no model load needed for the empty-string short-circuit
    assert judge.entails("anything here", "") is False


# ── NLIJudge — the strict upgrade (directional entailment) ────────────


@pytest.fixture(scope="module")
def nli_scorer():
    """Build the NLI scorer; skip if the model can't load offline."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    s = nli_entailment_scorer(threshold=0.5)
    try:
        s.loss("warm up the model.", "warm up.")
    except Exception as e:  # noqa: BLE001 - model not cached / offline
        pytest.skip(f"NLI model not available offline: {e}")
    return s


# A fluent, on-topic CONTRADICTION — high embedding similarity, but it
# negates the source. This is the case the embedding judge waves through
# and the NLI judge must catch.
CONTRADICTION = "No printing press was ever introduced to Europe, and literacy did not change."


def test_nli_catches_contradiction_embedding_misses(nli_scorer, scorer):
    """The upgrade, demonstrated as COMPLEMENTARITY: the NLI judge scores
    a fluent on-topic contradiction as HIGH loss; the embedding judge, a
    similarity proxy, MISSES it (credits it as moderate, like a tag). This
    is the hallucination-robustness only directional entailment buys, and
    MENLI's argument for blending NLI with similarity."""
    nli_loss = nli_scorer.loss(SOURCE, CONTRADICTION)
    emb_loss = scorer.loss(SOURCE, CONTRADICTION)  # `scorer` = embedding fixture
    assert nli_loss > 0.5, f"NLI must flag the contradiction; got {nli_loss:.3f}"
    assert emb_loss < 0.5, f"embedding is expected to MISS it; got {emb_loss:.3f}"


def test_nli_ranks_contradiction_worse_than_paraphrase(nli_scorer):
    """NLI must rank a contradiction strictly worse than a faithful (if
    lossy) compression — the directional signal lexical/embedding lack."""
    assert nli_scorer.loss(SOURCE, CONTRADICTION) > nli_scorer.loss(SOURCE, PARAPHRASE)


def test_nli_identity_is_zero(nli_scorer):
    assert nli_scorer.loss(SOURCE, SOURCE) == pytest.approx(0.0, abs=1e-9)


def test_nli_is_deterministic(nli_scorer):
    a = nli_scorer.loss(SOURCE, PARAPHRASE)
    b = nli_scorer.loss(SOURCE, PARAPHRASE)
    assert a == b


def test_nli_scorer_name_records_the_model(nli_scorer):
    assert nli_scorer.name.startswith("bidirectional-entailment[nli:")
