"""Default-judge model revisions are pinned (mutable-ref hardening).

HF model IDs are mutable refs: the upstream repo can be force-pushed or
re-uploaded under the same ID (all-MiniLM-L6-v2 was last modified
upstream 2026-06-01), silently changing what the default judge computes
on a fresh cache. ``_resolve_revision`` pins the DEFAULT models to the
exact revision the code was validated against, while a custom model_id
stays unpinned (another repo would 404 on our commit SHA).

Pure-function tests — no torch / transformers / network needed, so this
runs in the ordinary suite (unlike test_local_judge.py, which needs the
[judge] extra plus a cached model).
"""
from __future__ import annotations

from sum_engine_internal.research.meaning.local_judge import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_NLI_MODEL_ID,
    DEFAULT_NLI_MODEL_REVISION,
    EmbeddingJudge,
    NLIJudge,
    _resolve_revision,
)


def test_default_model_gets_pinned_revision():
    assert (
        _resolve_revision(
            DEFAULT_MODEL_ID, None, DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
        )
        == DEFAULT_MODEL_REVISION
    )


def test_custom_model_is_not_pinned_to_our_revision():
    assert (
        _resolve_revision(
            "someone/else-model", None, DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
        )
        is None
    )


def test_explicit_revision_always_wins():
    assert (
        _resolve_revision(
            DEFAULT_MODEL_ID, "my-branch", DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
        )
        == "my-branch"
    )
    assert (
        _resolve_revision(
            "someone/else-model",
            "abc123",
            DEFAULT_MODEL_ID,
            DEFAULT_MODEL_REVISION,
        )
        == "abc123"
    )


def test_pinned_revisions_are_full_commit_shas():
    # A branch name here would silently reintroduce the mutable ref.
    for rev in (DEFAULT_MODEL_REVISION, DEFAULT_NLI_MODEL_REVISION):
        assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)


def test_judges_expose_a_revision_field_defaulting_to_none():
    # None means "resolve at load time" (pinned for defaults); the field
    # exists so a caller can reproduce a run against any revision.
    assert EmbeddingJudge.__dataclass_fields__["revision"].default is None
    assert NLIJudge.__dataclass_fields__["revision"].default is None
    assert DEFAULT_NLI_MODEL_ID != DEFAULT_MODEL_ID


def test_blank_hypothesis_never_entailed_even_at_zero_threshold():
    """The blank-hypothesis guard is unconditional: a pathological
    threshold <= 0 must not flip the historical contract. Pure test —
    the blank path returns before any model load."""
    assert NLIJudge(threshold=-1.0).entails("Alice has a cat.", "   ") is False
    assert EmbeddingJudge(threshold=-1.0).entails("Alice has a cat.", " \t\n") is False
    # NOTE deliberately NOT tested as blank: a zero-width-only hypothesis
    # (U+200B) — .strip() keeps it, so it reaches the model, exactly as it
    # always has. Changing that would alter scorer semantics without a
    # version bump; the receipt layer's _has_visible_text handles the
    # disclosure-field variant of this gotcha.
