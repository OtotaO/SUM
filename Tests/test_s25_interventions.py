"""Tests for the §2.5 generator-side intervention module.

Three layers of coverage:

  1. **Pure-function behaviour.** Prompt construction and vocabulary
     summary helpers are pure; they are tested against fixed inputs.
  2. **Pydantic schema construction.** The constrained-extraction
     schema must accept in-vocabulary triples, reject out-of-vocabulary
     triples, and remain JSON-schema-serialisable for OpenAI structured
     output.
  3. **Empty-source fail-closed.** A source with zero axioms produces
     a schema where any non-empty triple list fails validation; the
     only legitimate response is ``triplets: []``. This is the
     correct behaviour when the constrained extractor is asked to
     extract from a fact-empty source.

The runner script ``scripts/bench/runners/s25_generator_side.py`` has
its own dry-run mode that exercises end-to-end serialisation; this
file covers the intervention primitives.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from sum_engine_internal.ensemble.s25_interventions import (
    CANONICAL_FIRST_SYS_PROMPT,
    DEFAULT_CANONICAL_PREDICATES,
    RECEIPT_SCHEMA_CANONICAL_FIRST,
    RECEIPT_SCHEMA_COMBINED,
    RECEIPT_SCHEMA_CONSTRAINED_EXTRACTOR,
    build_canonical_first_user_prompt,
    build_constrained_extraction_schema,
    vocabulary_summary,
)


# --------------------------------------------------------------------------
# Receipt schema identifiers — pinned at module level
# --------------------------------------------------------------------------


def test_receipt_schemas_are_distinct_and_versioned():
    """Each ablation MUST land under a distinct schema so a future
    reader of the receipt can identify which intervention produced
    which numbers without inspection."""
    schemas = {
        RECEIPT_SCHEMA_CANONICAL_FIRST,
        RECEIPT_SCHEMA_CONSTRAINED_EXTRACTOR,
        RECEIPT_SCHEMA_COMBINED,
    }
    assert len(schemas) == 3
    for s in schemas:
        assert s.startswith("sum.s25_")
        assert s.endswith(".v1")


# --------------------------------------------------------------------------
# Canonical-first prompt construction
# --------------------------------------------------------------------------


def test_canonical_first_sys_prompt_demands_verbatim_surface():
    """The prompt's load-bearing instruction is 'surface each source
    fact in canonical form FIRST, using EXACT tokens from the input'.
    This is the mechanism by which the intervention addresses the
    generator-elaboration failure mode."""
    assert "EXACT" in CANONICAL_FIRST_SYS_PROMPT
    assert "FIRST" in CANONICAL_FIRST_SYS_PROMPT
    assert "FAILED render" in CANONICAL_FIRST_SYS_PROMPT


def test_canonical_first_user_prompt_includes_facts():
    prompt = build_canonical_first_user_prompt(
        target_axioms=["alice likes cats", "bob owns dog"],
    )
    assert "alice likes cats" in prompt
    assert "bob owns dog" in prompt
    assert "EXACT tokens" in prompt


def test_canonical_first_user_prompt_threads_negative_constraints():
    prompt = build_canonical_first_user_prompt(
        target_axioms=["alice likes cats"],
        negative_constraints=["alice hates_cats"],
    )
    assert "alice hates_cats" in prompt
    assert "NEGATIVE CONSTRAINTS" in prompt


def test_canonical_first_user_prompt_omits_negative_section_when_empty():
    prompt = build_canonical_first_user_prompt(
        target_axioms=["alice likes cats"],
        negative_constraints=None,
    )
    assert "NEGATIVE CONSTRAINTS" not in prompt


# --------------------------------------------------------------------------
# Constrained-extraction Pydantic schema
# --------------------------------------------------------------------------


@pytest.fixture
def src_axioms():
    return [
        ("alice", "likes", "cats"),
        ("bob", "owns", "dog"),
    ]


def test_constrained_schema_accepts_in_vocabulary_triple(src_axioms):
    Schema = build_constrained_extraction_schema(src_axioms)
    parsed = Schema(triplets=[
        {"subject": "alice", "predicate": "likes", "object": "cats"},
    ])
    assert len(parsed.triplets) == 1
    assert parsed.triplets[0].subject == "alice"


def test_constrained_schema_rejects_out_of_vocabulary_subject(src_axioms):
    Schema = build_constrained_extraction_schema(src_axioms)
    with pytest.raises(ValidationError):
        Schema(triplets=[
            {"subject": "WRONG", "predicate": "likes", "object": "cats"},
        ])


def test_constrained_schema_rejects_out_of_vocabulary_object(src_axioms):
    Schema = build_constrained_extraction_schema(src_axioms)
    with pytest.raises(ValidationError):
        Schema(triplets=[
            {"subject": "alice", "predicate": "likes", "object": "WRONG"},
        ])


def test_constrained_schema_admits_canonical_predicate(src_axioms):
    """Predicates that are not in the source axiom set but are in
    DEFAULT_CANONICAL_PREDICATES (e.g. ``discover``, ``invent``) MUST
    be admitted. This widens the predicate vocabulary just enough to
    cover canonical scientific verbs without admitting paraphrases."""
    Schema = build_constrained_extraction_schema(src_axioms)
    # `discover` is in DEFAULT_CANONICAL_PREDICATES.
    parsed = Schema(triplets=[
        {"subject": "alice", "predicate": "discover", "object": "cats"},
    ])
    assert parsed.triplets[0].predicate == "discover"


def test_constrained_schema_excludes_source_predicate_lemmas():
    """Residual-closure fix: when a source predicate is an inflected
    form (e.g. ``proposed``, ``contains``, ``described``,
    ``discovered``), the constrained schema MUST exclude its lemma
    from the canonical-padding set so the LLM extractor cannot pick
    the lemma over the source surface form. This was the single root
    cause of the 5/50 residual after the first §2.5 closure receipt;
    every failing doc had a source predicate whose lemma sat in
    DEFAULT_CANONICAL_PREDICATES.

    Verified via the JSON schema export — a Pydantic ValidationError
    on the `propose` literal is the load-bearing assertion."""
    from pydantic import ValidationError

    Schema = build_constrained_extraction_schema(
        [("einstein", "proposed", "relativity")]
    )
    # Source surface form admitted.
    Schema(triplets=[
        {"subject": "einstein", "predicate": "proposed", "object": "relativity"},
    ])
    # Lemma excluded.
    with pytest.raises(ValidationError):
        Schema(triplets=[
            {"subject": "einstein", "predicate": "propose", "object": "relativity"},
        ])

    # Same shape for `-s` / `-es` 3rd-person singular.
    Schema = build_constrained_extraction_schema(
        [("water", "contains", "hydrogen")]
    )
    Schema(triplets=[
        {"subject": "water", "predicate": "contains", "object": "hydrogen"},
    ])
    with pytest.raises(ValidationError):
        Schema(triplets=[
            {"subject": "water", "predicate": "contain", "object": "hydrogen"},
        ])

    # Compound predicate: head verb in isolation also excluded.
    Schema = build_constrained_extraction_schema(
        [("birds", "build_nests", "nests")]
    )
    Schema(triplets=[
        {"subject": "birds", "predicate": "build_nests", "object": "nests"},
    ])
    with pytest.raises(ValidationError):
        Schema(triplets=[
            {"subject": "birds", "predicate": "build", "object": "nests"},
        ])


def test_constrained_schema_unrelated_canonical_predicates_survive_lemma_filter():
    """Lemma exclusion must NOT remove canonical predicates that
    are unrelated to the source's predicate form. For
    ``proposed``-as-source, ``discover`` stays in the enum because
    it is not a lemma of ``proposed``."""
    Schema = build_constrained_extraction_schema(
        [("einstein", "proposed", "relativity")]
    )
    # `discover` is in DEFAULT_CANONICAL_PREDICATES and is unrelated
    # to `proposed`'s lemma chain — it must survive.
    Schema(triplets=[
        {"subject": "einstein", "predicate": "discover", "object": "relativity"},
    ])


def test_constrained_schema_empty_source_fails_closed():
    """A source with zero axioms produces a schema where the only
    legitimate response is an empty triplets list."""
    Schema = build_constrained_extraction_schema([])
    parsed = Schema(triplets=[])
    assert parsed.triplets == []
    with pytest.raises(ValidationError):
        Schema(triplets=[
            {"subject": "alice", "predicate": "likes", "object": "cats"},
        ])


def test_constrained_schema_serialises_to_openai_structured_output(src_axioms):
    """OpenAI's structured-output validator consumes the JSON schema
    of the response_format class. The schema MUST be JSON-serialisable
    and contain enum constraints on subject, predicate, and object."""
    Schema = build_constrained_extraction_schema(src_axioms)
    js = Schema.model_json_schema()
    defs = js.get("$defs", {}) or js.get("definitions", {})
    assert defs, "schema must have nested triplet definition"
    triplet_def = next(iter(defs.values()))
    props = triplet_def["properties"]
    assert "enum" in props["subject"]
    assert "enum" in props["predicate"]
    assert "enum" in props["object"]
    # Subject enum is exactly the source-axiom subjects.
    assert set(props["subject"]["enum"]) == {"alice", "bob"}
    # Predicate enum is source ∪ canonical.
    assert "likes" in props["predicate"]["enum"]
    assert "owns" in props["predicate"]["enum"]
    assert any(p in props["predicate"]["enum"] for p in DEFAULT_CANONICAL_PREDICATES)


# --------------------------------------------------------------------------
# Vocabulary summary
# --------------------------------------------------------------------------


def test_vocabulary_summary_counts_unique_tokens(src_axioms):
    summary = vocabulary_summary(src_axioms)
    assert summary["n_source_axioms"] == 2
    assert summary["n_unique_subjects"] == 2
    assert summary["n_unique_predicates"] == 2
    assert summary["n_unique_objects"] == 2
    assert summary["n_predicates_with_canonical_padding"] >= len(DEFAULT_CANONICAL_PREDICATES)


def test_vocabulary_summary_handles_duplicates():
    """Duplicate axioms collapse in the unique-token counts."""
    src = [
        ("alice", "likes", "cats"),
        ("alice", "likes", "cats"),  # exact duplicate
        ("alice", "likes", "dogs"),  # same subj/pred, new obj
    ]
    summary = vocabulary_summary(src)
    assert summary["n_unique_subjects"] == 1
    assert summary["n_unique_predicates"] == 1
    assert summary["n_unique_objects"] == 2
