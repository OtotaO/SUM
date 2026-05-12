"""Extract transform contract tests.

T2 coverage:
  1. Registry has `extract` registered.
  2. Parameter validation + canonicalisation.
  3. Identity path (triples in → triples out, normalised + sorted).
  4. Sieve path (text in → triples out via DeterministicSieve).
  5. Output canonicalisation: byte-stable + componentwise-sorted.
  6. Tag-set max_tags trimming.
  7. multi_school + llm + auto-llm deferred until follow-ups.
"""
from __future__ import annotations

import asyncio

import pytest

from sum_engine_internal.transforms import (
    TransformEnv,
    TransformResult,
    get_transform,
    list_transforms,
)
from sum_engine_internal.transforms.extract import (
    EXTRACT_TRANSFORM,
    _normalize_triple,
    _unique_sorted,
    _validate_parameters,
)


# ─── Registry shape ─────────────────────────────────────────────────


def test_extract_is_registered():
    assert "extract" in list_transforms()
    assert get_transform("extract") is EXTRACT_TRANSFORM


def test_extract_has_correct_metadata():
    assert EXTRACT_TRANSFORM.name == "extract"
    assert EXTRACT_TRANSFORM.requires_llm is False
    assert EXTRACT_TRANSFORM.digital_source_type == "algorithmicMedia"


# ─── Parameter validation ───────────────────────────────────────────


def test_parameter_validation_defaults():
    """No parameters supplied → fills sensible defaults."""
    norm = _validate_parameters({})
    assert norm == {"extractor": "sieve", "max_tags": 64, "multi_school": False}


def test_parameter_validation_rejects_unknown_extractor():
    with pytest.raises(ValueError, match="extractor"):
        _validate_parameters({"extractor": "magic"})


def test_parameter_validation_rejects_bad_max_tags():
    with pytest.raises(ValueError, match="max_tags"):
        _validate_parameters({"max_tags": 0})
    with pytest.raises(ValueError, match="max_tags"):
        _validate_parameters({"max_tags": 100_000})
    with pytest.raises(ValueError, match="max_tags"):
        _validate_parameters({"max_tags": "many"})


# ─── Helper functions ────────────────────────────────────────────────


def test_normalize_triple_lowercases_and_strips():
    assert _normalize_triple(("  Alice  ", "LIKES", "Cats\n")) == (
        "alice", "likes", "cats",
    )


def test_unique_sorted_dedups_and_sorts():
    """Duplicates removed; remaining triples sorted component-wise.
    Insertion order doesn't affect output."""
    a = _unique_sorted([
        ("bob", "owns", "dog"),
        ("alice", "likes", "cats"),
        ("alice", "likes", "cats"),  # dup
    ])
    b = _unique_sorted([
        ("alice", "likes", "cats"),
        ("alice", "likes", "cats"),
        ("bob", "owns", "dog"),
    ])
    assert a == b == [
        ("alice", "likes", "cats"),
        ("bob", "owns", "dog"),
    ]


# ─── Canonicalisation byte-stability ────────────────────────────────


def test_canonicalize_parameters_normalises_before_hashing():
    """Empty params dict (defaults filled) and explicit-defaults dict
    produce byte-identical canonical bytes."""
    a = EXTRACT_TRANSFORM.canonicalize_parameters({})
    b = EXTRACT_TRANSFORM.canonicalize_parameters({
        "extractor": "sieve", "max_tags": 64, "multi_school": False,
    })
    assert a == b


def test_canonicalize_input_text_is_utf8():
    out = EXTRACT_TRANSFORM.canonicalize_input({"text": "Hello, world."})
    assert out.decode("utf-8") == "Hello, world."


def test_canonicalize_input_triples_sorts_and_normalises():
    """Two callers with different insertion order + different casing
    produce byte-identical input_hash."""
    a = EXTRACT_TRANSFORM.canonicalize_input({"triples": [
        ("alice", "likes", "cats"),
        ("bob", "owns", "dog"),
    ]})
    b = EXTRACT_TRANSFORM.canonicalize_input({"triples": [
        ("Bob", "OWNS", "Dog"),
        ("Alice", "Likes", "Cats"),
    ]})
    assert a == b


def test_canonicalize_input_rejects_bad_shape():
    with pytest.raises(ValueError, match="dict"):
        EXTRACT_TRANSFORM.canonicalize_input("just text")
    with pytest.raises(ValueError, match="'text' or 'triples'"):
        EXTRACT_TRANSFORM.canonicalize_input({"prose": "..."})


def test_canonicalize_output_sorts_and_normalises():
    """Output canonicalisation is the same as input-triples
    canonicalisation: sorted unique normalised triples."""
    a = EXTRACT_TRANSFORM.canonicalize_output([
        ("alice", "likes", "cats"),
        ("bob", "owns", "dog"),
    ])
    b = EXTRACT_TRANSFORM.canonicalize_output([
        ("Bob", "Owns", "DOG"),
        ("ALICE", "likes", "Cats"),
    ])
    assert a == b


def test_canonicalize_output_rejects_bad_shape():
    with pytest.raises(ValueError, match="list"):
        EXTRACT_TRANSFORM.canonicalize_output("not a list")


# ─── Identity path: triples in → triples out ────────────────────────


def test_identity_path_returns_normalised_sorted_unique():
    """Input already has triples. Transform is the identity over the
    triple set, normalised + sorted + deduped + capped at max_tags."""
    result = asyncio.run(EXTRACT_TRANSFORM.apply(
        input={"triples": [
            ("Bob", "OWNS", "Dog"),
            ("alice", "likes", "cats"),
            ("alice", "likes", "cats"),  # dup
        ]},
        parameters={},
        env=TransformEnv(),
    ))
    assert isinstance(result, TransformResult)
    assert result.provider == "canonical-path"
    assert result.model == "canonical-deterministic-v0"
    assert result.llm_calls_made == 0
    # Sorted alphabetically, deduped, normalised:
    assert result.output == [
        ["alice", "likes", "cats"],
        ["bob", "owns", "dog"],
    ]
    assert result.extra["extractor_used"] == "identity"
    assert result.extra["tag_count"] == 2


def test_identity_path_respects_max_tags():
    """max_tags trims after sort+dedup."""
    triples = [(f"e{i}", "rel", f"v{i}") for i in range(10)]
    result = asyncio.run(EXTRACT_TRANSFORM.apply(
        input={"triples": triples},
        parameters={"max_tags": 3},
        env=TransformEnv(),
    ))
    assert len(result.output) == 3


# ─── Sieve path: text in → triples out ──────────────────────────────


def test_sieve_path_extracts_triples_from_text():
    """The sieve path runs DeterministicSieve over the text and
    returns the canonical tag set. Requires spaCy + en_core_web_sm
    (the [sieve] extra)."""
    pytest.importorskip("spacy", reason="[sieve] extra not installed")
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        pytest.skip("spaCy en_core_web_sm model not available")

    result = asyncio.run(EXTRACT_TRANSFORM.apply(
        input={"text": "Alice likes cats. Bob owns a dog."},
        parameters={"extractor": "sieve"},
        env=TransformEnv(),
    ))
    assert result.provider == "canonical-path"
    assert result.extra["extractor_used"] == "sieve"
    # Should produce at least one triple from each sentence; some
    # spaCy versions may parse "likes" / "owns" slightly differently,
    # so we assert tag_count >= 1 rather than a specific tuple set.
    assert result.extra["tag_count"] >= 1
    # Every output entry is a 3-tuple of strings.
    for tag in result.output:
        assert len(tag) == 3
        assert all(isinstance(c, str) for c in tag)


# ─── Deferred-mode rejections ───────────────────────────────────────


def test_multi_school_returns_dict_entries():
    """As of T6, multi_school=True returns a list of
    {"triple", "extractors"} dicts rather than raising. The detailed
    contract is pinned in test_transform_extract_multi_school.py;
    here we just confirm that the previous NotImplementedError no
    longer fires."""
    result = asyncio.run(EXTRACT_TRANSFORM.apply(
        input={"text": "Alice likes cats."},
        parameters={"multi_school": True},
        env=TransformEnv(),
    ))
    assert isinstance(result.output, list)
    # Multi-school output is the dict-shape, not the bare triple-list.
    if result.output:
        assert isinstance(result.output[0], dict)
        assert "triple" in result.output[0]
        assert "extractors" in result.output[0]


def test_explicit_llm_extractor_raises_not_implemented():
    """extractor='llm' is deferred; the legacy LiveLLMAdapter remains
    the path for LLM-based extraction until T2-follow-up."""
    with pytest.raises(NotImplementedError, match="extractor='sieve'"):
        asyncio.run(EXTRACT_TRANSFORM.apply(
            input={"text": "hi"},
            parameters={"extractor": "llm"},
            env=TransformEnv(),
        ))


def test_auto_llm_when_key_present_raises_not_implemented():
    """extractor='auto' with an LLM key present picks LLM — also
    deferred. Without keys, falls through to sieve (handled in the
    sieve path test)."""
    with pytest.raises(NotImplementedError, match="extractor='sieve'"):
        asyncio.run(EXTRACT_TRANSFORM.apply(
            input={"text": "hi"},
            parameters={"extractor": "auto"},
            env=TransformEnv(anthropic_api_key="sk-test"),
        ))
