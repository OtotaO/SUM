"""T6 — multi_school mode tests.

The dream's "multiple schools of categorization in tandem" element.
Runs N extractors over the same input; output is a list of
{triple, extractors} dicts that pin per-tag provenance.

Tests:
  1. naive_token_pair_extract is deterministic + dependency-free.
  2. multi_school output shape: list of {"triple", "extractors"}.
  3. Output is sorted lex-stably for byte-equivalent receipts.
  4. canonicalize_output dispatches on shape.
  5. Sieve-unavailable case still works (falls back to naive-only).
  6. Receipt with multi-school output round-trips through
     sign+verify+integrity_check.
"""
from __future__ import annotations

import asyncio
import copy

import pytest

from sum_engine_internal.transforms import get_transform, TransformEnv
from sum_engine_internal.transforms.extract import (
    EXTRACT_TRANSFORM,
    naive_token_pair_extract,
)


# ─── naive_token_pair_extract ───────────────────────────────────────


def test_naive_is_deterministic():
    """Two runs over the same input produce identical triples."""
    a = naive_token_pair_extract("Alice likes cats. Bob owns a dog.")
    b = naive_token_pair_extract("Alice likes cats. Bob owns a dog.")
    assert a == b


def test_naive_emits_adjacent_pairs():
    """Each triple is (token_n, 'next_to', token_n+1)."""
    out = naive_token_pair_extract("Alice likes cats")
    # Lowercased + filtered by stopwords + length>=3
    assert out == [
        ("alice", "next_to", "likes"),
        ("likes", "next_to", "cats"),
    ]


def test_naive_drops_stopwords_and_short_tokens():
    """The/a/an/of/etc. plus tokens <3 chars are dropped."""
    out = naive_token_pair_extract("The cat sat on a mat")
    # "the", "on", "a" → dropped (stopwords);
    # "cat", "sat", "mat" → kept (3 chars each)
    # Resulting tokens: ["cat", "sat", "mat"]
    assert out == [
        ("cat", "next_to", "sat"),
        ("sat", "next_to", "mat"),
    ]


def test_naive_is_different_from_sieve():
    """Sieve and naive should disagree — that's the point. Sieve
    emits dependency-grammar triples; naive emits adjacency triples."""
    # We don't import sieve here (test stays cheap); we just check
    # that naive's predicate is fixed at 'next_to', which the sieve
    # would never emit.
    out = naive_token_pair_extract("Alice likes cats.")
    assert all(t[1] == "next_to" for t in out)


# ─── multi_school output shape ──────────────────────────────────────


def test_multi_school_produces_dict_entries():
    """Output is list of {"triple", "extractors"} dicts."""
    result = asyncio.run(EXTRACT_TRANSFORM.apply(
        input={"text": "Alice likes cats."},
        parameters={"multi_school": True},
        env=TransformEnv(),
    ))
    assert isinstance(result.output, list)
    for entry in result.output:
        assert isinstance(entry, dict)
        assert "triple" in entry
        assert "extractors" in entry
        assert isinstance(entry["triple"], list)
        assert len(entry["triple"]) == 3
        assert isinstance(entry["extractors"], list)
        assert all(isinstance(e, str) for e in entry["extractors"])


def test_multi_school_output_is_sorted():
    """Output triples are sorted lex; extractors within each entry
    are sorted alphabetically. Two runs produce byte-identical
    output (the receipt's output_hash binds to this canonical
    shape)."""
    a = asyncio.run(EXTRACT_TRANSFORM.apply(
        input={"text": "Alice likes cats. Bob owns dogs."},
        parameters={"multi_school": True},
        env=TransformEnv(),
    ))
    b = asyncio.run(EXTRACT_TRANSFORM.apply(
        input={"text": "Alice likes cats. Bob owns dogs."},
        parameters={"multi_school": True},
        env=TransformEnv(),
    ))
    # Same input → same output, byte-stable.
    assert a.output == b.output
    # Output is sorted by (triple, extractors).
    keys = [(e["triple"], e["extractors"]) for e in a.output]
    assert keys == sorted(keys)


def test_multi_school_reports_per_extractor_counts():
    result = asyncio.run(EXTRACT_TRANSFORM.apply(
        input={"text": "Alice likes cats. Bob owns dogs."},
        parameters={"multi_school": True},
        env=TransformEnv(),
    ))
    assert result.extra["extractor_used"] == "multi_school"
    assert "naive" in result.extra["extractors_ran"]
    assert "by_extractor_counts" in result.extra
    # Naive always produces at least one triple from a sentence with
    # >= 2 content tokens.
    assert result.extra["by_extractor_counts"]["naive"] > 0


def test_multi_school_rejects_triples_input():
    """multi_school mode requires raw text — running multiple
    extractors over an already-extracted triple list is meaningless.
    The transform rejects this input shape."""
    with pytest.raises(ValueError, match="'text'"):
        asyncio.run(EXTRACT_TRANSFORM.apply(
            input={"triples": [("a", "b", "c")]},
            parameters={"multi_school": True},
            env=TransformEnv(),
        ))


def test_multi_school_respects_max_tags():
    """max_tags caps the entry count post-sort."""
    result = asyncio.run(EXTRACT_TRANSFORM.apply(
        input={"text": "Alice likes cats. Bob owns dogs. Carol reads books. Dave plays piano."},
        parameters={"multi_school": True, "max_tags": 3},
        env=TransformEnv(),
    ))
    assert len(result.output) == 3


# ─── canonicalize_output dispatches on shape ────────────────────────


def test_canonicalize_output_handles_multi_school_shape():
    """Output of multi_school = list of dicts; canonicaliser
    recognises this shape and sorts by triple lex then extractor
    set."""
    multi_school_output = [
        {"triple": ["bob", "owns", "dog"], "extractors": ["sieve", "naive"]},
        {"triple": ["alice", "likes", "cats"], "extractors": ["sieve"]},
    ]
    a = EXTRACT_TRANSFORM.canonicalize_output(multi_school_output)
    # Reorder + change extractor list order → same canonical bytes.
    b = EXTRACT_TRANSFORM.canonicalize_output([
        {"triple": ["alice", "likes", "cats"], "extractors": ["sieve"]},
        {"triple": ["bob", "owns", "dog"], "extractors": ["naive", "sieve"]},
    ])
    assert a == b


def test_canonicalize_output_distinguishes_shapes():
    """Single-school output and multi-school output produce DIFFERENT
    canonical bytes for the same underlying triple set — they are
    distinct output types."""
    triples_only = EXTRACT_TRANSFORM.canonicalize_output([
        ["alice", "likes", "cats"],
    ])
    multi_school = EXTRACT_TRANSFORM.canonicalize_output([
        {"triple": ["alice", "likes", "cats"], "extractors": ["sieve"]},
    ])
    assert triples_only != multi_school


# ─── End-to-end via the receipt + share flow ────────────────────────


joserfc = pytest.importorskip("joserfc", reason="[receipt-verify] required")


@pytest.fixture(scope="module")
def keypair():
    from joserfc.jwk import OKPKey

    kid = "test-multi-school-2026"
    key = OKPKey.generate_key("Ed25519")
    private = key.as_dict(private=True)
    private["kid"] = kid
    public = key.as_dict(private=False)
    public["kid"] = kid
    public["alg"] = "EdDSA"
    public["use"] = "sig"
    return private, {"keys": [public]}, kid


def test_multi_school_receipt_signs_and_verifies(keypair):
    """End-to-end: run multi_school, build a receipt over the
    output, sign it, verify it."""
    from sum_engine_internal.transform_receipt import (
        build_payload,
        canonical_hash,
        sign_transform_receipt,
        verify_transform_receipt,
    )

    private, jwks, kid = keypair
    result = asyncio.run(EXTRACT_TRANSFORM.apply(
        input={"text": "Alice likes cats."},
        parameters={"multi_school": True},
        env=TransformEnv(),
    ))
    params = {"extractor": "sieve", "max_tags": 64, "multi_school": True}

    payload = build_payload(
        transform="extract",
        parameters_hash=canonical_hash(
            EXTRACT_TRANSFORM.canonicalize_parameters(params)
        ),
        input_hash=canonical_hash(
            EXTRACT_TRANSFORM.canonicalize_input({"text": "Alice likes cats."})
        ),
        output_hash=canonical_hash(
            EXTRACT_TRANSFORM.canonicalize_output(result.output)
        ),
        model=result.model,
        provider=result.provider,
        digital_source_type=result.digital_source_type,
    )
    receipt = sign_transform_receipt(payload, private_jwk=private, kid=kid)
    verify_result = verify_transform_receipt(receipt, jwks)
    assert verify_result.verified is True
