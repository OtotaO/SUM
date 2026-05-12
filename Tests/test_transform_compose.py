"""Compose transform contract tests.

T3 coverage:
  1. Registry has `compose` registered.
  2. Parameter validation.
  3. Multi-bundle union: dedup across bundles; sort + normalise.
  4. State integer recomputed from merged set; matches what
     encode_chunk_state would produce.
  5. Empty compose is the identity (no inputs → state=1, empty axioms).
  6. Idempotence: composing the same bundle twice gives the same output.
  7. Commutativity: (A, B) merged equals (B, A) merged (same OUTPUT,
     may differ on INPUT order in the hash — separately tested).
  8. Canonicalisation byte-stability.
  9. intersect / diff strategies deferred → NotImplementedError.
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
from sum_engine_internal.transforms.compose import (
    COMPOSE_TRANSFORM,
    _bundle_triples,
    _diff_triples,
    _intersect_triples,
    _normalize_triple,
    _union_triples,
    _validate_parameters,
)


# ─── Registry shape ─────────────────────────────────────────────────


def test_compose_is_registered():
    assert "compose" in list_transforms()
    assert get_transform("compose") is COMPOSE_TRANSFORM


def test_compose_metadata():
    assert COMPOSE_TRANSFORM.name == "compose"
    assert COMPOSE_TRANSFORM.requires_llm is False
    assert COMPOSE_TRANSFORM.digital_source_type == "algorithmicMedia"


# ─── Parameter validation ───────────────────────────────────────────


def test_parameter_validation_default_is_lcm():
    assert _validate_parameters({}) == {"merge_strategy": "lcm"}


def test_parameter_validation_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="merge_strategy"):
        _validate_parameters({"merge_strategy": "xor"})


# ─── Bundle triple extraction ───────────────────────────────────────


def test_bundle_triples_accepts_triples_key():
    triples = _bundle_triples({"triples": [("a", "b", "c")]})
    assert triples == [("a", "b", "c")]


def test_bundle_triples_accepts_axioms_dicts():
    """CanonicalBundle uses 'axioms' with subject/predicate/object dict
    shape."""
    triples = _bundle_triples({"axioms": [
        {"subject": "alice", "predicate": "likes", "object": "cats"},
    ]})
    assert triples == [("alice", "likes", "cats")]


def test_bundle_triples_accepts_axioms_tuples():
    triples = _bundle_triples({"axioms": [("alice", "likes", "cats")]})
    assert triples == [("alice", "likes", "cats")]


def test_bundle_triples_rejects_missing_keys():
    with pytest.raises(ValueError, match="triples' or 'axioms'"):
        _bundle_triples({"random": "stuff"})


# ─── Union / intersect / diff helpers ───────────────────────────────


def test_union_deduplicates_across_bundles():
    """A triple present in two bundles appears once in the union."""
    out = _union_triples([
        {"triples": [("alice", "likes", "cats")]},
        {"triples": [("alice", "likes", "cats"), ("bob", "owns", "dog")]},
    ])
    assert out == [("alice", "likes", "cats"), ("bob", "owns", "dog")]


def test_union_normalises_casing():
    """Same fact different casing → single triple."""
    out = _union_triples([
        {"triples": [("Alice", "Likes", "Cats")]},
        {"triples": [("ALICE", "LIKES", "CATS")]},
    ])
    assert out == [("alice", "likes", "cats")]


def test_intersect_returns_only_common_triples():
    out = _intersect_triples([
        {"triples": [("a", "r", "b"), ("c", "r", "d")]},
        {"triples": [("a", "r", "b"), ("e", "r", "f")]},
    ])
    assert out == [("a", "r", "b")]


def test_diff_returns_first_bundles_unique():
    out = _diff_triples([
        {"triples": [("a", "r", "b"), ("c", "r", "d")]},
        {"triples": [("c", "r", "d"), ("e", "r", "f")]},
    ])
    assert out == [("a", "r", "b")]


# ─── Apply paths ────────────────────────────────────────────────────


def test_compose_empty_is_identity():
    """Composing zero bundles → state=1, empty axioms."""
    result = asyncio.run(COMPOSE_TRANSFORM.apply(
        input={"bundles": []},
        parameters={},
        env=TransformEnv(),
    ))
    assert result.output == {"axioms": [], "state_integer": 1}
    assert result.extra["bundle_count"] == 0


def test_compose_single_bundle_is_normalisation():
    """One bundle → output = its triples (normalised + sorted)."""
    result = asyncio.run(COMPOSE_TRANSFORM.apply(
        input={"bundles": [{"triples": [
            ("Bob", "OWNS", "Dog"),
            ("alice", "likes", "cats"),
        ]}]},
        parameters={},
        env=TransformEnv(),
    ))
    assert result.output["axioms"] == [
        ["alice", "likes", "cats"],
        ["bob", "owns", "dog"],
    ]
    assert result.output["state_integer"] > 1


def test_compose_multi_bundle_unions_and_recomputes_state():
    """Two bundles with overlap → union, state recomputed from union."""
    result = asyncio.run(COMPOSE_TRANSFORM.apply(
        input={"bundles": [
            {"triples": [("alice", "likes", "cats"), ("bob", "owns", "dog")]},
            {"triples": [("alice", "likes", "cats"), ("carol", "reads", "books")]},
        ]},
        parameters={},
        env=TransformEnv(),
    ))
    assert isinstance(result, TransformResult)
    assert result.provider == "canonical-path"
    # Three unique triples, lex-sorted, lowercased.
    assert result.output["axioms"] == [
        ["alice", "likes", "cats"],
        ["bob", "owns", "dog"],
        ["carol", "reads", "books"],
    ]
    # State integer is LCM of the three primes — non-trivial integer.
    assert result.output["state_integer"] > 10
    assert result.extra == {
        "merge_strategy": "lcm",
        "bundle_count": 2,
        "axiom_count": 3,
        "input_axiom_counts": [2, 2],
    }


def test_compose_idempotent_on_duplicate_bundles():
    """Composing the same bundle twice equals composing it once.
    LCM property: lcm(p, p) == p."""
    bundle = {"triples": [("alice", "likes", "cats"), ("bob", "owns", "dog")]}

    once = asyncio.run(COMPOSE_TRANSFORM.apply(
        input={"bundles": [bundle]},
        parameters={}, env=TransformEnv(),
    ))
    twice = asyncio.run(COMPOSE_TRANSFORM.apply(
        input={"bundles": [bundle, bundle]},
        parameters={}, env=TransformEnv(),
    ))
    assert once.output["axioms"] == twice.output["axioms"]
    assert once.output["state_integer"] == twice.output["state_integer"]


def test_compose_output_state_matches_independent_recompute():
    """The state integer in the merged output equals what you'd get
    if you re-ran encode_chunk_state on the merged triple list
    independently. This is the load-bearing invariant: receipts'
    output_hash binds to a bundle whose state is consistent with
    its axioms."""
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra

    result = asyncio.run(COMPOSE_TRANSFORM.apply(
        input={"bundles": [
            {"triples": [("alice", "likes", "cats")]},
            {"triples": [("bob", "owns", "dog")]},
        ]},
        parameters={}, env=TransformEnv(),
    ))
    merged = [tuple(t) for t in result.output["axioms"]]
    algebra = GodelStateAlgebra()
    independent_state = algebra.encode_chunk_state(merged)
    assert result.output["state_integer"] == independent_state


# ─── Canonicalisation byte-stability ────────────────────────────────


def test_canonicalize_input_sorts_within_bundles_preserves_outer_order():
    """Inner ordering of a bundle's triples is invisible; outer
    ordering of bundles IS significant. This matches user intent:
    'compose A then B' is a different operation than 'compose B
    then A' even though the merged output is the same."""
    a_then_b = COMPOSE_TRANSFORM.canonicalize_input({"bundles": [
        {"triples": [("alice", "likes", "cats"), ("bob", "owns", "dog")]},
        {"triples": [("carol", "reads", "books")]},
    ]})
    b_then_a = COMPOSE_TRANSFORM.canonicalize_input({"bundles": [
        {"triples": [("carol", "reads", "books")]},
        {"triples": [("alice", "likes", "cats"), ("bob", "owns", "dog")]},
    ]})
    # Outer reordering ⇒ different input_hash.
    assert a_then_b != b_then_a

    # Inner reordering ⇒ identical input_hash.
    a_then_b_alt = COMPOSE_TRANSFORM.canonicalize_input({"bundles": [
        {"triples": [("bob", "owns", "dog"), ("alice", "likes", "cats")]},  # inner swap
        {"triples": [("carol", "reads", "books")]},
    ]})
    assert a_then_b == a_then_b_alt


def test_canonicalize_output_is_byte_stable_under_dict_reordering():
    """Canonical output bytes are invariant to dict key insertion
    order on the OUTPUT dict."""
    a = COMPOSE_TRANSFORM.canonicalize_output({
        "axioms": [["alice", "likes", "cats"]],
        "state_integer": 42,
    })
    b = COMPOSE_TRANSFORM.canonicalize_output({
        "state_integer": 42,
        "axioms": [["alice", "likes", "cats"]],
    })
    assert a == b


def test_canonicalize_output_rejects_bad_shape():
    with pytest.raises(ValueError, match="dict"):
        COMPOSE_TRANSFORM.canonicalize_output("not a dict")
    with pytest.raises(ValueError, match="'axioms' and 'state_integer'"):
        COMPOSE_TRANSFORM.canonicalize_output({"axioms": []})


# ─── Deferred strategies ────────────────────────────────────────────


def test_intersect_strategy_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="intersect"):
        asyncio.run(COMPOSE_TRANSFORM.apply(
            input={"bundles": [{"triples": [("a", "r", "b")]}]},
            parameters={"merge_strategy": "intersect"},
            env=TransformEnv(),
        ))


def test_diff_strategy_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="diff"):
        asyncio.run(COMPOSE_TRANSFORM.apply(
            input={"bundles": [{"triples": [("a", "r", "b")]}]},
            parameters={"merge_strategy": "diff"},
            env=TransformEnv(),
        ))


# ─── Input shape validation ─────────────────────────────────────────


def test_apply_rejects_missing_bundles_key():
    with pytest.raises(ValueError, match="'bundles'"):
        asyncio.run(COMPOSE_TRANSFORM.apply(
            input={"random": "stuff"},
            parameters={}, env=TransformEnv(),
        ))


def test_apply_rejects_bundles_not_a_list():
    with pytest.raises(ValueError, match="must be a list"):
        asyncio.run(COMPOSE_TRANSFORM.apply(
            input={"bundles": "not a list"},
            parameters={}, env=TransformEnv(),
        ))
