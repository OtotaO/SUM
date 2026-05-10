"""Bundle metadata: axiom_graph_entropy field.

Pins the substrate-integration contract for the von Neumann
graph entropy kernel (PR #184). The field:

  - Is computed automatically at every `export_bundle` call
  - Matches the result of `graph_entropy(triples)` on the same
    underlying axiom set
  - Is OUTSIDE the signed payload — its presence does not
    change `signature` / `public_signature`
  - Is None for empty bundles (and stripped from the JSON
    output by the existing None-stripping logic)
  - Failure to compute entropy MUST NOT block attestation
"""
from __future__ import annotations

import pytest


@pytest.fixture
def codec_and_algebra():
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec
    algebra = GodelStateAlgebra()
    gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(algebra, gen, signing_key="contract_test_key")
    return codec, algebra


def _state_from(algebra, triples):
    return algebra.encode_chunk_state(triples)


def test_bundle_includes_axiom_graph_entropy_field(codec_and_algebra):
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("alice", "knows", "bob"),
        ("bob", "owns", "rex"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    assert "axiom_graph_entropy" in bundle
    assert isinstance(bundle["axiom_graph_entropy"], float)


def test_axiom_graph_entropy_matches_direct_computation(codec_and_algebra):
    """The bundle's entropy field must equal what
    `graph_entropy()` returns when called directly on the same
    triples — guarantees the wiring doesn't transform the data."""
    from sum_engine_internal.graph_store import Triple
    from sum_engine_internal.research.spectral_entropy import graph_entropy

    codec, algebra = codec_and_algebra
    triples = [
        ("alice", "knows", "bob"),
        ("bob", "owns", "rex"),
        ("carol", "writes", "code"),
        ("dave", "drives", "tesla"),
    ]
    state = _state_from(algebra, triples)
    bundle = codec.export_bundle(state, branch="t")

    direct = graph_entropy([Triple(*t) for t in triples])
    assert abs(bundle["axiom_graph_entropy"] - direct) < 1e-9


def test_empty_bundle_omits_axiom_graph_entropy_field(codec_and_algebra):
    """state=1 means "no axioms"; entropy is None and gets
    stripped from the dict output by the existing None-stripping
    pass."""
    codec, _ = codec_and_algebra
    bundle = codec.export_bundle(1, branch="empty")
    assert bundle.get("axiom_count") == 0
    assert "axiom_graph_entropy" not in bundle


def test_signature_unchanged_by_entropy_field(codec_and_algebra):
    """The headline non-breaking property: adding the entropy
    field MUST NOT alter `signature` or `public_signature` —
    those cover only `canonical_tome|state_integer|timestamp`,
    not bundle metadata. If this test fails, the wiring has
    sneaked the entropy into the signed payload (a wire-format
    break)."""
    codec, algebra = codec_and_algebra
    triples = [("alice", "knows", "bob"), ("bob", "owns", "rex")]
    state = _state_from(algebra, triples)
    bundle = codec.export_bundle(state, branch="t")

    # Independently re-sign the same payload and compare
    sig_check = codec._sign(
        bundle["canonical_tome"],
        bundle["state_integer"],
        bundle["timestamp"],
    )
    assert sig_check == bundle["signature"]


def test_round_trip_import_succeeds_with_entropy_field(codec_and_algebra):
    """import_bundle must accept bundles with the new field —
    it should be invisible to the importer (which only reads
    canonical_tome / state_integer / timestamp / signature)."""
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("alice", "knows", "bob"),
        ("bob", "owns", "rex"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    recovered = codec.import_bundle(bundle)
    assert recovered == state


def test_entropy_is_invariant_under_axiom_set_permutation(codec_and_algebra):
    """The same triple set in different orders produces the same
    Gödel state and therefore the same entropy — pinning that
    no insertion-order bias has crept in via the wiring."""
    codec, algebra = codec_and_algebra
    triples_a = [
        ("alice", "knows", "bob"),
        ("bob", "owns", "rex"),
        ("carol", "writes", "code"),
    ]
    triples_b = list(reversed(triples_a))
    bundle_a = codec.export_bundle(_state_from(algebra, triples_a), branch="t")
    bundle_b = codec.export_bundle(_state_from(algebra, triples_b), branch="t")
    assert bundle_a["axiom_graph_entropy"] == bundle_b["axiom_graph_entropy"]


def test_entropy_field_does_not_appear_in_signature_payload(codec_and_algebra):
    """Belt-and-suspenders for the signature invariant: even if
    the codec is asked to re-sign with the entropy field
    explicitly excluded, the resulting signature must equal the
    one in the bundle."""
    codec, algebra = codec_and_algebra
    triples = [("a", "p", "b")]
    state = _state_from(algebra, triples)
    bundle = codec.export_bundle(state, branch="t")

    # The signed payload format is fixed at
    # `canonical_tome|state_integer|timestamp` (per
    # canonical_codec.py:141). Independently re-sign and verify.
    sig_recompute = codec._sign(
        bundle["canonical_tome"],
        bundle["state_integer"],
        bundle["timestamp"],
    )
    assert sig_recompute == bundle["signature"]


def test_failed_entropy_computation_does_not_block_attestation(monkeypatch):
    """If the entropy helper raises, attestation MUST still
    succeed — the entropy is a non-load-bearing metadata
    signal."""
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec
    from sum_engine_internal.infrastructure import canonical_codec as cc

    def _broken(*a, **kw):
        raise RuntimeError("forced failure for test")
    monkeypatch.setattr(cc, "_compute_axiom_graph_entropy", _broken)

    algebra = GodelStateAlgebra()
    gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(algebra, gen, signing_key="k")
    state = algebra.encode_chunk_state([("a", "p", "b")])
    # Should NOT raise — the bundle still attests
    bundle = codec.export_bundle(state, branch="t")
    assert bundle["axiom_count"] == 1
    # Round-trip still works
    assert codec.import_bundle(bundle) == state
