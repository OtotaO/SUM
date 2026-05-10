"""Bundle metadata: axiom_consistency_check field.

Pins the substrate-integration contract for wire #3 — Z3 SMT
consistency check shipped as bundle metadata. The field:

  - Is a dict ``{consistent, unsat_core, n_predicates_checked,
    z3_check_ms}`` for non-empty bundles
  - Is computed automatically at every `export_bundle` call
  - Uses the operator-curated `SUBSTRATE_PREDICATE_LIBRARY`
  - Is OUTSIDE the signed payload — its presence does not
    change `signature` / `public_signature`
  - Is None for empty bundles (and stripped from the JSON
    output by the existing None-stripping logic)
  - Failure to compute (e.g. Z3 missing) MUST NOT block
    attestation
  - **Additive shape**: UNSAT bundles still round-trip via
    `import_bundle` — downstream consumers (not the codec)
    decide whether to trust an UNSAT bundle
"""
from __future__ import annotations

import pytest

# Skip if z3-solver isn't installed — the field will be None
# in that environment and most assertions become vacuous.
z3 = pytest.importorskip("z3")


@pytest.fixture
def codec_and_algebra():
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec
    algebra = GodelStateAlgebra()
    gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(algebra, gen, signing_key="contract_key")
    return codec, algebra


def _state_from(algebra, triples):
    return algebra.encode_chunk_state(triples)


def test_bundle_includes_axiom_consistency_check_field(codec_and_algebra):
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("alice", "build", "house"),
        ("bob", "write", "book"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    assert "axiom_consistency_check" in bundle
    check = bundle["axiom_consistency_check"]
    assert isinstance(check, dict)
    assert "consistent" in check
    assert "unsat_core" in check
    assert "n_predicates_checked" in check
    assert "z3_check_ms" in check


def test_clean_axioms_yield_consistent_true(codec_and_algebra):
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("alice", "build", "house"),
        ("bob", "write", "book"),
        ("carol", "discover", "fact"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    check = bundle["axiom_consistency_check"]
    assert check["consistent"] is True
    assert check["unsat_core"] == []
    assert check["n_predicates_checked"] >= 1


def test_mutual_contain_yields_consistent_false_with_unsat_core(codec_and_algebra):
    """`contain` is in the curated library as antisymmetric +
    irreflexive + transitive. Mutual contain is a textbook UNSAT."""
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("alice", "build", "house"),
        ("box", "contain", "ball"),
        ("ball", "contain", "box"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    check = bundle["axiom_consistency_check"]
    assert check["consistent"] is False
    # UNSAT core should reference the two contradicting axioms; the
    # exact indices depend on Gödel-extraction order, so check the
    # core is non-empty and small (≤ 3 axioms is a minimal core for
    # this contradiction)
    assert len(check["unsat_core"]) >= 2
    assert len(check["unsat_core"]) <= 3


def test_unsat_bundle_still_round_trips(codec_and_algebra):
    """Wire #3 is ADDITIVE: UNSAT bundles emit `consistent: False`
    in metadata but `import_bundle` still verifies them. Downstream
    consumers (not the codec) decide what to do with UNSAT
    bundles. If this test fails, the wiring has become
    aggressive — re-check whether that was intentional."""
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("box", "contain", "ball"),
        ("ball", "contain", "box"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    assert bundle["axiom_consistency_check"]["consistent"] is False
    # But round-trip still succeeds:
    recovered = codec.import_bundle(bundle)
    assert recovered == state


def test_empty_bundle_omits_consistency_field(codec_and_algebra):
    codec, _ = codec_and_algebra
    bundle = codec.export_bundle(1, branch="empty")
    assert bundle.get("axiom_count") == 0
    assert "axiom_consistency_check" not in bundle


def test_signature_unchanged_by_consistency_field(codec_and_algebra):
    """The CI field MUST NOT alter `signature` — it's outside the
    signed payload (`canonical_tome | state_integer | timestamp`).
    If this fails, the wiring has sneaked into the signed payload."""
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [("alice", "build", "house")])
    bundle = codec.export_bundle(state, branch="t")
    sig_check = codec._sign(
        bundle["canonical_tome"],
        bundle["state_integer"],
        bundle["timestamp"],
    )
    assert sig_check == bundle["signature"]


def test_failed_consistency_check_does_not_block_attestation(monkeypatch):
    """If the Z3 helper raises, attestation MUST still succeed —
    the consistency check is non-load-bearing metadata.
    Defense-in-depth at the call site catches even a broken
    helper."""
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec
    from sum_engine_internal.infrastructure import canonical_codec as cc

    def _broken(*a, **kw):
        raise RuntimeError("forced failure for test")
    monkeypatch.setattr(cc, "_compute_axiom_consistency_check", _broken)

    algebra = GodelStateAlgebra()
    gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(algebra, gen, signing_key="k")
    state = algebra.encode_chunk_state([("a", "build", "b")])
    # Should NOT raise — bundle still attests, consistency_check is None
    bundle = codec.export_bundle(state, branch="t")
    assert bundle["axiom_count"] == 1
    assert "axiom_consistency_check" not in bundle  # None → stripped
    # Round-trip still works
    assert codec.import_bundle(bundle) == state


def test_predicates_outside_curated_library_dont_break_check(codec_and_algebra):
    """If all of a bundle's predicates are uncurated, Z3 still
    runs but trivially returns SAT (no constraints to violate).
    The field should be present + report n_predicates_checked=0."""
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        # 'be' and 'know' are intentionally NOT in the curated library
        ("alice", "be", "happy"),
        ("bob", "know", "carol"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    check = bundle["axiom_consistency_check"]
    assert check["consistent"] is True
    assert check["n_predicates_checked"] == 0


def test_consistency_field_independent_of_entropy_field(codec_and_algebra):
    """Wires #1, #2, #3 produce three independent metadata fields.
    Pin that they don't interact — the consistency check shouldn't
    affect the entropy / entropy_ci fields."""
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("box", "contain", "ball"),
        ("ball", "contain", "box"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    # All three fields present
    assert "axiom_graph_entropy" in bundle
    assert "axiom_graph_entropy_ci" in bundle
    assert "axiom_consistency_check" in bundle
    # Consistency=False but entropy fields populated
    assert bundle["axiom_consistency_check"]["consistent"] is False
    assert isinstance(bundle["axiom_graph_entropy"], float)
    assert isinstance(bundle["axiom_graph_entropy_ci"], list)
