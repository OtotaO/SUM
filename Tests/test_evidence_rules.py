"""Inference rules + non-leaf evidence chains.

Pins:
  - EvidenceLink: leaf vs non-leaf shape (derivation_rule +
    derived_from must be both set or both empty)
  - verify_chain_well_formed: dangling-derived_from rejected,
    self-derivation rejected, half-populated link rejected
  - TransitiveClosureRule: closure on substrate's TRANSITIVE
    predicates (currently `contain`); skips non-transitive
    predicates; idempotent at fixpoint
  - derive_non_leaf_links: returns ONLY new claims, properly
    tags derivation_rule + derived_from, provenance points back
    to a leaf source
  - compose_bundle_with_evidence(rules=[...]): bundle's chain
    contains both leaf and non-leaf links; coverage still holds;
    well-formedness still holds; signature unaffected
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sum_engine_internal.evidence import (
    EvidenceLink, EvidenceChainError, TransitiveClosureRule,
    compose_bundle_with_evidence, derive_non_leaf_links,
    verify_chain_well_formed,
)
from sum_engine_internal.infrastructure.provenance import ProvenanceRecord


def _prov(byte_start=0, byte_end=10, excerpt="x"):
    return ProvenanceRecord(
        source_uri="sha256:" + "ab" * 32,
        byte_start=byte_start,
        byte_end=byte_end,
        extractor_id="sum.test:v1",
        timestamp=datetime.now(timezone.utc).isoformat(),
        text_excerpt=excerpt,
    )


def _leaf(claim, **kw):
    return EvidenceLink(claim=claim, provenance=_prov(**kw))


def _derived(claim, supports, rule_id="transitive_closure"):
    return EvidenceLink(
        claim=claim,
        provenance=_prov(),
        derived_from=tuple(supports),
        derivation_rule=rule_id,
    )


# -- EvidenceLink: leaf vs non-leaf -----------------------------------


def test_leaf_link_is_leaf_property_true():
    link = _leaf("a||contain||b")
    assert link.is_leaf is True
    assert link.derived_from == ()
    assert link.derivation_rule is None


def test_derived_link_is_leaf_property_false():
    link = _derived("a||contain||c", ["a||contain||b", "b||contain||c"])
    assert link.is_leaf is False
    assert link.derivation_rule == "transitive_closure"
    assert link.derived_from == ("a||contain||b", "b||contain||c")


def test_derived_link_to_dict_includes_extras():
    link = _derived("a||contain||c", ["a||contain||b", "b||contain||c"])
    d = link.to_dict()
    assert d["derivation_rule"] == "transitive_closure"
    assert d["derived_from"] == ["a||contain||b", "b||contain||c"]


def test_leaf_link_to_dict_omits_extras():
    """derived_from + derivation_rule absent from the wire dict
    when the link is a leaf — keeps the on-wire payload minimal
    for the common case."""
    link = _leaf("a||contain||b")
    d = link.to_dict()
    assert "derived_from" not in d
    assert "derivation_rule" not in d


# -- verify_chain_well_formed: non-leaf invariants --------------------


def test_well_formed_accepts_valid_non_leaf_chain():
    leaf_a = _leaf("a||contain||b", byte_start=0, byte_end=10)
    leaf_b = _leaf("b||contain||c", byte_start=10, byte_end=20)
    derived = _derived("a||contain||c", [leaf_a.claim, leaf_b.claim])
    verify_chain_well_formed([leaf_a, leaf_b, derived])  # no raise


def test_well_formed_rejects_dangling_derived_from():
    leaf = _leaf("a||contain||b", byte_start=0, byte_end=10)
    derived = _derived("a||contain||c", [leaf.claim, "ghost||contain||c"])
    with pytest.raises(EvidenceChainError, match="does not appear"):
        verify_chain_well_formed([leaf, derived])


def test_well_formed_rejects_self_derivation():
    leaf = _leaf("a||contain||b")
    bad = _derived("a||contain||c", ["a||contain||c"])  # self-ref
    with pytest.raises(EvidenceChainError, match="self-derivation"):
        verify_chain_well_formed([leaf, bad])


def test_well_formed_rejects_rule_without_supports():
    """derivation_rule set but derived_from empty is malformed —
    a rule must have antecedents."""
    bad = EvidenceLink(
        claim="a||contain||b",
        provenance=_prov(),
        derivation_rule="some_rule",
        # derived_from defaults to ()
    )
    with pytest.raises(EvidenceChainError, match="both set or both empty"):
        verify_chain_well_formed([bad])


def test_well_formed_rejects_supports_without_rule():
    """derived_from populated but derivation_rule None is malformed
    — supports without a rule is meaningless."""
    leaf = _leaf("a||contain||b", byte_start=0, byte_end=10)
    bad = EvidenceLink(
        claim="x||contain||y",
        provenance=_prov(byte_start=20, byte_end=30),
        derived_from=(leaf.claim,),
        derivation_rule=None,
    )
    with pytest.raises(EvidenceChainError, match="both set or both empty"):
        verify_chain_well_formed([leaf, bad])


# -- TransitiveClosureRule --------------------------------------------


def test_transitive_closure_chains_on_contain():
    rule = TransitiveClosureRule(
        transitive_predicates=frozenset({"contain"}),
    )
    claims = frozenset({"a||contain||b", "b||contain||c"})
    derived = list(rule.derive_from(claims))
    assert len(derived) == 1
    derived_claim, supports = derived[0]
    assert derived_claim == "a||contain||c"
    assert set(supports) == {"a||contain||b", "b||contain||c"}


def test_transitive_closure_skips_non_transitive_predicates():
    """Predicates NOT declared TRANSITIVE must not be closed
    over — `like` is conversational, not transitive in general."""
    rule = TransitiveClosureRule(
        transitive_predicates=frozenset({"contain"}),
    )
    claims = frozenset({"a||like||b", "b||like||c"})
    derived = list(rule.derive_from(claims))
    assert derived == []


def test_transitive_closure_idempotent_at_fixpoint():
    """Once the closure includes (a, p, c), running the rule
    again should not produce anything new."""
    rule = TransitiveClosureRule(
        transitive_predicates=frozenset({"contain"}),
    )
    closed = frozenset({
        "a||contain||b", "b||contain||c", "a||contain||c",
    })
    derived = list(rule.derive_from(closed))
    assert derived == []


def test_transitive_closure_skips_self_loops():
    """A → B → A would derive (A, p, A) which is a cycle; the
    transitive predicates in the substrate library are also
    irreflexive so this would be inconsistent. The rule
    explicitly skips it."""
    rule = TransitiveClosureRule(
        transitive_predicates=frozenset({"contain"}),
    )
    claims = frozenset({"a||contain||b", "b||contain||a"})
    derived = list(rule.derive_from(claims))
    derived_claims = {d for d, _ in derived}
    assert "a||contain||a" not in derived_claims
    assert "b||contain||b" not in derived_claims


def test_transitive_closure_from_substrate_library_picks_up_contain():
    """Sanity check the auto-build path picks up the substrate's
    curated transitive set (currently just `contain`)."""
    rule = TransitiveClosureRule.from_substrate_library()
    assert "contain" in rule.transitive_predicates


# -- derive_non_leaf_links --------------------------------------------


def test_derive_non_leaf_returns_only_new_claims():
    leaf_a = _leaf("a||contain||b", byte_start=0, byte_end=10)
    leaf_b = _leaf("b||contain||c", byte_start=10, byte_end=20)
    rule = TransitiveClosureRule(transitive_predicates=frozenset({"contain"}))
    new = derive_non_leaf_links([leaf_a, leaf_b], [rule])
    new_claims = {link.claim for link in new}
    assert new_claims == {"a||contain||c"}
    derived = next(iter(new))
    assert derived.is_leaf is False
    assert derived.derivation_rule == "transitive_closure"
    assert set(derived.derived_from) == {leaf_a.claim, leaf_b.claim}


def test_derive_non_leaf_full_fixpoint_three_link_chain():
    """A→B→C→D should derive A→C, B→D, A→D in a single
    derive_non_leaf_links call (driven to fixpoint)."""
    leaves = [
        _leaf("a||contain||b", byte_start=0, byte_end=10),
        _leaf("b||contain||c", byte_start=10, byte_end=20),
        _leaf("c||contain||d", byte_start=20, byte_end=30),
    ]
    rule = TransitiveClosureRule(transitive_predicates=frozenset({"contain"}))
    new = derive_non_leaf_links(leaves, [rule])
    new_claims = {link.claim for link in new}
    assert new_claims == {
        "a||contain||c", "b||contain||d", "a||contain||d",
    }


def test_derive_non_leaf_no_rules_returns_empty():
    leaf = _leaf("a||contain||b")
    assert derive_non_leaf_links([leaf], []) == []


def test_derive_non_leaf_provenance_anchors_to_leaf_source():
    """Derived link's provenance source_uri must come from a
    LEAF link's source — chasing through derived ancestors would
    obscure the actual source text."""
    leaves = [
        _leaf("a||contain||b", byte_start=0, byte_end=10),
        _leaf("b||contain||c", byte_start=10, byte_end=20),
        _leaf("c||contain||d", byte_start=20, byte_end=30),
    ]
    rule = TransitiveClosureRule(transitive_predicates=frozenset({"contain"}))
    new = derive_non_leaf_links(leaves, [rule])
    leaf_uris = {link.provenance.source_uri for link in leaves}
    for link in new:
        assert link.provenance.source_uri in leaf_uris
        assert "rule=" in link.provenance.extractor_id


# -- compose_bundle_with_evidence(rules=[...]) integration ------------


@pytest.fixture
def codec():
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec
    algebra = GodelStateAlgebra()
    gen = AutoregressiveTomeGenerator(algebra)
    return CanonicalCodec(algebra, gen, signing_key="rules_test_key")


_TEXT_CONTAINMENT = (
    "The library contains the wing. "
    "The wing contains the room. "
    "The room contains the shelf."
)


def test_composed_bundle_with_no_rules_is_leaf_only(codec):
    """Backward compat — rules=None yields the v0 leaf-only chain."""
    bundle = compose_bundle_with_evidence(codec, _TEXT_CONTAINMENT, branch="t")
    chain = bundle["axiom_evidence_chain"]
    for link in chain:
        assert "derivation_rule" not in link
        assert "derived_from" not in link


def test_composed_bundle_with_transitive_closure_adds_non_leaf_links(codec):
    rule = TransitiveClosureRule.from_substrate_library()
    bundle = compose_bundle_with_evidence(
        codec, _TEXT_CONTAINMENT, branch="t", rules=[rule],
    )
    chain = bundle["axiom_evidence_chain"]
    leaf_count = sum(1 for link in chain if "derivation_rule" not in link)
    derived_count = sum(1 for link in chain if "derivation_rule" in link)
    # 3 leaf claims + 3 derived (a→c, b→d, a→d)
    assert leaf_count >= 3
    assert derived_count >= 1, (
        f"expected ≥1 derived links from transitive containment; "
        f"got chain={chain}"
    )


def test_composed_bundle_with_rules_round_trips(codec):
    """Adding non-leaf links must not break import (chain is
    OUTSIDE signed payload)."""
    rule = TransitiveClosureRule.from_substrate_library()
    bundle = compose_bundle_with_evidence(
        codec, _TEXT_CONTAINMENT, branch="t", rules=[rule],
    )
    state_int = int(bundle["state_integer"])
    assert codec.import_bundle(bundle) == state_int


def test_composed_bundle_with_rules_signature_unaffected(codec):
    rule = TransitiveClosureRule.from_substrate_library()
    bundle = compose_bundle_with_evidence(
        codec, _TEXT_CONTAINMENT, branch="t", rules=[rule],
    )
    sig_recompute = codec._sign(
        bundle["canonical_tome"],
        bundle["state_integer"],
        bundle["timestamp"],
    )
    assert sig_recompute == bundle["signature"]
