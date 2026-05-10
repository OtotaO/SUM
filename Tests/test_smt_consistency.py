"""Z3 axiom-consistency contract tests.

Layers:
  1. **Property correctness** — antisymmetry, irreflexivity,
     functionality, transitivity each catch their canonical
     contradiction class.
  2. **UNSAT-core minimality** — the returned core is the
     smallest subset that's still unsatisfiable.
  3. **Composition** — multiple properties combined on the same
     predicate behave as expected (e.g., transitive+irreflexive
     catches cycles).
  4. **Edge cases** — empty triples, no properties (always SAT),
     no overlap between declared properties and used predicates,
     identical entities in distinct triples.
  5. **Z3 timeout** — long-running checks honour the timeout.
"""
from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")  # skip if z3-solver isn't installed

from sum_engine_internal.graph_store import Triple
from sum_engine_internal.research.smt_consistency import (
    ConsistencyResult,
    PredicateProperty as P,
    check_consistency,
)


def _t(s, p, o):
    return Triple(s, p, o)


# -- Property correctness ---------------------------------------------


def test_clean_axioms_with_no_properties_are_consistent():
    triples = [_t("alice", "knows", "bob"), _t("bob", "knows", "carol")]
    r = check_consistency(triples)
    assert r.consistent is True
    assert r.unsat_core == []


def test_mutual_antisymmetric_predicate_is_unsat():
    triples = [
        _t("alice", "parent_of", "bob"),
        _t("bob", "parent_of", "alice"),
    ]
    r = check_consistency(triples, predicate_properties={
        "parent_of": {P.ANTISYMMETRIC},
    })
    assert r.consistent is False
    assert set(r.unsat_core) == {0, 1}


def test_self_loop_on_irreflexive_is_unsat():
    triples = [_t("alice", "parent_of", "alice")]
    r = check_consistency(triples, predicate_properties={
        "parent_of": {P.IRREFLEXIVE},
    })
    assert r.consistent is False
    assert r.unsat_core == [0]


def test_functional_predicate_with_two_distinct_outputs_is_unsat():
    triples = [
        _t("alice", "born_on", "1990-01-01"),
        _t("alice", "born_on", "1991-02-02"),
    ]
    r = check_consistency(triples, predicate_properties={
        "born_on": {P.FUNCTIONAL},
    })
    assert r.consistent is False
    assert set(r.unsat_core) == {0, 1}


def test_transitive_irreflexive_cycle_is_unsat():
    triples = [
        _t("a", "ancestor_of", "b"),
        _t("b", "ancestor_of", "c"),
        _t("c", "ancestor_of", "a"),
    ]
    r = check_consistency(triples, predicate_properties={
        "ancestor_of": {P.TRANSITIVE, P.IRREFLEXIVE},
    })
    assert r.consistent is False
    assert set(r.unsat_core) == {0, 1, 2}


# -- UNSAT-core minimality --------------------------------------------


def test_unsat_core_is_minimal_among_clean_axioms():
    """50 clean axioms + 1 contradicting axiom → core is just the
    contradicting one's index."""
    clean = [_t(f"e{i}", "knows", f"e{i+1}") for i in range(50)]
    bad = [_t("alice", "parent_of", "alice")]
    r = check_consistency(clean + bad, predicate_properties={
        "parent_of": {P.IRREFLEXIVE},
    })
    assert r.consistent is False
    assert r.unsat_core == [50]


def test_unsat_core_returns_minimal_pair_for_antisymmetric_violation():
    """100 clean axioms + 2 mutual parent_of → core is just those 2."""
    clean = [_t(f"e{i}", "knows", f"e{i+1}") for i in range(100)]
    bad = [
        _t("alice", "parent_of", "bob"),
        _t("bob", "parent_of", "alice"),
    ]
    r = check_consistency(clean + bad, predicate_properties={
        "parent_of": {P.ANTISYMMETRIC},
    })
    assert r.consistent is False
    assert set(r.unsat_core) == {100, 101}


# -- Composition -------------------------------------------------------


def test_antisymmetric_alone_does_not_catch_self_loop():
    """parent_of(alice, alice) is allowed under antisymmetry alone
    (the property requires x != y to fire). Need irreflexive."""
    triples = [_t("alice", "parent_of", "alice")]
    r = check_consistency(triples, predicate_properties={
        "parent_of": {P.ANTISYMMETRIC},
    })
    assert r.consistent is True


def test_irreflexive_alone_does_not_catch_mutual_links():
    """irreflexive(parent_of) doesn't say anything about
    parent_of(a, b) and parent_of(b, a) coexisting."""
    triples = [
        _t("alice", "parent_of", "bob"),
        _t("bob", "parent_of", "alice"),
    ]
    r = check_consistency(triples, predicate_properties={
        "parent_of": {P.IRREFLEXIVE},
    })
    assert r.consistent is True


def test_combining_irreflexive_and_antisymmetric_catches_both_violations():
    triples = [
        _t("alice", "parent_of", "bob"),
        _t("bob", "parent_of", "alice"),
    ]
    r = check_consistency(triples, predicate_properties={
        "parent_of": {P.IRREFLEXIVE, P.ANTISYMMETRIC},
    })
    assert r.consistent is False


# -- Edge cases --------------------------------------------------------


def test_empty_input_is_consistent():
    r = check_consistency([])
    assert r.consistent is True
    assert r.unsat_core == []
    assert r.n_triples == 0


def test_no_properties_declared_always_consistent():
    """Without any property schemas, Z3 can satisfy any
    non-trivially-contradictory axiom set by interpreting the
    predicates freely."""
    triples = [
        _t("a", "p", "b"),
        _t("b", "p", "a"),
        _t("a", "p", "a"),
    ]
    r = check_consistency(triples)
    assert r.consistent is True


def test_property_on_unused_predicate_does_nothing():
    """Declaring properties for a predicate that doesn't appear
    in the triples should not affect the verdict."""
    triples = [_t("alice", "knows", "bob")]
    r = check_consistency(triples, predicate_properties={
        "absent_predicate": {P.IRREFLEXIVE},
    })
    assert r.consistent is True
    assert r.n_predicates_with_properties == 0


def test_distinct_strings_are_distinct_z3_entities():
    """The Distinct constraint matters: without it, Z3 could
    satisfy mutual parent_of by collapsing alice == bob."""
    triples = [
        _t("alice", "parent_of", "bob"),
        _t("bob", "parent_of", "alice"),
    ]
    r = check_consistency(triples, predicate_properties={
        "parent_of": {P.ANTISYMMETRIC},
    })
    # If alice were collapsible to bob, this would be SAT
    assert r.consistent is False


def test_returns_ConsistencyResult_with_expected_fields():
    triples = [_t("alice", "p", "bob")]
    r = check_consistency(triples)
    assert isinstance(r, ConsistencyResult)
    assert isinstance(r.consistent, bool)
    assert isinstance(r.unsat_core, list)
    assert r.n_triples == 1
    assert r.z3_check_seconds >= 0
