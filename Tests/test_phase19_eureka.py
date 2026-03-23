"""
Tests — Phase 19: The Sovereign Edge & Automated Scientist

Verifies:
  1. GodelStateAlgebra.get_active_axioms() factorisation
  2. CausalDiscoveryEngine transitive closure inference
  3. Inverse predicate inference (inhibits → treats)
  4. No duplicate discoveries (already-entailed facts filtered)

Author: ototao
License: Apache License 2.0
"""

import math
import pytest
from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.algorithms.causal_discovery import CausalDiscoveryEngine


# ─── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def algebra():
    return GodelStateAlgebra()


@pytest.fixture
def engine(algebra):
    return CausalDiscoveryEngine(algebra)


# ─── State Factorisation ─────────────────────────────────────────────

def test_get_active_axioms_empty(algebra):
    """Empty state (1) has no active axioms."""
    assert algebra.get_active_axioms(1) == []


def test_get_active_axioms_single(algebra):
    """Single axiom is correctly extracted."""
    p = algebra.get_or_mint_prime("alice", "likes", "cats")
    state = p
    active = algebra.get_active_axioms(state)
    assert len(active) == 1
    assert "alice||likes||cats" in active


def test_get_active_axioms_multiple(algebra):
    """Multiple axioms are factorised correctly."""
    p1 = algebra.get_or_mint_prime("alice", "likes", "cats")
    p2 = algebra.get_or_mint_prime("bob", "likes", "dogs")
    p3 = algebra.get_or_mint_prime("carol", "knows", "alice")
    state = math.lcm(p1, math.lcm(p2, p3))

    active = algebra.get_active_axioms(state)
    assert len(active) == 3
    assert "alice||likes||cats" in active
    assert "bob||likes||dogs" in active
    assert "carol||knows||alice" in active


# ─── Causal Discovery ────────────────────────────────────────────────

def test_causal_transitive_closure(algebra, engine):
    """A causes B, B causes C  ⟹  A causes C."""
    p1 = algebra.get_or_mint_prime("smoking", "causes", "cancer")
    p2 = algebra.get_or_mint_prime("cancer", "causes", "death")
    state = math.lcm(p1, p2)

    discoveries = engine.sweep_for_discoveries(state)
    triplets = [(s, p, o) for s, p, o in discoveries]
    assert ("smoking", "causes", "death") in triplets


def test_inverse_inhibits_to_treats(algebra, engine):
    """A inhibits B, B causes C  ⟹  A treats C."""
    p1 = algebra.get_or_mint_prime("chemical_x", "inhibits", "enzyme_y")
    p2 = algebra.get_or_mint_prime("enzyme_y", "causes", "disease_z")
    state = math.lcm(p1, p2)

    discoveries = engine.sweep_for_discoveries(state)
    triplets = [(s, p, o) for s, p, o in discoveries]
    assert ("chemical_x", "treats", "disease_z") in triplets


def test_multi_hop_chain(algebra, engine):
    """Three-hop chain: A→B→C→D yields A→C, B→D, A→D (via 2-hop)."""
    p1 = algebra.get_or_mint_prime("chemical_x", "inhibits", "enzyme_y")
    p2 = algebra.get_or_mint_prime("enzyme_y", "causes", "disease_z")
    p3 = algebra.get_or_mint_prime("disease_z", "leads_to", "symptom_w")
    state = math.lcm(p1, math.lcm(p2, p3))

    discoveries = engine.sweep_for_discoveries(state)
    triplets = set((s, p, o) for s, p, o in discoveries)

    # Direct 2-hop inferences
    assert ("chemical_x", "treats", "disease_z") in triplets
    assert ("enzyme_y", "causes", "symptom_w") in triplets


def test_no_duplicate_discoveries(algebra, engine):
    """Already-entailed facts are not rediscovered."""
    p1 = algebra.get_or_mint_prime("a", "causes", "b")
    p2 = algebra.get_or_mint_prime("b", "causes", "c")
    # Pre-mint the transitive closure
    p3 = algebra.get_or_mint_prime("a", "causes", "c")
    state = math.lcm(p1, math.lcm(p2, p3))

    discoveries = engine.sweep_for_discoveries(state)
    # "a causes c" should NOT appear — it's already in the state
    novel_keys = [f"{s}||{p}||{o}" for s, p, o in discoveries]
    assert "a||causes||c" not in novel_keys


def test_no_self_loops(algebra, engine):
    """A→B→A should not infer A→A."""
    p1 = algebra.get_or_mint_prime("x", "causes", "y")
    p2 = algebra.get_or_mint_prime("y", "causes", "x")
    state = math.lcm(p1, p2)

    discoveries = engine.sweep_for_discoveries(state)
    for s, p, o in discoveries:
        assert s != o, f"Self-loop discovered: {s}||{p}||{o}"


def test_non_transitive_predicates_ignored(algebra, engine):
    """Non-transitive predicates (e.g. 'likes') produce no inferences."""
    p1 = algebra.get_or_mint_prime("alice", "likes", "bob")
    p2 = algebra.get_or_mint_prime("bob", "likes", "carol")
    state = math.lcm(p1, p2)

    discoveries = engine.sweep_for_discoveries(state)
    assert len(discoveries) == 0


# ─── Automated Scientist Integration ─────────────────────────────────

def test_automated_scientist_import():
    """Ensure the daemon can be imported without side effects."""
    from internal.ensemble.automated_scientist import AutomatedScientistDaemon
    assert AutomatedScientistDaemon is not None
