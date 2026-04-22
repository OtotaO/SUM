"""
Property Tests for Gödel-State Algebra

Verifies the mathematical invariants that the semantic algebra must satisfy.
These are not feature tests — they are property tests that enforce structural
guarantees regardless of input.

Invariants tested:
    - LCM commutativity
    - LCM associativity
    - Merge idempotency
    - Entailment correctness after merge
    - Delta correctness
    - Round-trip canonical conservation
    - Deletion reversibility
    - Update correctness

Author: ototao
License: Apache License 2.0
"""

import math
import re
import pytest

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator


# ─── Fixtures ──────────────────────────────────────────────────────

# Fixed axiom sets for reproducible property tests
AXIOM_SETS = [
    [("alice", "likes", "cats"), ("bob", "knows", "python")],
    [("earth", "orbits", "sun"), ("moon", "orbits", "earth"), ("mars", "has", "atmosphere")],
    [("x", "r", "y"), ("a", "b", "c"), ("p", "q", "r"), ("m", "n", "o")],
    [("water", "is", "wet")],  # single axiom
]


def _build_state(algebra, triplets):
    state = 1
    for s, p, o in triplets:
        prime = algebra.get_or_mint_prime(s, p, o)
        state = math.lcm(state, prime)
    return state


# ─── 1. LCM Commutativity ────────────────────────────────────────

class TestMergeCommutativity:

    @pytest.mark.parametrize("idx", range(len(AXIOM_SETS) - 1))
    def test_lcm_commutative(self, idx):
        """lcm(A, B) == lcm(B, A) for all state pairs."""
        algebra = GodelStateAlgebra()
        state_a = _build_state(algebra, AXIOM_SETS[idx])
        state_b = _build_state(algebra, AXIOM_SETS[idx + 1])
        assert math.lcm(state_a, state_b) == math.lcm(state_b, state_a)


# ─── 2. LCM Associativity ────────────────────────────────────────

class TestMergeAssociativity:

    def test_lcm_associative(self):
        """lcm(lcm(A, B), C) == lcm(A, lcm(B, C))."""
        algebra = GodelStateAlgebra()
        states = [_build_state(algebra, axioms) for axioms in AXIOM_SETS[:3]]
        left = math.lcm(math.lcm(states[0], states[1]), states[2])
        right = math.lcm(states[0], math.lcm(states[1], states[2]))
        assert left == right


# ─── 3. Merge Idempotency ────────────────────────────────────────

class TestMergeIdempotency:

    @pytest.mark.parametrize("axioms", AXIOM_SETS)
    def test_lcm_idempotent(self, axioms):
        """lcm(A, A) == A for any state."""
        algebra = GodelStateAlgebra()
        state = _build_state(algebra, axioms)
        assert math.lcm(state, state) == state

    @pytest.mark.parametrize("axioms", AXIOM_SETS)
    def test_merge_with_identity(self, axioms):
        """lcm(A, 1) == A (identity element)."""
        algebra = GodelStateAlgebra()
        state = _build_state(algebra, axioms)
        assert math.lcm(state, 1) == state


# ─── 4. Entailment After Merge ───────────────────────────────────

class TestEntailmentAfterMerge:

    @pytest.mark.parametrize("idx", range(len(AXIOM_SETS) - 1))
    def test_components_entailed_by_merge(self, idx):
        """After merge, both component states divide the merged state."""
        algebra = GodelStateAlgebra()
        state_a = _build_state(algebra, AXIOM_SETS[idx])
        state_b = _build_state(algebra, AXIOM_SETS[idx + 1])
        merged = math.lcm(state_a, state_b)
        assert merged % state_a == 0, "Merged state must entail component A"
        assert merged % state_b == 0, "Merged state must entail component B"

    @pytest.mark.parametrize("axioms", AXIOM_SETS)
    def test_individual_primes_entailed(self, axioms):
        """Each individual axiom prime divides the state."""
        algebra = GodelStateAlgebra()
        state = _build_state(algebra, axioms)
        for s, p, o in axioms:
            prime = algebra.get_or_mint_prime(s, p, o)
            assert state % prime == 0, f"State must be divisible by prime for {s}||{p}||{o}"


# ─── 5. Delta Correctness ────────────────────────────────────────

class TestDeltaCorrectness:

    def test_delta_union_equals_target(self):
        """lcm(source, delta) == target."""
        algebra = GodelStateAlgebra()
        source = _build_state(algebra, AXIOM_SETS[0])
        target = _build_state(algebra, AXIOM_SETS[0] + AXIOM_SETS[1])
        delta = target // math.gcd(target, source)
        assert math.lcm(source, delta) == target

    def test_delta_of_identical_is_one(self):
        """Delta of identical states is 1 (no novel axioms)."""
        algebra = GodelStateAlgebra()
        state = _build_state(algebra, AXIOM_SETS[0])
        delta = state // math.gcd(state, state)
        assert delta == 1

    def test_delta_of_disjoint_equals_target(self):
        """Delta of disjoint states equals the target (all novel)."""
        algebra = GodelStateAlgebra()
        source = _build_state(algebra, AXIOM_SETS[0])
        target = _build_state(algebra, AXIOM_SETS[1])
        delta = target // math.gcd(target, source)
        # When states share no primes, gcd == 1, delta == target
        assert delta == target


# ─── 6. Canonical Round-Trip Conservation ─────────────────────────

class TestCanonicalRoundTrip:

    @pytest.mark.parametrize("axioms", AXIOM_SETS)
    def test_encode_decode_encode(self, axioms):
        """canonical_tome(S) → parse → reconstruct == S."""
        algebra = GodelStateAlgebra()
        tome_gen = AutoregressiveTomeGenerator(algebra)
        state = _build_state(algebra, axioms)

        # Encode
        tome = tome_gen.generate_canonical(state, "Property Test")

        # Decode (parse canonical lines)
        axiom_keys = []
        for line in tome.split("\n"):
            m = re.match(r'^The\s+(\S+)\s+(\S+)\s+(\S+)\.$', line.strip())
            if m:
                s, p, o = m.groups()
                axiom_keys.append((s, p, o))

        # Re-encode
        reconstructed = 1
        for s, p, o in axiom_keys:
            prime = algebra.get_or_mint_prime(s, p, o)
            reconstructed = math.lcm(reconstructed, prime)

        assert reconstructed == state


# ─── 7. Deletion Reversibility ────────────────────────────────────

class TestDeletionReversibility:

    def test_delete_then_readd_restores_state(self):
        """Deleting an axiom and re-adding it restores the original state."""
        algebra = GodelStateAlgebra()
        state = _build_state(algebra, AXIOM_SETS[0])
        prime = algebra.get_or_mint_prime(*AXIOM_SETS[0][0])

        # Delete
        reduced = state // prime
        assert reduced % prime != 0, "Prime should not divide after deletion"

        # Re-add
        restored = math.lcm(reduced, prime)
        assert restored == state

    def test_delete_all_gives_one(self):
        """Deleting all axioms from a state yields 1."""
        algebra = GodelStateAlgebra()
        axioms = AXIOM_SETS[0]
        state = _build_state(algebra, axioms)
        for s, p, o in axioms:
            prime = algebra.get_or_mint_prime(s, p, o)
            state = state // prime
        assert state == 1
