"""
Tests for Semantic Prime Number Theorem (SPNT) & Gödel-State Algebra.

Covers:
    - SPNT asymptotic bound formula
    - Prime minting (uniqueness, idempotency, normalisation)
    - Gödel encoding (product of primes)
    - LCM merge (deduplication, commutativity)
    - Entailment verification (modulo check)
    - Curvature paradox detection
    - Mass Semantic Engine end-to-end pipeline
"""

import sys
import os
import math
import asyncio
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from internal.algorithms.semantic_arithmetic import (
    SemanticPrimeNumberTheorem,
    GodelStateAlgebra,
)


# ─── SPNT Bound ────────────────────────────────────────────────────────

class TestSemanticPrimeNumberTheorem:

    def test_asymptotic_bound_basic(self):
        """N / ln(N) for standard values."""
        assert SemanticPrimeNumberTheorem.asymptotic_bound(100) == int(100 / math.log(100))
        assert SemanticPrimeNumberTheorem.asymptotic_bound(1000) == int(1000 / math.log(1000))
        assert SemanticPrimeNumberTheorem.asymptotic_bound(10_000) == int(10_000 / math.log(10_000))

    def test_asymptotic_bound_edge_cases(self):
        """N <= 3 should return N directly."""
        assert SemanticPrimeNumberTheorem.asymptotic_bound(0) == 0
        assert SemanticPrimeNumberTheorem.asymptotic_bound(1) == 1
        assert SemanticPrimeNumberTheorem.asymptotic_bound(2) == 2
        assert SemanticPrimeNumberTheorem.asymptotic_bound(3) == 3


# ─── Prime Minting ─────────────────────────────────────────────────────

class TestGodelPrimeMinting:

    def test_mint_prime_uniqueness(self):
        """Distinct axioms get distinct primes."""
        alg = GodelStateAlgebra()
        p1 = alg.get_or_mint_prime("Alice", "age", "30")
        p2 = alg.get_or_mint_prime("Alice", "age", "31")
        p3 = alg.get_or_mint_prime("Bob", "age", "30")
        assert p1 != p2
        assert p1 != p3
        assert p2 != p3

    def test_mint_prime_idempotency(self):
        """Same axiom always returns the same prime."""
        alg = GodelStateAlgebra()
        p1 = alg.get_or_mint_prime("Alice", "age", "30")
        p2 = alg.get_or_mint_prime("Alice", "age", "30")
        assert p1 == p2

    def test_mint_prime_case_normalization(self):
        """Case and whitespace are normalised."""
        alg = GodelStateAlgebra()
        p1 = alg.get_or_mint_prime("Alice", "age", "30")
        p2 = alg.get_or_mint_prime("  ALICE  ", "  Age  ", "  30  ")
        assert p1 == p2


# ─── Encoding ──────────────────────────────────────────────────────────

class TestGodelEncoding:

    def test_encode_chunk_state(self):
        """State is the product of its axiom primes."""
        alg = GodelStateAlgebra()
        axioms = [
            ("Alice", "age", "30"),
            ("Bob", "role", "admin"),
        ]
        state = alg.encode_chunk_state(axioms)

        p_alice = alg.get_or_mint_prime("Alice", "age", "30")
        p_bob = alg.get_or_mint_prime("Bob", "role", "admin")

        assert state == p_alice * p_bob


# ─── LCM Merge ─────────────────────────────────────────────────────────

class TestGodelMerge:

    def test_merge_parallel_states_lcm(self):
        """LCM deduplicates overlapping axioms."""
        alg = GodelStateAlgebra()

        shared_axiom = ("Alice", "key", "8F4C")
        unique_a = ("Bob", "role", "admin")
        unique_b = ("Carol", "level", "5")

        state_a = alg.encode_chunk_state([shared_axiom, unique_a])
        state_b = alg.encode_chunk_state([shared_axiom, unique_b])

        merged = alg.merge_parallel_states([state_a, state_b])

        # Merged state should contain all three unique primes exactly once
        p_shared = alg.get_or_mint_prime(*shared_axiom)
        p_a = alg.get_or_mint_prime(*unique_a)
        p_b = alg.get_or_mint_prime(*unique_b)

        expected = p_shared * p_a * p_b
        assert merged == expected

    def test_merge_commutativity(self):
        """merge([A, B]) == merge([B, A])."""
        alg = GodelStateAlgebra()

        state_a = alg.encode_chunk_state([("X", "is", "1")])
        state_b = alg.encode_chunk_state([("Y", "is", "2")])

        assert alg.merge_parallel_states([state_a, state_b]) == \
               alg.merge_parallel_states([state_b, state_a])

    def test_merge_empty(self):
        """Merging an empty list returns identity (1)."""
        alg = GodelStateAlgebra()
        assert alg.merge_parallel_states([]) == 1


# ─── Entailment ────────────────────────────────────────────────────────

class TestGodelEntailment:

    def test_verify_entailment_true(self):
        """A state containing the hypothesis should entail it."""
        alg = GodelStateAlgebra()
        axioms = [("A", "is", "1"), ("B", "is", "2"), ("C", "is", "3")]
        global_state = alg.encode_chunk_state(axioms)

        hypothesis = alg.encode_chunk_state([("A", "is", "1"), ("B", "is", "2")])
        assert alg.verify_entailment(global_state, hypothesis) is True

    def test_verify_entailment_false(self):
        """A state missing an axiom should not entail it."""
        alg = GodelStateAlgebra()
        global_state = alg.encode_chunk_state([("A", "is", "1")])
        hypothesis = alg.encode_chunk_state([("A", "is", "1"), ("Z", "is", "99")])
        assert alg.verify_entailment(global_state, hypothesis) is False


# ─── Paradox Detection ─────────────────────────────────────────────────

class TestGodelCurvature:

    def test_detect_paradox_conflict(self):
        """Mutually exclusive values for the same subject+predicate."""
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([
            ("task", "status", "pending"),
            ("task", "status", "done"),
        ])
        paradoxes = alg.detect_curvature_paradoxes(state)
        assert len(paradoxes) == 1
        assert "task||status" in paradoxes[0]

    def test_detect_paradox_no_conflict(self):
        """No paradox when different predicates have different values."""
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([
            ("Alice", "age", "30"),
            ("Alice", "name", "Alice"),
        ])
        paradoxes = alg.detect_curvature_paradoxes(state)
        assert len(paradoxes) == 0


# ─── Mass Semantic Engine (end-to-end) ─────────────────────────────────

class TestMassSemanticEngine:

    def test_full_pipeline(self):
        """MAP → ENCODE → REDUCE → AUDIT end-to-end."""
        from internal.legacy_api.mass_semantic_engine import MassSemanticEngine

        # Mock extractor: returns fixed triplets per chunk
        async def mock_extractor(chunk: str):
            if "alice" in chunk.lower():
                return [("Alice", "key", "8F4C"), ("Alice", "role", "admin")]
            return [("Bob", "key", "A1B2"), ("Alice", "key", "8F4C")]

        engine = MassSemanticEngine(extractor_llm_func=mock_extractor)

        result = asyncio.get_event_loop().run_until_complete(
            engine.tomes_to_tags(
                raw_claims_count=20,
                chunks=["Alice has key 8F4C", "Bob has key A1B2"],
            )
        )

        assert result["global_state"] > 1
        assert result["total_unique_primes"] == 3  # Alice/key/8F4C + Alice/role/admin + Bob/key/A1B2
        assert isinstance(result["compression_ok"], bool)
        assert isinstance(result["paradoxes"], list)

    def test_spnt_overclaim_warning(self):
        """Pipeline flags overclaim when primes exceed SPNT bound."""
        from internal.legacy_api.mass_semantic_engine import MassSemanticEngine

        # Return many unique axioms from a single chunk
        async def greedy_extractor(chunk: str):
            return [(f"E{i}", "rel", f"V{i}") for i in range(50)]

        engine = MassSemanticEngine(extractor_llm_func=greedy_extractor)

        # Claim only 10 raw claims but extract 50 unique primes
        result = asyncio.get_event_loop().run_until_complete(
            engine.tomes_to_tags(raw_claims_count=10, chunks=["big chunk"])
        )

        assert result["compression_ok"] is False
        assert result["total_unique_primes"] > result["spnt_limit"]
