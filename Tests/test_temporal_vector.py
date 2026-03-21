"""
Tests for Temporal Evolution (Delete/Update) and Vector Bridge.

Validates:
    - O(1) axiom deletion via prime division
    - O(1) axiom update (delete + mint)
    - Edge cases (delete non-existent, malformed update)
    - Vector bridge indexing
    - Semantic search only returns alive primes
    - Deleted primes vanish from search results
"""

import sys
import os
import math
import asyncio
import pytest
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.ensemble.vector_bridge import ContinuousDiscreteBridge


# ─── Temporal Deletion ─────────────────────────────────────────────────

class TestDeleteAxiom:

    def test_delete_existing_axiom(self):
        """Deleting an axiom removes its prime from the state."""
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([
            ("Alice", "age", "30"),
            ("Bob", "age", "40"),
        ])

        p_alice = alg.axiom_to_prime["alice||age||30"]
        assert state % p_alice == 0  # Alice is in the state

        new_state = alg.delete_axiom(state, "alice||age||30")
        assert new_state % p_alice != 0  # Alice is gone

        # Bob should still be present
        p_bob = alg.axiom_to_prime["bob||age||40"]
        assert new_state % p_bob == 0

    def test_delete_nonexistent_axiom(self):
        """Deleting a never-minted axiom is a no-op."""
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([("Alice", "age", "30")])
        new_state = alg.delete_axiom(state, "eve||role||hacker")
        assert new_state == state

    def test_delete_normalises_key(self):
        """Deletion normalises case and whitespace."""
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([("Alice", "age", "30")])
        new_state = alg.delete_axiom(state, "  ALICE||AGE||30  ")
        p = alg.axiom_to_prime["alice||age||30"]
        assert new_state % p != 0

    def test_delete_all_axioms_gives_one(self):
        """Deleting every axiom leaves state = 1 (empty)."""
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([("A", "is", "1")])
        new_state = alg.delete_axiom(state, "a||is||1")
        assert new_state == 1


# ─── Temporal Update ───────────────────────────────────────────────────

class TestUpdateAxiom:

    def test_update_replaces_fact(self):
        """Update swaps old prime for new prime."""
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([
            ("Alice", "age", "30"),
            ("Bob", "age", "40"),
        ])

        new_state = alg.update_axiom(
            state, "alice||age||30", "alice||age||31"
        )

        # Old fact is gone
        p_old = alg.axiom_to_prime["alice||age||30"]
        assert new_state % p_old != 0

        # New fact is present
        p_new = alg.axiom_to_prime["alice||age||31"]
        assert new_state % p_new == 0

        # Bob still intact
        p_bob = alg.axiom_to_prime["bob||age||40"]
        assert new_state % p_bob == 0

    def test_update_invalid_format_raises(self):
        """Malformed new_axiom_key raises ValueError."""
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([("Alice", "age", "30")])

        with pytest.raises(ValueError, match="Invalid axiom format"):
            alg.update_axiom(state, "alice||age||30", "bad_format")

    def test_update_nonexistent_old_is_pure_insert(self):
        """Updating a non-existent axiom is effectively a pure insert."""
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([("Alice", "age", "30")])
        new_state = alg.update_axiom(
            state, "eve||role||hacker", "carol||level||5"
        )

        # Carol is now in the state
        p_carol = alg.axiom_to_prime["carol||level||5"]
        assert new_state % p_carol == 0

        # Alice still intact
        p_alice = alg.axiom_to_prime["alice||age||30"]
        assert new_state % p_alice == 0


# ─── Mock Embedding Model ─────────────────────────────────────────────

def make_deterministic_embedder(dim: int = 8):
    """
    Returns a mock async embedding function that produces deterministic
    vectors based on a hash of the input text.  Allows cosine similarity
    to behave meaningfully in tests.
    """
    async def embed(text: str):
        # Seed from text hash for reproducibility
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim).astype(np.float32)
        # Normalise to unit sphere
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()
    return embed


# ─── Vector Bridge ─────────────────────────────────────────────────────

class TestVectorBridge:

    def _run(self, coro):
        """Helper to run async code in tests."""
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_index_new_primes(self):
        """All primes get indexed into the vector space."""
        alg = GodelStateAlgebra()
        alg.encode_chunk_state([
            ("Alice", "age", "30"),
            ("Bob", "role", "admin"),
        ])

        bridge = ContinuousDiscreteBridge(alg, make_deterministic_embedder())
        count = self._run(bridge.index_new_primes())

        assert count == 2
        assert len(bridge.prime_embeddings) == 2

        # Re-indexing is idempotent
        count2 = self._run(bridge.index_new_primes())
        assert count2 == 0

    def test_search_returns_alive_primes(self):
        """Search only returns axioms whose primes divide the state."""
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([
            ("Alice", "age", "30"),
            ("Bob", "role", "admin"),
            ("Carol", "skill", "python"),
        ])

        bridge = ContinuousDiscreteBridge(alg, make_deterministic_embedder())
        self._run(bridge.index_new_primes())

        results = self._run(
            bridge.semantic_search_godel_state(state, "how old is alice", top_k=10)
        )

        # All 3 axioms should be returned (they're all alive)
        returned_keys = [r[0] for r in results]
        assert len(returned_keys) == 3

    def test_deleted_prime_vanishes_from_search(self):
        """After deleting an axiom, it no longer appears in search."""
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([
            ("Alice", "age", "30"),
            ("Bob", "role", "admin"),
        ])

        bridge = ContinuousDiscreteBridge(alg, make_deterministic_embedder())
        self._run(bridge.index_new_primes())

        # Delete Alice
        new_state = alg.delete_axiom(state, "alice||age||30")

        results = self._run(
            bridge.semantic_search_godel_state(new_state, "alice age", top_k=10)
        )

        returned_keys = [r[0] for r in results]
        assert "alice||age||30" not in returned_keys
        assert "bob||role||admin" in returned_keys

    def test_updated_axiom_reflected_in_search(self):
        """After updating, old axiom is gone and new axiom appears."""
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([("Alice", "age", "30")])

        bridge = ContinuousDiscreteBridge(alg, make_deterministic_embedder())
        self._run(bridge.index_new_primes())

        # Update: Alice's age 30 → 31
        new_state = alg.update_axiom(state, "alice||age||30", "alice||age||31")

        # Index the new prime
        self._run(bridge.index_new_primes())

        results = self._run(
            bridge.semantic_search_godel_state(new_state, "alice age", top_k=10)
        )

        returned_keys = [r[0] for r in results]
        assert "alice||age||30" not in returned_keys
        assert "alice||age||31" in returned_keys

    def test_empty_state_returns_nothing(self):
        """State = 1 (empty) returns no results."""
        alg = GodelStateAlgebra()
        alg.encode_chunk_state([("Alice", "age", "30")])

        bridge = ContinuousDiscreteBridge(alg, make_deterministic_embedder())
        self._run(bridge.index_new_primes())

        results = self._run(
            bridge.semantic_search_godel_state(1, "alice", top_k=10)
        )
        assert results == []
