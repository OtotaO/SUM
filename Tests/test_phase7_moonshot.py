"""
Phase 7 Tests — Quantum GraphRAG & Autonomous Crystallization

Validates:
    - Node registry is populated when primes are minted
    - O(1) GraphRAG returns correct 1-hop neighborhood
    - Autonomous Crystallizer compacts dense nodes above threshold
    - Compaction is idempotent for nodes below threshold
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from sum_engine_internal.ensemble.autonomous_agent import AutonomousCrystallizer


# ─── Mock helpers ─────────────────────────────────────────────────────

class MockLedger:
    """In-memory mock ledger for test isolation."""
    async def append_event(self, op, prime, axiom=""):
        pass


async def mock_summarizer(axioms):
    return "is_highly_active"


# ─── 1. Quantum GraphRAG ─────────────────────────────────────────────

class TestQuantumGraphRAG:

    def test_node_registry_populated_on_mint(self):
        """Minting a prime auto-registers both subject and object nodes."""
        algebra = GodelStateAlgebra()
        p = algebra.get_or_mint_prime("alice", "likes", "apples")

        assert "alice" in algebra.node_registry
        assert "apples" in algebra.node_registry
        assert algebra.node_registry["alice"] % p == 0
        assert algebra.node_registry["apples"] % p == 0

    def test_graph_rag_1hop_neighborhood(self):
        """1-hop query returns only edges connected to the queried node."""
        algebra = GodelStateAlgebra()

        p1 = algebra.get_or_mint_prime("alice", "likes", "apples")
        p2 = algebra.get_or_mint_prime("alice", "lives_in", "paris")
        p3 = algebra.get_or_mint_prime("bob", "hates", "apples")

        global_state = p1 * p2 * p3

        # Query Alice's neighborhood
        context = algebra.get_quantum_neighborhood(global_state, ["alice"])

        assert context % p1 == 0  # alice->apples
        assert context % p2 == 0  # alice->paris
        assert context % p3 != 0  # bob's edge excluded

    def test_graph_rag_filters_deleted_axioms(self):
        """Deleted axioms don't appear in GraphRAG results."""
        algebra = GodelStateAlgebra()

        p1 = algebra.get_or_mint_prime("x", "is", "1")
        p2 = algebra.get_or_mint_prime("x", "is", "2")

        global_state = p1 * p2

        # Delete p2 from global state
        deleted_state = algebra.delete_axiom(global_state, "x||is||2")

        context = algebra.get_quantum_neighborhood(deleted_state, ["x"])
        assert context % p1 == 0  # still alive
        assert context % p2 != 0  # deleted

    def test_graph_rag_multi_node_query(self):
        """Querying multiple nodes returns the union of neighborhoods."""
        algebra = GodelStateAlgebra()

        p1 = algebra.get_or_mint_prime("a", "rel", "b")
        p2 = algebra.get_or_mint_prime("c", "rel", "d")

        global_state = p1 * p2

        context = algebra.get_quantum_neighborhood(
            global_state, ["a", "c"]
        )

        assert context % p1 == 0
        assert context % p2 == 0


# ─── 2. Autonomous Crystallization ───────────────────────────────────

class TestAutonomousCrystallizer:

    @pytest.mark.asyncio
    async def test_compaction_above_threshold(self):
        """Nodes with ≥ threshold edges get compressed into macro-prime."""
        algebra = GodelStateAlgebra()

        primes = []
        for i in range(5):
            p = algebra.get_or_mint_prime("charlie", f"action_{i}", "thing")
            primes.append(p)

        global_state = math.prod(primes)

        agent = AutonomousCrystallizer(
            algebra, MockLedger(), mock_summarizer
        )
        new_state = await agent.run_compaction_cycle(
            global_state, threshold=5
        )

        # Micro-primes should be gone
        for p in primes:
            assert new_state % p != 0

        # Macro-prime should be present
        macro_key = "charlie||is_highly_active||compressed_cluster"
        assert macro_key in algebra.axiom_to_prime
        assert new_state % algebra.axiom_to_prime[macro_key] == 0

    @pytest.mark.asyncio
    async def test_no_compaction_below_threshold(self):
        """Nodes below the threshold are left untouched."""
        algebra = GodelStateAlgebra()

        p1 = algebra.get_or_mint_prime("dave", "likes", "coffee")
        p2 = algebra.get_or_mint_prime("dave", "age", "25")

        global_state = p1 * p2

        agent = AutonomousCrystallizer(
            algebra, MockLedger(), mock_summarizer
        )
        new_state = await agent.run_compaction_cycle(
            global_state, threshold=5
        )

        # Nothing should change
        assert new_state == global_state
        assert new_state % p1 == 0
        assert new_state % p2 == 0
