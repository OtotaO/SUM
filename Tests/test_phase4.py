"""
Phase 4 Tests — Temporal CRUD, Akashic Fidelity & Vector Bridge

Validates:
    - O(1) update (LCM-based gauge transformation) and delete (prime division)
    - Crash recovery via event-sourced SQLite ledger (Fidelity Axiom)
    - Semantic search returns only alive primes; deleted primes vanish
"""

import sys
import os
import pytest
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from sum_engine_internal.infrastructure.akashic_ledger import AkashicLedger
from sum_engine_internal.ensemble.vector_bridge import ContinuousDiscreteBridge


# ─── Mock Embedder ─────────────────────────────────────────────────────

class MockEmbedder:
    """Deterministic mock returning axis-aligned vectors for keywords."""

    async def get_embedding(self, text: str):
        if "alice" in text.lower():
            return [1.0, 0.0, 0.0]
        if "bob" in text.lower():
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


# ─── 1. Temporal CRUD ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_temporal_crud():
    """Update replaces old prime via LCM; delete removes it; state → 1."""
    algebra = GodelStateAlgebra()
    p1 = algebra.get_or_mint_prime("Task", "status", "pending")
    state = p1

    # UPDATE: pending → done
    state = algebra.update_axiom(state, "task||status||pending", "task||status||done")
    assert state % p1 != 0, "Old prime should be removed"

    p2 = algebra.axiom_to_prime["task||status||done"]
    assert state % p2 == 0, "New prime should be present"

    # DELETE: done
    state = algebra.delete_axiom(state, "task||status||done")
    assert state % p2 != 0, "Deleted prime should be gone"
    assert state == 1, "State should collapse to identity"


# ─── 2. Akashic Fidelity ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_akashic_fidelity():
    """Event trace faithfully rebuilds global state after simulated crash."""
    db_path = "test_akashic.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    try:
        algebra = GodelStateAlgebra()
        ledger = AkashicLedger(db_path)

        # MINT + MUL Alice as Engineer
        p_alice = algebra.get_or_mint_prime("Alice", "job", "Engineer")
        await ledger.append_event("MINT", p_alice, "alice||job||engineer")

        global_state = p_alice
        await ledger.append_event("MUL", p_alice)

        # UPDATE: Engineer → Director (traces: MINT new, DIV old, MUL new)
        global_state = algebra.update_axiom(
            global_state, "alice||job||engineer", "alice||job||Director"
        )
        p_new = algebra.axiom_to_prime["alice||job||director"]
        await ledger.append_event("MINT", p_new, "alice||job||director")
        await ledger.append_event("DIV", p_alice)
        await ledger.append_event("MUL", p_new)

        # ── Simulate crash & recovery ──
        recovered_algebra = GodelStateAlgebra()
        recovered_state = await ledger.rebuild_state(recovered_algebra)

        assert recovered_state == global_state, (
            f"Recovered {recovered_state} ≠ live {global_state}"
        )
        assert recovered_state % p_alice != 0, "Old prime should not divide state"
        assert recovered_state % p_new == 0, "New prime should divide state"

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


# ─── 3. Vector Bridge ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_vector_bridge():
    """Deleted primes mathematically vanish from semantic search results."""
    algebra = GodelStateAlgebra()
    bridge = ContinuousDiscreteBridge(algebra, MockEmbedder().get_embedding)

    p1 = algebra.get_or_mint_prime("Alice", "status", "active")
    p2 = algebra.get_or_mint_prime("Bob", "status", "inactive")

    state = p1 * p2
    await bridge.index_new_primes()

    # Search — Alice should appear
    res = await bridge.semantic_search_godel_state(state, "Alice", top_k=1)
    returned_keys = [r[0] for r in res]
    assert "alice||status||active" in returned_keys

    # Delete Alice — she should vanish from search
    state = algebra.delete_axiom(state, "alice||status||active")
    res_after = await bridge.semantic_search_godel_state(state, "Alice", top_k=1)
    returned_keys_after = [r[0] for r in res_after]
    assert "alice||status||active" not in returned_keys_after
