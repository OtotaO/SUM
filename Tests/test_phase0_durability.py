"""
Phase 0: Restart-Semantics Tests — Durability Contract

These tests define the persistence contract mechanically.
Each test verifies that a specific workflow survives a simulated
restart (new AkashicLedger instance on the same DB file).

These are the tests Carmack said to write BEFORE touching
the implementation. They ensure that feature semantics and
storage semantics are aligned.

Author: ototao
License: Apache License 2.0
"""

import math
import os
import tempfile
import pytest

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from sum_engine_internal.infrastructure.akashic_ledger import AkashicLedger


@pytest.fixture
def durability_env():
    """Provide a fresh ledger + algebra in a temp directory, returning the db_path for restart simulation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "durability_test.db")
        ledger = AkashicLedger(db_path=db_path)
        algebra = GodelStateAlgebra()
        yield ledger, algebra, db_path


class TestIngestDurability:

    @pytest.mark.asyncio
    async def test_ingest_survives_restart(self, durability_env):
        """
        Ingest axioms → restart → axioms still alive in main branch.
        This is the baseline: MINT+MUL events are durable today.
        """
        ledger, algebra, db_path = durability_env

        p1 = algebra.get_or_mint_prime("earth", "orbits", "sun")
        p2 = algebra.get_or_mint_prime("moon", "orbits", "earth")

        await ledger.append_event("MINT", p1, "earth||orbits||sun")
        await ledger.append_event("MUL", p1)
        await ledger.append_event("MINT", p2, "moon||orbits||earth")
        await ledger.append_event("MUL", p2)

        # Save main branch head
        expected_state = math.lcm(p1, p2)
        await ledger.save_branch_head("main", expected_state)

        # --- SIMULATE RESTART ---
        ledger2 = AkashicLedger(db_path=db_path)
        fresh_algebra = GodelStateAlgebra()
        rebuilt = await ledger2.rebuild_state(fresh_algebra)

        assert rebuilt == expected_state
        assert rebuilt % p1 == 0, "earth||orbits||sun should survive"
        assert rebuilt % p2 == 0, "moon||orbits||earth should survive"

    @pytest.mark.asyncio
    async def test_main_branch_always_survives(self, durability_env):
        """Any operation on main → restart → main is correct."""
        ledger, algebra, db_path = durability_env

        p = algebra.get_or_mint_prime("gravity", "is", "fundamental")
        await ledger.append_event("MINT", p, "gravity||is||fundamental")
        await ledger.append_event("MUL", p)
        await ledger.save_branch_head("main", p)

        # --- SIMULATE RESTART ---
        ledger2 = AkashicLedger(db_path=db_path)
        heads = await ledger2.load_branch_heads()

        assert "main" in heads
        assert heads["main"] == p


class TestBranchDurability:

    @pytest.mark.asyncio
    async def test_branch_survives_restart(self, durability_env):
        """
        Create branch → restart → branch exists with correct state.
        """
        ledger, algebra, db_path = durability_env

        # Set up main branch
        p1 = algebra.get_or_mint_prime("alice", "knows", "bob")
        await ledger.append_event("MINT", p1, "alice||knows||bob")
        await ledger.append_event("MUL", p1)
        await ledger.save_branch_head("main", p1)

        # Create branch (copies main's state)
        branch_state = p1
        await ledger.save_branch_head("experimental", branch_state)

        # --- SIMULATE RESTART ---
        ledger2 = AkashicLedger(db_path=db_path)
        heads = await ledger2.load_branch_heads()

        assert "experimental" in heads, "Branch should survive restart"
        assert heads["experimental"] == branch_state

    @pytest.mark.asyncio
    async def test_merge_survives_restart(self, durability_env):
        """
        Merge branches → restart → merged state correct.
        """
        ledger, algebra, db_path = durability_env

        p1 = algebra.get_or_mint_prime("fact", "one", "alpha")
        p2 = algebra.get_or_mint_prime("fact", "two", "beta")

        await ledger.append_event("MINT", p1, "fact||one||alpha")
        await ledger.append_event("MUL", p1)
        await ledger.append_event("MINT", p2, "fact||two||beta")
        await ledger.append_event("MUL", p2, branch="dev")

        # Simulate merge: target gets LCM of both
        merged = math.lcm(p1, p2)
        await ledger.save_branch_head("main", merged)

        # --- SIMULATE RESTART ---
        ledger2 = AkashicLedger(db_path=db_path)
        heads = await ledger2.load_branch_heads()

        assert heads["main"] == merged
        assert merged % p1 == 0
        assert merged % p2 == 0


class TestImportSyncDurability:

    @pytest.mark.asyncio
    async def test_import_survives_restart(self, durability_env):
        """
        Import bundle primes → restart → imported axioms present.
        """
        ledger, algebra, db_path = durability_env

        # Simulate pre-existing state
        p_existing = algebra.get_or_mint_prime("known", "fact", "alpha")
        await ledger.append_event("MINT", p_existing, "known||fact||alpha")
        await ledger.append_event("MUL", p_existing)

        # Simulate import: new axiom comes from a signed bundle
        p_imported = algebra.get_or_mint_prime("imported", "fact", "beta")
        await ledger.append_event("MINT", p_imported, "imported||fact||beta")
        await ledger.append_event("MUL", p_imported, branch="user123")

        # Save the merged user branch head
        user_state = math.lcm(p_existing, p_imported)
        await ledger.save_branch_head("user123", user_state)

        # --- SIMULATE RESTART ---
        ledger2 = AkashicLedger(db_path=db_path)
        heads = await ledger2.load_branch_heads()

        assert "user123" in heads
        assert heads["user123"] == user_state
        assert user_state % p_imported == 0, "Imported axiom should survive"

    @pytest.mark.asyncio
    async def test_sync_survives_restart(self, durability_env):
        """
        Sync peer state → restart → merged state present.
        """
        ledger, algebra, db_path = durability_env

        p_local = algebra.get_or_mint_prime("local", "knowledge", "alpha")
        p_peer = algebra.get_or_mint_prime("peer", "knowledge", "beta")

        # Local axiom
        await ledger.append_event("MINT", p_local, "local||knowledge||alpha")
        await ledger.append_event("MUL", p_local)

        # Peer sync: MINT the new axiom + MUL into branch
        await ledger.append_event("MINT", p_peer, "peer||knowledge||beta")
        await ledger.append_event("MUL", p_peer, branch="node_A")

        synced_state = math.lcm(p_local, p_peer)
        await ledger.save_branch_head("node_A", synced_state)

        # --- SIMULATE RESTART ---
        ledger2 = AkashicLedger(db_path=db_path)
        heads = await ledger2.load_branch_heads()

        assert "node_A" in heads
        assert heads["node_A"] == synced_state
        assert synced_state % p_peer == 0, "Synced peer axiom should survive"


class TestTimeTravelEphemeral:

    @pytest.mark.asyncio
    async def test_time_travel_is_ephemeral(self, durability_env):
        """
        Time-travel branch → restart → branch NOT present.
        Ephemeral branches are intentionally transient.
        """
        ledger, algebra, db_path = durability_env

        p = algebra.get_or_mint_prime("historical", "fact", "gamma")
        await ledger.append_event("MINT", p, "historical||fact||gamma")
        await ledger.append_event("MUL", p)

        # Save time-travel branch as ephemeral
        await ledger.save_branch_head("past_tick_42", p, is_ephemeral=True)

        # --- SIMULATE RESTART ---
        ledger2 = AkashicLedger(db_path=db_path)
        heads = await ledger2.load_branch_heads()

        assert "past_tick_42" not in heads, (
            "Ephemeral time-travel branches should NOT survive restart"
        )


class TestSnapshotIntegrity:

    @pytest.mark.asyncio
    async def test_branch_head_snapshot_matches_replay(self, durability_env):
        """
        Saved head matches replayed state — snapshot is consistent with events.
        """
        ledger, algebra, db_path = durability_env

        p1 = algebra.get_or_mint_prime("verify", "snapshot", "alpha")
        p2 = algebra.get_or_mint_prime("verify", "snapshot", "beta")

        await ledger.append_event("MINT", p1, "verify||snapshot||alpha")
        await ledger.append_event("MUL", p1)
        await ledger.append_event("MINT", p2, "verify||snapshot||beta")
        await ledger.append_event("MUL", p2)

        expected = math.lcm(p1, p2)
        await ledger.save_branch_head("main", expected)

        # --- SIMULATE RESTART ---
        ledger2 = AkashicLedger(db_path=db_path)

        # Load from snapshot
        heads = await ledger2.load_branch_heads()
        snapshot_state = heads["main"]

        # Replay from events
        fresh_algebra = GodelStateAlgebra()
        replayed_state = await ledger2.rebuild_state(fresh_algebra)

        assert snapshot_state == replayed_state, (
            f"Snapshot ({snapshot_state}) must match replay ({replayed_state})"
        )
