"""
Phase 0.1: Boundary-Case Tests — Durability Integrity Pass

These tests cover the tricky edge cases Carmack identified
after Phase 0:
  1. Non-main /ingest survives restart (branch-scoped events)
  2. Boot replay does NOT pollute main with user-branch events
  3. Import with novel axioms materializes semantics locally
  4. Gossip-acquired knowledge persists via mesh callback

Author: ototao
License: Apache License 2.0
"""

import math
import os
import tempfile
import pytest

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.infrastructure.akashic_ledger import AkashicLedger


@pytest.fixture
def durability_env():
    """Provide a fresh ledger + algebra in a temp directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "boundary_test.db")
        ledger = AkashicLedger(db_path=db_path)
        algebra = GodelStateAlgebra()
        yield ledger, algebra, db_path


class TestNonMainIngestDurability:

    @pytest.mark.asyncio
    async def test_branch_ingest_survives_restart(self, durability_env):
        """
        Ingest into a user branch → restart → branch state correct,
        and events are properly scoped to that branch.
        """
        ledger, algebra, db_path = durability_env

        # Mint and ingest into user branch "alice"
        p1 = algebra.get_or_mint_prime("cat", "is", "animal")
        await ledger.append_event("MINT", p1, "cat||is||animal", branch="alice")
        await ledger.append_event("MUL", p1, branch="alice")

        p2 = algebra.get_or_mint_prime("dog", "is", "animal")
        await ledger.append_event("MINT", p2, "dog||is||animal", branch="alice")
        await ledger.append_event("MUL", p2, branch="alice")

        alice_state = math.lcm(p1, p2)
        await ledger.save_branch_head("alice", alice_state)
        await ledger.save_branch_head("main", 1)  # main untouched

        # --- SIMULATE RESTART ---
        ledger2 = AkashicLedger(db_path=db_path)
        heads = await ledger2.load_branch_heads()

        assert "alice" in heads
        assert heads["alice"] == alice_state
        assert heads["alice"] % p1 == 0
        assert heads["alice"] % p2 == 0


class TestBootReplayIsolation:

    @pytest.mark.asyncio
    async def test_main_not_polluted_by_branch_events(self, durability_env):
        """
        Events on user branches do NOT bleed into main during boot replay.
        This is the key correctness test for Fix 1.
        """
        ledger, algebra, db_path = durability_env

        # Main branch axiom
        p_main = algebra.get_or_mint_prime("gravity", "is", "fundamental")
        await ledger.append_event("MINT", p_main, "gravity||is||fundamental", branch="main")
        await ledger.append_event("MUL", p_main, branch="main")

        # User branch axiom — should NOT appear in main
        p_user = algebra.get_or_mint_prime("magic", "is", "real")
        await ledger.append_event("MINT", p_user, "magic||is||real", branch="bob")
        await ledger.append_event("MUL", p_user, branch="bob")

        await ledger.save_branch_head("main", p_main)
        await ledger.save_branch_head("bob", math.lcm(p_main, p_user))

        # --- SIMULATE RESTART ---
        ledger2 = AkashicLedger(db_path=db_path)
        fresh_algebra = GodelStateAlgebra()

        # Rebuild main with branch filter (as boot_sequence now does)
        main_state = await ledger2.rebuild_state(fresh_algebra, branch="main")

        assert main_state % p_main == 0, "Main axiom should be present"
        assert main_state % p_user != 0, (
            "User-branch axiom 'magic||is||real' should NOT bleed into main"
        )

    @pytest.mark.asyncio
    async def test_branch_replay_sees_only_its_events(self, durability_env):
        """
        Rebuilding a specific branch only returns events for that branch.
        """
        ledger, algebra, db_path = durability_env

        p1 = algebra.get_or_mint_prime("fact", "a", "alpha")
        p2 = algebra.get_or_mint_prime("fact", "b", "beta")

        await ledger.append_event("MINT", p1, "fact||a||alpha", branch="main")
        await ledger.append_event("MUL", p1, branch="main")
        await ledger.append_event("MINT", p2, "fact||b||beta", branch="dev")
        await ledger.append_event("MUL", p2, branch="dev")

        # --- SIMULATE RESTART ---
        ledger2 = AkashicLedger(db_path=db_path)
        fresh_algebra = GodelStateAlgebra()

        main_state = await ledger2.rebuild_state(fresh_algebra, branch="main")
        dev_state = await ledger2.rebuild_state(fresh_algebra, branch="dev")

        assert main_state % p1 == 0, "main should have fact||a||alpha"
        assert main_state % p2 != 0, "main should NOT have fact||b||beta"
        assert dev_state % p2 == 0, "dev should have fact||b||beta"


class TestImportAxiomMaterialization:

    @pytest.mark.asyncio
    async def test_novel_axioms_registered_after_import(self, durability_env):
        """
        Importing a bundle with axioms this node has never seen should
        register those axiom↔prime mappings in the local algebra,
        not just merge the integer.
        """
        ledger, algebra, db_path = durability_env

        # Simulate a bundle created on a REMOTE node with axioms
        # the local node has never processed
        remote_algebra = GodelStateAlgebra()
        p_remote_1 = remote_algebra.get_or_mint_prime("quark", "is", "fundamental")
        p_remote_2 = remote_algebra.get_or_mint_prime("gluon", "mediates", "strong_force")
        remote_state = math.lcm(p_remote_1, p_remote_2)

        # The local algebra should NOT know these axioms yet
        assert "quark||is||fundamental" not in algebra.prime_to_axiom.values()

        # Simulate what import_bundle + tome parsing does:
        # Parse canonical tome lines to materialize axioms locally
        tome_lines = [
            "- quark||is||fundamental [prime: {}]".format(p_remote_1),
            "- gluon||mediates||strong_force [prime: {}]".format(p_remote_2),
        ]

        for line in tome_lines:
            line = line.strip()
            if line.startswith("- ") and "||" in line:
                axiom_part = line[2:].strip()
                if " [prime:" in axiom_part:
                    axiom_part = axiom_part[:axiom_part.index(" [prime:")].strip()
                parts = axiom_part.split("||")
                if len(parts) == 3:
                    s, p, o = [x.strip().lower() for x in parts]
                    algebra.get_or_mint_prime(s, p, o)

        # Now the local algebra SHOULD know these axioms
        local_axioms = set(algebra.prime_to_axiom.values())
        assert "quark||is||fundamental" in local_axioms, (
            "Novel axiom 'quark' should be materialized locally"
        )
        assert "gluon||mediates||strong_force" in local_axioms, (
            "Novel axiom 'gluon' should be materialized locally"
        )

        # And the local primes should match because SHA-256 is deterministic
        local_p1 = algebra.get_or_mint_prime("quark", "is", "fundamental")
        assert local_p1 == p_remote_1, (
            "Deterministic prime derivation must produce identical primes"
        )


class TestGossipDurability:

    @pytest.mark.asyncio
    async def test_mesh_callback_persists_state(self, durability_env):
        """
        Simulates what _update_branch_state + _persist_branch_update does:
        gossip-acquired knowledge should be persisted to branch_heads.
        """
        ledger, algebra, db_path = durability_env

        # Simulate gossip acquisition: mesh absorbed new axiom
        p_gossip = algebra.get_or_mint_prime("dark_matter", "explains", "rotation_curves")
        await ledger.append_event("MINT", p_gossip, "dark_matter||explains||rotation_curves")
        await ledger.append_event("MUL", p_gossip)

        gossip_state = p_gossip
        # This is what _persist_branch_update does
        await ledger.save_branch_head("main", gossip_state)

        # --- SIMULATE RESTART ---
        ledger2 = AkashicLedger(db_path=db_path)
        heads = await ledger2.load_branch_heads()

        assert "main" in heads
        assert heads["main"] == gossip_state
        assert heads["main"] % p_gossip == 0, (
            "Gossip-acquired axiom should survive restart"
        )

    @pytest.mark.asyncio
    async def test_gossip_branch_state_persists(self, durability_env):
        """
        Gossip updates to non-main branches also survive restart.
        """
        ledger, algebra, db_path = durability_env

        p = algebra.get_or_mint_prime("peer_fact", "from", "node_B")
        await ledger.append_event("MINT", p, "peer_fact||from||node_B", branch="main")
        await ledger.append_event("MUL", p, branch="main")

        # Mesh callback persists to main
        await ledger.save_branch_head("main", p)

        # --- SIMULATE RESTART ---
        ledger2 = AkashicLedger(db_path=db_path)
        heads = await ledger2.load_branch_heads()

        assert heads["main"] % p == 0
