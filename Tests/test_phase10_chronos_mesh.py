"""
Phase 10 Tests — The Chronos Engine & The Holographic Mesh

Validates:
    - Zero-Knowledge Semantic Proofs (ZK-SP)
    - Chronos Engine (Time Travel via Akashic Ledger replay)
    - P2P Holographic Mesh (Gossip-based Gödel Integer sync)
    - Proof verification (tamper detection)
    - Historical state isolation from present
    - Branch creation from time-travel snapshots
"""

import sys
import os
import math
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from sum_engine_internal.algorithms.zk_semantics import ZKSemanticProver
from sum_engine_internal.infrastructure.akashic_ledger import AkashicLedger
from sum_engine_internal.infrastructure.p2p_mesh import EpistemicMeshNetwork


# ─── 1. Zero-Knowledge Semantic Proofs ───────────────────────────────

class TestZeroKnowledgeProofs:

    def test_valid_proof_generation(self):
        """Generate and verify a valid ZK-SP."""
        algebra = GodelStateAlgebra()
        p1 = algebra.get_or_mint_prime("earth", "orbits", "sun")
        p2 = algebra.get_or_mint_prime("moon", "orbits", "earth")

        state = p1 * p2

        proof = ZKSemanticProver.generate_proof(state, p1)

        assert "commitment" in proof
        assert "salt" in proof
        assert proof["prime"] == p1
        assert int(proof["quotient"]) == p2

    def test_proof_verifies_correctly(self):
        """A valid proof passes verification."""
        algebra = GodelStateAlgebra()
        p1 = algebra.get_or_mint_prime("sky", "color", "blue")
        state = p1

        proof = ZKSemanticProver.generate_proof(state, p1)
        assert ZKSemanticProver.verify_proof(proof) is True

    def test_tampered_proof_fails(self):
        """Modifying the quotient invalidates the proof."""
        algebra = GodelStateAlgebra()
        p1 = algebra.get_or_mint_prime("earth", "orbits", "sun")
        p2 = algebra.get_or_mint_prime("moon", "orbits", "earth")
        state = p1 * p2

        proof = ZKSemanticProver.generate_proof(state, p1)
        assert ZKSemanticProver.verify_proof(proof) is True

        # Tamper with the quotient
        proof["quotient"] = "999"
        assert ZKSemanticProver.verify_proof(proof) is False

    def test_proof_rejects_non_entailed_prime(self):
        """Cannot generate a proof for a prime not in the state."""
        algebra = GodelStateAlgebra()
        p1 = algebra.get_or_mint_prime("a", "is", "1")
        p2 = algebra.get_or_mint_prime("b", "is", "2")

        state = p1  # Only contains p1

        with pytest.raises(ValueError, match="does not entail"):
            ZKSemanticProver.generate_proof(state, p2)

    def test_proof_hides_full_state(self):
        """The proof does not contain the full state integer."""
        algebra = GodelStateAlgebra()
        primes = []
        for i in range(10):
            p = algebra.get_or_mint_prime(f"fact_{i}", "is", "true")
            primes.append(p)

        state = 1
        for p in primes:
            state *= p

        proof = ZKSemanticProver.generate_proof(state, primes[0])

        # Quotient is State // prime, not State itself
        quotient = int(proof["quotient"])
        assert quotient != state
        assert quotient * primes[0] == state

    def test_unique_salts_per_proof(self):
        """Each proof gets a unique salt for security."""
        algebra = GodelStateAlgebra()
        p1 = algebra.get_or_mint_prime("x", "is", "1")
        state = p1

        proof1 = ZKSemanticProver.generate_proof(state, p1)
        proof2 = ZKSemanticProver.generate_proof(state, p1)

        assert proof1["salt"] != proof2["salt"]
        assert proof1["commitment"] != proof2["commitment"]


# ─── 2. Chronos Engine (Time Travel) ────────────────────────────────

class TestChronosEngine:

    @pytest.fixture
    def db_path(self, tmp_path):
        return str(tmp_path / "chronos_test.db")

    @pytest.mark.asyncio
    async def test_time_travel_to_past_tick(self, db_path):
        """Rebuilding at a past tick produces the exact historical state."""
        algebra = GodelStateAlgebra()
        ledger = AkashicLedger(db_path)

        # Epoch 1: mint and multiply alice
        p1 = algebra.get_or_mint_prime("alice", "job", "engineer")
        await ledger.append_event("MINT", p1, "alice||job||engineer")
        await ledger.append_event("MUL", p1)

        tick_1 = await ledger.get_latest_tick()

        # Epoch 2: mint and multiply bob
        p2 = algebra.get_or_mint_prime("bob", "job", "doctor")
        await ledger.append_event("MINT", p2, "bob||job||doctor")
        await ledger.append_event("MUL", p2)

        # Present state should contain both
        present_algebra = GodelStateAlgebra()
        present = await ledger.rebuild_state(present_algebra)
        assert present % p1 == 0
        assert present % p2 == 0

        # Time travel to tick 1 — should only contain alice
        past_algebra = GodelStateAlgebra()
        past = await ledger.rebuild_state(past_algebra, max_seq_id=tick_1)
        assert past % p1 == 0
        assert past % p2 != 0

    @pytest.mark.asyncio
    async def test_time_travel_to_tick_zero(self, db_path):
        """Traveling to tick 0 yields an empty state."""
        algebra = GodelStateAlgebra()
        ledger = AkashicLedger(db_path)

        p1 = algebra.get_or_mint_prime("alice", "job", "engineer")
        await ledger.append_event("MINT", p1, "alice||job||engineer")
        await ledger.append_event("MUL", p1)

        empty_algebra = GodelStateAlgebra()
        state = await ledger.rebuild_state(empty_algebra, max_seq_id=0)
        assert state == 1  # Empty universe

    @pytest.mark.asyncio
    async def test_get_latest_tick(self, db_path):
        """Latest tick advances with each event."""
        algebra = GodelStateAlgebra()
        ledger = AkashicLedger(db_path)

        assert await ledger.get_latest_tick() == 0

        p1 = algebra.get_or_mint_prime("x", "is", "1")
        await ledger.append_event("MINT", p1, "x||is||1")
        t1 = await ledger.get_latest_tick()
        assert t1 == 1

        await ledger.append_event("MUL", p1)
        t2 = await ledger.get_latest_tick()
        assert t2 == 2

    @pytest.mark.asyncio
    async def test_time_travel_preserves_delete(self, db_path):
        """Deletions in the past are correctly replayed during time travel."""
        algebra = GodelStateAlgebra()
        ledger = AkashicLedger(db_path)

        p1 = algebra.get_or_mint_prime("alice", "job", "engineer")
        p2 = algebra.get_or_mint_prime("bob", "job", "doctor")

        await ledger.append_event("MINT", p1, "alice||job||engineer")
        await ledger.append_event("MUL", p1)
        await ledger.append_event("MINT", p2, "bob||job||doctor")
        await ledger.append_event("MUL", p2)

        # Delete alice
        await ledger.append_event("DIV", p1)
        tick_after_delete = await ledger.get_latest_tick()

        # Add charlie
        p3 = algebra.get_or_mint_prime("charlie", "job", "artist")
        await ledger.append_event("MINT", p3, "charlie||job||artist")
        await ledger.append_event("MUL", p3)

        # Time travel to just after delete
        past_alg = GodelStateAlgebra()
        past = await ledger.rebuild_state(past_alg, max_seq_id=tick_after_delete)

        assert past % p1 != 0  # Deleted
        assert past % p2 == 0  # Present
        assert past % p3 != 0  # Not yet created


# ─── 3. P2P Holographic Mesh ────────────────────────────────────────

class TestHolographicMesh:

    def test_add_and_remove_peers(self):
        """Peer management works correctly."""
        algebra = GodelStateAlgebra()
        mesh = EpistemicMeshNetwork(algebra, lambda b: 1, lambda b, s: None)

        mesh.add_peer("http://node-a:8000")
        mesh.add_peer("http://node-b:8000/")

        assert len(mesh.peers) == 2
        assert "http://node-a:8000" in mesh.peers
        assert "http://node-b:8000" in mesh.peers  # Trailing slash stripped

        mesh.remove_peer("http://node-a:8000")
        assert len(mesh.peers) == 1

    def test_duplicate_peers_ignored(self):
        """Adding the same peer twice doesn't duplicate."""
        algebra = GodelStateAlgebra()
        mesh = EpistemicMeshNetwork(algebra, lambda b: 1, lambda b, s: None)

        mesh.add_peer("http://node-a:8000")
        mesh.add_peer("http://node-a:8000")

        assert len(mesh.peers) == 1

    @pytest.mark.asyncio
    async def test_sync_absorbs_novel_axioms(self):
        """Sync with a remote node absorbs missing axioms via LCM."""
        algebra = GodelStateAlgebra()

        p_local = algebra.get_or_mint_prime("local", "is", "online")
        state = {"main": p_local}

        def get_state(b):
            return state[b]

        def set_state(b, s):
            state[b] = s

        mesh = EpistemicMeshNetwork(algebra, get_state, set_state)

        # Mock the remote sync response
        p_remote = algebra.get_or_mint_prime("remote", "is", "online")

        import httpx

        class MockResponse:
            status_code = 200

            def json(self):
                return {
                    "new_global_state": str(p_local * p_remote),
                    "delta": {
                        "add": ["remote||is||online"],
                        "delete": [],
                    },
                }

        original_post = httpx.AsyncClient.post

        async def mock_post(self_client, *args, **kwargs):
            return MockResponse()

        httpx.AsyncClient.post = mock_post
        try:
            await mesh._sync_with_peer("http://fake-peer:8000", "main")
        finally:
            httpx.AsyncClient.post = original_post

        # Local state should now contain the remote axiom
        assert state["main"] % p_remote == 0
        assert state["main"] % p_local == 0

    @pytest.mark.asyncio
    async def test_sync_with_identical_state_is_noop(self):
        """Sync with a peer at the same state does nothing."""
        algebra = GodelStateAlgebra()

        p1 = algebra.get_or_mint_prime("fact", "is", "true")
        state = {"main": p1}

        mesh = EpistemicMeshNetwork(
            algebra, lambda b: state[b], lambda b, s: state.__setitem__(b, s)
        )

        import httpx

        class MockResponse:
            status_code = 200

            def json(self):
                return {
                    "new_global_state": str(p1),
                    "delta": {"add": [], "delete": []},
                }

        original_post = httpx.AsyncClient.post

        async def mock_post(self_client, *args, **kwargs):
            return MockResponse()

        httpx.AsyncClient.post = mock_post
        try:
            await mesh._sync_with_peer("http://fake-peer:8000", "main")
        finally:
            httpx.AsyncClient.post = original_post

        # State should be unchanged
        assert state["main"] == p1

    def test_daemon_stop(self):
        """Daemon stop flag works correctly."""
        algebra = GodelStateAlgebra()
        mesh = EpistemicMeshNetwork(algebra, lambda b: 1, lambda b, s: None)

        assert not mesh.is_running
        mesh.is_running = True
        mesh.stop_daemon()
        assert not mesh.is_running


# ─── 4. Integration: ZK + Branching + Time Travel ───────────────────

class TestPhase10Integration:

    @pytest.mark.asyncio
    async def test_zk_proof_across_branches(self):
        """ZK proofs work on branch-specific states."""
        algebra = GodelStateAlgebra()

        p_base = algebra.get_or_mint_prime("base", "is", "truth")
        p_exp = algebra.get_or_mint_prime("experiment", "is", "novel")

        branches = {"main": p_base}
        branches["experiment"] = math.lcm(branches["main"], p_exp)

        # Can prove experiment axiom on experiment branch
        proof = ZKSemanticProver.generate_proof(branches["experiment"], p_exp)
        assert ZKSemanticProver.verify_proof(proof) is True

        # Cannot prove experiment axiom on main branch
        with pytest.raises(ValueError):
            ZKSemanticProver.generate_proof(branches["main"], p_exp)

    @pytest.mark.asyncio
    async def test_time_travel_creates_provable_branch(self, tmp_path):
        """Time-travel branches support ZK proofs."""
        db_path = str(tmp_path / "integration.db")
        algebra = GodelStateAlgebra()
        ledger = AkashicLedger(db_path)

        p1 = algebra.get_or_mint_prime("alice", "age", "30")
        await ledger.append_event("MINT", p1, "alice||age||30")
        await ledger.append_event("MUL", p1)

        tick_1 = await ledger.get_latest_tick()

        p2 = algebra.get_or_mint_prime("alice", "age", "31")
        await ledger.append_event("MINT", p2, "alice||age||31")
        await ledger.append_event("MUL", p2)

        # Time travel to when alice was 30
        past_alg = GodelStateAlgebra()
        past_state = await ledger.rebuild_state(past_alg, max_seq_id=tick_1)

        # Can prove alice was 30 in the past
        proof_30 = ZKSemanticProver.generate_proof(past_state, p1)
        assert ZKSemanticProver.verify_proof(proof_30) is True

        # Cannot prove alice was 31 in the past
        with pytest.raises(ValueError):
            ZKSemanticProver.generate_proof(past_state, p2)
