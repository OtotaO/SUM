"""
Phase 5 Tests — Gödel Sync Protocol & Quantum API

Validates:
    - O(1) network delta correctly identifies add/delete axioms
    - Identical states produce empty deltas
    - FastAPI /state and /sync endpoints return correct responses
    - Pre-boot 503 protection
"""

import sys
import os
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra


# ─── 1. Gödel Sync Protocol (Pure Math) ──────────────────────────────

class TestGodelSyncProtocol:

    def test_network_delta_add_and_delete(self):
        """Delta correctly identifies axioms to add and delete."""
        algebra = GodelStateAlgebra()

        p_alice = algebra.get_or_mint_prime("alice", "job", "engineer")
        p_bob = algebra.get_or_mint_prime("bob", "job", "doctor")
        p_charlie = algebra.get_or_mint_prime("charlie", "job", "artist")

        # Client knows Alice and Bob
        client_state = p_alice * p_bob

        # Server knows Alice and Charlie (Bob was deleted, Charlie is new)
        server_state = p_alice * p_charlie

        delta = algebra.calculate_network_delta(server_state, client_state)

        assert "charlie||job||artist" in delta["add"]
        assert "bob||job||doctor" in delta["delete"]
        # Alice is shared — should not appear in either list
        assert "alice||job||engineer" not in delta["add"]
        assert "alice||job||engineer" not in delta["delete"]

    def test_network_delta_identical_states(self):
        """Identical client and server states produce empty delta."""
        algebra = GodelStateAlgebra()
        p = algebra.get_or_mint_prime("x", "is", "1")
        state = p

        delta = algebra.calculate_network_delta(state, state)
        assert delta == {"add": [], "delete": []}

    def test_network_delta_fresh_client(self):
        """A fresh client (state=1) receives everything the server has."""
        algebra = GodelStateAlgebra()
        p1 = algebra.get_or_mint_prime("a", "is", "1")
        p2 = algebra.get_or_mint_prime("b", "is", "2")

        server_state = p1 * p2
        delta = algebra.calculate_network_delta(server_state, 1)

        assert "a||is||1" in delta["add"]
        assert "b||is||2" in delta["add"]
        assert delta["delete"] == []

    def test_network_delta_client_ahead(self):
        """If client has axioms the server deleted, they appear in delete."""
        algebra = GodelStateAlgebra()
        p1 = algebra.get_or_mint_prime("a", "is", "1")
        p2 = algebra.get_or_mint_prime("b", "is", "2")

        # Client has both, server only has p1
        client_state = p1 * p2
        server_state = p1

        delta = algebra.calculate_network_delta(server_state, client_state)
        assert delta["add"] == []
        assert "b||is||2" in delta["delete"]


# ─── 2. FastAPI Endpoint Tests ────────────────────────────────────────

class TestQuantumAPI:

    @pytest.fixture(autouse=True)
    def _setup_client(self):
        """Create a fresh TestClient for each test."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import api.quantum_router as qr

        # Reset the singleton so each test starts clean
        qr.GlobalKnowledgeOS._instance = None
        qr.kos = qr.GlobalKnowledgeOS()
        self.kos = qr.kos

        app = FastAPI()
        app.include_router(qr.router)
        self.client = TestClient(app)

    def test_state_returns_503_before_boot(self):
        """Pre-boot requests get a 503."""
        response = self.client.get("/api/v1/quantum/state")
        assert response.status_code == 503

    def test_state_endpoint(self):
        """After boot, /state returns the integer as a string."""
        self.kos.is_booted = True
        self.kos.global_state = 42

        response = self.client.get("/api/v1/quantum/state")
        assert response.status_code == 200

        data = response.json()
        assert data["global_state_integer"] == "42"

    def test_sync_endpoint(self):
        """The /sync endpoint returns correct delta."""
        algebra = GodelStateAlgebra()
        p1 = algebra.get_or_mint_prime("x", "is", "1")
        p2 = algebra.get_or_mint_prime("y", "is", "2")

        self.kos.algebra = algebra
        self.kos.global_state = p1 * p2
        self.kos.is_booted = True

        # Client only has p1
        response = self.client.post(
            "/api/v1/quantum/sync",
            json={"client_state_integer": str(p1)},
        )
        assert response.status_code == 200

        data = response.json()
        assert "y||is||2" in data["delta"]["add"]
        assert data["delta"]["delete"] == []
        assert data["new_global_state"] == str(p1 * p2)
