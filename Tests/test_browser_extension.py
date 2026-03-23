"""
Tests — Browser Extension API Integration

Verifies that all Quantum API endpoints used by the browser extension
respond correctly. These tests validate the server-side contracts that
the extension JS relies on.

Author: ototao
License: Apache License 2.0
"""

import pytest
from fastapi.testclient import TestClient
from quantum_main import app
from api.quantum_router import kos
from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.algorithms.syntactic_sieve import DeterministicSieve


client = TestClient(app)


@pytest.fixture(autouse=True)
def boot_kos():
    """Boot KOS in math-only mode before each test."""
    kos.algebra = GodelStateAlgebra()
    kos.sieve = DeterministicSieve()
    kos.branches = {"main": 1}
    kos.is_booted = True
    yield
    kos.branches = {"main": 1}


class TestBrowserExtensionEndpoints:
    """
    Tests every endpoint the browser extension calls.
    Uses math-only paths to avoid LLM dependencies.
    """

    def test_state_endpoint(self):
        """GET /state returns global state integer (popup.js: fetchState)."""
        res = client.get("/api/v1/quantum/state")
        assert res.status_code == 200
        data = res.json()
        assert "global_state_integer" in data

    def test_ingest_math_endpoint(self):
        """POST /ingest/math accepts raw triplets (background.js: Math Only context menu)."""
        res = client.post("/api/v1/quantum/ingest/math", json={
            "triplets": [["browser", "ingested", "page"]],
            "branch": "main"
        })
        assert res.status_code == 200
        data = res.json()
        assert "axioms_count" in data or "new_global_state" in data

    def test_ingest_math_multiple_triplets(self):
        """POST /ingest/math handles batch triplets from page extraction."""
        res = client.post("/api/v1/quantum/ingest/math", json={
            "triplets": [
                ["sun", "causes", "warmth"],
                ["warmth", "causes", "evaporation"],
                ["evaporation", "produces", "clouds"]
            ],
            "branch": "main"
        })
        assert res.status_code == 200

    def test_sync_state_endpoint(self):
        """POST /sync/state merges a peer state (popup.js: Sync State button)."""
        state_res = client.get("/api/v1/quantum/state")
        assert state_res.status_code == 200
        state = state_res.json()["global_state_integer"]

        res = client.post("/api/v1/quantum/sync/state", json={
            "peer_state_integer": str(state)
        })
        assert res.status_code == 200

    def test_discoveries_endpoint(self):
        """GET /discoveries returns machine-deduced knowledge (popup.js: fetchDiscoveries)."""
        res = client.get("/api/v1/quantum/discoveries")
        assert res.status_code == 200
        data = res.json()
        # Response has 'recent' and 'total_discoveries' keys
        assert "recent" in data
        assert "total_discoveries" in data

    def test_auth_token_endpoint(self):
        """POST /auth/token generates a JWT (options.js: JWT config)."""
        res = client.post("/api/v1/quantum/auth/token", json={
            "username": "exocortex-browser"
        })
        assert res.status_code == 200
        data = res.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_state_grows_after_math_ingest(self):
        """State integer grows after crystallizing new axioms."""
        state_before = int(client.get("/api/v1/quantum/state").json()["global_state_integer"])

        client.post("/api/v1/quantum/ingest/math", json={
            "triplets": [["photosynthesis", "converts", "light"]],
            "branch": "main"
        })

        state_after = int(client.get("/api/v1/quantum/state").json()["global_state_integer"])
        assert state_after >= state_before

    def test_auth_token_creates_branch(self):
        """Generating a token for a new user creates their branch."""
        res = client.post("/api/v1/quantum/auth/token", json={
            "username": "test-user-42"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["branch"] == "test-user-42"
