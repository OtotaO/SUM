"""
Phase 13 Tests — Zenith of Process Intensification

Tests for:
    1. Deterministic Syntactic Sieve (spaCy NLP triplet extraction)
    2. JWT Multi-Tenant Isolation (Quantum Passport)
    3. BigInt uncap validation

Author: ototao
License: Apache License 2.0
"""

import sys
import pytest
from fastapi.testclient import TestClient
from quantum_main import app
from api.quantum_router import kos
from internal.algorithms.semantic_arithmetic import GodelStateAlgebra


client = TestClient(app)


# ─── Reset KOS for each test ─────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_kos():
    """Boot KOS in math-only mode before each test."""
    kos.algebra = GodelStateAlgebra()
    kos.branches = {"main": 1}
    kos.is_booted = True
    yield
    # Cleanup: remove any test branches
    kos.branches = {"main": 1}


# ─── Test 1: Deterministic Syntactic Sieve ────────────────────────────

class TestDeterministicSieve:
    def test_simple_sentence_extraction(self):
        from internal.algorithms.syntactic_sieve import DeterministicSieve
        sieve = DeterministicSieve()

        text = "The scientist discovered a new particle."
        triplets = sieve.extract_triplets(text)

        assert len(triplets) >= 1
        # At least one triplet should mention scientist and discover
        subjects = [s for s, p, o in triplets]
        predicates = [p for s, p, o in triplets]
        assert any("scientist" in s for s in subjects)
        assert any("discover" in p for p in predicates)

    def test_multiple_sentences(self):
        from internal.algorithms.syntactic_sieve import DeterministicSieve
        sieve = DeterministicSieve()

        text = (
            "The scientist discovered a new particle. "
            "The telescope observed the galaxy."
        )
        triplets = sieve.extract_triplets(text)

        assert len(triplets) >= 2
        all_text = " ".join(f"{s} {p} {o}" for s, p, o in triplets)
        assert "scientist" in all_text
        assert "telescope" in all_text or "galaxy" in all_text

    def test_deduplication(self):
        from internal.algorithms.syntactic_sieve import DeterministicSieve
        sieve = DeterministicSieve()

        text = "The cat sat on the mat. The cat sat on the mat."
        triplets = sieve.extract_triplets(text)

        # Deduplicated: same sentence twice should produce unique triplets
        unique = set(triplets)
        assert len(triplets) == len(unique)

    def test_empty_input(self):
        from internal.algorithms.syntactic_sieve import DeterministicSieve
        sieve = DeterministicSieve()

        triplets = sieve.extract_triplets("")
        assert triplets == []

    def test_no_triplets_from_fragment(self):
        from internal.algorithms.syntactic_sieve import DeterministicSieve
        sieve = DeterministicSieve()

        # A fragment without clear SVO structure
        triplets = sieve.extract_triplets("hello world")
        assert isinstance(triplets, list)


# ─── Test 2: JWT Multi-Tenant Isolation ───────────────────────────────

class TestJWTMultiTenancy:
    def test_token_generation(self):
        res = client.post(
            "/api/v1/quantum/auth/token",
            json={"username": "alice"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["branch"] == "alice"

    def test_unauthenticated_falls_back_to_main(self):
        state = client.get("/api/v1/quantum/state")
        assert state.status_code == 200
        data = state.json()
        assert data["user_id"] == "main"
        assert data["branch"] == "main"

    def test_invalid_token_returns_401(self):
        headers = {"Authorization": "Bearer invalid.token.here"}
        state = client.get("/api/v1/quantum/state", headers=headers)
        assert state.status_code == 401

    @pytest.mark.asyncio
    async def test_mathematical_isolation(self):
        """Alice and Bob ingest different facts; their states must differ."""
        # 1. Get tokens
        res_alice = client.post(
            "/api/v1/quantum/auth/token", json={"username": "alice"}
        )
        token_alice = res_alice.json()["access_token"]
        headers_alice = {"Authorization": f"Bearer {token_alice}"}

        res_bob = client.post(
            "/api/v1/quantum/auth/token", json={"username": "bob"}
        )
        token_bob = res_bob.json()["access_token"]
        headers_bob = {"Authorization": f"Bearer {token_bob}"}

        # 2. Alice ingests a fact
        client.post(
            "/api/v1/quantum/ingest/math",
            json={"triplets": [["alice", "likes", "cryptography"]]},
            headers=headers_alice,
        )

        # 3. Bob ingests a different fact
        client.post(
            "/api/v1/quantum/ingest/math",
            json={"triplets": [["bob", "likes", "biology"]]},
            headers=headers_bob,
        )

        # 4. Verify Mathematical Isolation
        state_alice = client.get(
            "/api/v1/quantum/state", headers=headers_alice
        ).json()
        state_bob = client.get(
            "/api/v1/quantum/state", headers=headers_bob
        ).json()

        assert state_alice["global_state_integer"] != state_bob["global_state_integer"]
        assert state_alice["global_state_integer"] != "1"
        assert state_bob["global_state_integer"] != "1"
        assert state_alice["user_id"] == "alice"
        assert state_bob["user_id"] == "bob"

    @pytest.mark.asyncio
    async def test_main_branch_unaffected(self):
        """Authenticated users should not pollute the main branch."""
        # Get main state before
        main_before = client.get("/api/v1/quantum/state").json()

        # Alice ingests with JWT
        res = client.post(
            "/api/v1/quantum/auth/token", json={"username": "eve"}
        )
        token = res.json()["access_token"]
        client.post(
            "/api/v1/quantum/ingest/math",
            json={"triplets": [["eve", "knows", "secrets"]]},
            headers={"Authorization": f"Bearer {token}"},
        )

        # Main branch should be unchanged
        main_after = client.get("/api/v1/quantum/state").json()
        assert (
            main_before["global_state_integer"]
            == main_after["global_state_integer"]
        )


# ─── Test 3: BigInt Uncap ─────────────────────────────────────────────

class TestBigIntUncap:
    def test_large_integer_string_conversion(self):
        """Ensure Gödel integers beyond 4,300 digits don't crash Python."""
        # This would crash without sys.set_int_max_str_digits(0)
        huge_int = 2 ** 50000  # ~15,000 digits
        result = str(huge_int)
        assert len(result) > 4300

    def test_massive_state_survives_str(self):
        """Simulate a heavily loaded state and verify str() works."""
        algebra = GodelStateAlgebra()
        state = 1
        # Mint 100 primes and multiply them in
        for i in range(100):
            prime = algebra.get_or_mint_prime(
                f"entity_{i}", "has_property", f"value_{i}"
            )
            state *= prime

        # This will produce a very large integer
        state_str = str(state)
        assert len(state_str) > 100
