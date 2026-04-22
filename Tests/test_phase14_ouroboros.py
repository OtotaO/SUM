"""
Phase 14 Tests — The Ouroboros Protocol (Lossless Semantic Rehydration)

Deterministic, offline tests for:
    1. Canonical Tome Generation (proof mode)
    2. Ouroboros Round-Trip Conservation
    3. Epistemic Delta Generation
    4. API endpoints (/rehydrate, /learn, /ouroboros/verify)

All tests run in math-only mode — no LLM or network dependencies.

Author: ototao
License: Apache License 2.0
"""

import math
import pytest
from fastapi.testclient import TestClient
from quantum_main import app
from api.quantum_router import kos
from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve

spacy = pytest.importorskip("spacy", reason="spacy not installed")
from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
from sum_engine_internal.ensemble.ouroboros import OuroborosVerifier

client = TestClient(app)


# ─── Reset KOS for each test ─────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_kos():
    """Boot KOS in math-only mode before each test."""
    kos.algebra = GodelStateAlgebra()
    kos.branches = {"main": 1}
    kos.is_booted = True
    kos.sieve = DeterministicSieve()
    kos.tome_generator = AutoregressiveTomeGenerator(kos.algebra)
    kos.ouroboros = OuroborosVerifier(kos.algebra, kos.sieve, kos.tome_generator)
    yield
    kos.branches = {"main": 1}


# ─── Helpers ──────────────────────────────────────────────────────────

def ingest_triplets(triplets, branch="main"):
    """Ingest triplets directly into a branch via math."""
    state = kos.branches.get(branch, 1)
    for s, p, o in triplets:
        prime = kos.algebra.get_or_mint_prime(s, p, o)
        if state % prime != 0:
            state = math.lcm(state, prime)
    kos.branches[branch] = state
    return state


# ═════════════════════════════════════════════════════════════════════
# Test 1: Canonical Tome Generator
# ═════════════════════════════════════════════════════════════════════

class TestCanonicalTomeGenerator:
    def test_empty_state_returns_empty_message(self):
        gen = AutoregressiveTomeGenerator(GodelStateAlgebra())
        tome = gen.generate_canonical(1)
        assert "empty" in tome.lower()

    def test_single_axiom_round_trips(self):
        algebra = GodelStateAlgebra()
        prime = algebra.get_or_mint_prime("alice", "likes", "cats")
        state = prime

        gen = AutoregressiveTomeGenerator(algebra)
        tome = gen.generate_canonical(state)

        assert "alice" in tome.lower()
        assert "likes" in tome.lower()
        assert "cats" in tome.lower()

    def test_multiple_subjects_grouped(self):
        algebra = GodelStateAlgebra()
        state = 1
        for s, p, o in [
            ("alice", "likes", "cats"),
            ("alice", "age", "30"),
            ("bob", "likes", "dogs"),
        ]:
            prime = algebra.get_or_mint_prime(s, p, o)
            state = math.lcm(state, prime)

        gen = AutoregressiveTomeGenerator(algebra)
        tome = gen.generate_canonical(state, "Test Codex")

        assert "# Test Codex" in tome
        assert "## Alice" in tome
        assert "## Bob" in tome

    def test_output_is_deterministic(self):
        """Same state always produces the same canonical text."""
        algebra = GodelStateAlgebra()
        state = 1
        for s, p, o in [
            ("x", "has", "1"),
            ("y", "has", "2"),
            ("x", "has", "3"),
        ]:
            prime = algebra.get_or_mint_prime(s, p, o)
            state = math.lcm(state, prime)

        gen = AutoregressiveTomeGenerator(algebra)
        tome_a = gen.generate_canonical(state)
        tome_b = gen.generate_canonical(state)
        assert tome_a == tome_b

    def test_extract_active_axioms(self):
        algebra = GodelStateAlgebra()
        p1 = algebra.get_or_mint_prime("a", "is", "b")
        p2 = algebra.get_or_mint_prime("c", "is", "d")
        state = p1 * p2

        gen = AutoregressiveTomeGenerator(algebra)
        active = gen.extract_active_axioms(state)
        assert len(active) == 2
        assert "a||is||b" in active
        assert "c||is||d" in active


# ═════════════════════════════════════════════════════════════════════
# Test 2: Ouroboros Verifier — Canonical Round-Trip
# ═════════════════════════════════════════════════════════════════════

class TestOuroborosVerifier:
    def test_lossless_round_trip_simple(self):
        """Encode manually → canonical decode → re-encode must equal."""
        algebra = GodelStateAlgebra()
        sieve = DeterministicSieve()

        # Build a state from known triplets
        state = 1
        for s, p, o in [("alice", "likes", "cats"), ("bob", "likes", "dogs")]:
            prime = algebra.get_or_mint_prime(s, p, o)
            state = math.lcm(state, prime)

        gen = AutoregressiveTomeGenerator(algebra)
        verifier = OuroborosVerifier(algebra, sieve, gen)

        proof = verifier.verify_from_state(state)
        assert proof.is_conserved is True
        assert proof.source_state == proof.reconstructed_state
        assert proof.missing_axioms == []
        assert proof.extra_axioms == []

    def test_conservation_diagnostics(self):
        """Proof object contains expected diagnostic fields."""
        algebra = GodelStateAlgebra()
        sieve = DeterministicSieve()

        state = 1
        for s, p, o in [("sun", "is", "star")]:
            prime = algebra.get_or_mint_prime(s, p, o)
            state = math.lcm(state, prime)

        gen = AutoregressiveTomeGenerator(algebra)
        verifier = OuroborosVerifier(algebra, sieve, gen)
        proof = verifier.verify_from_state(state)

        assert proof.source_axiom_count == 1
        assert proof.canonical_tome != ""
        assert "sun" in proof.canonical_tome.lower()

    def test_proof_to_dict_serialization(self):
        algebra = GodelStateAlgebra()
        sieve = DeterministicSieve()
        prime = algebra.get_or_mint_prime("x", "has", "y")

        gen = AutoregressiveTomeGenerator(algebra)
        verifier = OuroborosVerifier(algebra, sieve, gen)
        proof = verifier.verify_from_state(prime)
        d = verifier.proof_to_dict(proof)

        assert "is_conserved" in d
        assert "source_axiom_count" in d
        assert "states_match" in d
        assert d["source_axiom_count"] == 1

    def test_multiple_axioms_conservation(self):
        """10 axioms round-trip losslessly through canonical path."""
        algebra = GodelStateAlgebra()
        sieve = DeterministicSieve()

        state = 1
        for i in range(10):
            prime = algebra.get_or_mint_prime(
                f"entity{i}", "has_property", f"value{i}"
            )
            state = math.lcm(state, prime)

        gen = AutoregressiveTomeGenerator(algebra)
        verifier = OuroborosVerifier(algebra, sieve, gen)
        proof = verifier.verify_from_state(state)

        assert proof.is_conserved is True
        assert proof.source_axiom_count == 10
        assert proof.reconstructed_axiom_count == 10


# ═════════════════════════════════════════════════════════════════════
# Test 3: Epistemic Delta Computation
# ═════════════════════════════════════════════════════════════════════

class TestEpistemicDelta:
    def test_delta_removes_known_facts(self):
        """delta = topic // gcd(topic, user) should exclude shared facts."""
        algebra = GodelStateAlgebra()

        # Shared fact
        p_shared = algebra.get_or_mint_prime("physics", "has", "gravity")
        # Topic-only fact
        p_topic = algebra.get_or_mint_prime("physics", "has", "entropy")
        # User-only fact
        p_user = algebra.get_or_mint_prime("user", "knows", "algebra")

        topic_state = p_shared * p_topic
        user_state = p_shared * p_user

        delta = topic_state // math.gcd(topic_state, user_state)

        # Delta should only contain the topic-exclusive fact
        assert delta % p_topic == 0  # entropy IS in the delta
        assert delta % p_shared != 0  # gravity is NOT (already known)

    def test_fully_known_returns_1(self):
        """If user already knows everything, delta = 1."""
        algebra = GodelStateAlgebra()
        p1 = algebra.get_or_mint_prime("a", "is", "b")
        p2 = algebra.get_or_mint_prime("c", "is", "d")

        topic = p1 * p2
        user = p1 * p2  # user knows everything

        delta = topic // math.gcd(topic, user)
        assert delta == 1


# ═════════════════════════════════════════════════════════════════════
# Test 4: API Endpoints
# ═════════════════════════════════════════════════════════════════════

class TestPhase14API:
    def test_rehydrate_empty_state(self):
        res = client.post(
            "/api/v1/quantum/rehydrate",
            json={"title": "Test Tome"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "empty" in data["tome"].lower()
        assert data["axiom_count"] == 0

    def test_rehydrate_with_data(self):
        ingest_triplets([
            ("alice", "likes", "cats"),
            ("bob", "likes", "dogs"),
        ])
        res = client.post(
            "/api/v1/quantum/rehydrate",
            json={"title": "My Codex", "mode": "proof"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "alice" in data["tome"].lower()
        assert "bob" in data["tome"].lower()
        assert data["axiom_count"] == 2
        assert data["mode"] == "proof"

    def test_learn_no_topic_knowledge(self):
        res = client.post(
            "/api/v1/quantum/learn",
            json={"target_topic_node": "nonexistent_entity"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "no knowledge" in data["educational_tome"].lower()

    def test_learn_with_delta(self):
        # Ingest topic knowledge into main
        ingest_triplets([
            ("quantum", "has", "superposition"),
            ("quantum", "has", "entanglement"),
        ])

        # Get a JWT for alice (who knows nothing)
        token_res = client.post(
            "/api/v1/quantum/auth/token",
            json={"username": "alice"},
        )
        token = token_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Alice learns about quantum
        res = client.post(
            "/api/v1/quantum/learn",
            json={"target_topic_node": "quantum"},
            headers=headers,
        )
        assert res.status_code == 200
        data = res.json()
        assert data["delta_axiom_count"] == 2

    def test_ouroboros_verify_endpoint(self):
        # First ingest some facts so the algebra has primes
        ingest_triplets([("sun", "is", "star")])

        res = client.post(
            "/api/v1/quantum/ouroboros/verify",
            json={"text": "The sun is star."},
        )
        assert res.status_code == 200
        data = res.json()
        assert "is_conserved" in data
        assert "canonical_tome" in data
        assert data["source_axiom_count"] >= 1
