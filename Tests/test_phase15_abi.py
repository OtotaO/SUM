"""
Phase 15 Test Suite: Canonical Semantic ABI

Tests:
    1. Canonical format versioning (version header emitted and parsed)
    2. Structured proof manifest (enriched fields, timestamp)
    3. Signed export/import bundles (round-trip, tamper detection)
    4. Delta bundle compression (novel axioms only)
    5. Multi-hop neighborhood traversal (1-hop vs 2-hop)
    6. JWT-aware API endpoints (/export, /import)

All tests are deterministic, offline, and LLM-free.

Author: ototao
License: Apache License 2.0
"""

import math
import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from quantum_main import app
from api.quantum_router import kos
from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.algorithms.syntactic_sieve import DeterministicSieve

spacy = pytest.importorskip("spacy", reason="spacy not installed")
from internal.ensemble.tome_generator import (
    AutoregressiveTomeGenerator,
    CANONICAL_FORMAT_VERSION,
)
from internal.ensemble.ouroboros import OuroborosVerifier
from internal.infrastructure.canonical_codec import (
    CanonicalCodec,
    InvalidSignatureError,
)


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def boot_kos():
    """Ensure KOS is booted before every test."""
    if not kos.is_booted:
        import asyncio
        asyncio.run(kos.boot_sequence())
    # Ensure Phase 15 components exist even if KOS was booted by earlier tests
    if not hasattr(kos, 'codec') or kos.codec is None:
        from api.quantum_router import SECRET_KEY
        kos.codec = CanonicalCodec(
            kos.algebra,
            kos.tome_generator,
            signing_key=SECRET_KEY,
        )
    yield


@pytest.fixture
def fresh_system():
    """Return a fresh algebra, sieve, tome_generator, ouroboros, codec."""
    algebra = GodelStateAlgebra()
    sieve = DeterministicSieve()
    tome_gen = AutoregressiveTomeGenerator(algebra)
    ouroboros = OuroborosVerifier(algebra, sieve, tome_gen)
    codec = CanonicalCodec(algebra, tome_gen, signing_key="test-key-phase15")
    return algebra, sieve, tome_gen, ouroboros, codec


def _mint_state(algebra, triplets):
    """Helper: encode triplets into a Gödel integer."""
    state = 1
    for s, p, o in triplets:
        prime = algebra.get_or_mint_prime(s, p, o)
        state = math.lcm(state, prime)
    return state


# ─── 1. Canonical Format Versioning ──────────────────────────────

class TestCanonicalVersioning:

    def test_version_header_emitted(self, fresh_system):
        """Canonical tome starts with @canonical_version header."""
        algebra, _, tome_gen, _, _ = fresh_system
        state = _mint_state(algebra, [("alice", "likes", "cats")])
        tome = tome_gen.generate_canonical(state, "Test")
        assert tome.startswith(f"@canonical_version: {CANONICAL_FORMAT_VERSION}")

    def test_version_header_in_empty_state(self, fresh_system):
        """Even empty state tomes include the version header."""
        _, _, tome_gen, _, _ = fresh_system
        tome = tome_gen.generate_canonical(1, "Empty")
        assert f"@canonical_version: {CANONICAL_FORMAT_VERSION}" in tome

    def test_version_parsed_by_verifier(self, fresh_system):
        """Verifier extracts and records the format version."""
        algebra, _, _, ouroboros, _ = fresh_system
        state = _mint_state(algebra, [("bob", "knows", "python")])
        proof = ouroboros.verify_from_state(state)
        assert proof.format_version == CANONICAL_FORMAT_VERSION

    def test_unknown_version_still_parses(self, fresh_system):
        """Verifier handles tomes with unknown versions gracefully."""
        algebra, _, tome_gen, ouroboros, _ = fresh_system
        state = _mint_state(algebra, [("x", "y", "z")])
        # Get canonical and tamper the version
        tome = tome_gen.generate_canonical(state, "Test")
        tampered = tome.replace(
            f"@canonical_version: {CANONICAL_FORMAT_VERSION}",
            "@canonical_version: 99.0.0"
        )
        _, _, version = ouroboros._reconstruct_from_canonical(tampered)
        assert version == "99.0.0"


# ─── 2. Structured Proof Manifest ────────────────────────────────

class TestProofManifest:

    def test_enriched_fields_present(self, fresh_system):
        """Proof contains format_version, proof_mode, timestamp, digit counts."""
        algebra, _, _, ouroboros, _ = fresh_system
        state = _mint_state(algebra, [("alice", "age", "30")])
        proof = ouroboros.verify_from_state(state)

        assert proof.format_version == CANONICAL_FORMAT_VERSION
        assert proof.proof_mode == "canonical"
        assert proof.state_a_digits > 0
        assert proof.state_b_digits > 0

    def test_timestamp_is_iso8601(self, fresh_system):
        """Proof timestamp is a valid ISO 8601 string."""
        algebra, _, _, ouroboros, _ = fresh_system
        state = _mint_state(algebra, [("x", "y", "z")])
        proof = ouroboros.verify_from_state(state)
        # Should parse without error
        dt = datetime.fromisoformat(proof.timestamp)
        assert dt.year >= 2026

    def test_proof_dict_has_all_keys(self, fresh_system):
        """proof_to_dict includes all enriched keys."""
        algebra, _, _, ouroboros, _ = fresh_system
        state = _mint_state(algebra, [("a", "b", "c")])
        proof = ouroboros.verify_from_state(state)
        d = ouroboros.proof_to_dict(proof)

        expected_keys = {
            "is_conserved", "format_version", "proof_mode", "timestamp",
            "state_a_digits", "state_b_digits", "source_axiom_count",
            "reconstructed_axiom_count", "missing_axioms", "extra_axioms",
            "states_match",
        }
        assert expected_keys <= set(d.keys())

    def test_conservation_with_enriched_proof(self, fresh_system):
        """Full conservation still works with enriched proof structure."""
        algebra, _, _, ouroboros, _ = fresh_system
        triplets = [
            ("earth", "orbits", "sun"),
            ("moon", "orbits", "earth"),
            ("mars", "has", "atmosphere"),
        ]
        state = _mint_state(algebra, triplets)
        proof = ouroboros.verify_from_state(state)

        assert proof.is_conserved
        assert proof.source_axiom_count == 3
        assert proof.state_a_digits == proof.state_b_digits


# ─── 3. Signed Export/Import Bundles ─────────────────────────────

class TestSignedBundle:

    def test_export_import_round_trip(self, fresh_system):
        """Export then import preserves the state integer."""
        algebra, _, _, _, codec = fresh_system
        state = _mint_state(algebra, [
            ("alice", "knows", "bob"),
            ("bob", "knows", "charlie"),
        ])
        bundle = codec.export_bundle(state, branch="test")
        imported_state = codec.import_bundle(bundle)
        assert imported_state == state

    def test_tampered_bundle_rejected(self, fresh_system):
        """Modifying bundle content causes signature failure."""
        algebra, _, _, _, codec = fresh_system
        state = _mint_state(algebra, [("x", "y", "z")])
        bundle = codec.export_bundle(state, branch="test")

        # Tamper with the state integer
        bundle["state_integer"] = "42"

        with pytest.raises(InvalidSignatureError):
            codec.import_bundle(bundle)

    def test_missing_fields_rejected(self, fresh_system):
        """Bundle with missing required fields raises ValueError."""
        _, _, _, _, codec = fresh_system
        with pytest.raises(ValueError, match="missing required fields"):
            codec.import_bundle({"state_integer": "1"})

    def test_bundle_contains_version(self, fresh_system):
        """Bundle includes bundle_version and canonical_format_version."""
        algebra, _, _, _, codec = fresh_system
        state = _mint_state(algebra, [("a", "b", "c")])
        bundle = codec.export_bundle(state)
        assert "bundle_version" in bundle
        assert bundle["canonical_format_version"] == CANONICAL_FORMAT_VERSION

    def test_wrong_key_rejected(self, fresh_system):
        """Bundle signed with different key is rejected."""
        algebra, _, tome_gen, _, _ = fresh_system
        state = _mint_state(algebra, [("a", "b", "c")])

        codec_a = CanonicalCodec(algebra, tome_gen, signing_key="key-a")
        codec_b = CanonicalCodec(algebra, tome_gen, signing_key="key-b")

        bundle = codec_a.export_bundle(state)
        with pytest.raises(InvalidSignatureError):
            codec_b.import_bundle(bundle)


# ─── 4. Delta Bundle Compression ─────────────────────────────────

class TestDeltaBundle:

    def test_delta_contains_only_novel(self, fresh_system):
        """Delta bundle encodes only axioms missing from source."""
        algebra, _, _, _, codec = fresh_system
        state_a = _mint_state(algebra, [("a", "knows", "b")])
        state_b = _mint_state(algebra, [
            ("a", "knows", "b"),
            ("c", "knows", "d"),
        ])

        delta = codec.compress_delta(state_a, state_b)
        delta_state = int(delta["state_integer"])

        # Delta should contain c||knows||d but NOT a||knows||b
        p_cd = algebra.axiom_to_prime["c||knows||d"]
        p_ab = algebra.axiom_to_prime["a||knows||b"]
        assert delta_state % p_cd == 0
        assert delta_state % p_ab != 0

    def test_identical_states_yield_empty_delta(self, fresh_system):
        """Delta of identical states is 1 (empty)."""
        algebra, _, _, _, codec = fresh_system
        state = _mint_state(algebra, [("x", "y", "z")])
        delta = codec.compress_delta(state, state)
        assert int(delta["state_integer"]) == 1


# ─── 5. Multi-Hop Neighborhood ───────────────────────────────────

class TestMultiHop:

    def test_1hop_gets_direct_neighbors(self, fresh_system):
        """1-hop returns only direct edges of the queried node."""
        algebra, _, _, _, _ = fresh_system
        state = _mint_state(algebra, [
            ("alice", "knows", "bob"),
            ("bob", "knows", "charlie"),
            ("charlie", "knows", "dave"),
        ])

        ctx = algebra.get_quantum_neighborhood(state, ["alice"], hops=1)
        # Should include alice→bob edge
        p_ab = algebra.axiom_to_prime["alice||knows||bob"]
        assert ctx % p_ab == 0

        # Should NOT include charlie→dave (2 hops away)
        p_cd = algebra.axiom_to_prime["charlie||knows||dave"]
        assert ctx % p_cd != 0

    def test_2hop_discovers_neighbors_of_neighbors(self, fresh_system):
        """2-hop expands to include edges of discovered neighbors."""
        algebra, _, _, _, _ = fresh_system
        state = _mint_state(algebra, [
            ("alice", "knows", "bob"),
            ("bob", "knows", "charlie"),
            ("charlie", "knows", "dave"),
        ])

        ctx = algebra.get_quantum_neighborhood(state, ["alice"], hops=2)
        # 2 hops from alice: alice→bob + bob→charlie
        p_ab = algebra.axiom_to_prime["alice||knows||bob"]
        p_bc = algebra.axiom_to_prime["bob||knows||charlie"]
        assert ctx % p_ab == 0
        assert ctx % p_bc == 0

    def test_3hop_reaches_full_chain(self, fresh_system):
        """3-hop traversal covers the full linear chain."""
        algebra, _, _, _, _ = fresh_system
        state = _mint_state(algebra, [
            ("alice", "knows", "bob"),
            ("bob", "knows", "charlie"),
            ("charlie", "knows", "dave"),
        ])

        ctx = algebra.get_quantum_neighborhood(state, ["alice"], hops=3)
        # All 3 edges should be reachable
        for axiom in ["alice||knows||bob", "bob||knows||charlie", "charlie||knows||dave"]:
            p = algebra.axiom_to_prime[axiom]
            assert ctx % p == 0

    def test_0hop_returns_empty(self, fresh_system):
        """0-hop returns 1 (empty state)."""
        algebra, _, _, _, _ = fresh_system
        state = _mint_state(algebra, [("a", "b", "c")])
        ctx = algebra.get_quantum_neighborhood(state, ["a"], hops=0)
        assert ctx == 1


# ─── 6. API Endpoints ────────────────────────────────────────────

class TestPhase15API:

    @pytest.fixture
    def client_with_auth(self):
        """TestClient with JWT auth headers."""
        import jwt as pyjwt
        from api.quantum_router import SECRET_KEY
        client = TestClient(app)
        token = pyjwt.encode(
            {"sub": "test-phase15-user"}, SECRET_KEY, algorithm="HS256"
        )
        return client, {"Authorization": f"Bearer {token}"}

    def test_export_returns_signed_bundle(self, client_with_auth):
        """GET /export returns a valid signed bundle."""
        client, headers = client_with_auth
        res = client.post("/api/v1/quantum/export", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert "signature" in data
        assert "canonical_tome" in data
        assert "bundle_version" in data

    def test_import_with_valid_bundle(self, client_with_auth):
        """POST /import with valid bundle returns success."""
        client, headers = client_with_auth

        # First export
        export_res = client.post("/api/v1/quantum/export", headers=headers)
        bundle = export_res.json()

        # Then import
        import_res = client.post(
            "/api/v1/quantum/import",
            headers=headers,
            json={"bundle": bundle},
        )
        assert import_res.status_code == 200
        data = import_res.json()
        assert data["imported"] is True

    def test_import_tampered_bundle_rejected(self, client_with_auth):
        """POST /import with tampered bundle returns 403."""
        client, headers = client_with_auth

        export_res = client.post("/api/v1/quantum/export", headers=headers)
        bundle = export_res.json()
        bundle["state_integer"] = "999999"  # Tamper

        import_res = client.post(
            "/api/v1/quantum/import",
            headers=headers,
            json={"bundle": bundle},
        )
        assert import_res.status_code == 403
