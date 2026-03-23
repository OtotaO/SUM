"""
Phase 21 Tests — Predicate Canonicalization + Knowledge Retrieval (/ask)

Tests:
    1. Predicate canonicalization: synonym → canonical mapping
    2. Predicate canonicalization: pass-through for unknown predicates
    3. GodelStateAlgebra integration: synonyms produce same prime
    4. /ask endpoint: keyword matching on ingested axioms
    5. /ask endpoint: empty state returns no matches
    6. /ask endpoint: math-only (no LLM) still works
    7. CausalDiscovery benefits: canonicalized predicates enable transitivity
"""

import math
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from internal.algorithms.predicate_canon import canonicalize, CANONICAL_MAP
from internal.algorithms.semantic_arithmetic import GodelStateAlgebra


# ─── Predicate Canonicalization Unit Tests ────────────────────────────

class TestPredicateCanon:
    """Tests for predicate_canon.canonicalize()."""

    def test_synonym_maps_to_canonical(self):
        """Known synonyms must map to their canonical form."""
        assert canonicalize("leads_to") == "causes"
        assert canonicalize("triggers") == "causes"
        assert canonicalize("results_in") == "causes"
        assert canonicalize("prevents") == "inhibits"
        assert canonicalize("blocks") == "inhibits"
        assert canonicalize("suggests") == "implies"
        assert canonicalize("needs") == "requires"
        assert canonicalize("is_type_of") == "is_a"
        assert canonicalize("cures") == "treats"

    def test_canonical_predicates_unchanged(self):
        """Canonical forms pass through unchanged."""
        assert canonicalize("causes") == "causes"
        assert canonicalize("inhibits") == "inhibits"
        assert canonicalize("implies") == "implies"
        assert canonicalize("is_a") == "is_a"

    def test_unknown_predicates_pass_through(self):
        """Unknown predicates are returned as-is (open-world)."""
        assert canonicalize("loves") == "loves"
        assert canonicalize("orbits_around") == "orbits_around"
        assert canonicalize("flavors") == "flavors"

    def test_whitespace_normalization(self):
        """Whitespace is stripped and spaces become underscores."""
        assert canonicalize("  leads to  ") == "causes"
        assert canonicalize("results in") == "causes"

    def test_case_normalization(self):
        """Predicates are lowercased before lookup."""
        assert canonicalize("LEADS_TO") == "causes"
        assert canonicalize("Prevents") == "inhibits"

    def test_all_mappings_resolve(self):
        """Every entry in CANONICAL_MAP resolves to a valid string."""
        for synonym, canonical in CANONICAL_MAP.items():
            assert isinstance(canonical, str)
            assert len(canonical) > 0
            assert canonicalize(synonym) == canonical


# ─── GodelStateAlgebra Integration ────────────────────────────────────

class TestCanonInAlgebra:
    """Canonical predicates produce identical primes in the algebra."""

    def test_synonym_same_prime(self):
        """'leads_to' and 'causes' must mint the same prime."""
        algebra = GodelStateAlgebra()
        p1 = algebra.get_or_mint_prime("sun", "causes", "warmth")
        p2 = algebra.get_or_mint_prime("sun", "leads_to", "warmth")
        assert p1 == p2, (
            f"Synonym predicates should produce the same prime, "
            f"got {p1} vs {p2}"
        )

    def test_multiple_synonyms_same_prime(self):
        """All synonyms in the 'causes' family must converge."""
        algebra = GodelStateAlgebra()
        base = algebra.get_or_mint_prime("rain", "causes", "flood")
        for syn in ["leads_to", "triggers", "results_in", "produces"]:
            p = algebra.get_or_mint_prime("rain", syn, "flood")
            assert p == base, f"'{syn}' did not canonicalize to 'causes'"

    def test_different_canonical_different_prime(self):
        """Different canonical predicates must produce different primes."""
        algebra = GodelStateAlgebra()
        p_causes = algebra.get_or_mint_prime("x", "causes", "y")
        p_inhibits = algebra.get_or_mint_prime("x", "inhibits", "y")
        assert p_causes != p_inhibits


# ─── CausalDiscovery Benefit ──────────────────────────────────────────

class TestCausalWithCanon:
    """CausalDiscovery now works through canonicalized predicates."""

    def test_transitive_after_canonicalization(self):
        """If A leads_to B (→causes) and B triggers C (→causes), discover A→C."""
        from internal.algorithms.causal_discovery import CausalDiscoveryEngine

        algebra = GodelStateAlgebra()
        # These will be canonicalized to "causes"
        p1 = algebra.get_or_mint_prime("chemical_x", "leads_to", "enzyme_y")
        p2 = algebra.get_or_mint_prime("enzyme_y", "triggers", "reaction_z")
        state = math.lcm(p1, p2)

        engine = CausalDiscoveryEngine(algebra)
        discoveries = engine.sweep_for_discoveries(state)

        # Should find chemical_x → causes → reaction_z
        found = any(
            s == "chemical_x" and o == "reaction_z"
            for s, p, o in discoveries
        )
        assert found, (
            f"Expected transitive discovery chemical_x→reaction_z, "
            f"got {discoveries}"
        )


# ─── /ask Endpoint Tests ─────────────────────────────────────────────

class TestAskEndpoint:
    """Integration tests for POST /ask."""

    @pytest.fixture
    def booted_app(self):
        """Create a test client with a booted KOS."""
        from api.quantum_router import kos
        orig_booted = kos.is_booted
        orig_branches = kos.branches
        orig_algebra = kos.algebra

        kos.is_booted = True
        kos.branches = {"main": 1}
        kos.algebra = GodelStateAlgebra()

        # Ingest some test axioms
        p1 = kos.algebra.get_or_mint_prime("python", "is_a", "programming_language")
        p2 = kos.algebra.get_or_mint_prime("python", "causes", "readability")
        p3 = kos.algebra.get_or_mint_prime("rust", "enables", "memory_safety")
        kos.branches["main"] = math.lcm(p1, math.lcm(p2, p3))

        from fastapi.testclient import TestClient
        from quantum_main import app
        yield TestClient(app)

        kos.is_booted = orig_booted
        kos.branches = orig_branches
        kos.algebra = orig_algebra

    def test_ask_returns_matches(self, booted_app):
        """Asking about 'python' should return python-related axioms."""
        resp = booted_app.post(
            "/api/v1/quantum/ask",
            json={"question": "What is python?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["question"] == "What is python?"
        assert len(data["matches"]) > 0
        subjects = [m["subject"] for m in data["matches"]]
        assert "python" in subjects

    def test_ask_empty_state(self, booted_app):
        """Empty state returns no matches."""
        from api.quantum_router import kos
        kos.branches["main"] = 1
        resp = booted_app.post(
            "/api/v1/quantum/ask",
            json={"question": "What is python?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "empty"
        assert data["matches"] == []

    def test_ask_mode_is_keyword(self, booted_app):
        """Without VectorBridge, mode should be 'keyword'."""
        resp = booted_app.post(
            "/api/v1/quantum/ask",
            json={"question": "Tell me about rust"},
        )
        data = resp.json()
        assert data["mode"] == "keyword"

    def test_ask_no_match(self, booted_app):
        """Querying for something not in state returns no matches."""
        resp = booted_app.post(
            "/api/v1/quantum/ask",
            json={"question": "What is quantum gravity?"},
        )
        data = resp.json()
        assert len(data["matches"]) == 0

    def test_ask_answer_contains_facts(self, booted_app):
        """Answer string should contain matched facts."""
        resp = booted_app.post(
            "/api/v1/quantum/ask",
            json={"question": "What is rust?"},
        )
        data = resp.json()
        assert "rust" in data["answer"].lower()
