"""
Phase 25 Tests — Semantic Deduplication

Tests:
    1-3.  Normalization: lowercase, whitespace, predicate synonyms
    4-5.  Jaccard similarity: high overlap, no overlap
    6-7.  Levenshtein ratio: near-identical, completely different
    8-9.  Predicate synonym dedup: orbits ≈ revolves_around, lives_in ≈ resides_in
    10.   Object normalization: "New York" ≈ "new york"
    11.   Below threshold: distinct axioms stay separate
    12.   Exact duplicate detection
    13-14. API: duplicate skipped, distinct accepted
    15.   Batch dedup: duplicates within a single batch
"""

import math
import pytest

from internal.ensemble.semantic_dedup import (
    SemanticDeduplicator,
    DedupResult,
    PREDICATE_SYNONYMS,
)
from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.infrastructure.akashic_ledger import AkashicLedger


# ─── Normalization Tests ─────────────────────────────────────────────

class TestNormalization:
    def setup_method(self):
        self.dedup = SemanticDeduplicator()

    def test_lowercase(self):
        assert self.dedup.normalize("EARTH||ORBITS||SUN") == "earth||orbits||sun"

    def test_whitespace_collapse(self):
        assert self.dedup.normalize("new  york||is_in||united  states") == "new_york||located_in||united_states"

    def test_predicate_synonym(self):
        assert self.dedup.normalize("earth||revolves_around||sun") == "earth||orbits||sun"

    def test_lives_in_canonical(self):
        assert self.dedup.normalize("alice||lives_in||london") == "alice||located_in||london"

    def test_created_by_synonym(self):
        assert self.dedup.normalize("painting||made_by||picasso") == "painting||created_by||picasso"


# ─── Similarity Tests ────────────────────────────────────────────────

class TestSimilarity:
    def test_jaccard_identical(self):
        assert SemanticDeduplicator.jaccard_similarity(
            "earth||orbits||sun", "earth||orbits||sun"
        ) == 1.0

    def test_jaccard_high_overlap(self):
        sim = SemanticDeduplicator.jaccard_similarity(
            "earth||orbits||sun", "earth||revolves||sun"
        )
        assert sim >= 0.5

    def test_jaccard_no_overlap(self):
        sim = SemanticDeduplicator.jaccard_similarity(
            "earth||orbits||sun", "alice||likes||cats"
        )
        assert sim == 0.0

    def test_levenshtein_identical(self):
        assert SemanticDeduplicator.levenshtein_ratio("hello", "hello") == 1.0

    def test_levenshtein_similar(self):
        ratio = SemanticDeduplicator.levenshtein_ratio("kitten", "sitting")
        assert 0.4 < ratio < 0.7

    def test_levenshtein_different(self):
        ratio = SemanticDeduplicator.levenshtein_ratio("abc", "xyz")
        assert ratio == 0.0


# ─── Deduplication Tests ─────────────────────────────────────────────

class TestDeduplication:
    def setup_method(self):
        self.dedup = SemanticDeduplicator()

    def test_exact_duplicate(self):
        result = self.dedup.deduplicate(
            "earth||orbits||sun",
            ["earth||orbits||sun", "mars||has||moons"],
        )
        assert result.is_duplicate is True
        assert result.method == "exact"
        assert result.similarity == 1.0

    def test_predicate_synonym_detected(self):
        """revolves_around should be deduped against orbits."""
        result = self.dedup.deduplicate(
            "earth||revolves_around||sun",
            ["earth||orbits||sun"],
        )
        assert result.is_duplicate is True
        assert result.duplicate_of == "earth||orbits||sun"

    def test_lives_in_resides_in(self):
        """lives_in and resides_in are both located_in."""
        result = self.dedup.deduplicate(
            "alice||resides_in||london",
            ["alice||lives_in||london"],
        )
        assert result.is_duplicate is True

    def test_case_insensitive_dedup(self):
        result = self.dedup.deduplicate(
            "Earth||Orbits||Sun",
            ["earth||orbits||sun"],
        )
        assert result.is_duplicate is True

    def test_distinct_axioms_not_deduped(self):
        """Completely different axioms should NOT be flagged."""
        result = self.dedup.deduplicate(
            "earth||orbits||sun",
            ["alice||likes||cats"],
        )
        assert result.is_duplicate is False
        assert result.method == "none"

    def test_same_subject_different_predicate_object(self):
        """Same subject but different predicate+object = not a duplicate."""
        result = self.dedup.deduplicate(
            "earth||has||atmosphere",
            ["earth||orbits||sun"],
        )
        assert result.is_duplicate is False

    def test_empty_existing(self):
        result = self.dedup.deduplicate(
            "earth||orbits||sun",
            [],
        )
        assert result.is_duplicate is False

    def test_find_near_duplicates_returns_sorted(self):
        results = self.dedup.find_near_duplicates(
            "earth||revolves_around||sun",
            ["earth||orbits||sun", "mars||orbits||sun", "alice||likes||cats"],
        )
        # First match should be the best
        assert len(results) > 0
        assert results[0][0] == "earth||orbits||sun"


# ─── API Integration Tests ──────────────────────────────────────────

class TestAPIDedupIntegration:
    """Test that /ingest/math skips near-duplicates."""

    @pytest.fixture
    def booted_app(self, tmp_path):
        from api.quantum_router import kos
        orig_ledger = kos.ledger
        orig_booted = kos.is_booted
        orig_branches = kos.branches
        orig_algebra = kos.algebra

        db_path = str(tmp_path / "test_dedup.db")
        kos.ledger = AkashicLedger(db_path)
        kos.is_booted = True
        kos.branches = {"main": 1}
        kos.algebra = GodelStateAlgebra()

        from fastapi.testclient import TestClient
        from quantum_main import app
        yield TestClient(app)

        kos.ledger = orig_ledger
        kos.is_booted = orig_booted
        kos.branches = orig_branches
        kos.algebra = orig_algebra

    def test_duplicate_skipped_on_second_ingest(self, booted_app):
        """Ingesting the same axiom twice: second is skipped."""
        # First ingest
        resp1 = booted_app.post(
            "/api/v1/quantum/ingest/math",
            json={"triplets": [["earth", "orbits", "sun"]]},
        )
        assert resp1.status_code == 200
        assert resp1.json()["axioms_added"] == 1

        # Second ingest with synonym predicate
        resp2 = booted_app.post(
            "/api/v1/quantum/ingest/math",
            json={"triplets": [["earth", "revolves_around", "sun"]]},
        )
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["axioms_added"] == 0
        assert data["duplicates_skipped"] == 1
        assert data["skipped_details"][0]["duplicate_of"] == "earth||orbits||sun"

    def test_distinct_axiom_accepted(self, booted_app):
        """Distinct axiom is accepted even after prior ingestion."""
        booted_app.post(
            "/api/v1/quantum/ingest/math",
            json={"triplets": [["earth", "orbits", "sun"]]},
        )
        resp = booted_app.post(
            "/api/v1/quantum/ingest/math",
            json={"triplets": [["mars", "has", "moons"]]},
        )
        assert resp.status_code == 200
        assert resp.json()["axioms_added"] == 1
        assert resp.json()["duplicates_skipped"] == 0

    def test_batch_within_request_dedup(self, booted_app):
        """Multiple triplets in one request: duplicates within batch are caught."""
        resp = booted_app.post(
            "/api/v1/quantum/ingest/math",
            json={
                "triplets": [
                    ["earth", "orbits", "sun"],
                    ["earth", "revolves_around", "sun"],
                    ["mars", "has", "moons"],
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # First axiom is added, second is deduped, third is added
        assert data["axioms_added"] == 2
        assert data["duplicates_skipped"] == 1
