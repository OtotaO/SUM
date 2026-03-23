"""
Phase 22 Tests — Provenance + Confidence

Tests:
    1. Schema migration idempotency (run _init_db twice)
    2. append_event stores provenance metadata
    3. get_axiom_provenance returns correct chain
    4. get_provenance_batch returns latest per axiom
    5. /ingest/math stores source_url in ledger
    6. /ask weights results by confidence
    7. /ask includes provenance in response
    8. /ask recency weighting: newer axioms rank higher
    9. /provenance/{axiom_key} returns provenance chain
    10. Default confidence is 0.5
    11. Empty source_url handled gracefully
    12. Unknown axiom returns empty provenance
"""

import math
import os
import asyncio
import sqlite3
import tempfile
import pytest

from internal.infrastructure.akashic_ledger import AkashicLedger
from internal.algorithms.semantic_arithmetic import GodelStateAlgebra


# ─── Akashic Ledger Unit Tests ───────────────────────────────────────

class TestLedgerProvenance:
    """Tests for the Akashic Ledger provenance columns and queries."""

    @pytest.fixture
    def ledger(self, tmp_path):
        """Create a fresh ledger in a temp directory."""
        db = str(tmp_path / "test_provenance.db")
        return AkashicLedger(db)

    def test_schema_migration_idempotent(self, tmp_path):
        """Running _init_db twice should not raise."""
        db = str(tmp_path / "idempotent.db")
        ledger1 = AkashicLedger(db)
        ledger2 = AkashicLedger(db)  # Second init triggers migration again
        # Verify columns exist
        with sqlite3.connect(db) as conn:
            cursor = conn.execute("PRAGMA table_info(semantic_events)")
            cols = {row[1] for row in cursor.fetchall()}
        assert "source_url" in cols
        assert "confidence" in cols
        assert "ingested_at" in cols

    @pytest.mark.asyncio
    async def test_append_event_with_provenance(self, ledger):
        """append_event stores source_url, confidence, ingested_at."""
        await ledger.append_event(
            "MINT", 7, "alice||likes||cats",
            source_url="https://example.com/article",
            confidence=0.9,
            ingested_at="2026-03-23T12:00:00",
        )
        with sqlite3.connect(ledger.db_path) as conn:
            row = conn.execute(
                "SELECT source_url, confidence, ingested_at "
                "FROM semantic_events WHERE axiom_key = ?",
                ("alice||likes||cats",),
            ).fetchone()
        assert row[0] == "https://example.com/article"
        assert row[1] == pytest.approx(0.9)
        assert row[2] == "2026-03-23T12:00:00"

    @pytest.mark.asyncio
    async def test_append_event_default_confidence(self, ledger):
        """Default confidence is 0.5 when not specified."""
        await ledger.append_event("MINT", 11, "bob||has||dog")
        with sqlite3.connect(ledger.db_path) as conn:
            row = conn.execute(
                "SELECT confidence FROM semantic_events "
                "WHERE axiom_key = ?",
                ("bob||has||dog",),
            ).fetchone()
        assert row[0] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_get_axiom_provenance(self, ledger):
        """get_axiom_provenance returns full chain for an axiom."""
        await ledger.append_event(
            "MINT", 7, "sun||causes||warmth",
            source_url="https://textbook.edu",
            confidence=0.95,
            ingested_at="2026-01-01T00:00:00",
        )
        await ledger.append_event(
            "MINT", 7, "sun||causes||warmth",
            source_url="https://wiki.org",
            confidence=0.7,
            ingested_at="2026-03-01T00:00:00",
        )
        chain = await ledger.get_axiom_provenance("sun||causes||warmth")
        assert len(chain) == 2
        assert chain[0]["source_url"] == "https://textbook.edu"
        assert chain[0]["confidence"] == pytest.approx(0.95)
        assert chain[1]["source_url"] == "https://wiki.org"

    @pytest.mark.asyncio
    async def test_get_axiom_provenance_empty(self, ledger):
        """Unknown axiom returns empty provenance."""
        chain = await ledger.get_axiom_provenance("nonexistent||pred||obj")
        assert chain == []

    @pytest.mark.asyncio
    async def test_get_provenance_batch(self, ledger):
        """Batch lookup returns latest provenance per axiom."""
        await ledger.append_event(
            "MINT", 7, "a||is||b",
            source_url="src1", confidence=0.3, ingested_at="2026-01-01T00:00:00",
        )
        await ledger.append_event(
            "MINT", 7, "a||is||b",
            source_url="src2", confidence=0.8, ingested_at="2026-03-01T00:00:00",
        )
        await ledger.append_event(
            "MINT", 11, "c||has||d",
            source_url="src3", confidence=0.6, ingested_at="2026-02-01T00:00:00",
        )
        result = await ledger.get_provenance_batch(["a||is||b", "c||has||d"])
        assert "a||is||b" in result
        assert "c||has||d" in result
        # Latest for a||is||b should be src2 (higher seq_id)
        assert result["a||is||b"]["source_url"] == "src2"
        assert result["a||is||b"]["confidence"] == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_get_provenance_batch_empty(self, ledger):
        """Empty list returns empty dict."""
        result = await ledger.get_provenance_batch([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_provenance_batch_unknown(self, ledger):
        """Unknown axioms are absent from result."""
        result = await ledger.get_provenance_batch(["unknown||pred||obj"])
        assert "unknown||pred||obj" not in result


# ─── API Endpoint Tests ──────────────────────────────────────────────

class TestIngestMathProvenance:
    """/ingest/math stores provenance in the ledger."""

    @pytest.fixture
    def booted_app(self, tmp_path):
        """Create a test client with a booted KOS using a temp DB."""
        from api.quantum_router import kos
        # Save original state
        orig_ledger = kos.ledger
        orig_booted = kos.is_booted
        orig_branches = kos.branches
        orig_algebra = kos.algebra

        # Use temp DB so tests don't pollute production
        db_path = str(tmp_path / "test_ingest.db")
        kos.ledger = AkashicLedger(db_path)
        kos.is_booted = True
        kos.branches = {"main": 1}
        kos.algebra = GodelStateAlgebra()

        from fastapi.testclient import TestClient
        from quantum_main import app
        yield TestClient(app)

        # Restore original state
        kos.ledger = orig_ledger
        kos.is_booted = orig_booted
        kos.branches = orig_branches
        kos.algebra = orig_algebra

    def test_ingest_math_stores_source_url(self, booted_app, tmp_path):
        """POST /ingest/math with source_url stores it in the ledger."""
        from api.quantum_router import kos
        resp = booted_app.post(
            "/api/v1/quantum/ingest/math",
            json={
                "triplets": [["earth", "orbits", "sun"]],
                "source_url": "https://nasa.gov/solar-system",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["axioms_added"] == 1

        # Verify ledger has provenance
        with sqlite3.connect(kos.ledger.db_path) as conn:
            row = conn.execute(
                "SELECT source_url, confidence, ingested_at "
                "FROM semantic_events WHERE operation = 'MINT' LIMIT 1"
            ).fetchone()
        assert row is not None
        assert row[0] == "https://nasa.gov/solar-system"
        assert row[1] == pytest.approx(0.5)  # default
        assert len(row[2]) > 0  # ingested_at is set

    def test_ingest_math_empty_source_url(self, booted_app, tmp_path):
        """POST /ingest/math without source_url defaults to empty string."""
        from api.quantum_router import kos
        resp = booted_app.post(
            "/api/v1/quantum/ingest/math",
            json={
                "triplets": [["mars", "has", "moons"]],
            },
        )
        assert resp.status_code == 200

        with sqlite3.connect(kos.ledger.db_path) as conn:
            row = conn.execute(
                "SELECT source_url FROM semantic_events "
                "WHERE operation = 'MINT' LIMIT 1"
            ).fetchone()
        assert row[0] == ""


class TestAskProvenance:
    """/ask returns provenance-weighted results."""

    @pytest.fixture
    def booted_app(self, tmp_path):
        """Create a test client with axioms that have different confidence."""
        from api.quantum_router import kos
        orig_ledger = kos.ledger
        orig_booted = kos.is_booted
        orig_branches = kos.branches
        orig_algebra = kos.algebra

        db_path = str(tmp_path / "test_ask.db")
        kos.ledger = AkashicLedger(db_path)
        kos.is_booted = True
        kos.branches = {"main": 1}
        kos.algebra = GodelStateAlgebra()

        # Ingest axioms with different confidences
        p1 = kos.algebra.get_or_mint_prime("python", "is_a", "language")
        p2 = kos.algebra.get_or_mint_prime("python", "enables", "scripting")
        kos.branches["main"] = math.lcm(p1, p2)

        # Write provenance directly to DB
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            # High confidence for "python is_a language"
            ax1 = "python||is_a||language"
            conn.execute(
                "INSERT INTO semantic_events "
                "(operation, prime, axiom_key, source_url, confidence, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("MINT", str(p1), ax1, "https://docs.python.org", 0.95,
                 "2026-03-23T00:00:00"),
            )
            # Low confidence for "python enables scripting"
            ax2 = "python||enables||scripting"
            conn.execute(
                "INSERT INTO semantic_events "
                "(operation, prime, axiom_key, source_url, confidence, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("MINT", str(p2), ax2, "https://random-blog.com", 0.2,
                 "2026-03-23T00:00:00"),
            )
            conn.commit()

        from fastapi.testclient import TestClient
        from quantum_main import app
        yield TestClient(app)

        kos.ledger = orig_ledger
        kos.is_booted = orig_booted
        kos.branches = orig_branches
        kos.algebra = orig_algebra

    def test_ask_includes_provenance(self, booted_app):
        """Matches include source_url and confidence."""
        resp = booted_app.post(
            "/api/v1/quantum/ask",
            json={"question": "What is python?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["matches"]) > 0
        m = data["matches"][0]
        assert "source_url" in m
        assert "confidence" in m
        assert "ingested_at" in m

    def test_ask_confidence_weighting(self, booted_app):
        """Higher-confidence axioms rank higher for same keyword score."""
        resp = booted_app.post(
            "/api/v1/quantum/ask",
            json={"question": "Tell me about python"},
        )
        data = resp.json()
        matches = data["matches"]
        assert len(matches) >= 2
        # The high-confidence one (0.95) should rank above low-confidence (0.2)
        confidences = [m["confidence"] for m in matches]
        assert confidences[0] > confidences[1], (
            f"Expected high-confidence first, got {confidences}"
        )


class TestProvenanceEndpoint:
    """/provenance/{axiom_key} returns the provenance chain."""

    @pytest.fixture
    def booted_app(self, tmp_path):
        from api.quantum_router import kos
        orig_ledger = kos.ledger
        orig_booted = kos.is_booted
        orig_branches = kos.branches
        orig_algebra = kos.algebra

        db_path = str(tmp_path / "test_prov.db")
        kos.ledger = AkashicLedger(db_path)
        kos.is_booted = True
        kos.branches = {"main": 1}
        kos.algebra = GodelStateAlgebra()

        p = kos.algebra.get_or_mint_prime("earth", "is_a", "planet")
        kos.branches["main"] = p

        import sqlite3
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO semantic_events "
                "(operation, prime, axiom_key, source_url, confidence, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("MINT", str(p), "earth||is_a||planet",
                 "https://nasa.gov", 0.99, "2026-03-23T00:00:00"),
            )
            conn.commit()

        from fastapi.testclient import TestClient
        from quantum_main import app
        yield TestClient(app)

        kos.ledger = orig_ledger
        kos.is_booted = orig_booted
        kos.branches = orig_branches
        kos.algebra = orig_algebra

    def test_provenance_returns_chain(self, booted_app):
        """GET /provenance/earth||is_a||planet returns the chain."""
        resp = booted_app.get(
            "/api/v1/quantum/provenance/earth||is_a||planet",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["axiom_key"] == "earth||is_a||planet"
        assert data["prime"] is not None
        assert data["total_sources"] == 1
        assert data["provenance_chain"][0]["source_url"] == "https://nasa.gov"
        assert data["provenance_chain"][0]["confidence"] == pytest.approx(0.99)

    def test_provenance_unknown_axiom(self, booted_app):
        """GET /provenance for unknown axiom returns empty chain."""
        resp = booted_app.get(
            "/api/v1/quantum/provenance/unknown||pred||obj",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_sources"] == 0
        assert data["provenance_chain"] == []
        assert data["prime"] is None
