"""
Phase 24 Tests — Automatic Confidence Calibration

Tests:
    1-7.  Source-type heuristic: academic, gov, arxiv, wikipedia, blog, social, unknown
    8.    Redundancy boost: multiple sources increase confidence
    9.    Redundancy boost: single source = no boost
    10.   Contradiction penalty: conflicting axiom halves confidence
    11.   Contradiction penalty: no conflict = 1.0
    12.   Full calibrate pipeline: combines all signals
    13.   Manual override: confidence_mode="manual" bypasses calibration
    14.   API integration: /ingest/math auto-calculates confidence from URL
    15.   API integration: manual mode preserves user-set value
"""

import sqlite3
import asyncio
import pytest

from internal.ensemble.confidence_calibrator import (
    ConfidenceCalibrator,
    DEFAULT_CONFIDENCE,
    CONTRADICTION_PENALTY,
    load_venn_abers_fixture,
)
from internal.ensemble.venn_abers import ConfidenceInterval, VennAbersCalibrator
from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.infrastructure.akashic_ledger import AkashicLedger


# ─── Source-Type Heuristic Tests ─────────────────────────────────────

class TestSourceTypeScore:
    """Unit tests for URL-based confidence scoring."""

    def setup_method(self):
        self.cal = ConfidenceCalibrator()

    def test_academic_edu(self):
        assert self.cal.source_type_score("https://cs.stanford.edu/paper") == 0.90

    def test_government_gov(self):
        assert self.cal.source_type_score("https://data.nasa.gov/dataset") == 0.85

    def test_arxiv(self):
        assert self.cal.source_type_score("https://arxiv.org/abs/2404.12345") == 0.88

    def test_wikipedia(self):
        assert self.cal.source_type_score("https://en.wikipedia.org/wiki/Earth") == 0.70

    def test_blog(self):
        assert self.cal.source_type_score("https://medium.com/article") == 0.40

    def test_social(self):
        assert self.cal.source_type_score("https://reddit.com/r/science") == 0.35

    def test_unknown_url(self):
        assert self.cal.source_type_score("https://randomsite.xyz/info") == DEFAULT_CONFIDENCE

    def test_empty_url(self):
        assert self.cal.source_type_score("") == DEFAULT_CONFIDENCE

    def test_no_url(self):
        assert self.cal.source_type_score(None) == DEFAULT_CONFIDENCE


# ─── Redundancy Boost Tests ─────────────────────────────────────────

class TestRedundancyBoost:
    """Tests for the multi-source redundancy signal."""

    @pytest.mark.asyncio
    async def test_no_ledger(self):
        """No ledger means no boost."""
        cal = ConfidenceCalibrator()
        boost = await cal.redundancy_boost("any||pred||obj", None)
        assert boost == 0.0

    @pytest.mark.asyncio
    async def test_single_source(self, tmp_path):
        """One source = no boost."""
        ledger = AkashicLedger(str(tmp_path / "r.db"))
        await ledger.append_event(
            "MINT", 7, "sun||is||star", source_url="https://nasa.gov"
        )
        cal = ConfidenceCalibrator()
        boost = await cal.redundancy_boost("sun||is||star", ledger)
        assert boost == 0.0

    @pytest.mark.asyncio
    async def test_multiple_sources(self, tmp_path):
        """Three unique sources = 0.10 boost."""
        ledger = AkashicLedger(str(tmp_path / "r2.db"))
        await ledger.append_event("MINT", 7, "sun||is||star", source_url="src1")
        await ledger.append_event("MINT", 7, "sun||is||star", source_url="src2")
        await ledger.append_event("MINT", 7, "sun||is||star", source_url="src3")
        cal = ConfidenceCalibrator()
        boost = await cal.redundancy_boost("sun||is||star", ledger)
        assert boost == pytest.approx(0.10)


# ─── Contradiction Penalty Tests ─────────────────────────────────────

class TestContradictionPenalty:
    """Tests for the conflict detection signal."""

    def test_no_conflict(self):
        """Non-conflicting axiom gets multiplier of 1.0."""
        cal = ConfidenceCalibrator()
        algebra = GodelStateAlgebra()
        p = algebra.get_or_mint_prime("earth", "orbits", "sun")
        state = p
        penalty = cal.contradiction_penalty("mars||has||moons", state, algebra)
        assert penalty == 1.0

    def test_contradiction_same_subj_pred(self):
        """Same subject+predicate but different object → penalty."""
        cal = ConfidenceCalibrator()
        algebra = GodelStateAlgebra()
        p = algebra.get_or_mint_prime("alice", "lives_in", "new_york")
        state = p
        penalty = cal.contradiction_penalty("alice||lives_in||london", state, algebra)
        assert penalty == CONTRADICTION_PENALTY

    def test_empty_state(self):
        """Empty state = no contradiction possible."""
        cal = ConfidenceCalibrator()
        penalty = cal.contradiction_penalty("any||pred||obj", 1, None)
        assert penalty == 1.0


# ─── Full Pipeline Tests ─────────────────────────────────────────────

class TestCalibratePipeline:
    """Integration tests for the calibrate() method."""

    @pytest.mark.asyncio
    async def test_auto_calibrate_academic_url(self):
        """Academic URL with no conflicts → high confidence."""
        cal = ConfidenceCalibrator()
        score = await cal.calibrate(
            axiom_key="earth||orbits||sun",
            source_url="https://astro.caltech.edu/paper",
        )
        assert score == pytest.approx(0.90)

    @pytest.mark.asyncio
    async def test_auto_calibrate_blog_url(self):
        """Blog URL → low confidence."""
        cal = ConfidenceCalibrator()
        score = await cal.calibrate(
            axiom_key="earth||is_flat||true",
            source_url="https://medium.com/conspiracies",
        )
        assert score == pytest.approx(0.40)

    @pytest.mark.asyncio
    async def test_auto_calibrate_with_contradiction(self):
        """Blog + contradiction → very low confidence."""
        cal = ConfidenceCalibrator()
        algebra = GodelStateAlgebra()
        p = algebra.get_or_mint_prime("earth", "shape", "sphere")
        state = p
        score = await cal.calibrate(
            axiom_key="earth||shape||flat",
            source_url="https://medium.com/conspiracies",
            current_state=state,
            algebra=algebra,
        )
        # 0.40 (blog) × 0.5 (contradiction) = 0.20
        assert score == pytest.approx(0.20)

    @pytest.mark.asyncio
    async def test_manual_override(self):
        """Manual confidence bypasses all signals."""
        cal = ConfidenceCalibrator()
        score = await cal.calibrate(
            axiom_key="any||pred||obj",
            source_url="https://cs.stanford.edu/paper",
            manual_confidence=0.42,
        )
        assert score == pytest.approx(0.42)

    @pytest.mark.asyncio
    async def test_manual_override_clamped(self):
        """Manual confidence above 1.0 is clamped."""
        cal = ConfidenceCalibrator()
        score = await cal.calibrate(
            axiom_key="any||pred||obj",
            manual_confidence=1.5,
        )
        assert score == 1.0


# ─── API Integration Tests ──────────────────────────────────────────

class TestAPIConfidenceCalibration:
    """/ingest/math auto-calibrates confidence based on source URL."""

    @pytest.fixture
    def booted_app(self, tmp_path):
        from api.quantum_router import kos
        orig_ledger = kos.ledger
        orig_booted = kos.is_booted
        orig_branches = kos.branches
        orig_algebra = kos.algebra

        db_path = str(tmp_path / "test_cal.db")
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

    def test_auto_confidence_from_academic_url(self, booted_app, tmp_path):
        """Ingest from .edu URL → confidence ≈ 0.90."""
        from api.quantum_router import kos
        resp = booted_app.post(
            "/api/v1/quantum/ingest/math",
            json={
                "triplets": [["earth", "orbits", "sun"]],
                "source_url": "https://astro.caltech.edu/paper",
            },
        )
        assert resp.status_code == 200
        with sqlite3.connect(kos.ledger.db_path) as conn:
            row = conn.execute(
                "SELECT confidence FROM semantic_events WHERE operation='MINT' LIMIT 1"
            ).fetchone()
        assert row[0] == pytest.approx(0.90)

    def test_auto_confidence_from_blog_url(self, booted_app, tmp_path):
        """Ingest from blog → confidence ≈ 0.40."""
        from api.quantum_router import kos
        resp = booted_app.post(
            "/api/v1/quantum/ingest/math",
            json={
                "triplets": [["cats", "like", "fish"]],
                "source_url": "https://medium.com/pets",
            },
        )
        assert resp.status_code == 200
        with sqlite3.connect(kos.ledger.db_path) as conn:
            row = conn.execute(
                "SELECT confidence FROM semantic_events WHERE operation='MINT' LIMIT 1"
            ).fetchone()
        assert row[0] == pytest.approx(0.40)

    def test_manual_mode(self, booted_app, tmp_path):
        """confidence_mode=manual preserves user-supplied value."""
        from api.quantum_router import kos
        resp = booted_app.post(
            "/api/v1/quantum/ingest/math",
            json={
                "triplets": [["pi", "approx", "3.14"]],
                "source_url": "https://medium.com/math",
                "confidence_mode": "manual",
                "confidence": 0.99,
            },
        )
        assert resp.status_code == 200
        with sqlite3.connect(kos.ledger.db_path) as conn:
            row = conn.execute(
                "SELECT confidence FROM semantic_events WHERE operation='MINT' LIMIT 1"
            ).fetchone()
        assert row[0] == pytest.approx(0.99)


# ─── Venn-Abers Interval Integration (Polytaxis Bucket A) ─────────────


class TestCalibrateIntervalWithoutFixture:
    """Without a Venn-Abers fixture, calibrate_interval returns zero-width."""

    def setup_method(self):
        self.cal = ConfidenceCalibrator()

    def test_returns_confidence_interval(self):
        result = asyncio.run(
            self.cal.calibrate_interval(
                axiom_key="alice||age||30",
                source_url="https://arxiv.org/abs/2404.12345",
            )
        )
        assert isinstance(result, ConfidenceInterval)

    def test_zero_width_when_no_fixture(self):
        result = asyncio.run(
            self.cal.calibrate_interval(
                axiom_key="alice||age||30",
                source_url="https://arxiv.org/abs/2404.12345",
            )
        )
        assert result.lower == result.upper
        assert result.width == 0.0

    def test_scalar_matches_calibrate(self):
        scalar = asyncio.run(
            self.cal.calibrate(
                axiom_key="alice||age||30",
                source_url="https://arxiv.org/abs/2404.12345",
            )
        )
        interval = asyncio.run(
            self.cal.calibrate_interval(
                axiom_key="alice||age||30",
                source_url="https://arxiv.org/abs/2404.12345",
            )
        )
        assert interval.lower == scalar
        assert interval.upper == scalar


class TestCalibrateIntervalWithFixture:
    """With a Venn-Abers fixture, calibrate_interval returns bounded intervals."""

    def setup_method(self):
        # Simple synthetic fixture: monotone scores with matching labels.
        # Enough mass near both ends that the interval at mid-score is bounded.
        scores = [i / 20 for i in range(21)]
        labels = [0 if s < 0.5 else 1 for s in scores]
        va = VennAbersCalibrator(scores, labels)
        self.cal = ConfidenceCalibrator(venn_abers=va)

    def test_interval_is_in_unit_range(self):
        result = asyncio.run(
            self.cal.calibrate_interval(
                axiom_key="alice||age||30",
                source_url="https://arxiv.org/abs/2404.12345",
            )
        )
        assert 0.0 <= result.lower <= result.upper <= 1.0

    def test_interval_may_be_non_zero_width(self):
        # With a small fixture, the Venn-Abers interval typically has
        # non-zero width. Not strict — on edge cases width can be 0.
        result = asyncio.run(
            self.cal.calibrate_interval(
                axiom_key="alice||age||30",
                source_url="https://wikipedia.org/wiki/Earth",
            )
        )
        assert isinstance(result, ConfidenceInterval)


class TestLoadVennAbersFixture:
    """load_venn_abers_fixture reads JSON and returns a calibrator."""

    def test_roundtrip(self, tmp_path):
        import json
        fixture = tmp_path / "cal.json"
        fixture.write_text(json.dumps({
            "scores": [0.1, 0.5, 0.9],
            "labels": [0, 1, 1],
        }))
        va = load_venn_abers_fixture(fixture)
        assert va.n_calibration == 3

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_venn_abers_fixture(tmp_path / "nonexistent.json")
