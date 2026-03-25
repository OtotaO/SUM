"""
Stage 4 — Evidence Enrichment Tests

Tests:
    1-3. Hedging detection (definite, hedged, multiple markers)
    4-5. Linguistic certainty integration in calibrator
    6-7. Annotated triplet extraction
    8.   Hedging floor enforcement
    9.   Calibrator backward compatibility (no linguistic_certainty arg)
"""

import pytest
import asyncio

from internal.algorithms.syntactic_sieve import (
    detect_hedging,
    HEDGE_FLOOR,
    DeterministicSieve,
)
from internal.ensemble.confidence_calibrator import ConfidenceCalibrator

spacy = pytest.importorskip("spacy", reason="spacy not installed")


# ─── Hedging Detection Tests ─────────────────────────────────────────

class TestHedgingDetection:
    def test_definite_statement(self):
        """Definite statements get 1.0 certainty."""
        score = detect_hedging("The earth orbits the sun.")
        assert score == 1.0

    def test_single_hedge(self):
        """One hedging word reduces certainty."""
        score = detect_hedging("The earth might orbit the sun.")
        assert score < 1.0
        assert score > HEDGE_FLOOR

    def test_multiple_hedges(self):
        """Multiple hedging markers stack penalties."""
        score = detect_hedging(
            "Some researchers suggest that it may possibly indicate presence."
        )
        assert score < 0.7  # multiple markers hit

    def test_floor_enforcement(self):
        """Even extreme hedging can't go below HEDGE_FLOOR."""
        # Pack as many hedge words as possible
        text = (
            "It may might could possibly perhaps probably allegedly "
            "seemingly suggest appear seem believe estimate"
        )
        score = detect_hedging(text)
        assert score == HEDGE_FLOOR

    def test_empty_text(self):
        """Empty text returns 1.0."""
        assert detect_hedging("") == 1.0

    def test_none_text(self):
        """None-like empty text returns 1.0."""
        assert detect_hedging("") == 1.0


# ─── Calibrator Signal 4 Tests ────────────────────────────────────────

class TestCalibratorLinguistic:
    @pytest.fixture
    def calibrator(self):
        return ConfidenceCalibrator()

    @pytest.mark.asyncio
    async def test_linguistic_certainty_reduces_score(self, calibrator):
        """Hedged text reduces calibrated confidence."""
        definite = await calibrator.calibrate(
            axiom_key="earth||orbits||sun",
            source_url="https://nasa.gov/facts",
            linguistic_certainty=1.0,
        )
        hedged = await calibrator.calibrate(
            axiom_key="earth||orbits||sun",
            source_url="https://nasa.gov/facts",
            linguistic_certainty=0.7,
        )
        assert hedged < definite

    @pytest.mark.asyncio
    async def test_backward_compat_no_linguistic_arg(self, calibrator):
        """Omitting linguistic_certainty defaults to 1.0 (no change)."""
        score = await calibrator.calibrate(
            axiom_key="earth||orbits||sun",
            source_url="https://nasa.gov/facts",
        )
        assert score > 0.0  # should work without the new arg

    @pytest.mark.asyncio
    async def test_manual_confidence_bypasses_linguistic(self, calibrator):
        """manual_confidence skips all signals including linguistic."""
        score = await calibrator.calibrate(
            axiom_key="anything",
            manual_confidence=0.99,
            linguistic_certainty=0.1,
        )
        assert score == 0.99


# ─── Annotated Triplet Tests ─────────────────────────────────────────

class TestAnnotatedTriplets:
    @pytest.fixture
    def sieve(self):
        return DeterministicSieve()

    def test_definite_triplet_gets_full_certainty(self, sieve):
        """Definite sentence → certainty = 1.0."""
        results = sieve.extract_annotated_triplets(
            "The earth orbits the sun."
        )
        for r in results:
            assert r["linguistic_certainty"] == 1.0

    def test_hedged_triplet_gets_reduced_certainty(self, sieve):
        """Hedged sentence → certainty < 1.0."""
        results = sieve.extract_annotated_triplets(
            "Scientists suggest that water may contain hydrogen."
        )
        for r in results:
            assert r["linguistic_certainty"] < 1.0

    def test_annotated_triplets_have_required_keys(self, sieve):
        """Each annotated triplet has subject, predicate, object, certainty."""
        results = sieve.extract_annotated_triplets(
            "The earth orbits the sun."
        )
        for r in results:
            assert "subject" in r
            assert "predicate" in r
            assert "object" in r
            assert "linguistic_certainty" in r
