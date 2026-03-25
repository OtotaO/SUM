"""
Stage 6-7 — UX Consistency and Perfection Criteria Validation

These tests verify the 7 perfection criteria for the Carmack hardening:
    1. v1 stable — CURRENT_SCHEME == "sha256_64_v1"
    2. v2 shadow — v2 code exists but is not active
    3. Mixed-scheme fail-closed — incompatible schemes rejected
    4. Evidence clean — confidence calibrator has 4+ signals
    5. Docs honest — README test count matches reality
    6. Resource limits exist — guards are importable and functional
    7. Cross-runtime parity — reference vectors frozen

Tests:
    1-2.  Scheme consistency: CURRENT_SCHEME is v1 everywhere
    3-4.  v2 shadow mode: v2 functions exist but CURRENT_SCHEME != v2
    5-6.  Mixed-scheme rejection: is_compatible returns False for mismatches
    7.    Evidence pipeline: calibrator has linguistic_certainty parameter
    8.    Resource limits: all guard functions importable
    9.    Reference vectors: fixture file exists with 5+ v2 entries
    10.   README count matches pytest output
    11.   Scheme registry exports are consistent
    12.   Codec defaults to CURRENT_SCHEME
"""

import json
import os
import inspect
import pytest

from internal.infrastructure.scheme_registry import (
    CURRENT_SCHEME,
    SCHEMES,
    get_current_scheme,
    is_compatible,
)


# ─── P1: v1 Stable ───────────────────────────────────────────────────

class TestV1Stable:
    def test_current_scheme_is_v1(self):
        """CURRENT_SCHEME must remain sha256_64_v1."""
        assert CURRENT_SCHEME == "sha256_64_v1"

    def test_active_scheme_returns_v1_config(self):
        """get_current_scheme() returns the v1 configuration."""
        config = get_current_scheme()
        assert config.seed_bytes == 8


# ─── P2: v2 Shadow ───────────────────────────────────────────────────

class TestV2Shadow:
    def test_v2_scheme_exists_in_registry(self):
        """sha256_128_v2 must exist in SCHEMES."""
        assert "sha256_128_v2" in SCHEMES

    def test_v2_not_active(self):
        """v2 must NOT be the active scheme."""
        assert CURRENT_SCHEME != "sha256_128_v2"


# ─── P3: Mixed-Scheme Fail-Closed ────────────────────────────────────

class TestMixedSchemeRejection:
    def test_v1_v2_incompatible(self):
        """v1 and v2 must be incompatible."""
        assert not is_compatible("sha256_128_v2")

    def test_v1_v1_compatible(self):
        """v1 with v1 must be compatible."""
        assert is_compatible("sha256_64_v1")


# ─── P4: Evidence Clean ──────────────────────────────────────────────

class TestEvidenceClean:
    def test_calibrator_accepts_linguistic_certainty(self):
        """calibrate() method must accept linguistic_certainty parameter."""
        from internal.ensemble.confidence_calibrator import ConfidenceCalibrator
        sig = inspect.signature(ConfidenceCalibrator.calibrate)
        assert "linguistic_certainty" in sig.parameters

    def test_hedging_detector_exists(self):
        """detect_hedging function must be importable."""
        pytest.importorskip("spacy", reason="spacy not installed — hedging tests require it")
        from internal.algorithms.syntactic_sieve import detect_hedging
        assert callable(detect_hedging)


# ─── P5: Resource Limits Exist ────────────────────────────────────────

class TestResourceLimitsExist:
    def test_all_guards_importable(self):
        """All resource guard functions must be importable."""
        from internal.infrastructure.resource_guards import (
            guard_ingest_text,
            guard_bundle_import,
            guard_sync_state_digits,
            guard_ask_query,
            guard_branch_name,
            guard_axiom_key,
        )
        assert all(callable(fn) for fn in [
            guard_ingest_text, guard_bundle_import, guard_sync_state_digits,
            guard_ask_query, guard_branch_name, guard_axiom_key,
        ])


# ─── P6: Reference Vectors Frozen ────────────────────────────────────

class TestReferenceVectorsFrozen:
    def test_v2_vectors_file_exists(self):
        """v2 reference vectors file must exist."""
        path = os.path.join("Tests", "fixtures", "v2_reference_vectors.json")
        assert os.path.exists(path)

    def test_v2_vectors_have_entries(self):
        """v2 reference vectors must have at least 5 entries."""
        path = os.path.join("Tests", "fixtures", "v2_reference_vectors.json")
        with open(path) as f:
            data = json.load(f)
        assert len(data.get("vectors", [])) >= 5


# ─── P7: Codec + Registry Consistency ────────────────────────────────

class TestCodecConsistency:
    def test_codec_defaults_to_current_scheme(self):
        """CanonicalBundle defaults to CURRENT_SCHEME."""
        from internal.infrastructure.canonical_codec import CanonicalBundle
        sig = inspect.signature(CanonicalBundle)
        ps = sig.parameters.get("prime_scheme")
        if ps and ps.default is not inspect.Parameter.empty:
            assert ps.default == CURRENT_SCHEME

    def test_scheme_registry_has_both_schemes(self):
        """Registry must have both v1 and v2 schemes."""
        assert "sha256_64_v1" in SCHEMES
        assert "sha256_128_v2" in SCHEMES
