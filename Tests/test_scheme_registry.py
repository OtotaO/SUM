"""
Stage 2 Tests — Prime Scheme Versioning & Negotiation

Tests:
    1-3.  Scheme registry: lookup, current scheme, unknown scheme
    4-5.  Compatibility: v1 compatible, v2 rejected
    6.    validate_scheme_or_raise: error message quality
    7-8.  Bundle export: prime_scheme + state_integer_hex present
    9.    Bundle import: compatible scheme accepted
    10.   Bundle import: incompatible scheme rejected
    11.   API: /state includes prime_scheme
    12.   API: /sync includes prime_scheme
"""

import pytest

from sum_engine_internal.infrastructure.scheme_registry import (
    CURRENT_SCHEME,
    SCHEMES,
    get_scheme,
    get_current_scheme,
    is_compatible,
    validate_scheme_or_raise,
)
from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from sum_engine_internal.infrastructure.akashic_ledger import AkashicLedger


# ─── Unit Tests: scheme_registry ─────────────────────────────────────

class TestSchemeRegistry:
    def test_current_scheme_is_sha256_64_v1(self):
        assert CURRENT_SCHEME == "sha256_64_v1"

    def test_get_current_scheme(self):
        scheme = get_current_scheme()
        assert scheme.scheme_id == "sha256_64_v1"
        assert scheme.seed_bytes == 8
        assert scheme.hash_algorithm == "sha256"

    def test_get_scheme_v1(self):
        scheme = get_scheme("sha256_64_v1")
        assert scheme.primality_test == "deterministic_miller_rabin_12_witnesses"

    def test_get_scheme_v2_exists(self):
        scheme = get_scheme("sha256_128_v2")
        assert scheme.seed_bytes == 16
        assert scheme.primality_test == "bpsw"

    def test_get_scheme_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown prime scheme"):
            get_scheme("sha512_256_v99")

    def test_compatible_v1(self):
        assert is_compatible("sha256_64_v1") is True

    def test_incompatible_v2(self):
        assert is_compatible("sha256_128_v2") is False

    def test_incompatible_unknown(self):
        assert is_compatible("made_up_scheme") is False

    def test_validate_compatible(self):
        validate_scheme_or_raise("sha256_64_v1", context="test")

    def test_validate_incompatible_raises(self):
        with pytest.raises(ValueError, match="Incompatible prime scheme"):
            validate_scheme_or_raise("sha256_128_v2", context="test import")

    def test_scheme_is_frozen(self):
        """Scheme dataclass is frozen — can't mutate."""
        scheme = get_scheme("sha256_64_v1")
        with pytest.raises(AttributeError):
            scheme.seed_bytes = 16  # type: ignore


# ─── Bundle Integration Tests ────────────────────────────────────────

class TestBundleScheme:
    @pytest.fixture
    def codec(self, tmp_path):
        from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
        from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec
        algebra = GodelStateAlgebra()
        tome_gen = AutoregressiveTomeGenerator(algebra)
        return CanonicalCodec(algebra, tome_gen, signing_key="test-key")

    def test_export_includes_scheme(self, codec):
        """Exported bundles include prime_scheme."""
        bundle = codec.export_bundle(1, branch="main")
        assert "prime_scheme" in bundle
        assert bundle["prime_scheme"] == "sha256_64_v1"

    def test_export_includes_hex(self, codec):
        """Exported bundles include state_integer_hex."""
        bundle = codec.export_bundle(42, branch="main")
        assert "state_integer_hex" in bundle
        assert bundle["state_integer_hex"] == "0x2a"
        assert bundle["state_integer"] == "42"

    def test_import_compatible_scheme(self, codec):
        """Import succeeds for compatible scheme."""
        bundle = codec.export_bundle(1)
        assert bundle["prime_scheme"] == "sha256_64_v1"
        state = codec.import_bundle(bundle)
        assert state == 1

    def test_import_incompatible_scheme_rejected(self, codec):
        """Import rejects incompatible scheme."""
        bundle = codec.export_bundle(1)
        bundle["prime_scheme"] = "sha256_128_v2"
        with pytest.raises(ValueError, match="Incompatible prime scheme"):
            codec.import_bundle(bundle)

    def test_import_missing_scheme_defaults_to_v1(self, codec):
        """Bundles without prime_scheme default to v1 (backward compat)."""
        bundle = codec.export_bundle(1)
        del bundle["prime_scheme"]
        state = codec.import_bundle(bundle)
        assert state == 1


# ─── API Integration Tests ───────────────────────────────────────────

class TestAPIScheme:
    @pytest.fixture
    def booted_app(self, tmp_path):
        from api.quantum_router import kos
        orig = {
            "ledger": kos.ledger,
            "booted": kos.is_booted,
            "branches": kos.branches,
            "algebra": kos.algebra,
        }
        kos.ledger = AkashicLedger(str(tmp_path / "test_scheme.db"))
        kos.is_booted = True
        kos.branches = {"main": 1}
        kos.algebra = GodelStateAlgebra()

        from fastapi.testclient import TestClient
        from quantum_main import app
        yield TestClient(app)

        kos.ledger = orig["ledger"]
        kos.is_booted = orig["booted"]
        kos.branches = orig["branches"]
        kos.algebra = orig["algebra"]

    def test_state_includes_scheme(self, booted_app):
        resp = booted_app.get("/api/v1/quantum/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "prime_scheme" in data
        assert data["prime_scheme"] == "sha256_64_v1"

    def test_sync_includes_scheme(self, booted_app):
        booted_app.post(
            "/api/v1/quantum/ingest/math",
            json={"triplets": [["earth", "orbits", "sun"]]},
        )
        resp = booted_app.post(
            "/api/v1/quantum/sync",
            json={"client_state_integer": "0x1", "branch": "main"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["prime_scheme"] == "sha256_64_v1"


# ─── End-to-End Protocol Enforcement Tests ────────────────────────────

class TestProtocolEnforcement:
    """These tests prove the scheme negotiation is real, not theater."""

    @pytest.fixture
    def booted_app(self, tmp_path):
        from api.quantum_router import kos
        orig = {
            "ledger": kos.ledger,
            "booted": kos.is_booted,
            "branches": kos.branches,
            "algebra": kos.algebra,
        }
        kos.ledger = AkashicLedger(str(tmp_path / "test_enforce.db"))
        kos.is_booted = True
        kos.branches = {"main": 1}
        kos.algebra = GodelStateAlgebra()

        from fastapi.testclient import TestClient
        from quantum_main import app
        yield TestClient(app)

        kos.ledger = orig["ledger"]
        kos.is_booted = orig["booted"]
        kos.branches = orig["branches"]
        kos.algebra = orig["algebra"]

    def test_sync_rejects_incompatible_scheme(self, booted_app):
        """/sync returns 409 when client sends an incompatible scheme."""
        resp = booted_app.post(
            "/api/v1/quantum/sync",
            json={
                "client_state_integer": "1",
                "branch": "main",
                "prime_scheme": "sha256_128_v2",
            },
        )
        assert resp.status_code == 409
        assert "Incompatible prime scheme" in resp.json()["detail"]

    def test_sync_accepts_compatible_scheme(self, booted_app):
        """/sync succeeds when client sends the current scheme."""
        resp = booted_app.post(
            "/api/v1/quantum/sync",
            json={
                "client_state_integer": "1",
                "branch": "main",
                "prime_scheme": "sha256_64_v1",
            },
        )
        assert resp.status_code == 200

    def test_sync_accepts_absent_scheme(self, booted_app):
        """/sync succeeds when no scheme is sent (backward compat)."""
        resp = booted_app.post(
            "/api/v1/quantum/sync",
            json={"client_state_integer": "1", "branch": "main"},
        )
        assert resp.status_code == 200

    def test_sync_state_rejects_incompatible_scheme(self, booted_app):
        """/sync/state returns 409 when peer sends an incompatible scheme."""
        resp = booted_app.post(
            "/api/v1/quantum/sync/state",
            json={
                "peer_state_integer": "1",
                "prime_scheme": "sha256_128_v2",
            },
        )
        assert resp.status_code == 409
        assert "Incompatible prime scheme" in resp.json()["detail"]

    def test_sync_state_accepts_compatible_scheme(self, booted_app):
        """/sync/state succeeds with the current scheme."""
        resp = booted_app.post(
            "/api/v1/quantum/sync/state",
            json={
                "peer_state_integer": "1",
                "prime_scheme": "sha256_64_v1",
            },
        )
        assert resp.status_code == 200

    def test_sync_state_accepts_absent_scheme(self, booted_app):
        """/sync/state succeeds when no scheme is sent (backward compat)."""
        resp = booted_app.post(
            "/api/v1/quantum/sync/state",
            json={"peer_state_integer": "1"},
        )
        assert resp.status_code == 200


class TestBundleHexCrossCheck:
    """Prove that state_integer_hex is not just advisory."""

    @pytest.fixture
    def codec(self, tmp_path):
        from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
        from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec
        algebra = GodelStateAlgebra()
        tome_gen = AutoregressiveTomeGenerator(algebra)
        return CanonicalCodec(algebra, tome_gen, signing_key="test-key")

    def test_hex_tamper_detected(self, codec):
        """If state_integer_hex is altered, import raises ValueError."""
        bundle = codec.export_bundle(42)
        assert bundle["state_integer"] == "42"
        assert bundle["state_integer_hex"] == "0x2a"
        # Tamper the hex
        bundle["state_integer_hex"] = "0xff"
        with pytest.raises(ValueError, match="does not match"):
            codec.import_bundle(bundle)

    def test_hex_consistent_passes(self, codec):
        """Consistent hex/decimal passes fine."""
        bundle = codec.export_bundle(255)
        assert bundle["state_integer_hex"] == "0xff"
        state = codec.import_bundle(bundle)
        assert state == 255

    def test_hex_absent_passes(self, codec):
        """Bundles without state_integer_hex still import fine."""
        bundle = codec.export_bundle(1)
        del bundle["state_integer_hex"]
        state = codec.import_bundle(bundle)
        assert state == 1
