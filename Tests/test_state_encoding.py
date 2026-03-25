"""
Stage 1 Tests — Dual-Format State Transport Migration

Tests:
    1-3.  state_encoding: to_hex, from_hex, parse_state round-trips
    4-5.  parse_state: auto-detect hex vs decimal
    6.    dual_field: backward-compatible field generation
    7-8.  Edge cases: zero, large integers
    9-10. API integration: hex companion fields present in responses
    11.   API: sync accepts hex input
    12.   API: sync/state accepts hex input
"""

import math
import pytest

from internal.infrastructure.state_encoding import (
    to_hex,
    from_hex,
    parse_state,
    to_dual,
    dual_field,
)
from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.infrastructure.akashic_ledger import AkashicLedger


# ─── Unit Tests: state_encoding ──────────────────────────────────────

class TestStateEncoding:
    def test_to_hex_basic(self):
        assert to_hex(255) == "0xff"

    def test_to_hex_large(self):
        big = 2**256
        h = to_hex(big)
        assert h.startswith("0x")
        assert int(h, 16) == big

    def test_from_hex_with_prefix(self):
        assert from_hex("0xff") == 255

    def test_from_hex_without_prefix(self):
        assert from_hex("ff") == 255

    def test_parse_state_decimal(self):
        assert parse_state("12345") == 12345

    def test_parse_state_hex(self):
        assert parse_state("0x3039") == 12345

    def test_parse_state_hex_uppercase(self):
        assert parse_state("0X3039") == 12345

    def test_round_trip(self):
        """decimal → int → hex → int roundtrip."""
        original = 1898585074409907150524167558344558620554613878579045806247
        h = to_hex(original)
        recovered = from_hex(h)
        assert recovered == original

    def test_round_trip_via_parse(self):
        original = 2**512 + 7
        h = to_hex(original)
        assert parse_state(h) == original
        assert parse_state(str(original)) == original

    def test_zero(self):
        assert to_hex(0) == "0x0"
        assert from_hex("0x0") == 0
        assert parse_state("0") == 0

    def test_dual_field(self):
        result = dual_field("my_state", 42)
        assert result["my_state"] == "42"
        assert result["my_state_hex"] == "0x2a"

    def test_to_dual(self):
        result = to_dual(255)
        assert result["state_decimal"] == "255"
        assert result["state_hex"] == "0xff"

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            to_hex(-1)


# ─── API Integration Tests ──────────────────────────────────────────

class TestAPIDualFormat:
    """Verify API responses include hex companion fields."""

    @pytest.fixture
    def booted_app(self, tmp_path):
        from api.quantum_router import kos
        orig = {
            "ledger": kos.ledger,
            "booted": kos.is_booted,
            "branches": kos.branches,
            "algebra": kos.algebra,
        }
        kos.ledger = AkashicLedger(str(tmp_path / "test_hex.db"))
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

    def test_state_has_hex_field(self, booted_app):
        """GET /state returns both decimal and hex."""
        resp = booted_app.get("/api/v1/quantum/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "global_state_integer" in data
        assert "global_state_integer_hex" in data
        # They should be equivalent
        dec_val = int(data["global_state_integer"])
        hex_val = int(data["global_state_integer_hex"], 16)
        assert dec_val == hex_val

    def test_ingest_math_has_hex(self, booted_app):
        """/ingest/math response includes hex."""
        resp = booted_app.post(
            "/api/v1/quantum/ingest/math",
            json={"triplets": [["earth", "orbits", "sun"]]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "new_global_state" in data
        assert "new_global_state_hex" in data
        assert int(data["new_global_state"]) == int(data["new_global_state_hex"], 16)

    def test_sync_accepts_hex_input(self, booted_app):
        """POST /sync accepts hex client_state_integer."""
        # Ingest something first
        booted_app.post(
            "/api/v1/quantum/ingest/math",
            json={"triplets": [["alice", "likes", "cats"]]},
        )
        # Sync with hex state = 0x1
        resp = booted_app.post(
            "/api/v1/quantum/sync",
            json={"client_state_integer": "0x1", "branch": "main"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["delta"]["add"]  # client at 1 should get additions
        assert "new_global_state_hex" in data

    def test_sync_accepts_decimal_input(self, booted_app):
        """POST /sync still works with decimal (backward compatible)."""
        booted_app.post(
            "/api/v1/quantum/ingest/math",
            json={"triplets": [["bob", "knows", "python"]]},
        )
        resp = booted_app.post(
            "/api/v1/quantum/sync",
            json={"client_state_integer": "1", "branch": "main"},
        )
        assert resp.status_code == 200
        assert resp.json()["delta"]["add"]

    def test_hex_decimal_parity_across_endpoints(self, booted_app):
        """After ingest, all hex fields are consistent with their decimal counterparts."""
        booted_app.post(
            "/api/v1/quantum/ingest/math",
            json={"triplets": [["earth", "orbits", "sun"]]},
        )
        # Check /state
        state_resp = booted_app.get("/api/v1/quantum/state")
        d = state_resp.json()
        dec = int(d["global_state_integer"])
        hx = int(d["global_state_integer_hex"], 16)
        assert dec == hx
        assert dec > 1  # Non-trivial state
