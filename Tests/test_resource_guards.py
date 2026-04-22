"""
Stage 5 — Resource Limits and Failure Observability Tests

Tests:
    1-3.  Ingest text guards (accept, reject chars, reject bytes)
    4-5.  Bundle import guards (accept, reject oversized)
    6-7.  Sync request guards (accept, reject)
    8-9.  /ask query guards (accept, reject)
    10.   Branch name guard
    11.   Axiom key guard
    12.   ResourceLimitError is HTTP 413
    13.   Failure messages are explicit and operator-readable
    14.   Scheme mismatch produces explicit 409 (regression)
"""

import pytest
from fastapi import HTTPException

from sum_engine_internal.infrastructure.resource_guards import (
    guard_ingest_text,
    guard_bundle_import,
    guard_sync_state_digits,
    guard_ask_query,
    guard_branch_name,
    guard_axiom_key,
    ResourceLimitError,
    MAX_INGEST_TEXT_CHARS,
    MAX_BUNDLE_SIZE_BYTES,
    MAX_SYNC_STATE_DIGITS,
    MAX_ASK_QUERY_LENGTH,
    MAX_BRANCH_NAME_LENGTH,
    MAX_AXIOM_KEY_LENGTH,
    MAX_STATE_INTEGER_DIGITS,
)


# ─── Ingest Guards ────────────────────────────────────────────────────

class TestIngestGuards:
    def test_normal_text_accepted(self):
        """Short text passes without exception."""
        guard_ingest_text("The earth orbits the sun. " * 10)

    def test_oversized_text_rejected_chars(self):
        """Text exceeding MAX_INGEST_TEXT_CHARS is rejected."""
        huge = "x" * (MAX_INGEST_TEXT_CHARS + 1)
        with pytest.raises(HTTPException) as exc:
            guard_ingest_text(huge)
        assert exc.value.status_code == 413
        assert "ingest_text_chars" in exc.value.detail

    def test_empty_text_accepted(self):
        """Empty text passes."""
        guard_ingest_text("")


# ─── Bundle Guards ────────────────────────────────────────────────────

class TestBundleGuards:
    def test_normal_bundle_accepted(self):
        """Reasonable bundle passes."""
        bundle = {
            "canonical_tome": "The alice likes cats.\n" * 10,
            "state_integer": "12345",
            "bundle_version": "1.0.0",
        }
        guard_bundle_import(bundle)

    def test_oversized_state_integer_rejected(self):
        """State integer with too many digits is rejected."""
        bundle = {
            "state_integer": "9" * (MAX_STATE_INTEGER_DIGITS + 1),
        }
        with pytest.raises(HTTPException) as exc:
            guard_bundle_import(bundle)
        assert exc.value.status_code == 413

    def test_too_many_tome_lines_rejected(self):
        """Bundle with too many canonical tome lines is rejected."""
        bundle = {
            "canonical_tome": "line\n" * 60_000,
            "state_integer": "1",
        }
        with pytest.raises(HTTPException) as exc:
            guard_bundle_import(bundle)
        assert exc.value.status_code == 413


# ─── Sync Guards ──────────────────────────────────────────────────────

class TestSyncGuards:
    def test_normal_sync_accepted(self):
        guard_sync_state_digits(100)

    def test_sync_rejects_oversized(self):
        with pytest.raises(ResourceLimitError) as exc:
            guard_sync_state_digits(MAX_STATE_INTEGER_DIGITS + 1)
        assert exc.value.status_code == 413


# ─── Ask Guards ───────────────────────────────────────────────────────

class TestAskGuards:
    def test_normal_query_accepted(self):
        guard_ask_query("What color is the sky?")

    def test_oversized_query_rejected(self):
        with pytest.raises(HTTPException) as exc:
            guard_ask_query("x" * (MAX_ASK_QUERY_LENGTH + 1))
        assert exc.value.status_code == 413


# ─── Name Guards ──────────────────────────────────────────────────────

class TestNameGuards:
    def test_normal_branch_accepted(self):
        guard_branch_name("main")

    def test_long_branch_rejected(self):
        with pytest.raises(HTTPException) as exc:
            guard_branch_name("x" * (MAX_BRANCH_NAME_LENGTH + 1))
        assert exc.value.status_code == 413

    def test_normal_axiom_key_accepted(self):
        guard_axiom_key("alice||likes||cats")

    def test_long_axiom_key_rejected(self):
        with pytest.raises(HTTPException) as exc:
            guard_axiom_key("x" * (MAX_AXIOM_KEY_LENGTH + 1))
        assert exc.value.status_code == 413


# ─── Error Quality ────────────────────────────────────────────────────

class TestErrorQuality:
    def test_resource_limit_error_is_413(self):
        """ResourceLimitError is a proper HTTP 413."""
        err = ResourceLimitError("test_resource", 100, 50)
        assert err.status_code == 413

    def test_error_message_is_explicit(self):
        """Error message names the resource, actual, and limit."""
        err = ResourceLimitError("sync_axiom_count", 15000, 10000)
        assert "sync_axiom_count" in err.detail
        assert "15000" in err.detail
        assert "10000" in err.detail

    def test_error_includes_advice(self):
        """Error detail includes actionable advice when provided."""
        err = ResourceLimitError(
            "ingest_text_chars", 300000, 200000,
            "Split large documents into smaller chunks."
        )
        assert "Split" in err.detail
