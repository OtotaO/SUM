"""
Behavioral Integration Tests — Final Integration Phase

Tests that prove OPERATIONAL enforcement, not just importability.
Covers Priorities 1-4 of the Carmack honesty directive.

Author: ototao
License: Apache License 2.0
"""
import json
import math
import os
import subprocess
import sys

import pytest


# ─── Priority 1: Resource Guards Fire in Live Handlers ───────────────

class TestGuardsWiredIntoHandlers:
    """Prove guard functions are called from production code, not just tests."""

    def test_guard_callsites_exist_in_quantum_router(self):
        """grep for guard_ calls in quantum_router.py — must find at least 6."""
        router_path = os.path.join("api", "quantum_router.py")
        with open(router_path) as f:
            source = f.read()

        guard_calls = [
            "guard_ingest_text(",
            "guard_bundle_import(",
            "guard_ask_query(",
            "guard_branch_name(",
            "guard_axiom_key(",
            "guard_sync_state_digits(",
        ]
        found = [g for g in guard_calls if g in source]
        assert len(found) == 6, (
            f"Expected 6 guard call sites in quantum_router.py, found {len(found)}: {found}"
        )

    def test_guard_import_in_router(self):
        """The router must import from resource_guards — not just define them elsewhere."""
        router_path = os.path.join("api", "quantum_router.py")
        with open(router_path) as f:
            source = f.read()
        assert "from internal.infrastructure.resource_guards import" in source

    def test_guards_fire_before_boot_check(self):
        """Guards must appear before 'if not kos.is_booted' in at least /ingest."""
        router_path = os.path.join("api", "quantum_router.py")
        with open(router_path) as f:
            lines = f.readlines()

        # Find /ingest handler
        in_ingest = False
        guard_line = None
        boot_line = None
        for i, line in enumerate(lines):
            if 'async def ingest_document(' in line:
                in_ingest = True
            if in_ingest and 'guard_ingest_text(' in line:
                guard_line = i
            if in_ingest and 'kos.is_booted' in line:
                boot_line = i
                break

        assert guard_line is not None, "guard_ingest_text not found in /ingest handler"
        assert boot_line is not None, "kos.is_booted not found in /ingest handler"
        assert guard_line < boot_line, (
            f"Guard (line {guard_line}) must fire BEFORE boot check (line {boot_line})"
        )

    def test_resource_limit_error_is_413(self):
        """ResourceLimitError must produce HTTP 413."""
        from internal.infrastructure.resource_guards import ResourceLimitError
        try:
            raise ResourceLimitError(
                resource="test", actual=999, limit=10, detail="reduce"
            )
        except Exception as err:
            assert err.status_code == 413

    def test_oversized_ingest_raises(self):
        """guard_ingest_text rejects text > limit."""
        from internal.infrastructure.resource_guards import (
            guard_ingest_text, ResourceLimitError, MAX_INGEST_TEXT_CHARS
        )
        big_text = "x" * (MAX_INGEST_TEXT_CHARS + 1)
        with pytest.raises(ResourceLimitError):
            guard_ingest_text(big_text)

    def test_oversized_ask_raises(self):
        """guard_ask_query rejects oversized queries."""
        from internal.infrastructure.resource_guards import (
            guard_ask_query, ResourceLimitError, MAX_ASK_QUERY_LENGTH
        )
        big_query = "q" * (MAX_ASK_QUERY_LENGTH + 1)
        with pytest.raises(ResourceLimitError):
            guard_ask_query(big_query)

    def test_pathological_branch_name_raises(self):
        """guard_branch_name rejects names with path traversal."""
        from internal.infrastructure.resource_guards import (
            guard_branch_name, ResourceLimitError
        )
        with pytest.raises(ResourceLimitError):
            guard_branch_name("../" * 100 + "evil")


# ─── Priority 2: Evidence Enrichment in Live Paths ───────────────────

class TestEvidenceInLivePaths:
    """Prove linguistic_certainty is threaded through live ingestion code."""

    def test_ingest_handler_calls_detect_hedging(self):
        """The /ingest handler must call detect_hedging on the raw text."""
        router_path = os.path.join("api", "quantum_router.py")
        with open(router_path) as f:
            source = f.read()
        assert "detect_hedging(request.text)" in source, (
            "/ingest handler must call detect_hedging on request.text"
        )

    def test_ingest_handler_passes_linguistic_certainty(self):
        """The /ingest handler must pass linguistic_certainty to calibrator."""
        router_path = os.path.join("api", "quantum_router.py")
        with open(router_path) as f:
            source = f.read()
        assert "linguistic_certainty=text_certainty" in source, (
            "/ingest handler must pass linguistic_certainty to calibrator.calibrate()"
        )

    def test_hedging_reduces_confidence(self):
        """A hedged sentence must produce lower calibrated confidence than a definite one."""
        from internal.algorithms.syntactic_sieve import detect_hedging
        from internal.ensemble.confidence_calibrator import ConfidenceCalibrator

        definite_certainty = detect_hedging("The Earth orbits the Sun.")
        hedged_certainty = detect_hedging("The Earth might orbit the Sun perhaps.")

        assert definite_certainty > hedged_certainty, (
            f"Definite ({definite_certainty}) must score higher than hedged ({hedged_certainty})"
        )

        # Prove that the calibrator actually uses this signal
        calibrator = ConfidenceCalibrator()
        # Both use same base inputs, differ only in linguistic_certainty
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        mock_ledger = MagicMock()
        mock_ledger.get_axiom_provenance = AsyncMock(return_value=[])
        mock_algebra = MagicMock()
        mock_algebra.axiom_to_prime = {}

        async def measure(certainty):
            return await calibrator.calibrate(
                axiom_key="earth||orbits||sun",
                source_url="https://example.com",
                current_state=1,
                algebra=mock_algebra,
                ledger=mock_ledger,
                linguistic_certainty=certainty,
            )

        conf_definite = asyncio.get_event_loop().run_until_complete(measure(definite_certainty))
        conf_hedged = asyncio.get_event_loop().run_until_complete(measure(hedged_certainty))

        assert conf_definite > conf_hedged, (
            f"Definite confidence ({conf_definite}) must exceed hedged ({conf_hedged})"
        )


# ─── Priority 3: Node.js Scheme-Aware Verification ──────────────────

class TestNodeSchemeAware:
    """Prove verify.js dispatches on prime_scheme."""

    def test_verify_js_has_scheme_dispatch(self):
        """verifyBundle must inspect bundle.prime_scheme."""
        verify_path = os.path.join("standalone_verifier", "verify.js")
        with open(verify_path) as f:
            source = f.read()
        assert "bundle.prime_scheme" in source
        assert "sha256_64_v1" in source
        assert "sha256_128_v2" in source

    def test_reconstruct_state_accepts_scheme(self):
        """reconstructState must accept a scheme parameter."""
        verify_path = os.path.join("standalone_verifier", "verify.js")
        with open(verify_path) as f:
            source = f.read()
        assert "function reconstructState(axiomKeys, scheme" in source

    def test_v2_collision_is_hard_failure(self):
        """v2 collision handling must throw PrimeCollisionError, not advance."""
        verify_path = os.path.join("standalone_verifier", "verify.js")
        with open(verify_path) as f:
            source = f.read()
        assert "PrimeCollisionError" in source

    def test_node_v1_selftest_passes(self):
        """node verify.js --self-test must pass."""
        r = subprocess.run(
            ["node", "standalone_verifier/verify.js", "--self-test"],
            capture_output=True, text=True, timeout=30,
        )
        assert r.returncode == 0, f"v1 self-test failed: {r.stderr}"
        assert "0 failed" in r.stdout

    def test_node_v2_parity_passes(self):
        """node verify.js --v2-test must pass."""
        r = subprocess.run(
            ["node", "standalone_verifier/verify.js", "--v2-test"],
            capture_output=True, text=True, timeout=60,
        )
        assert r.returncode == 0, f"v2 parity test failed: {r.stderr}"
        assert "0 failed" in r.stdout


# ─── Priority 4: Activation Strategy ────────────────────────────────

class TestActivationStrategy:
    """Prove SUM_PRIME_SCHEME env var controls activation in a fresh process."""

    def _run_scheme_check(self, env_val=None, expect_success=True):
        """Helper: spawn subprocess with SUM_PRIME_SCHEME set."""
        env = os.environ.copy()
        if env_val is not None:
            env["SUM_PRIME_SCHEME"] = env_val
        else:
            env.pop("SUM_PRIME_SCHEME", None)

        r = subprocess.run(
            [sys.executable, "-c",
             "from internal.infrastructure.scheme_registry import CURRENT_SCHEME; print(CURRENT_SCHEME)"],
            capture_output=True, text=True, env=env, timeout=10,
        )
        return r

    def test_default_is_v1(self):
        """Unset SUM_PRIME_SCHEME → v1."""
        r = self._run_scheme_check(env_val="")
        assert r.returncode == 0
        assert r.stdout.strip() == "sha256_64_v1"

    def test_explicit_v1(self):
        """SUM_PRIME_SCHEME=sha256_64_v1 → v1."""
        r = self._run_scheme_check(env_val="sha256_64_v1")
        assert r.returncode == 0
        assert r.stdout.strip() == "sha256_64_v1"

    def test_explicit_v2(self):
        """SUM_PRIME_SCHEME=sha256_128_v2 → v2."""
        r = self._run_scheme_check(env_val="sha256_128_v2")
        assert r.returncode == 0
        assert r.stdout.strip() == "sha256_128_v2"

    def test_unknown_scheme_crashes(self):
        """SUM_PRIME_SCHEME=garbage → crash with FATAL."""
        r = self._run_scheme_check(env_val="sha256_999_v99")
        assert r.returncode != 0, "Unknown scheme must crash"
        assert "FATAL" in r.stderr, f"Must say FATAL: {r.stderr}"

    def test_v2_process_is_compatible_with_v2(self):
        """A v2 process must accept v2 peers and reject v1."""
        env = os.environ.copy()
        env["SUM_PRIME_SCHEME"] = "sha256_128_v2"
        r = subprocess.run(
            [sys.executable, "-c",
             "from internal.infrastructure.scheme_registry import is_compatible; "
             "print(is_compatible('sha256_128_v2'), is_compatible('sha256_64_v1'))"],
            capture_output=True, text=True, env=env, timeout=10,
        )
        assert r.returncode == 0
        assert r.stdout.strip() == "True False"
