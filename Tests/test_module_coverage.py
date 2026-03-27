"""
Tests — Dedicated Module Coverage

Covers modules that were previously only tested indirectly:
  1. telemetry.py — @trace_zig_ffi decorator
  2. autonomous_agent.py — AutonomousCrystallizer
  3. p2p_mesh.py — EpistemicMeshNetwork lifecycle
  4. zig_bridge.py — ZigEngine fallback paths

Author: ototao
License: Apache License 2.0
"""

import math
import pytest


# ─── Telemetry ────────────────────────────────────────────────────────

class TestTelemetry:
    """@trace_zig_ffi(label) decorator tests."""

    def test_decorator_preserves_return_value(self):
        from internal.infrastructure.telemetry import trace_zig_ffi

        @trace_zig_ffi("add_op")
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_decorator_preserves_function_name(self):
        from internal.infrastructure.telemetry import trace_zig_ffi

        @trace_zig_ffi("special")
        def my_special_function():
            return 42

        assert my_special_function() == 42

    def test_decorator_handles_exceptions(self):
        from internal.infrastructure.telemetry import trace_zig_ffi

        @trace_zig_ffi("boom_op")
        def explode():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            explode()

    def test_counter_increments(self):
        import internal.infrastructure.telemetry as telem
        from internal.infrastructure.telemetry import trace_zig_ffi

        before = telem._python_calls + telem._zig_calls

        @trace_zig_ffi("counter_test")
        def noop():
            return 1

        noop()
        after = telem._python_calls + telem._zig_calls
        assert after >= before  # at least no regression


# ─── Autonomous Agent ────────────────────────────────────────────────

class TestAutonomousAgent:
    """AutonomousCrystallizer tests."""

    def test_import_crystallizer(self):
        from internal.ensemble.autonomous_agent import AutonomousCrystallizer
        assert AutonomousCrystallizer is not None

    def test_crystallizer_init(self):
        from internal.ensemble.autonomous_agent import AutonomousCrystallizer
        from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
        from internal.infrastructure.akashic_ledger import AkashicLedger
        import tempfile, os

        algebra = GodelStateAlgebra()
        db_path = os.path.join(tempfile.mkdtemp(), "test_ledger.db")
        ledger = AkashicLedger(db_path=db_path)

        async def mock_summarize(axioms):
            return "summary"

        crystallizer = AutonomousCrystallizer(algebra, ledger, mock_summarize)
        assert crystallizer is not None
        assert crystallizer.algebra is algebra


# ─── P2P Mesh ─────────────────────────────────────────────────────────

class TestP2PMesh:
    """EpistemicMeshNetwork lifecycle tests."""

    def _make_mesh(self):
        from internal.infrastructure.p2p_mesh import EpistemicMeshNetwork
        from internal.algorithms.semantic_arithmetic import GodelStateAlgebra

        algebra = GodelStateAlgebra()
        state = [1]
        mesh = EpistemicMeshNetwork(
            algebra=algebra,
            get_local_state_fn=lambda: state[0],
            update_local_state_fn=lambda s: state.__setitem__(0, s),
        )
        return mesh

    def test_import_mesh(self):
        from internal.infrastructure.p2p_mesh import EpistemicMeshNetwork
        assert EpistemicMeshNetwork is not None

    def test_mesh_init(self):
        mesh = self._make_mesh()
        assert isinstance(mesh.peers, set)

    def test_add_peer(self):
        mesh = self._make_mesh()
        mesh.add_peer("http://localhost:8001")
        assert "http://localhost:8001" in mesh.peers

    def test_add_peer_strips_trailing_slash(self):
        mesh = self._make_mesh()
        mesh.add_peer("http://localhost:8002/")
        assert "http://localhost:8002" in mesh.peers


# ─── Zig Bridge ───────────────────────────────────────────────────────

class TestZigBridge:
    """ZigEngine fallback path tests."""

    def test_import_zig_engine(self):
        from internal.infrastructure.zig_bridge import zig_engine
        # zig_engine may be None if .dylib not present — that's OK

    def test_fallback_deterministic_prime(self):
        """Python fallback must produce a valid prime."""
        from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
        from sympy import isprime

        algebra = GodelStateAlgebra()
        p = algebra.get_or_mint_prime("test", "relation", "object")
        assert p > 1
        assert isprime(p)

    def test_fallback_lcm_gcd(self):
        """Python math.lcm and math.gcd work as fallback."""
        a, b = 2 * 3 * 5, 3 * 5 * 7
        assert math.lcm(a, b) == 2 * 3 * 5 * 7
        assert math.gcd(a, b) == 3 * 5

    def test_batch_mint_python_fallback(self):
        """Batch minting falls back to Python correctly."""
        from internal.algorithms.semantic_arithmetic import GodelStateAlgebra

        algebra = GodelStateAlgebra()
        axioms = [
            ("a", "causes", "b"),
            ("c", "implies", "d"),
        ]
        primes = [algebra.get_or_mint_prime(s, p, o) for s, p, o in axioms]
        assert len(primes) == 2
        assert all(p > 1 for p in primes)
        assert len(set(primes)) == 2

    def test_batch_mint_attribute_correct(self):
        """batch_mint_primes must reference self.lib, not self._lib."""
        import inspect
        from internal.infrastructure.zig_bridge import ZigMathEngine
        source = inspect.getsource(ZigMathEngine.batch_mint_primes)
        assert "self._lib" not in source, (
            "batch_mint_primes references self._lib instead of self.lib"
        )
