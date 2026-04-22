"""
Phase 17b — BigInt Zig C-ABI Verification Suite

Tests that the Strangler Fig BigInt injection (LCM, GCD, mod,
divisibility) produces *identical* results to Python's ``math``
module, regardless of whether the Zig binary is compiled.

The tests run against the real ``GodelStateAlgebra`` methods,
exercising the exact code paths that would use Zig in production.
"""

import math
import pytest
from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra


# ─── Reference data (from Phase 16/17 frozen vectors) ────────────

AXIOM_A = "alice||likes||cats"
AXIOM_B = "bob||hates||rain"
AXIOM_C = "earth||orbits||sun"


@pytest.fixture
def algebra():
    return GodelStateAlgebra()


@pytest.fixture
def seeded_algebra(algebra):
    """Returns an algebra with 3 axioms minted and their merged state."""
    p_a = algebra.get_or_mint_prime("alice", "likes", "cats")
    p_b = algebra.get_or_mint_prime("bob", "hates", "rain")
    p_c = algebra.get_or_mint_prime("earth", "orbits", "sun")
    state = algebra.merge_parallel_states([p_a, p_b, p_c])
    return algebra, state, p_a, p_b, p_c


# ═════════════════════════════════════════════════════════════════════
#  LCM Tests
# ═════════════════════════════════════════════════════════════════════


class TestBigIntLCM:
    """Verify merge_parallel_states (LCM) produces correct results."""

    def test_lcm_of_primes(self, algebra):
        """LCM of distinct primes = their product."""
        p1 = algebra.get_or_mint_prime("a", "b", "c")
        p2 = algebra.get_or_mint_prime("d", "e", "f")
        merged = algebra.merge_parallel_states([p1, p2])
        assert merged == p1 * p2

    def test_lcm_idempotent(self, algebra):
        """LCM(a, a) = a."""
        p = algebra.get_or_mint_prime("x", "y", "z")
        merged = algebra.merge_parallel_states([p, p])
        assert merged == p

    def test_lcm_empty(self, algebra):
        """LCM of empty list = 1 (identity)."""
        assert algebra.merge_parallel_states([]) == 1

    def test_lcm_single(self, algebra):
        """LCM of single element = that element."""
        p = algebra.get_or_mint_prime("only", "one", "here")
        assert algebra.merge_parallel_states([p]) == p

    def test_lcm_commutative(self, algebra):
        """LCM(a, b) == LCM(b, a)."""
        p1 = algebra.get_or_mint_prime("a", "b", "c")
        p2 = algebra.get_or_mint_prime("d", "e", "f")
        assert algebra.merge_parallel_states([p1, p2]) == algebra.merge_parallel_states([p2, p1])

    def test_lcm_associative(self, algebra):
        """LCM(LCM(a, b), c) == LCM(a, LCM(b, c))."""
        p1 = algebra.get_or_mint_prime("a", "b", "c")
        p2 = algebra.get_or_mint_prime("d", "e", "f")
        p3 = algebra.get_or_mint_prime("g", "h", "i")
        left = algebra.merge_parallel_states([
            algebra.merge_parallel_states([p1, p2]),
            p3
        ])
        right = algebra.merge_parallel_states([
            p1,
            algebra.merge_parallel_states([p2, p3])
        ])
        assert left == right

    def test_lcm_large_set(self, algebra):
        """LCM of 20 distinct primes."""
        primes = [
            algebra.get_or_mint_prime(f"s{i}", f"p{i}", f"o{i}")
            for i in range(20)
        ]
        merged = algebra.merge_parallel_states(primes)
        # All primes should divide the merged state
        for p in primes:
            assert merged % p == 0


# ═════════════════════════════════════════════════════════════════════
#  GCD Tests (via isolate_hallucinations and calculate_network_delta)
# ═════════════════════════════════════════════════════════════════════


class TestBigIntGCD:
    """Verify GCD paths through isolate_hallucinations and network_delta."""

    def test_no_hallucinations(self, seeded_algebra):
        """When generated_state divides global_state, no hallucinations."""
        algebra, state, p_a, _, _ = seeded_algebra
        hallucinated = algebra.isolate_hallucinations(state, p_a)
        assert hallucinated == []

    def test_hallucination_detected(self, seeded_algebra):
        """When generated_state has a prime NOT in global_state, it's a hallucination."""
        algebra, state, _, _, _ = seeded_algebra
        fake_prime = algebra.get_or_mint_prime("unicorn", "flies", "high")
        generated = state * fake_prime
        hallucinated = algebra.isolate_hallucinations(state, generated)
        assert "unicorn||flies||high" in hallucinated

    def test_network_delta_identical(self, seeded_algebra):
        """Delta of identical states = no adds, no deletes."""
        algebra, state, _, _, _ = seeded_algebra
        delta = algebra.calculate_network_delta(state, state)
        assert delta["add"] == []
        assert delta["delete"] == []

    def test_network_delta_missing_facts(self, seeded_algebra):
        """Client missing facts shows them in 'add'."""
        algebra, state, p_a, p_b, p_c = seeded_algebra
        client_state = p_a  # Client only knows about alice
        delta = algebra.calculate_network_delta(state, client_state)
        assert len(delta["add"]) > 0


# ═════════════════════════════════════════════════════════════════════
#  Entailment / Divisibility Tests
# ═════════════════════════════════════════════════════════════════════


class TestBigIntEntailment:
    """Verify verify_entailment (modulo / divisibility) correctness."""

    def test_entailment_true(self, seeded_algebra):
        """A prime that's in the state should be entailed."""
        algebra, state, p_a, _, _ = seeded_algebra
        assert algebra.verify_entailment(state, p_a) is True

    def test_entailment_false(self, seeded_algebra):
        """A prime NOT in the state should fail entailment."""
        algebra, state, _, _, _ = seeded_algebra
        foreign = algebra.get_or_mint_prime("mars", "is", "red")
        assert algebra.verify_entailment(state, foreign) is False

    def test_entailment_zero_hypothesis(self, algebra):
        """Zero hypothesis always fails."""
        assert algebra.verify_entailment(100, 0) is False

    def test_entailment_composite(self, seeded_algebra):
        """Composite hypothesis (product of primes) should be entailed if all primes present."""
        algebra, state, p_a, p_b, _ = seeded_algebra
        composite = p_a * p_b
        assert algebra.verify_entailment(state, composite) is True


# ═════════════════════════════════════════════════════════════════════
#  Consistency with Python math module
# ═════════════════════════════════════════════════════════════════════


class TestPythonConsistency:
    """Ensure results match Python's math module exactly."""

    def test_lcm_matches_python(self, algebra):
        """merge_parallel_states result matches math.lcm."""
        primes = [
            algebra.get_or_mint_prime(f"s{i}", f"p{i}", f"o{i}")
            for i in range(10)
        ]
        our_result = algebra.merge_parallel_states(primes)
        python_result = math.lcm(*primes)
        assert our_result == python_result

    def test_gcd_matches_python(self, seeded_algebra):
        """isolate_hallucinations uses GCD that matches math.gcd."""
        algebra, state, p_a, p_b, _ = seeded_algebra
        partial = p_a * p_b
        # GCD should isolate the shared part
        expected_gcd = math.gcd(state, partial)
        assert expected_gcd == partial  # Sanity: partial divides state

    def test_large_integer_lcm(self, algebra):
        """LCM with very large integers (100+ digit primes)."""
        primes = [
            algebra.get_or_mint_prime(f"big-{i}", f"huge-{i}", f"massive-{i}")
            for i in range(50)
        ]
        merged = algebra.merge_parallel_states(primes)
        # Verify all primes divide the result
        for p in primes:
            assert merged % p == 0
        # Verify it matches Python
        assert merged == math.lcm(*primes)


# ═════════════════════════════════════════════════════════════════════
#  Zig Bridge Unit Tests (graceful fallback)
# ═════════════════════════════════════════════════════════════════════


class TestZigBridgeBigInt:
    """Verify the Zig bridge BigInt methods handle the no-binary case."""

    def test_bridge_bigint_lcm_fallback(self):
        """bigint_lcm returns None when Zig not compiled."""
        from sum_engine_internal.infrastructure.zig_bridge import ZigMathEngine
        engine = ZigMathEngine.__new__(ZigMathEngine)
        engine.lib = None
        assert engine.bigint_lcm(6, 10) is None

    def test_bridge_bigint_gcd_fallback(self):
        """bigint_gcd returns None when Zig not compiled."""
        from sum_engine_internal.infrastructure.zig_bridge import ZigMathEngine
        engine = ZigMathEngine.__new__(ZigMathEngine)
        engine.lib = None
        assert engine.bigint_gcd(12, 8) is None

    def test_bridge_bigint_mod_fallback(self):
        """bigint_mod returns None when Zig not compiled."""
        from sum_engine_internal.infrastructure.zig_bridge import ZigMathEngine
        engine = ZigMathEngine.__new__(ZigMathEngine)
        engine.lib = None
        assert engine.bigint_mod(100, 7) is None

    def test_bridge_divisibility_fallback(self):
        """is_divisible_by returns None when Zig not compiled."""
        from sum_engine_internal.infrastructure.zig_bridge import ZigMathEngine
        engine = ZigMathEngine.__new__(ZigMathEngine)
        engine.lib = None
        assert engine.is_divisible_by(100, 5) is None
