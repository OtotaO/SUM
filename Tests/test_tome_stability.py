"""
Canonical Tome Ordering Stability Tests

Verifies that the canonical tome produced for the same state integer
is always identical — byte-for-byte deterministic. This is essential
because the HMAC signature covers the tome text, so any reordering
would invalidate signatures and break P2P sync.

Author: ototao
License: Apache License 2.0
"""

import math
import pytest
from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.ensemble.tome_generator import AutoregressiveTomeGenerator


@pytest.fixture
def algebra_with_axioms():
    algebra = GodelStateAlgebra()
    axioms = [
        ("alice", "likes", "cats"),
        ("bob", "knows", "python"),
        ("earth", "orbits", "sun"),
        ("mars", "has", "moons"),
        ("water", "is", "wet"),
    ]
    for s, p, o in axioms:
        algebra.get_or_mint_prime(s, p, o)
    state = 1
    for prime in algebra.axiom_to_prime.values():
        state = math.lcm(state, prime)
    return algebra, state


class TestTomeOrderingStability:

    def test_repeated_generation_identical(self, algebra_with_axioms):
        """Same state → same tome, 100 consecutive times."""
        algebra, state = algebra_with_axioms
        gen = AutoregressiveTomeGenerator(algebra)
        baseline = gen.generate_canonical(state, "Stability Test")

        for i in range(100):
            result = gen.generate_canonical(state, "Stability Test")
            assert result == baseline, f"Tome diverged on iteration {i}"

    def test_two_generators_same_output(self, algebra_with_axioms):
        """Two independent generators produce identical tomes."""
        algebra, state = algebra_with_axioms
        gen1 = AutoregressiveTomeGenerator(algebra)
        gen2 = AutoregressiveTomeGenerator(algebra)

        tome1 = gen1.generate_canonical(state, "Gen1")
        tome2 = gen2.generate_canonical(state, "Gen1")
        assert tome1 == tome2

    def test_empty_state_stable(self):
        """State=1 produces deterministic empty tome."""
        algebra = GodelStateAlgebra()
        gen = AutoregressiveTomeGenerator(algebra)
        t1 = gen.generate_canonical(1, "Empty")
        t2 = gen.generate_canonical(1, "Empty")
        assert t1 == t2

    def test_single_axiom_stable(self):
        """Single-axiom state produces stable tome."""
        algebra = GodelStateAlgebra()
        p = algebra.get_or_mint_prime("solo", "fact", "here")
        gen = AutoregressiveTomeGenerator(algebra)
        t1 = gen.generate_canonical(p, "Solo")
        t2 = gen.generate_canonical(p, "Solo")
        assert t1 == t2

    def test_subset_state_tome_subset(self, algebra_with_axioms):
        """Superset state's tome contains all lines from subset state's tome."""
        algebra, full_state = algebra_with_axioms
        gen = AutoregressiveTomeGenerator(algebra)

        # Get just the first two axiom primes
        primes = list(algebra.axiom_to_prime.values())
        partial_state = math.lcm(primes[0], primes[1])

        full_tome = gen.generate_canonical(full_state, "Full")
        partial_tome = gen.generate_canonical(partial_state, "Partial")

        # Every fact line in partial should appear in full
        partial_facts = [l for l in partial_tome.split("\n") if l.startswith("The ")]
        full_facts = [l for l in full_tome.split("\n") if l.startswith("The ")]

        for fact in partial_facts:
            assert fact in full_facts, f"Missing fact: {fact}"

    def test_tome_hash_deterministic(self, algebra_with_axioms):
        """SHA-256 of tome text is deterministic."""
        import hashlib
        algebra, state = algebra_with_axioms
        gen = AutoregressiveTomeGenerator(algebra)

        h1 = hashlib.sha256(gen.generate_canonical(state, "Hash").encode()).hexdigest()
        h2 = hashlib.sha256(gen.generate_canonical(state, "Hash").encode()).hexdigest()
        assert h1 == h2
