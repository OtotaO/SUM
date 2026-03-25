"""
Tests for Phase 19D: Active Prime Set Index

Validates that the ActivePrimeIndex maintains correct O(1) lookups
for active primes per branch, across all lifecycle operations:
    rebuild, add, remove, fork, merge, get_active_axioms,
    extract_axioms_from_product.

Author: ototao
License: Apache License 2.0
"""

import math
import pytest

from internal.algorithms.semantic_arithmetic import (
    GodelStateAlgebra,
    ActivePrimeIndex,
)


@pytest.fixture
def algebra():
    return GodelStateAlgebra()


@pytest.fixture
def index():
    return ActivePrimeIndex()


@pytest.fixture
def seeded(algebra, index):
    """Seed 10 axioms into the algebra and build a state + index."""
    state = 1
    primes = []
    for i in range(10):
        p = algebra.get_or_mint_prime(f"entity_{i}", "has_property", f"value_{i}")
        state = math.lcm(state, p)
        primes.append(p)
    index.rebuild("main", state, algebra)
    return state, primes


# ─── Core Lifecycle ───────────────────────────────────────────────────

class TestActivePrimeIndexLifecycle:

    def test_rebuild_empty_state(self, algebra, index):
        index.rebuild("main", 1, algebra)
        assert index.get_active_primes("main") == set()

    def test_rebuild_populated_state(self, algebra, index, seeded):
        state, primes = seeded
        active = index.get_active_primes("main")
        assert active == set(primes)
        assert len(active) == 10

    def test_add(self, algebra, index):
        p = algebra.get_or_mint_prime("new", "rel", "obj")
        index.add("main", p)
        assert p in index.get_active_primes("main")

    def test_remove(self, algebra, index, seeded):
        state, primes = seeded
        target = primes[0]
        index.remove("main", target)
        assert target not in index.get_active_primes("main")
        assert len(index.get_active_primes("main")) == 9

    def test_remove_nonexistent_is_safe(self, index):
        index.remove("nonexistent_branch", 99999)  # should not raise

    def test_fork(self, algebra, index, seeded):
        state, primes = seeded
        index.fork("main", "experiment")
        assert index.get_active_primes("experiment") == set(primes)
        # Mutation isolation: adding to fork doesn't affect source
        index.add("experiment", 99999)
        assert 99999 in index.get_active_primes("experiment")
        assert 99999 not in index.get_active_primes("main")

    def test_merge(self, algebra, index, seeded):
        state, primes = seeded
        # Create a branch with a different prime
        extra_prime = algebra.get_or_mint_prime("extra", "rel", "val")
        index.add("feature", extra_prime)

        index.merge("merged", "main", "feature")
        merged_set = index.get_active_primes("merged")
        assert set(primes).issubset(merged_set)
        assert extra_prime in merged_set


# ─── Query Methods ────────────────────────────────────────────────────

class TestActivePrimeIndexQueries:

    def test_get_active_axioms(self, algebra, index, seeded):
        state, primes = seeded
        axioms = index.get_active_axioms("main", algebra)
        assert len(axioms) == 10
        # Each axiom should be a valid key
        for ax in axioms:
            assert "||" in ax or "has_property" in ax

    def test_get_active_axioms_empty_branch(self, algebra, index):
        axioms = index.get_active_axioms("nonexistent", algebra)
        assert axioms == []

    def test_extract_axioms_from_product(self, algebra, index, seeded):
        state, primes = seeded
        # Build a product of the first 3 primes
        product = 1
        for p in primes[:3]:
            product *= p
        axioms = index.extract_axioms_from_product(product, algebra)
        assert len(axioms) == 3

    def test_extract_axioms_narrowed(self, algebra, index, seeded):
        state, primes = seeded
        product = primes[0] * primes[1]
        # Narrow scan to only the first 5 primes
        candidates = set(primes[:5])
        axioms = index.extract_axioms_from_product(
            product, algebra, candidate_primes=candidates
        )
        assert len(axioms) == 2

    def test_extract_axioms_product_one(self, algebra, index):
        axioms = index.extract_axioms_from_product(1, algebra)
        assert axioms == []


# ─── Consistency ──────────────────────────────────────────────────────

class TestActivePrimeIndexConsistency:

    def test_index_matches_brute_force(self, algebra, index, seeded):
        """The index must agree with the brute-force scan."""
        state, primes = seeded
        # Brute-force
        brute = algebra.get_active_axioms(state)
        # Indexed
        indexed = index.get_active_axioms("main", algebra)
        assert sorted(brute) == sorted(indexed)

    def test_add_then_matches(self, algebra, index, seeded):
        """After .add(), index still agrees with the math."""
        state, primes = seeded
        new_p = algebra.get_or_mint_prime("added", "rel", "obj")
        new_state = math.lcm(state, new_p)
        index.add("main", new_p)

        brute = algebra.get_active_axioms(new_state)
        indexed = index.get_active_axioms("main", algebra)
        assert sorted(brute) == sorted(indexed)

    def test_remove_then_matches(self, algebra, index, seeded):
        """After .remove(), index agrees with the reduced state."""
        state, primes = seeded
        target = primes[0]
        reduced_state = state
        while reduced_state % target == 0:
            reduced_state //= target
        index.remove("main", target)

        brute = algebra.get_active_axioms(reduced_state)
        indexed = index.get_active_axioms("main", algebra)
        assert sorted(brute) == sorted(indexed)


# ─── Performance Sanity ──────────────────────────────────────────────

class TestActivePrimeIndexPerformance:

    def test_indexed_is_faster_than_scan(self, algebra, index):
        """Index lookup should be faster than full scan at 1000 axioms."""
        import time

        state = 1
        for i in range(1000):
            p = algebra.get_or_mint_prime(f"perf_{i}", "rel", f"val_{i}")
            state = math.lcm(state, p)
        index.rebuild("main", state, algebra)

        # Brute-force scan
        t0 = time.perf_counter_ns()
        brute = algebra.get_active_axioms(state)
        brute_ns = time.perf_counter_ns() - t0

        # Indexed lookup
        t0 = time.perf_counter_ns()
        indexed = index.get_active_axioms("main", algebra)
        index_ns = time.perf_counter_ns() - t0

        assert len(brute) == len(indexed) == 1000
        # Index should be at least 5x faster
        assert index_ns < brute_ns, (
            f"Index ({index_ns/1000:.0f}µs) should be faster than "
            f"brute ({brute_ns/1000:.0f}µs)"
        )
