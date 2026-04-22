"""
Phase 17c — Performance Benchmarks: Zig vs Python Fallback

Measures the throughput of every Strangler Fig call site to quantify
the bare-metal speedup from the Zig C-ABI core.

Each benchmark runs N iterations and reports mean/median/min/max.
When Zig is not compiled, results reflect pure-Python performance
and serve as the baseline for comparison.

Author: ototao
License: Apache License 2.0
"""

import math
import time
import statistics
import hashlib
import pytest
from typing import List, Tuple

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_algebra_with_primes(n: int) -> Tuple[GodelStateAlgebra, List[int]]:
    """Create an algebra and mint n deterministic primes."""
    algebra = GodelStateAlgebra()
    primes = []
    for i in range(n):
        p = algebra.get_or_mint_prime(f"bench_s{i}", "relates_to", f"bench_o{i}")
        primes.append(p)
    return algebra, primes


def _bench(fn, iterations: int = 1000) -> dict:
    """Run fn() iterations times and return timing statistics."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
    return {
        "iterations": iterations,
        "mean_ns": statistics.mean(times),
        "median_ns": statistics.median(times),
        "min_ns": min(times),
        "max_ns": max(times),
        "stddev_ns": statistics.stdev(times) if len(times) > 1 else 0,
    }


# ---------------------------------------------------------------------------
# Benchmark Tests
# ---------------------------------------------------------------------------

class TestBenchLCM:
    """LCM benchmarks — merge_parallel_states hot path."""

    def test_lcm_small_primes(self):
        """LCM of two 64-bit primes (u64 fast path)."""
        algebra, primes = _make_algebra_with_primes(4)
        state_a = math.lcm(primes[0], primes[1])
        state_b = math.lcm(primes[2], primes[3])

        result = _bench(lambda: algebra.merge_parallel_states([state_a, state_b]))
        print(f"\n  LCM small:  {result['mean_ns']/1000:.1f} µs (median {result['median_ns']/1000:.1f} µs)")
        assert result["mean_ns"] > 0

    def test_lcm_large_composites(self):
        """LCM of two 512-bit composite integers."""
        algebra, primes = _make_algebra_with_primes(20)
        state_a = 1
        state_b = 1
        for p in primes[:10]:
            state_a = math.lcm(state_a, p)
        for p in primes[10:]:
            state_b = math.lcm(state_b, p)

        assert state_a.bit_length() > 200  # Confirm BigInt territory

        result = _bench(lambda: algebra.merge_parallel_states([state_a, state_b]), iterations=500)
        print(f"\n  LCM large:  {result['mean_ns']/1000:.1f} µs (median {result['median_ns']/1000:.1f} µs)")
        assert result["mean_ns"] > 0


class TestBenchGCD:
    """GCD benchmarks — isolate_hallucinations hot path."""

    def test_gcd_small(self):
        """GCD of two 64-bit integers."""
        algebra, primes = _make_algebra_with_primes(4)
        # Create overlapping states (shared prime p[0])
        state_a = primes[0] * primes[1]
        state_b = primes[0] * primes[2]

        result = _bench(lambda: algebra.isolate_hallucinations(state_a, state_b))
        print(f"\n  GCD small:  {result['mean_ns']/1000:.1f} µs (median {result['median_ns']/1000:.1f} µs)")
        assert result["mean_ns"] > 0

    def test_gcd_large(self):
        """GCD of two 512-bit composites with partial overlap."""
        algebra, primes = _make_algebra_with_primes(30)
        state_a = 1
        state_b = 1
        # 10 shared primes, 10 unique each
        for p in primes[:20]:
            state_a = math.lcm(state_a, p)
        for p in primes[10:]:
            state_b = math.lcm(state_b, p)

        result = _bench(lambda: algebra.calculate_network_delta(state_a, state_b), iterations=500)
        print(f"\n  GCD large:  {result['mean_ns']/1000:.1f} µs (median {result['median_ns']/1000:.1f} µs)")
        assert result["mean_ns"] > 0


class TestBenchEntailment:
    """Entailment benchmarks — verify_entailment hot path."""

    def test_entailment_single_prime(self):
        """Divisibility check with a single u64 prime (fast path)."""
        algebra, primes = _make_algebra_with_primes(5)
        state = 1
        for p in primes:
            state = math.lcm(state, p)

        # Entailed prime
        target = primes[2]
        result = _bench(lambda: algebra.verify_entailment(state, target))
        print(f"\n  Entail u64: {result['mean_ns']/1000:.1f} µs (median {result['median_ns']/1000:.1f} µs)")
        assert result["mean_ns"] > 0

    def test_entailment_composite(self):
        """Divisibility check with a composite > 64 bits (BigInt mod path)."""
        algebra, primes = _make_algebra_with_primes(10)
        state = 1
        for p in primes:
            state = math.lcm(state, p)

        hypothesis = 1
        for p in primes[:5]:
            hypothesis = math.lcm(hypothesis, p)

        assert hypothesis.bit_length() > 64  # Confirm BigInt path

        result = _bench(lambda: algebra.verify_entailment(state, hypothesis))
        print(f"\n  Entail big: {result['mean_ns']/1000:.1f} µs (median {result['median_ns']/1000:.1f} µs)")
        assert result["mean_ns"] > 0


class TestBenchPrimeGeneration:
    """Prime generation benchmark — _deterministic_prime hot path."""

    def test_prime_generation(self):
        """Generate a deterministic prime from an axiom key."""
        algebra = GodelStateAlgebra()

        i = [0]
        def gen():
            p = algebra._deterministic_prime(f"bench||key||{i[0]}")
            i[0] += 1
            return p

        result = _bench(gen)
        print(f"\n  Prime gen:  {result['mean_ns']/1000:.1f} µs (median {result['median_ns']/1000:.1f} µs)")
        assert result["mean_ns"] > 0


class TestBenchSummary:
    """Aggregated benchmark summary for the experiment ledger."""

    def test_print_summary_table(self):
        """Print a formatted summary table of all benchmarks."""
        algebra, primes = _make_algebra_with_primes(20)

        # Build states
        small_a = math.lcm(primes[0], primes[1])
        small_b = math.lcm(primes[2], primes[3])
        large_a = 1
        large_b = 1
        for p in primes[:10]:
            large_a = math.lcm(large_a, p)
        for p in primes[10:]:
            large_b = math.lcm(large_b, p)

        benchmarks = {
            "LCM small":   _bench(lambda: algebra.merge_parallel_states([small_a, small_b]), 500),
            "LCM large":   _bench(lambda: algebra.merge_parallel_states([large_a, large_b]), 500),
            "GCD small":   _bench(lambda: algebra.isolate_hallucinations(small_a, small_b), 500),
            "GCD large":   _bench(lambda: algebra.calculate_network_delta(large_a, large_b), 500),
            "Entail u64":  _bench(lambda: algebra.verify_entailment(large_a, primes[0]), 500),
            "Entail big":  _bench(lambda: algebra.verify_entailment(large_a, small_a), 500),
        }

        # Detect Zig status
        try:
            from sum_engine_internal.infrastructure.zig_bridge import zig_engine
            zig_status = "⚡ ZIG ACTIVE" if zig_engine and zig_engine.lib else "🐍 Python fallback"
        except ImportError:
            zig_status = "🐍 Python fallback"

        print(f"\n{'='*60}")
        print(f"  SUM Performance Benchmarks — {zig_status}")
        print(f"{'='*60}")
        print(f"  {'Operation':<15} {'Mean µs':>10} {'Median µs':>10} {'Min µs':>10}")
        print(f"  {'-'*45}")
        for name, stats in benchmarks.items():
            print(f"  {name:<15} {stats['mean_ns']/1000:>10.1f} {stats['median_ns']/1000:>10.1f} {stats['min_ns']/1000:>10.1f}")
        print(f"{'='*60}")

        assert len(benchmarks) == 6
