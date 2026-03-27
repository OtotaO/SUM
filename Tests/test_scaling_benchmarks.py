"""
Scaling Benchmarks — Empirical Characterization

Measures actual performance at 100, 1K, 10K, and 50K axioms to
establish honest scaling characteristics for the README.

These are NOT pass/fail tests — they measure and report.
The ONLY assertion is that operations complete without error.
"""

import math
import time
import statistics
import pytest

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra


def _bench(fn, iterations=10):
    """Run fn() N times, return stats dict."""
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {
        "mean_ms": round(statistics.mean(times) * 1000, 2),
        "median_ms": round(statistics.median(times) * 1000, 2),
        "p95_ms": round(sorted(times)[int(len(times) * 0.95)] * 1000, 2),
        "max_ms": round(max(times) * 1000, 2),
    }


SCALE_POINTS = [100, 1_000, 10_000]


class TestScalingCharacteristics:
    """Measure, don't assert. Report honest numbers."""

    @pytest.fixture(params=SCALE_POINTS, ids=[f"{n}_axioms" for n in SCALE_POINTS])
    def scaled_state(self, request):
        n = request.param
        algebra = GodelStateAlgebra()
        state = 1
        primes = []
        for i in range(n):
            p = algebra.get_or_mint_prime(f"e{i}", "rel", f"v{i}")
            state = math.lcm(state, p)
            primes.append(p)
        return algebra, state, primes, n

    def test_state_bit_length(self, scaled_state):
        algebra, state, primes, n = scaled_state
        bits = state.bit_length()
        digits = len(str(state))
        print(f"\n  [{n} axioms] bit_length={bits}, decimal_digits={digits}")
        assert bits > 0

    def test_entailment_check(self, scaled_state):
        algebra, state, primes, n = scaled_state
        target = primes[n // 2]
        r = _bench(lambda: state % target == 0, iterations=100)
        print(f"\n  [{n} axioms] entailment: {r['mean_ms']:.3f}ms (p95={r['p95_ms']:.3f}ms)")
        assert True

    def test_lcm_merge_single(self, scaled_state):
        algebra, state, primes, n = scaled_state
        new_p = algebra.get_or_mint_prime("fresh", "new", f"axiom_{n}")
        r = _bench(lambda: math.lcm(state, new_p), iterations=50)
        print(f"\n  [{n} axioms] lcm_merge: {r['mean_ms']:.3f}ms (p95={r['p95_ms']:.3f}ms)")
        assert True

    def test_gcd_sync_delta(self, scaled_state):
        algebra, state, primes, n = scaled_state
        client_state = 1
        for p in primes[:int(n * 0.7)]:
            client_state = math.lcm(client_state, p)
        r = _bench(lambda: math.gcd(state, client_state), iterations=20)
        print(f"\n  [{n} axioms] gcd_delta: {r['mean_ms']:.3f}ms (p95={r['p95_ms']:.3f}ms)")
        assert True

    def test_get_active_axioms_scan(self, scaled_state):
        algebra, state, primes, n = scaled_state
        r = _bench(lambda: algebra.get_active_axioms(state), iterations=10)
        print(f"\n  [{n} axioms] get_active_axioms: {r['mean_ms']:.3f}ms (p95={r['p95_ms']:.3f}ms)")
        assert True

    def test_paradox_detection(self, scaled_state):
        algebra, state, primes, n = scaled_state
        r = _bench(lambda: algebra.detect_curvature_paradoxes(state), iterations=10)
        print(f"\n  [{n} axioms] paradox_scan: {r['mean_ms']:.3f}ms (p95={r['p95_ms']:.3f}ms)")
        assert True


class TestScalingSummary:

    def test_print_scaling_table(self):
        print(f"\n{'='*80}")
        print(f"  SUM Scaling Characterization")
        print(f"{'='*80}")
        print(f"  {'N':>8} | {'Bits':>10} | {'Entail':>10} | {'LCM':>10} | {'GCD':>10} | {'Scan':>10}")
        print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

        for n in [100, 1_000, 5_000]:
            algebra = GodelStateAlgebra()
            state = 1
            primes = []
            for i in range(n):
                p = algebra.get_or_mint_prime(f"s{i}", "r", f"o{i}")
                state = math.lcm(state, p)
                primes.append(p)

            t0 = time.perf_counter()
            for _ in range(100):
                _ = state % primes[n//2] == 0
            entail_us = (time.perf_counter() - t0) / 100 * 1e6

            fresh = algebra.get_or_mint_prime("x", "x", f"x{n}")
            t0 = time.perf_counter()
            for _ in range(50):
                _ = math.lcm(state, fresh)
            lcm_us = (time.perf_counter() - t0) / 50 * 1e6

            client = 1
            for p in primes[:n//2]:
                client = math.lcm(client, p)
            t0 = time.perf_counter()
            for _ in range(20):
                _ = math.gcd(state, client)
            gcd_us = (time.perf_counter() - t0) / 20 * 1e6

            t0 = time.perf_counter()
            for _ in range(5):
                _ = algebra.get_active_axioms(state)
            scan_us = (time.perf_counter() - t0) / 5 * 1e6

            print(f"  {n:>8} | {state.bit_length():>10} | {entail_us:>8.0f}µs | {lcm_us:>8.0f}µs | {gcd_us:>8.0f}µs | {scan_us:>8.0f}µs")

        print(f"{'='*80}")
        assert True
