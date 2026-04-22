#!/usr/bin/env python3
"""
SUM Hot-Path Profiling Harness

Benchmarks the core algebra operations that scan the prime registry,
to determine whether Phase 19D (Active Prime Set Index) is justified.

Key question: Is the O(n) scan over prime_to_axiom actually hot?

Usage:
    .venv/bin/python scripts/profile_hot_paths.py

Author: ototao
License: Apache License 2.0
"""

import math
import time
import statistics
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra


# ─── Configuration ────────────────────────────────────────────────────

SCALE_FACTORS = [10, 100, 500, 1000, 5000]
ITERATIONS = 50  # repetitions per measurement
SYNC_OVERLAP_RATIO = 0.7  # 70% overlap between server and client states


# ─── Helpers ──────────────────────────────────────────────────────────

def _build_state(algebra: GodelStateAlgebra, n: int) -> tuple:
    """Mint n axioms and return (state_integer, algebra)."""
    state = 1
    for i in range(n):
        prime = algebra.get_or_mint_prime(
            f"entity_{i}", "has_property", f"value_{i}"
        )
        state = math.lcm(state, prime)
    return state


def _timeit(fn, iterations=ITERATIONS) -> dict:
    """Time a function over N iterations, returning stats in microseconds."""
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        result = fn()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1_000)  # nanoseconds → microseconds
    return {
        "mean_us": round(statistics.mean(times), 1),
        "median_us": round(statistics.median(times), 1),
        "p95_us": round(sorted(times)[int(len(times) * 0.95)], 1),
        "min_us": round(min(times), 1),
        "max_us": round(max(times), 1),
    }


# ─── Benchmark Functions ─────────────────────────────────────────────

def bench_get_active_axioms(algebra, state, n):
    """O(n) scan: check each known prime for divisibility."""
    return _timeit(lambda: algebra.get_active_axioms(state))


def bench_get_quantum_neighborhood(algebra, state, hops):
    """GraphRAG traversal: node lookup + filtered expansion."""
    # Pick a node that exists
    node = "entity_0"
    return _timeit(lambda: algebra.get_quantum_neighborhood(state, [node], hops))


def bench_detect_curvature_paradoxes(algebra, state):
    """Paradox scan: iterate exclusion zones, check divisibility."""
    return _timeit(lambda: algebra.detect_curvature_paradoxes(state))


def bench_calculate_network_delta(algebra, server_state, client_state):
    """Sync delta: GCD + axiom extraction from quotient."""
    return _timeit(lambda: algebra.calculate_network_delta(server_state, client_state))


def bench_prime_minting(n):
    """Measure cold-start prime minting cost (SHA-256 → nextprime)."""
    algebra = GodelStateAlgebra()
    def mint_all():
        for i in range(n):
            algebra.get_or_mint_prime(f"fresh_{i}", "rel", f"obj_{i}")
    # Only 5 iterations — destructive
    return _timeit(mint_all, iterations=5)


def bench_lcm_merge(algebra, state, n):
    """LCM merge of a new axiom into an existing large state."""
    fresh_prime = algebra.get_or_mint_prime("zz_fresh", "zz_rel", "zz_obj")
    return _timeit(lambda: math.lcm(state, fresh_prime))


def bench_entailment_check(algebra, state):
    """Single modulo check (entailment verification)."""
    # Pick a prime that divides the state
    some_prime = list(algebra.prime_to_axiom.keys())[0]
    return _timeit(lambda: state % some_prime == 0)


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  SUM Hot-Path Profiling Harness")
    print("  Measuring algebra operations at varying scale")
    print("=" * 72)
    print()

    results = {}

    for n in SCALE_FACTORS:
        print(f"─── Scale: {n} axioms ────────────────────────────────")
        algebra = GodelStateAlgebra()

        # Build state
        t0 = time.perf_counter()
        state = _build_state(algebra, n)
        build_time = time.perf_counter() - t0
        print(f"  State built in {build_time:.3f}s  "
              f"(bit-length={state.bit_length()}, "
              f"primes={len(algebra.prime_to_axiom)})")

        results[n] = {"build_time_s": round(build_time, 3)}

        # 1. get_active_axioms — THE 19D CANDIDATE
        r = bench_get_active_axioms(algebra, state, n)
        results[n]["get_active_axioms"] = r
        print(f"  get_active_axioms:       {r['mean_us']:>10.1f} µs  "
              f"(p95={r['p95_us']:.1f})")

        # 2. get_quantum_neighborhood (1-hop)
        r = bench_get_quantum_neighborhood(algebra, state, hops=1)
        results[n]["neighborhood_1hop"] = r
        print(f"  neighborhood(1-hop):     {r['mean_us']:>10.1f} µs  "
              f"(p95={r['p95_us']:.1f})")

        # 3. get_quantum_neighborhood (2-hop)
        r = bench_get_quantum_neighborhood(algebra, state, hops=2)
        results[n]["neighborhood_2hop"] = r
        print(f"  neighborhood(2-hop):     {r['mean_us']:>10.1f} µs  "
              f"(p95={r['p95_us']:.1f})")

        # 4. detect_curvature_paradoxes
        r = bench_detect_curvature_paradoxes(algebra, state)
        results[n]["detect_paradoxes"] = r
        print(f"  detect_paradoxes:        {r['mean_us']:>10.1f} µs  "
              f"(p95={r['p95_us']:.1f})")

        # 5. calculate_network_delta (70% overlap)
        overlap_count = int(n * SYNC_OVERLAP_RATIO)
        client_algebra = GodelStateAlgebra()
        client_state = _build_state(client_algebra, overlap_count)
        # Merge client primes into server algebra for fair comparison
        for key, prime in client_algebra.axiom_to_prime.items():
            if key not in algebra.axiom_to_prime:
                algebra.axiom_to_prime[key] = prime
                algebra.prime_to_axiom[prime] = key
        r = bench_calculate_network_delta(algebra, state, client_state)
        results[n]["network_delta"] = r
        print(f"  calculate_network_delta: {r['mean_us']:>10.1f} µs  "
              f"(p95={r['p95_us']:.1f})")

        # 6. LCM merge (single axiom)
        r = bench_lcm_merge(algebra, state, n)
        results[n]["lcm_merge"] = r
        print(f"  lcm_merge(1 axiom):      {r['mean_us']:>10.1f} µs  "
              f"(p95={r['p95_us']:.1f})")

        # 7. Entailment check (single modulo)
        r = bench_entailment_check(algebra, state)
        results[n]["entailment_check"] = r
        print(f"  entailment_check:        {r['mean_us']:>10.1f} µs  "
              f"(p95={r['p95_us']:.1f})")

        print()

    # ─── Summary: Is 19D Justified? ───────────────────────────────────
    print("=" * 72)
    print("  VERDICT: Should Phase 19D (Active Prime Set Index) be implemented?")
    print("=" * 72)
    print()

    # Check if get_active_axioms is > 1ms at any scale
    hot = False
    for n in SCALE_FACTORS:
        mean = results[n]["get_active_axioms"]["mean_us"]
        if mean > 1000:
            hot = True
            print(f"  ⚠️  get_active_axioms at {n} axioms = {mean:.0f} µs "
                  f"({mean/1000:.1f} ms) — HOT")
        else:
            print(f"  ✅  get_active_axioms at {n} axioms = {mean:.0f} µs — OK")

    print()
    if hot:
        print("  RECOMMENDATION: 19D is JUSTIFIED — prime scans are materializing as")
        print("  a cost center at scale. An active prime set index would eliminate the")
        print("  O(n) scan over prime_to_axiom.")
    else:
        print("  RECOMMENDATION: 19D is NOT JUSTIFIED yet — prime scans are fast enough.")
        print("  Revisit when axiom count exceeds the hot threshold observed above.")

    print()
    print("  Profile complete.")


if __name__ == "__main__":
    main()
