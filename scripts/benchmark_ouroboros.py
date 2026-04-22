#!/usr/bin/env python3
"""
Ouroboros Performance Benchmark

Measures encode/decode/verify costs at various axiom counts:
    - Encode time  (triplets → Gödel integer via LCM)
    - Canonical decode time  (integer → tome text)
    - Round-trip verify time  (full Ouroboros proof)
    - Export bundle time  (including HMAC-SHA256 signature)

Usage:
    python scripts/benchmark_ouroboros.py

Phase 15: Canonical Semantic ABI.
"""

import math
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve
from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
from sum_engine_internal.ensemble.ouroboros import OuroborosVerifier
from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec


def generate_synthetic_triplets(n: int):
    """Generate n synthetic (subject, predicate, object) triplets."""
    triplets = []
    subjects = [f"entity_{i // 5}" for i in range(n)]
    predicates = ["has_property", "relates_to", "contains", "produces", "requires"]
    for i in range(n):
        s = subjects[i]
        p = predicates[i % len(predicates)]
        o = f"value_{i}"
        triplets.append((s, p, o))
    return triplets


def benchmark_encode(algebra, triplets):
    """Encode triplets → Gödel integer."""
    start = time.perf_counter()
    state = 1
    for s, p, o in triplets:
        prime = algebra.get_or_mint_prime(s, p, o)
        if state % prime != 0:
            state = math.lcm(state, prime)
    elapsed = time.perf_counter() - start
    return state, elapsed


def benchmark_decode(tome_generator, state, title="Benchmark Tome"):
    """Canonical decode: integer → tome text."""
    start = time.perf_counter()
    tome = tome_generator.generate_canonical(state, title)
    elapsed = time.perf_counter() - start
    return tome, elapsed


def benchmark_verify(ouroboros, state):
    """Full Ouroboros round-trip verification."""
    start = time.perf_counter()
    proof = ouroboros.verify_from_state(state)
    elapsed = time.perf_counter() - start
    return proof, elapsed


def benchmark_export(codec, state, branch="benchmark"):
    """Export bundle (including HMAC-SHA256 signature)."""
    start = time.perf_counter()
    bundle = codec.export_bundle(state, branch=branch)
    elapsed = time.perf_counter() - start
    return bundle, elapsed


def run_benchmarks():
    sizes = [10, 100, 1000, 5000]

    print("=" * 72)
    print("  Ouroboros Performance Benchmark — Phase 15")
    print("=" * 72)
    print()
    print(f"{'Axioms':>8} │ {'Encode':>10} │ {'Decode':>10} │ {'Verify':>10} │ {'Export':>10} │ {'Conserved':>9} │ {'Digits':>8}")
    print("─" * 8 + "─┼─" + "─" * 10 + "─┼─" + "─" * 10 + "─┼─" + "─" * 10 + "─┼─" + "─" * 10 + "─┼─" + "─" * 9 + "─┼─" + "─" * 8)

    for n in sizes:
        # Fresh algebra per size to avoid cross-contamination
        algebra = GodelStateAlgebra()
        sieve = DeterministicSieve()
        tome_gen = AutoregressiveTomeGenerator(algebra)
        ouroboros = OuroborosVerifier(algebra, sieve, tome_gen)
        codec = CanonicalCodec(algebra, tome_gen, signing_key="benchmark-key")

        triplets = generate_synthetic_triplets(n)

        state, t_encode = benchmark_encode(algebra, triplets)
        _, t_decode = benchmark_decode(tome_gen, state)
        proof, t_verify = benchmark_verify(ouroboros, state)
        _, t_export = benchmark_export(codec, state)

        digits = len(str(state))
        conserved = "✅" if proof.is_conserved else "❌"

        print(
            f"{n:>8} │ {t_encode*1000:>8.2f}ms │ {t_decode*1000:>8.2f}ms │ "
            f"{t_verify*1000:>8.2f}ms │ {t_export*1000:>8.2f}ms │ "
            f"{conserved:>9} │ {digits:>8}"
        )

    print()
    print("=" * 72)
    print("  All benchmarks complete.")
    print("=" * 72)


if __name__ == "__main__":
    run_benchmarks()
