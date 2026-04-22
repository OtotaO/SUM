from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..schema import PerfOperation, PerformanceMetrics

DEFAULT_SIZES: tuple[int, ...] = (1_000, 5_000, 10_000)


@dataclass(frozen=True)
class SumPerformanceRunner:
    """Measures wall-clock p50/p99 for the four core algebra operations at
    increasing corpus sizes. Synthetic axioms of the form (s{i}, p, o{i}) give
    deterministic, non-colliding primes and exercise the pure-Python path when
    the Zig core is absent.
    """

    name: str = "sum.perf"
    corpus_sizes: Sequence[int] = field(default_factory=lambda: DEFAULT_SIZES)
    samples_per_op: int = 200
    warmup: int = 20
    seed: int = 42

    def run(self) -> Sequence[PerformanceMetrics]:
        from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra

        results: list[PerformanceMetrics] = []
        for size in self.corpus_sizes:
            results.extend(self._measure_size(GodelStateAlgebra, size))
        return tuple(results)

    def _measure_size(
        self, algebra_cls: type, size: int
    ) -> list[PerformanceMetrics]:
        rng = random.Random(self.seed + size)
        triples: list[tuple[str, str, str]] = [
            (f"s{i}", "p", f"o{i}") for i in range(size)
        ]

        ingest_times_ns = self._time_ingest(algebra_cls, triples)

        # Fully-minted algebra for encode/merge/entail timings
        algebra = algebra_cls()
        for s, p, o in triples:
            algebra.get_or_mint_prime(s, p, o)
        global_state = algebra.encode_chunk_state(triples)

        encode_times_ns = self._time_op(
            lambda: algebra.encode_chunk_state(triples),
            samples=self.samples_per_op,
            warmup=self.warmup,
        )

        partitions = 4
        sub_states = [
            algebra.encode_chunk_state(triples[i::partitions])
            for i in range(partitions)
        ]
        merge_times_ns = self._time_op(
            lambda: algebra.merge_parallel_states(sub_states),
            samples=self.samples_per_op,
            warmup=self.warmup,
        )

        # Entailment: pick random single-prime hypotheses from the set
        hypotheses = [
            algebra.get_or_mint_prime(*triples[rng.randrange(size)])
            for _ in range(self.samples_per_op + self.warmup)
        ]
        entail_times_ns: list[int] = []
        for i, hyp in enumerate(hypotheses):
            t0 = time.monotonic_ns()
            algebra.verify_entailment(global_state, hyp)
            elapsed = time.monotonic_ns() - t0
            if i >= self.warmup:
                entail_times_ns.append(elapsed)

        return [
            self._to_metric("ingest", size, ingest_times_ns),
            self._to_metric("encode", size, encode_times_ns),
            self._to_metric("merge", size, merge_times_ns),
            self._to_metric("entail", size, entail_times_ns),
        ]

    @staticmethod
    def _time_ingest(
        algebra_cls: type, triples: list[tuple[str, str, str]]
    ) -> list[int]:
        """Ingest is measured as full-corpus runs amortized per triple.
        Order-dependence of minting makes per-op sampling unreliable; report
        per-triple mean across a handful of fresh-algebra runs.
        """
        times_ns: list[int] = []
        n = max(1, len(triples))
        for _ in range(5):
            algebra = algebra_cls()
            t0 = time.monotonic_ns()
            for s, p, o in triples:
                algebra.get_or_mint_prime(s, p, o)
            elapsed = time.monotonic_ns() - t0
            times_ns.append(elapsed // n)
        return times_ns

    @staticmethod
    def _time_op(
        fn: Callable[[], object], samples: int, warmup: int
    ) -> list[int]:
        times_ns: list[int] = []
        for i in range(samples + warmup):
            t0 = time.monotonic_ns()
            fn()
            elapsed = time.monotonic_ns() - t0
            if i >= warmup:
                times_ns.append(elapsed)
        return times_ns

    @staticmethod
    def _to_metric(
        op: PerfOperation, size: int, times_ns: list[int]
    ) -> PerformanceMetrics:
        if not times_ns:
            return PerformanceMetrics(
                operation=op,
                corpus_size=size,
                p50_ms=0.0,
                p99_ms=0.0,
                n_samples=0,
            )
        sorted_ns = sorted(times_ns)
        p50 = statistics.median(sorted_ns)
        p99_idx = max(0, min(len(sorted_ns) - 1, int(len(sorted_ns) * 0.99)))
        p99 = sorted_ns[p99_idx]
        return PerformanceMetrics(
            operation=op,
            corpus_size=size,
            p50_ms=p50 / 1_000_000.0,
            p99_ms=p99 / 1_000_000.0,
            n_samples=len(times_ns),
        )
