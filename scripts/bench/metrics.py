from __future__ import annotations

from typing import Protocol, Sequence

from .corpus import Corpus
from .schema import (
    ExtractionMetrics,
    PerformanceMetrics,
    RegenerationMetrics,
    RoundtripMetrics,
)


class ExtractionRunner(Protocol):
    name: str

    def run(self, corpus: Corpus) -> ExtractionMetrics: ...


class RegenerationRunner(Protocol):
    name: str

    def run(self, corpus: Corpus) -> Sequence[RegenerationMetrics]: ...


class RoundtripRunner(Protocol):
    name: str

    def run(self, corpus: Corpus) -> Sequence[RoundtripMetrics]: ...


class PerformanceRunner(Protocol):
    name: str
    corpus_sizes: Sequence[int]

    def run(self) -> Sequence[PerformanceMetrics]: ...
