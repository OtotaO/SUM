"""Tests for scripts.bench.runners.regeneration — per-doc attribution logging.

The runner has two responsibilities: process one document (extract → generate →
check entailment → record which claims fail) and aggregate across a corpus.
Both are exercised with stub generator/checker objects so no API key is needed
and the tests are deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pytest

from scripts.bench.corpus import CorpusDocument, GoldTriple, JsonCorpus
from scripts.bench.runners.regeneration import (
    _aggregate,
    _process_doc,
)
from scripts.bench.schema import PerDocRegeneration, RegenerationMetrics


@dataclass
class _StubSieve:
    triples_by_text: dict[str, list[tuple[str, str, str]]]

    def extract_triplets(self, text: str) -> list[tuple[str, str, str]]:
        return self.triples_by_text.get(text, [])


@dataclass
class _StubGenerator:
    narrative: str

    async def generate_text(
        self, axiom_keys: list[str], tags: list[str]
    ) -> str:
        return self.narrative


@dataclass
class _StubCheckerResult:
    entailed: bool


@dataclass
class _StubChecker:
    """Entails by default; any triple listed in `unsupported` is rejected."""

    unsupported: set[tuple[str, str, str]]

    async def check(
        self, passage: str, claim: tuple[str, str, str]
    ) -> _StubCheckerResult:
        return _StubCheckerResult(entailed=claim not in self.unsupported)


def _doc(doc_id: str, text: str) -> CorpusDocument:
    return CorpusDocument(id=doc_id, text=text, gold_triples=())


# ─── _process_doc ────────────────────────────────────────────────────


class TestProcessDoc:
    @pytest.mark.asyncio
    async def test_all_supported_records_full_rate(self) -> None:
        triples = [("alice", "knows", "bob"), ("bob", "likes", "cats")]
        sieve = _StubSieve({"t": triples})
        gen = _StubGenerator("some prose")
        chk = _StubChecker(unsupported=set())

        record = await _process_doc(sieve, gen, chk, _doc("d1", "t"))

        assert record is not None
        assert record.doc_id == "d1"
        assert record.n_claims == 2
        assert record.n_supported == 2
        assert record.per_claim_rate == 1.0
        assert record.unsupported_claims == ()

    @pytest.mark.asyncio
    async def test_partial_support_names_the_failures(self) -> None:
        triples = [
            ("alice", "knows", "bob"),
            ("bob", "likes", "cats"),
            ("cats", "eat", "fish"),
        ]
        bad = ("cats", "eat", "fish")
        sieve = _StubSieve({"t": triples})
        gen = _StubGenerator("prose mentions alice, bob, cats")
        chk = _StubChecker(unsupported={bad})

        record = await _process_doc(sieve, gen, chk, _doc("d2", "t"))

        assert record is not None
        assert record.n_claims == 3
        assert record.n_supported == 2
        assert record.per_claim_rate == pytest.approx(2 / 3)
        assert record.unsupported_claims == (bad,)

    @pytest.mark.asyncio
    async def test_empty_extraction_returns_none(self) -> None:
        sieve = _StubSieve({})
        gen = _StubGenerator("unused")
        chk = _StubChecker(unsupported=set())

        record = await _process_doc(sieve, gen, chk, _doc("d3", "t"))

        assert record is None

    @pytest.mark.asyncio
    async def test_empty_narrative_marks_all_unsupported(self) -> None:
        triples = [("a", "b", "c"), ("d", "e", "f")]
        sieve = _StubSieve({"t": triples})
        gen = _StubGenerator("")
        chk = _StubChecker(unsupported=set())

        record = await _process_doc(sieve, gen, chk, _doc("d4", "t"))

        assert record is not None
        assert record.n_supported == 0
        assert record.per_claim_rate == 0.0
        assert set(record.unsupported_claims) == set(triples)
        assert record.narrative_excerpt == ""

    @pytest.mark.asyncio
    async def test_excerpt_truncates_long_narrative(self) -> None:
        triples = [("a", "b", "c")]
        long_text = "x" * 1000
        sieve = _StubSieve({"t": triples})
        gen = _StubGenerator(long_text)
        chk = _StubChecker(unsupported=set())

        record = await _process_doc(sieve, gen, chk, _doc("d5", "t"))

        assert record is not None
        assert len(record.narrative_excerpt) == 200
        assert record.narrative_excerpt == "x" * 200


# ─── _aggregate ──────────────────────────────────────────────────────


class TestAggregate:
    def test_mean_rate_across_docs(self) -> None:
        per_doc: Sequence[PerDocRegeneration] = (
            PerDocRegeneration("d1", 2, 2, 1.0),
            PerDocRegeneration("d2", 2, 1, 0.5),
        )
        m = _aggregate("c1", per_doc)

        assert isinstance(m, RegenerationMetrics)
        assert m.corpus_id == "c1"
        assert m.path == "freeform"
        assert m.factscore == pytest.approx(0.75)
        assert m.minicheck_entailment_rate == pytest.approx(0.75)
        assert m.n_generations == 2
        assert m.n_supported_claims == 3
        assert m.n_total_claims == 4
        assert m.epistemic_status == "empirical-benchmark"
        assert len(m.per_doc) == 2

    def test_empty_corpus(self) -> None:
        m = _aggregate("c_empty", ())

        assert m.factscore == 0.0
        assert m.n_generations == 0
        assert m.n_total_claims == 0
        assert m.per_doc == ()


# ─── integration with a tiny JsonCorpus ──────────────────────────────


class TestIntegrationWithJsonCorpus:
    @pytest.mark.asyncio
    async def test_end_to_end_with_stubs(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        corpus_file = tmp_path / "mini.json"
        corpus_file.write_text(
            '{"id": "mini", "documents": ['
            '{"id": "a", "text": "TEXT_A", "gold_triples": [["a","b","c"]]},'
            '{"id": "b", "text": "TEXT_B", "gold_triples": [["d","e","f"]]}'
            "]}",
            encoding="utf-8",
        )
        corpus = JsonCorpus.load(corpus_file)
        sieve = _StubSieve(
            {"TEXT_A": [("a", "b", "c")], "TEXT_B": [("d", "e", "f")]}
        )
        gen = _StubGenerator("prose")
        bad = ("d", "e", "f")
        chk = _StubChecker(unsupported={bad})

        per_doc = []
        for doc in corpus.documents:
            rec = await _process_doc(sieve, gen, chk, doc)
            assert rec is not None
            per_doc.append(rec)

        m = _aggregate(corpus.id, per_doc)

        assert m.corpus_id == "mini"
        assert m.factscore == pytest.approx(0.5)
        assert m.n_supported_claims == 1
        assert m.n_total_claims == 2
        assert [p.doc_id for p in m.per_doc] == ["a", "b"]
        assert m.per_doc[0].unsupported_claims == ()
        assert m.per_doc[1].unsupported_claims == (bad,)
        # GoldTriple is imported to exercise the corpus loader alongside the
        # runner — keeps this test honest about JsonCorpus shape.
        assert corpus.documents[0].gold_triples == (GoldTriple("a", "b", "c"),)
