"""Tests for scripts.bench.runners.llm_roundtrip — LLM narrative full-loop drift.

Exercises _process_doc, _aggregate, and a small integration with JsonCorpus
using stub extractor/generator objects. No network I/O, no API key required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pytest

from scripts.bench.corpus import CorpusDocument, JsonCorpus
from scripts.bench.runners.llm_roundtrip import (
    _aggregate,
    _as_set,
    _process_doc,
)
from scripts.bench.schema import LlmRoundtripMetrics, PerDocLlmRoundtrip


@dataclass
class _StubExtractor:
    """Returns a predefined triples list for each input passage."""

    by_passage: dict[str, list[tuple[str, str, str]]] = field(default_factory=dict)

    async def extract_triplets(
        self, chunk: str
    ) -> list[tuple[str, str, str]]:
        return list(self.by_passage.get(chunk, []))


@dataclass
class _StubGenerator:
    narrative: str

    async def generate_text(
        self, target_axioms: list[str], negative_constraints: list[str]
    ) -> str:
        return self.narrative


def _doc(doc_id: str, text: str) -> CorpusDocument:
    return CorpusDocument(id=doc_id, text=text, gold_triples=())


class TestAsSet:
    def test_canonicalizes_case_and_whitespace(self) -> None:
        triples = [
            ("Alice", "KNOWS", "Bob"),
            (" alice ", "knows", "bob"),
        ]
        assert _as_set(triples) == {("alice", "knows", "bob")}


# ─── _process_doc ────────────────────────────────────────────────────


class TestProcessDoc:
    @pytest.mark.asyncio
    async def test_perfect_preservation_is_zero_drift(self) -> None:
        triples = [("a", "b", "c"), ("d", "e", "f")]
        extractor = _StubExtractor({"src": triples, "PROSE": triples})
        generator = _StubGenerator("PROSE")

        record = await _process_doc(extractor, generator, _doc("d1", "src"))

        assert record is not None
        assert record.doc_id == "d1"
        assert record.n_source_axioms == 2
        assert record.n_reconstructed_axioms == 2
        assert record.drift_pct == 0.0
        assert record.missing_claims == ()
        assert record.extra_claims == ()
        assert record.narrative_excerpt == "PROSE"

    @pytest.mark.asyncio
    async def test_missing_claim_is_attributed(self) -> None:
        source = [("a", "b", "c"), ("d", "e", "f")]
        recon = [("a", "b", "c")]  # (d,e,f) lost in the narrative
        extractor = _StubExtractor({"src": source, "PROSE": recon})
        generator = _StubGenerator("PROSE")

        record = await _process_doc(extractor, generator, _doc("d2", "src"))

        assert record is not None
        assert record.n_source_axioms == 2
        assert record.n_reconstructed_axioms == 1
        assert record.drift_pct == pytest.approx(50.0)
        assert set(record.missing_claims) == {("d", "e", "f")}
        assert record.extra_claims == ()

    @pytest.mark.asyncio
    async def test_extra_hallucination_is_attributed(self) -> None:
        source = [("a", "b", "c")]
        recon = [("a", "b", "c"), ("x", "y", "z")]  # hallucinated extra
        extractor = _StubExtractor({"src": source, "PROSE": recon})
        generator = _StubGenerator("PROSE")

        record = await _process_doc(extractor, generator, _doc("d3", "src"))

        assert record is not None
        assert record.missing_claims == ()
        assert set(record.extra_claims) == {("x", "y", "z")}
        # denom = max(1, 2) = 2; sym-diff = 1 → 50%
        assert record.drift_pct == pytest.approx(50.0)

    @pytest.mark.asyncio
    async def test_mixed_missing_and_extra(self) -> None:
        source = [("a", "b", "c"), ("d", "e", "f")]
        recon = [("a", "b", "c"), ("x", "y", "z")]
        extractor = _StubExtractor({"src": source, "PROSE": recon})
        generator = _StubGenerator("PROSE")

        record = await _process_doc(extractor, generator, _doc("d4", "src"))

        assert record is not None
        # sym-diff = 2 (one missing + one extra); denom = 2 → 100%
        assert record.drift_pct == pytest.approx(100.0)
        assert set(record.missing_claims) == {("d", "e", "f")}
        assert set(record.extra_claims) == {("x", "y", "z")}

    @pytest.mark.asyncio
    async def test_empty_source_returns_none(self) -> None:
        extractor = _StubExtractor({"src": []})
        generator = _StubGenerator("unused")

        record = await _process_doc(extractor, generator, _doc("d5", "src"))

        assert record is None

    @pytest.mark.asyncio
    async def test_empty_narrative_marks_all_missing(self) -> None:
        triples = [("a", "b", "c"), ("d", "e", "f")]
        extractor = _StubExtractor({"src": triples})
        generator = _StubGenerator("")

        record = await _process_doc(extractor, generator, _doc("d6", "src"))

        assert record is not None
        assert record.drift_pct == 100.0
        assert record.n_reconstructed_axioms == 0
        assert set(record.missing_claims) == set(triples)
        assert record.extra_claims == ()
        assert record.narrative_excerpt == ""

    @pytest.mark.asyncio
    async def test_canonicalizes_before_compare(self) -> None:
        # Source returned lowercased, reconstruction in mixed case with padding;
        # canonicalization should make them equal (zero drift).
        source = [("alice", "knows", "bob")]
        recon = [("ALICE", "knows ", " BOB")]
        extractor = _StubExtractor({"src": source, "PROSE": recon})
        generator = _StubGenerator("PROSE")

        record = await _process_doc(extractor, generator, _doc("d7", "src"))

        assert record is not None
        assert record.drift_pct == 0.0

    @pytest.mark.asyncio
    async def test_excerpt_truncates(self) -> None:
        source = [("a", "b", "c")]
        extractor = _StubExtractor({"src": source, "y" * 1000: source})
        generator = _StubGenerator("y" * 1000)

        record = await _process_doc(extractor, generator, _doc("d8", "src"))

        assert record is not None
        assert len(record.narrative_excerpt) == 200


# ─── _aggregate ──────────────────────────────────────────────────────


class TestAggregate:
    def test_mean_drift(self) -> None:
        per_doc: Sequence[PerDocLlmRoundtrip] = (
            PerDocLlmRoundtrip("d1", 2, 2, 0.0),
            PerDocLlmRoundtrip("d2", 2, 1, 50.0),
        )
        m = _aggregate("c1", per_doc)

        assert isinstance(m, LlmRoundtripMetrics)
        assert m.corpus_id == "c1"
        assert m.drift_pct == pytest.approx(25.0)
        assert m.n_roundtrips == 2
        assert m.n_source_axioms_total == 4
        assert m.n_reconstructed_axioms_total == 3
        assert m.epistemic_status == "empirical-benchmark"
        assert len(m.per_doc) == 2

    def test_empty(self) -> None:
        m = _aggregate("c_empty", ())

        assert m.drift_pct == 0.0
        assert m.n_roundtrips == 0
        assert m.per_doc == ()


# ─── integration with JsonCorpus ─────────────────────────────────────


class TestIntegrationWithJsonCorpus:
    @pytest.mark.asyncio
    async def test_end_to_end_with_stubs(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        corpus_file = tmp_path / "mini.json"
        corpus_file.write_text(
            '{"id": "mini", "documents": ['
            '{"id": "p", "text": "SRC_P", "gold_triples": [["a","b","c"]]},'
            '{"id": "q", "text": "SRC_Q", "gold_triples": [["d","e","f"]]}'
            "]}",
            encoding="utf-8",
        )
        corpus = JsonCorpus.load(corpus_file)
        extractor = _StubExtractor(
            {
                "SRC_P": [("a", "b", "c")],
                "PROSE_P": [("a", "b", "c")],  # preserved
                "SRC_Q": [("d", "e", "f")],
                "PROSE_Q": [],  # lost
            }
        )

        # One generator per-doc to vary narrative output; run sequentially.
        per_doc: list[PerDocLlmRoundtrip] = []
        for doc, prose in zip(corpus.documents, ["PROSE_P", "PROSE_Q"]):
            gen = _StubGenerator(prose)
            rec = await _process_doc(extractor, gen, doc)
            assert rec is not None
            per_doc.append(rec)

        m = _aggregate(corpus.id, per_doc)

        assert m.corpus_id == "mini"
        # doc p: 0% drift; doc q: 100% drift → mean 50%
        assert m.drift_pct == pytest.approx(50.0)
        assert m.n_roundtrips == 2
        assert m.n_source_axioms_total == 2
        assert m.n_reconstructed_axioms_total == 1
        assert [p.doc_id for p in m.per_doc] == ["p", "q"]
        assert m.per_doc[0].missing_claims == ()
        assert set(m.per_doc[1].missing_claims) == {("d", "e", "f")}
