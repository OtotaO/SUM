"""LLM narrative round-trip runner — the full-loop drift measurement.

For each corpus document:
    1. LLM.extract(doc.text)          -> source axioms
    2. LLM.generate(source axioms)    -> prose narrative
    3. LLM.extract(narrative)         -> reconstructed axioms
    4. drift = 100 * |A Δ A'| / max(|A|, |A'|)

Both extractions pass through the same LLM extractor, so canonicalization is
consistent across the comparison — the reported drift reflects what the
generator+extractor pair preserves through a prose intermediary, not
sieve-vs-LLM tokenization disagreement.

Requires pinned model IDs (SystemExit if unpinned — see run_bench.py):
    generator_model_id  SUM_BENCH_MODEL (or SUM_BENCH_GENERATOR_MODEL override)
    extractor_model_id  SUM_BENCH_MODEL (or SUM_BENCH_EXTRACTOR_MODEL override)
"""
from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass
from typing import Protocol, Sequence

from ..corpus import Corpus, CorpusDocument
from ..schema import LlmRoundtripMetrics, PerDocLlmRoundtrip

_EXCERPT_LEN = 200


class _ExtractorLike(Protocol):
    async def extract_triplets(
        self, chunk: str
    ) -> list[tuple[str, str, str]]: ...


class _GeneratorLike(Protocol):
    async def generate_text(
        self, target_axioms: list[str], negative_constraints: list[str]
    ) -> str: ...


def _as_set(
    triples: list[tuple[str, str, str]],
) -> set[tuple[str, str, str]]:
    # Stable canonicalization: lowercased + whitespace-stripped across fields.
    return {
        (s.strip().lower(), p.strip().lower(), o.strip().lower())
        for s, p, o in triples
    }


async def _process_doc(
    extractor: _ExtractorLike,
    generator: _GeneratorLike,
    doc: CorpusDocument,
) -> PerDocLlmRoundtrip | None:
    source_list = await extractor.extract_triplets(doc.text)
    source = _as_set(source_list)
    if not source:
        return None

    axiom_keys = [f"{s}||{p}||{o}" for s, p, o in source]
    narrative = await generator.generate_text(axiom_keys, [])
    excerpt = (narrative or "")[:_EXCERPT_LEN]

    if not narrative:
        return PerDocLlmRoundtrip(
            doc_id=doc.id,
            n_source_axioms=len(source),
            n_reconstructed_axioms=0,
            drift_pct=100.0,
            missing_claims=tuple(source),
            extra_claims=(),
            narrative_excerpt=excerpt,
        )

    reconstructed_list = await extractor.extract_triplets(narrative)
    reconstructed = _as_set(reconstructed_list)
    missing = tuple(source - reconstructed)
    extra = tuple(reconstructed - source)
    denom = max(len(source), len(reconstructed), 1)
    drift = 100.0 * (len(missing) + len(extra)) / denom
    return PerDocLlmRoundtrip(
        doc_id=doc.id,
        n_source_axioms=len(source),
        n_reconstructed_axioms=len(reconstructed),
        drift_pct=drift,
        missing_claims=missing,
        extra_claims=extra,
        narrative_excerpt=excerpt,
    )


def _aggregate(
    corpus_id: str, per_doc: Sequence[PerDocLlmRoundtrip]
) -> LlmRoundtripMetrics:
    drifts = [p.drift_pct for p in per_doc]
    return LlmRoundtripMetrics(
        corpus_id=corpus_id,
        drift_pct=statistics.mean(drifts) if drifts else 0.0,
        n_roundtrips=len(per_doc),
        n_source_axioms_total=sum(p.n_source_axioms for p in per_doc),
        n_reconstructed_axioms_total=sum(
            p.n_reconstructed_axioms for p in per_doc
        ),
        epistemic_status="empirical-benchmark",
        per_doc=tuple(per_doc),
    )


@dataclass(frozen=True)
class LlmRoundtripRunner:
    """Orchestrates the LLM narrative full-loop drift measurement.

    The generator and extractor may be pinned to the same OpenAI model snapshot
    or to different ones; both are constructed from LiveLLMAdapter. The runner
    itself is a thin wrapper around _process_doc / _aggregate so the unit tests
    can drive the per-doc pipeline without any network I/O.
    """

    name: str = "sum.roundtrip.llm"
    generator_model_id: str = ""
    extractor_model_id: str = ""

    def run(self, corpus: Corpus) -> Sequence[LlmRoundtripMetrics]:
        return asyncio.run(self._run_async(corpus))

    async def _run_async(
        self, corpus: Corpus
    ) -> Sequence[LlmRoundtripMetrics]:
        from internal.ensemble.live_llm_adapter import LiveLLMAdapter

        generator = LiveLLMAdapter(model=self.generator_model_id)
        extractor = LiveLLMAdapter(
            model=self.extractor_model_id or self.generator_model_id
        )

        per_doc: list[PerDocLlmRoundtrip] = []
        for doc in corpus.documents:
            record = await _process_doc(extractor, generator, doc)
            if record is not None:
                per_doc.append(record)

        return (_aggregate(corpus.id, per_doc),)
