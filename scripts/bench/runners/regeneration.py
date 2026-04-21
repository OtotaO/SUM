from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass
from typing import Protocol, Sequence

from ..corpus import Corpus, CorpusDocument
from ..schema import PerDocRegeneration, RegenerationMetrics

_EXCERPT_LEN = 200


class _SieveLike(Protocol):
    def extract_triplets(self, text: str) -> list[tuple[str, str, str]]: ...


class _GeneratorLike(Protocol):
    async def generate_text(
        self, axiom_keys: list[str], tags: list[str]
    ) -> str: ...


class _EntailmentResultLike(Protocol):
    @property
    def entailed(self) -> bool: ...


class _CheckerLike(Protocol):
    async def check(
        self, passage: str, claim: tuple[str, str, str]
    ) -> _EntailmentResultLike: ...


async def _process_doc(
    sieve: _SieveLike,
    generator: _GeneratorLike,
    checker: _CheckerLike,
    doc: CorpusDocument,
) -> PerDocRegeneration | None:
    """Process one document. Returns None if no triples were extracted."""
    triples = sieve.extract_triplets(doc.text)
    if not triples:
        return None

    axiom_keys = [f"{s}||{p}||{o}" for s, p, o in triples]
    narrative = await generator.generate_text(axiom_keys, [])
    n_claims = len(triples)
    excerpt = (narrative or "")[:_EXCERPT_LEN]

    if not narrative:
        return PerDocRegeneration(
            doc_id=doc.id,
            n_claims=n_claims,
            n_supported=0,
            per_claim_rate=0.0,
            unsupported_claims=tuple(triples),
            narrative_excerpt=excerpt,
        )

    unsupported: list[tuple[str, str, str]] = []
    for t in triples:
        result = await checker.check(narrative, t)
        if not result.entailed:
            unsupported.append(t)
    n_supported = n_claims - len(unsupported)
    return PerDocRegeneration(
        doc_id=doc.id,
        n_claims=n_claims,
        n_supported=n_supported,
        per_claim_rate=n_supported / n_claims,
        unsupported_claims=tuple(unsupported),
        narrative_excerpt=excerpt,
    )


def _aggregate(
    corpus_id: str, per_doc: Sequence[PerDocRegeneration]
) -> RegenerationMetrics:
    n_supported_total = sum(p.n_supported for p in per_doc)
    n_claims_total = sum(p.n_claims for p in per_doc)
    rates = [p.per_claim_rate for p in per_doc]
    rate = statistics.mean(rates) if rates else 0.0
    return RegenerationMetrics(
        corpus_id=corpus_id,
        path="freeform",
        factscore=rate,
        minicheck_entailment_rate=rate,
        n_generations=len(per_doc),
        n_supported_claims=n_supported_total,
        n_total_claims=n_claims_total,
        epistemic_status="empirical-benchmark",
        per_doc=tuple(per_doc),
    )


@dataclass(frozen=True)
class OpenAiRegenerationRunner:
    """Measures LLM regeneration faithfulness against the extracted axiom set.

    For each document:
      1. Extract triples via DeterministicSieve (same path as extraction runner).
      2. Render prose narrative via LiveLLMAdapter.generate_text().
      3. Independently check entailment of each source triple against the
         generated prose via LlmEntailmentChecker.
      4. Emit a PerDocRegeneration record naming the unsupported triples, so
         the aggregate FActScore gap is attributable at the doc level; aggregate
         across the corpus.

    Output:
      - factscore: mean per-document entailment rate (in [0, 1])
      - minicheck_entailment_rate: same measurement, different name, one checker
      - n_supported_claims / n_total_claims: aggregate support counts
      - n_generations: docs where narrative was produced (non-empty input)
      - per_doc: per-document attribution (which claims failed, which doc)

    Requires pinned model IDs, resolved by run_bench.py from the
    environment. Default source: SUM_BENCH_MODEL (one pinned snapshot
    used for every role). Per-role overrides still honored:

      - generator_model_id  SUM_BENCH_GENERATOR_MODEL (narrative synthesis)
      - entailment_model_id SUM_BENCH_MINICHECK_MODEL (entailment judgement)

    The SUM_BENCH_FACTSCORE_MODEL env var is reserved for a future
    atomic-claim decomposition stage. On the current seed_v1 corpus
    (one triple per doc) decomposition is identity, so the
    generator+entailment pair is sufficient.
    """

    name: str = "sum.regeneration.openai"
    generator_model_id: str = ""
    entailment_model_id: str = ""

    def run(self, corpus: Corpus) -> Sequence[RegenerationMetrics]:
        return asyncio.run(self._run_async(corpus))

    async def _run_async(
        self, corpus: Corpus
    ) -> Sequence[RegenerationMetrics]:
        from internal.algorithms.syntactic_sieve import DeterministicSieve
        from internal.ensemble.live_llm_adapter import LiveLLMAdapter
        from internal.ensemble.llm_entailment import LlmEntailmentChecker

        sieve = DeterministicSieve()  # type: ignore[no-untyped-call]
        generator = LiveLLMAdapter(model=self.generator_model_id)
        checker = LlmEntailmentChecker(model_id=self.entailment_model_id)

        per_doc: list[PerDocRegeneration] = []
        for doc in corpus.documents:
            record = await _process_doc(sieve, generator, checker, doc)
            if record is not None:
                per_doc.append(record)

        return (_aggregate(corpus.id, per_doc),)
