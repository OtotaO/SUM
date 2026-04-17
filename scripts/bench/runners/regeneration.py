from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass
from typing import Sequence

from ..corpus import Corpus
from ..schema import RegenerationMetrics


@dataclass(frozen=True)
class OpenAiRegenerationRunner:
    """Measures LLM regeneration faithfulness against the extracted axiom set.

    For each document:
      1. Extract triples via DeterministicSieve (same path as extraction runner).
      2. Render prose narrative via LiveLLMAdapter.generate_text().
      3. Independently check entailment of each source triple against the
         generated prose via LlmEntailmentChecker.
      4. Report per-document rate; aggregate across corpus.

    Output:
      - factscore: mean per-document entailment rate (in [0, 1])
      - minicheck_entailment_rate: same measurement, different name, one checker
      - n_supported_claims / n_total_claims: aggregate support counts
      - n_generations: docs where narrative was produced (non-empty input)

    Requires pinned model IDs:
      - generator_model_id  (narrative synthesis — SUM_BENCH_GENERATOR_MODEL)
      - entailment_model_id (entailment judgement — SUM_BENCH_MINICHECK_MODEL)

    The SUM_BENCH_FACTSCORE_MODEL env var is reserved for a future atomic-claim
    decomposition stage. On the current seed_v1 corpus (one triple per doc)
    decomposition is identity, so the generator+entailment pair is sufficient.
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

        per_doc_rates: list[float] = []
        n_supported_total = 0
        n_claims_total = 0
        n_generations = 0

        for doc in corpus.documents:
            triples = sieve.extract_triplets(doc.text)
            if not triples:
                continue

            axiom_keys = [f"{s}||{p}||{o}" for s, p, o in triples]
            narrative = await generator.generate_text(axiom_keys, [])
            n_generations += 1
            if not narrative:
                per_doc_rates.append(0.0)
                n_claims_total += len(triples)
                continue

            n_supported_doc = 0
            for t in triples:
                result = await checker.check(narrative, t)
                if result.entailed:
                    n_supported_doc += 1

            n_claims_total += len(triples)
            n_supported_total += n_supported_doc
            per_doc_rates.append(
                n_supported_doc / len(triples) if triples else 0.0
            )

        rate = statistics.mean(per_doc_rates) if per_doc_rates else 0.0

        return (
            RegenerationMetrics(
                corpus_id=corpus.id,
                path="freeform",
                factscore=rate,
                minicheck_entailment_rate=rate,
                n_generations=n_generations,
                n_supported_claims=n_supported_total,
                n_total_claims=n_claims_total,
                epistemic_status="empirical-benchmark",
            ),
        )
