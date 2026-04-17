from __future__ import annotations

from dataclasses import dataclass

from ..corpus import Corpus
from ..schema import ExtractionMetrics


def _norm_key(doc_id: str, s: str, p: str, o: str) -> str:
    return (
        f"{doc_id}::"
        f"{s.strip().lower()}||"
        f"{p.strip().lower()}||"
        f"{o.strip().lower()}"
    )


@dataclass(frozen=True)
class SumExtractionRunner:
    """Runs the current SUM sieve against a corpus and scores triple-set F1.

    Note on normalization: `DeterministicSieve` lowercases and lemmatizes its
    output. Gold triples are expected to be pre-canonicalized the same way.
    This runner does NOT reconcile lemmatization mismatches — those count as
    false negatives, by design. That is the honest measurement.
    """

    name: str = "sum.syntactic_sieve"

    def run(self, corpus: Corpus) -> ExtractionMetrics:
        from internal.algorithms.syntactic_sieve import DeterministicSieve

        sieve = DeterministicSieve()  # type: ignore[no-untyped-call]

        predicted_keys: set[str] = set()
        gold_keys: set[str] = set()

        for doc in corpus.documents:
            for s, p, o in sieve.extract_triplets(doc.text):
                predicted_keys.add(_norm_key(doc.id, s, p, o))
            for g in doc.gold_triples:
                gold_keys.add(_norm_key(doc.id, g.subject, g.predicate, g.object))

        tp = len(predicted_keys & gold_keys)
        fp = len(predicted_keys - gold_keys)
        fn = len(gold_keys - predicted_keys)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return ExtractionMetrics(
            corpus_id=corpus.id,
            precision=precision,
            recall=recall,
            f1=f1,
            n_predicted=len(predicted_keys),
            n_gold=len(gold_keys),
            n_correct=tp,
        )


@dataclass(frozen=True)
class KgGenExtractionRunner:
    name: str = "kggen.v1"
    model_id: str = ""

    def run(self, corpus: Corpus) -> ExtractionMetrics:
        raise NotImplementedError(
            "STATE 4-C: KGGen baseline wiring — requires pinned model_id"
        )
