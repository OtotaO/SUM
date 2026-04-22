from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

from ..corpus import Corpus
from ..schema import RoundtripMetrics


@dataclass(frozen=True)
class SumRoundtripRunner:
    """Measures round-trip conservation on two input paths.

    ``input_kind="canonical"`` — uses OuroborosVerifier.verify_from_state() to
    confirm lossless conservation of the canonical (deterministic-template)
    layer. By construction (§1.1 of PROOF_BOUNDARY), drift is zero; this path
    exists to catch regressions in the canonical codec on every CI run.
    Reported with epistemic_status="provable".

    ``input_kind="prose"`` — measures empirical drift when the canonical-
    rendered tome is re-ingested via the NLP sieve. This is NOT the full LLM
    narrative round-trip (which requires an extrapolator); it is a
    lower-bound estimate of sieve behavior on the system's own emitted text.
    Reported with epistemic_status="empirical-benchmark".

    Documents producing zero triples from the sieve are skipped — there is
    no basis for a round-trip measurement on an empty extraction.
    """

    name: str = "sum.roundtrip"

    def run(self, corpus: Corpus) -> Sequence[RoundtripMetrics]:
        from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
        from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve
        from sum_engine_internal.ensemble.ouroboros import OuroborosVerifier
        from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator

        algebra = GodelStateAlgebra()  # type: ignore[no-untyped-call]
        sieve = DeterministicSieve()  # type: ignore[no-untyped-call]
        generator = AutoregressiveTomeGenerator(algebra)
        verifier = OuroborosVerifier(algebra, sieve, generator)

        canonical_drifts: list[float] = []
        prose_drifts: list[float] = []
        source_axiom_counts: list[int] = []
        prose_reconstructed_counts: list[int] = []

        for doc in corpus.documents:
            triples = sieve.extract_triplets(doc.text)
            if not triples:
                continue

            source_axioms = {f"{s}||{p}||{o}" for s, p, o in triples}
            source_axiom_counts.append(len(source_axioms))
            state_a = algebra.encode_chunk_state(list(triples))

            proof = verifier.verify_from_state(state_a)
            canonical_sym_diff = set(proof.missing_axioms) | set(proof.extra_axioms)
            canonical_denom = max(
                proof.source_axiom_count, proof.reconstructed_axiom_count, 1
            )
            canonical_drifts.append(
                100.0 * len(canonical_sym_diff) / canonical_denom
            )

            canonical_text = generator.generate_canonical(state_a)
            reextracted = sieve.extract_triplets(canonical_text)
            reextracted_axioms = {f"{s}||{p}||{o}" for s, p, o in reextracted}
            prose_reconstructed_counts.append(len(reextracted_axioms))
            prose_sym_diff = source_axioms ^ reextracted_axioms
            prose_denom = max(len(source_axioms), len(reextracted_axioms), 1)
            prose_drifts.append(100.0 * len(prose_sym_diff) / prose_denom)

        n = len(source_axiom_counts)
        source_avg = statistics.mean(source_axiom_counts) if source_axiom_counts else 0.0
        prose_recon_avg = (
            statistics.mean(prose_reconstructed_counts)
            if prose_reconstructed_counts
            else 0.0
        )

        return (
            RoundtripMetrics(
                corpus_id=corpus.id,
                input_kind="canonical",
                axiom_drift_pct=statistics.mean(canonical_drifts) if canonical_drifts else 0.0,
                n_roundtrips=n,
                source_axioms_avg=source_avg,
                reconstructed_axioms_avg=source_avg,
                epistemic_status="provable",
            ),
            RoundtripMetrics(
                corpus_id=corpus.id,
                input_kind="prose",
                axiom_drift_pct=statistics.mean(prose_drifts) if prose_drifts else 0.0,
                n_roundtrips=n,
                source_axioms_avg=source_avg,
                reconstructed_axioms_avg=prose_recon_avg,
                epistemic_status="empirical-benchmark",
            ),
        )
