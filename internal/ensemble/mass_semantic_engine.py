"""
Mass Semantic Engine — MapReduce Gödel-State Parallelization

Wire the SPNT bounding and Gödel-State Algebra into a fully async
MAP → ENCODE → REDUCE → AUDIT pipeline for mass-parallel semantic
extraction.

Architecture:
    1. MAP:    asyncio.gather all chunk extractions (lock-free)
    2. ENCODE: Convert string triplets to Gödel integers
    3. REDUCE: LCM merge all chunk states into one global integer
    4. AUDIT:  Paradox detection + SPNT compression bound check

Author: ototao
License: Apache License 2.0
"""

import asyncio
import logging
from typing import Callable, Awaitable, List, Tuple, Dict, Any

from internal.algorithms.semantic_arithmetic import (
    GodelStateAlgebra,
    SemanticPrimeNumberTheorem,
)
from internal.ensemble.extraction_validator import ExtractionValidator

logger = logging.getLogger(__name__)


class MassSemanticEngine:
    """
    Async MapReduce engine that extracts semantic triplets from parallel
    text chunks, encodes them as Gödel integers, and merges them via LCM.

    Complements the existing MassDocumentEngine (string-based hierarchical
    summarization) with a mathematical, lock-free alternative.
    """

    def __init__(
        self,
        extractor_llm_func: Callable[
            [str], Awaitable[List[Tuple[str, str, str]]]
        ],
    ):
        """
        Args:
            extractor_llm_func: An async callable that accepts a text chunk
                and returns a list of (subject, predicate, object) triplets.
        """
        self.extract_triplets = extractor_llm_func
        self.algebra = GodelStateAlgebra()

    async def tomes_to_tags(
        self,
        raw_claims_count: int,
        chunks: List[str],
    ) -> Dict[str, Any]:
        """
        Mass-parallel extraction mapping text chunks to a single Global
        Integer State.

        Args:
            raw_claims_count: Estimated total raw claims across all chunks.
            chunks:           List of text chunks to extract from in parallel.

        Returns:
            Dictionary with:
                global_state        – the merged Gödel integer
                total_unique_primes – count of unique axioms in the state
                spnt_limit          – theoretical compression bound
                compression_ok      – True if within SPNT bound
                paradoxes           – list of detected curvature conflicts
        """
        # ── 1. MAP: Mass-parallel lock-free extraction ──────────────
        tasks = [self.extract_triplets(chunk) for chunk in chunks]
        extracted_chunks: List[List[Tuple[str, str, str]]] = list(
            await asyncio.gather(*tasks)
        )

        # ── 1.5 VALIDATE: Structural gate (Phase 19A) ─────────────
        validator = ExtractionValidator()
        validated_chunks = []
        total_rejected = 0
        all_rejection_reasons = []

        for chunk_triplets in extracted_chunks:
            result = validator.validate_batch(chunk_triplets)
            validated_chunks.append(result.accepted)
            total_rejected += result.rejected_count
            for rej in result.rejected:
                all_rejection_reasons.append({
                    "triplet": f"{rej.subject}||{rej.predicate}||{rej.object_}",
                    "reason": rej.reason,
                })

        # ── 2. ENCODE: Convert validated triplets to Gödel integers ──
        chunk_states = [
            self.algebra.encode_chunk_state(triplets)
            for triplets in validated_chunks
        ]

        # ── 3. REDUCE: Mathematical Merge via LCM ──────────────────
        global_state = self.algebra.merge_parallel_states(chunk_states)

        # ── 4. AUDIT: Paradox detection & compression bounds ───────
        paradoxes = self.algebra.detect_curvature_paradoxes(global_state)
        if paradoxes:
            logger.warning("Curvature paradoxes detected: %s", paradoxes)

        # Count unique axioms present in the global state
        total_unique_primes = sum(
            1
            for p in self.algebra.prime_to_axiom
            if global_state % p == 0
        )

        spnt_limit = SemanticPrimeNumberTheorem.asymptotic_bound(
            raw_claims_count
        )
        compression_ok = total_unique_primes <= spnt_limit

        if not compression_ok:
            logger.warning(
                "SPNT compression failed: %d primes exceeds bound %d",
                total_unique_primes,
                spnt_limit,
            )

        return {
            "global_state": global_state,
            "total_unique_primes": total_unique_primes,
            "spnt_limit": spnt_limit,
            "compression_ok": compression_ok,
            "paradoxes": paradoxes,
            "rejected_count": total_rejected,
            "rejection_reasons": all_rejection_reasons,
        }
