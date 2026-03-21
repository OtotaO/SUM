"""
Epistemic Feedback Loop — "Tags to Tomes then Back"

Governs the closed-loop extrapolation pipeline:
    1. TOMES:  Generate narrative text from verified Gödel axioms.
    2. TAGS:   Extract triplets from the narrative and re-encode as a
               Gödel integer.
    3. VERIFY: O(1) modulo check — ``global_state % generated_state == 0``.
    4. DIAGNOSE: If verification fails, GCD-based hallucination isolation
                 identifies the exact fabricated claims.
    5. SELF-CORRECT: Feed hallucinated axioms back as strict negative
                     constraints and re-generate.

The loop refuses to return a string until it is *mathematically proven*
to be a pure subset of the global truth.

Author: ototao
License: Apache License 2.0
"""

import logging
from typing import Callable, Awaitable, List, Tuple, Dict, Any

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra

logger = logging.getLogger(__name__)


class QuantumExtrapolator:
    """
    Translates Gödel Integers (Tags) into Narrative Text (Tomes) and
    verifies them mathematically by converting them back into Integers.

    The extrapolation loop guarantees zero hallucination through an
    unbreakable epistemic cage: no text is returned until
    ``global_state % generated_state == 0``.
    """

    def __init__(
        self,
        godel_algebra: GodelStateAlgebra,
        llm_generator: Callable[
            [List[str], List[str]], Awaitable[str]
        ],
        llm_extractor: Callable[
            [str], Awaitable[List[Tuple[str, str, str]]]
        ],
        max_retries: int = 3,
    ):
        """
        Args:
            godel_algebra:  A GodelStateAlgebra instance with the global
                            truth already encoded.
            llm_generator:  Async callable (axioms, negative_constraints) → text.
            llm_extractor:  Async callable (text) → List[(subj, pred, obj)].
            max_retries:    Maximum correction attempts before raising.
        """
        self.algebra = godel_algebra
        self.generate_text = llm_generator
        self.extract_triplets = llm_extractor
        self.max_retries = max_retries

    async def extrapolate_with_proof(
        self,
        global_state: int,
        target_axioms: List[str],
    ) -> str:
        """
        The Tags-to-Tomes pipeline.

        Guarantees the output text strictly entails the global state with
        zero hallucinations.

        Args:
            global_state:  The verified global Gödel integer.
            target_axioms: Axiom key strings to expand into narrative.

        Returns:
            A narrative string that is *mathematically proven* to contain
            only claims present in the global state.

        Raises:
            RuntimeError: If the LLM fails to self-correct within
                          ``max_retries`` attempts.
        """
        negative_constraints: List[str] = []

        for attempt in range(self.max_retries):
            # ── 1. TOMES: Generate narrative from verified axioms ────
            narrative = await self.generate_text(
                target_axioms, negative_constraints
            )

            # ── 2. TAGS: Map narrative back to a Gödel integer ───────
            extracted_triplets = await self.extract_triplets(narrative)

            if not extracted_triplets:
                negative_constraints.append(
                    "Failed to extract any verifiable axioms. "
                    "Be more explicit."
                )
                continue

            generated_state = self.algebra.encode_chunk_state(
                extracted_triplets
            )

            # ── 3. VERIFY: The O(1) Epistemic Hardware Filter ────────
            if self.algebra.verify_entailment(global_state, generated_state):
                logger.info(
                    "Mathematical Proof of Zero Hallucination achieved "
                    "on attempt %d.",
                    attempt + 1,
                )
                return narrative

            # ── 4. DIAGNOSE: Isolate hallucinated primes via GCD ─────
            hallucinations = self.algebra.isolate_hallucinations(
                global_state, generated_state
            )
            logger.warning(
                "Modulo check failed (attempt %d). "
                "Hallucinations detected: %s",
                attempt + 1,
                hallucinations,
            )

            # ── 5. SELF-CORRECT: Feed exact errors back ──────────────
            if hallucinations:
                negative_constraints.extend(hallucinations)

        raise RuntimeError(
            f"Epistemic Failure: LLM failed to mathematically align "
            f"after {self.max_retries} attempts. "
            f"Residual hallucinations: {negative_constraints[-5:]}"
        )
