"""
Tests for the Epistemic Feedback Loop ("Tags to Tomes then Back").

Validates:
    - GCD-based hallucination isolation (Sieve of Hallucinations)
    - Self-correcting extrapolation loop
    - Mathematical proof of zero hallucination
    - Failure after max retries
"""

import sys
import os
import pytest
import asyncio

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.ensemble.epistemic_loop import QuantumExtrapolator


# ─── Hallucination Isolation (unit) ────────────────────────────────────

class TestIsolateHallucinations:

    def test_no_hallucinations(self):
        """Clean generated state returns empty list."""
        alg = GodelStateAlgebra()
        truth = alg.encode_chunk_state([
            ("Alice", "age", "30"),
            ("Bob", "age", "40"),
        ])
        hypothesis = alg.encode_chunk_state([("Alice", "age", "30")])

        assert alg.isolate_hallucinations(truth, hypothesis) == []

    def test_single_hallucination(self):
        """One fabricated axiom is isolated exactly."""
        alg = GodelStateAlgebra()
        truth = alg.encode_chunk_state([
            ("Alice", "age", "30"),
            ("Bob", "age", "40"),
        ])
        # Generated state includes a hallucinated claim
        generated = alg.encode_chunk_state([
            ("Alice", "age", "30"),
            ("Alice", "lives_in", "Paris"),  # hallucinated
        ])

        hallucinated = alg.isolate_hallucinations(truth, generated)
        assert len(hallucinated) == 1
        assert "alice||lives_in||paris" in hallucinated[0]

    def test_multiple_hallucinations(self):
        """Multiple fabricated axioms are all isolated."""
        alg = GodelStateAlgebra()
        truth = alg.encode_chunk_state([("Alice", "age", "30")])
        generated = alg.encode_chunk_state([
            ("Alice", "age", "30"),
            ("Alice", "job", "spy"),       # hallucinated
            ("Alice", "pet", "dragon"),    # hallucinated
        ])

        hallucinated = alg.isolate_hallucinations(truth, generated)
        assert len(hallucinated) == 2
        keys = set(hallucinated)
        assert "alice||job||spy" in keys
        assert "alice||pet||dragon" in keys

    def test_entirely_hallucinated(self):
        """Generated state shares zero primes with truth."""
        alg = GodelStateAlgebra()
        truth = alg.encode_chunk_state([("Alice", "age", "30")])
        generated = alg.encode_chunk_state([("Eve", "role", "hacker")])

        hallucinated = alg.isolate_hallucinations(truth, generated)
        assert len(hallucinated) == 1
        assert "eve||role||hacker" in hallucinated[0]


# ─── Mock LLMs ─────────────────────────────────────────────────────────

class MockGenerators:
    """Simulates an LLM that halluccinates on first attempt then self-corrects."""

    def __init__(self):
        self.attempt = 0

    async def mock_generator(self, axioms, constraints):
        self.attempt += 1
        if self.attempt == 1:
            # First attempt: introduces "Paris" hallucination
            return "Alice is 30 and she lives in Paris."
        # Second attempt: corrected after receiving negative constraint
        return "Alice is 30."

    async def mock_extractor(self, text):
        triplets = []
        if "Alice is 30" in text:
            triplets.append(("Alice", "age", "30"))
        if "Paris" in text:
            triplets.append(("Alice", "lives_in", "Paris"))
        return triplets


class StubbornMockGenerators:
    """Simulates an LLM that never stops hallucinating."""

    async def mock_generator(self, axioms, constraints):
        return "Alice is 30 and she lives in Paris."

    async def mock_extractor(self, text):
        return [
            ("Alice", "age", "30"),
            ("Alice", "lives_in", "Paris"),
        ]


class EmptyMockGenerators:
    """Simulates an LLM that produces no extractable axioms."""

    async def mock_generator(self, axioms, constraints):
        return "This text has no verifiable claims."

    async def mock_extractor(self, text):
        return []


# ─── End-to-End Epistemic Loop ─────────────────────────────────────────

class TestEpistemicLoop:

    def test_tags_to_tomes_and_back(self):
        """Full loop: hallucinate → diagnose → self-correct → prove."""
        algebra = GodelStateAlgebra()

        # Build verified Global State (Tags)
        truth_triplets = [("Alice", "age", "30"), ("Bob", "age", "40")]
        global_state = algebra.encode_chunk_state(truth_triplets)

        target_axioms = ["alice||age||30"]

        mock_llms = MockGenerators()
        extrapolator = QuantumExtrapolator(
            godel_algebra=algebra,
            llm_generator=mock_llms.mock_generator,
            llm_extractor=mock_llms.mock_extractor,
        )

        verified = asyncio.run(
            extrapolator.extrapolate_with_proof(global_state, target_axioms)
        )

        # The corrected narrative
        assert verified == "Alice is 30."
        # Proves it caught the error, diagnosed via GCD, and retried
        assert mock_llms.attempt == 2

        # Verify the hallucinated prime was properly registered
        assert "alice||lives_in||paris" in algebra.axiom_to_prime

    def test_epistemic_failure_after_max_retries(self):
        """RuntimeError raised when LLM refuses to stop hallucinating."""
        algebra = GodelStateAlgebra()
        global_state = algebra.encode_chunk_state([("Alice", "age", "30")])

        stubborn = StubbornMockGenerators()
        extrapolator = QuantumExtrapolator(
            godel_algebra=algebra,
            llm_generator=stubborn.mock_generator,
            llm_extractor=stubborn.mock_extractor,
            max_retries=2,
        )

        with pytest.raises(RuntimeError, match="Epistemic Failure"):
            asyncio.run(
                extrapolator.extrapolate_with_proof(global_state, ["alice||age||30"])
            )

    def test_empty_extraction_retries(self):
        """Loop retries when extractor returns no triplets."""
        algebra = GodelStateAlgebra()
        global_state = algebra.encode_chunk_state([("Alice", "age", "30")])

        empty = EmptyMockGenerators()
        extrapolator = QuantumExtrapolator(
            godel_algebra=algebra,
            llm_generator=empty.mock_generator,
            llm_extractor=empty.mock_extractor,
            max_retries=2,
        )

        with pytest.raises(RuntimeError, match="Epistemic Failure"):
            asyncio.run(
                extrapolator.extrapolate_with_proof(global_state, ["alice||age||30"])
            )
