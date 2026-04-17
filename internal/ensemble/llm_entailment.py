"""
LLM Entailment Checker — Structured Entailment for Regeneration Faithfulness

Wraps an LLM call behind a strict entailment-check interface. Given a passage
and a claim triple, returns a boolean entailed + confidence score.

This is the symbolic-boundary verifier for the regeneration path: once an LLM
has rendered prose from a structured axiom set, each source axiom is
independently checked for entailment against the prose. Non-entailed axioms
are counted as drift for FActScore / MiniCheck-equivalent metrics surfaced by
the bench harness.

The model MUST be pinned (with date suffix); unpinned identifiers raise at
construction time so reproducibility is preserved.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from openai import AsyncOpenAI
from pydantic import BaseModel, Field


class EntailmentJudgment(BaseModel):
    """Pydantic-enforced LLM output schema for one entailment decision."""

    entailed: bool = Field(description="Does the passage support the claim?")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model's confidence in its judgment, 0.0-1.0",
    )


@dataclass(frozen=True)
class EntailmentResult:
    """Decoded entailment outcome for one (passage, claim) pair."""

    entailed: bool
    confidence: float
    claim_sentence: str


class LlmEntailmentChecker:
    """Structured entailment verifier via OpenAI structured-output parsing.

    Single method: ``check(passage, claim_triple) -> EntailmentResult``.
    The claim triple is rendered as ``"{s} {p} {o}"`` and submitted alongside
    the passage. The model answers with a boolean ``entailed`` + a confidence
    in [0, 1]. The judgement prompt is intentionally conservative — paraphrases
    of the same fact count as entailed; reinterpretations or unsupported
    inferences do not.
    """

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
    ) -> None:
        if not model_id or not model_id.strip():
            raise ValueError(
                "LlmEntailmentChecker requires a pinned model_id "
                "(e.g. 'gpt-4o-2024-08-06')."
            )
        self.model_id = model_id
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )

    async def check(
        self, passage: str, claim: tuple[str, str, str]
    ) -> EntailmentResult:
        s, p, o = claim
        claim_sentence = f"{s} {p} {o}"

        response = await self.client.beta.chat.completions.parse(
            model=self.model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict entailment checker. Given a "
                        "passage and a claim, decide whether the passage "
                        "supports the claim. Be conservative: set "
                        "entailed=true only if the passage explicitly or "
                        "strongly implies the claim. Paraphrases of the "
                        "same fact count as entailed; reinterpretations or "
                        "unsupported inferences do not."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"PASSAGE:\n{passage}\n\n"
                        f"CLAIM: {claim_sentence}\n\n"
                        f"Does the passage entail the claim?"
                    ),
                },
            ],
            response_format=EntailmentJudgment,
        )
        judgment = response.choices[0].message.parsed
        if judgment is None:
            return EntailmentResult(
                entailed=False,
                confidence=0.0,
                claim_sentence=claim_sentence,
            )
        return EntailmentResult(
            entailed=judgment.entailed,
            confidence=judgment.confidence,
            claim_sentence=claim_sentence,
        )
