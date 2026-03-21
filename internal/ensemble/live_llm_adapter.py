"""
Live LLM Adapter — The Reality Bridge

Replaces mock generators and extractors with real AI models.
Uses Pydantic structured outputs to enforce strict (subject, predicate,
object) schemas, and OpenAI's embedding endpoint for the Vector Bridge.

Author: ototao
License: Apache License 2.0
"""

import os
import logging
from typing import List, Tuple

from pydantic import BaseModel, Field
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# ─── Pydantic schemas for structured LLM output ──────────────────────

class SemanticTriplet(BaseModel):
    """A single irreducible fact as a subject-predicate-object triple."""
    subject: str = Field(description="The core entity or subject")
    predicate: str = Field(description="The relational verb or attribute")
    object_: str = Field(alias="object", description="The target entity or value")

    model_config = {"populate_by_name": True}


class ExtractionResponse(BaseModel):
    """Structured output wrapper for a list of extracted triplets."""
    triplets: List[SemanticTriplet]


# ─── Adapter ─────────────────────────────────────────────────────────

class LiveLLMAdapter:
    """
    Production connector that maps natural language to the Gödel universe
    via constrained LLM calls.

    Three capabilities:
        1. ``extract_triplets``  — text → List[(subj, pred, obj)]
        2. ``generate_text``     — axioms + negative constraints → narrative
        3. ``get_embedding``     — text → List[float] (continuous vector)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ):
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.model = model
        self.embedding_model = embedding_model

    # ------------------------------------------------------------------
    # Extraction (Tags)
    # ------------------------------------------------------------------

    async def extract_triplets(
        self, chunk: str
    ) -> List[Tuple[str, str, str]]:
        """
        Maps natural language into strict topological triplets via
        Pydantic-constrained structured output.
        """
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract all distinct factual claims from the text "
                        "as subject-predicate-object triplets. Keep strings "
                        "highly concise and lowercased."
                    ),
                },
                {"role": "user", "content": chunk},
            ],
            response_format=ExtractionResponse,
        )

        parsed = response.choices[0].message.parsed
        return [
            (t.subject.lower(), t.predicate.lower(), t.object_.lower())
            for t in parsed.triplets
        ]

    # ------------------------------------------------------------------
    # Generation (Tomes)
    # ------------------------------------------------------------------

    async def generate_text(
        self,
        target_axioms: List[str],
        negative_constraints: List[str],
    ) -> str:
        """
        The Tomes generator for the Epistemic Loop.

        Produces a cohesive narrative from verified axioms while honouring
        negative constraints (previously identified hallucinations).
        """
        sys_prompt = (
            "You are a precise technical writer. Extrapolate the "
            "following absolute facts into a cohesive narrative. "
            "Do not invent facts."
        )
        user_prompt = (
            f"FACTS TO INCLUDE:\n{chr(10).join(target_axioms)}\n\n"
        )

        if negative_constraints:
            user_prompt += (
                "CRITICAL NEGATIVE CONSTRAINTS "
                "(DO NOT INCLUDE THESE HALLUCINATIONS):\n"
                f"{chr(10).join(negative_constraints)}"
            )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Embeddings (Vector Bridge)
    # ------------------------------------------------------------------

    async def get_embedding(self, text: str) -> List[float]:
        """Continuous mapping for the Continuous-Discrete Vector Bridge."""
        response = await self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding
