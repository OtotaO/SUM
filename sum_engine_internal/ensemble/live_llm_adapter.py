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
from typing import List, Tuple, Optional

from pydantic import BaseModel, Field
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# ─── Pydantic schemas for structured LLM output ──────────────────────

class SemanticTriplet(BaseModel):
    """A single irreducible fact as a subject-predicate-object triple."""
    subject: str = Field(
        min_length=2, max_length=200,
        description="The core entity or subject (lowercased, concise)",
    )
    predicate: str = Field(
        min_length=2, max_length=200,
        description="The relational verb or attribute (snake_case)",
    )
    object_: str = Field(
        alias="object",
        min_length=2, max_length=200,
        description="The target entity or value (lowercased, concise)",
    )
    # Phase 19A: Metadata fields — do NOT alter algebra semantics
    source_span: Optional[str] = Field(
        default=None,
        description="The exact text span this triplet was extracted from",
    )
    certainty: Optional[str] = Field(
        default=None,
        description="Model's assessment: 'definite', 'hedged', or 'speculative'",
    )
    extraction_notes: Optional[str] = Field(
        default=None,
        description="Any caveats about this extraction (negation, conditional, etc.)",
    )

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

        Phase 19A: Enhanced prompt with negation awareness, certainty
        metadata, and source span tracking.
        """
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract all distinct factual claims from the text "
                        "as subject-predicate-object triplets.\n\n"
                        "Rules:\n"
                        "- Keep subject, predicate, and object concise and lowercased\n"
                        "- Use snake_case for multi-word predicates (e.g., 'is_part_of')\n"
                        "- Do NOT extract opinions, questions, or hypotheticals as facts\n"
                        "- If a statement is negated (e.g., 'X does NOT cause Y'), "
                        "set certainty to 'speculative' and note 'negation' in extraction_notes\n"
                        "- If language is hedged ('may', 'might', 'possibly'), "
                        "set certainty to 'hedged'\n"
                        "- For definite factual statements, set certainty to 'definite'\n"
                        "- Include the source_span: the exact phrase from the text"
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
            # Phase 19A: Skip speculative extractions (negations)
            if t.certainty != "speculative"
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
