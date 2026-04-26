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
from typing import List, Optional, Tuple

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


class EntailmentResponse(BaseModel):
    """v0.4 NLI audit schema. Used to disentangle real fact loss from
    embedding-similarity false negatives in the slider drift bench."""

    is_supported: bool = Field(
        description=(
            "True if and only if the passage explicitly states or "
            "directly implies the given (subject, predicate, object) "
            "fact. Be strict: rephrasings that preserve meaning count "
            "as supported; loose associations or topic-similarity do "
            "not."
        ),
    )
    rationale: str = Field(
        default="",
        description=(
            "One-sentence justification. Helps audit tracing when the "
            "judgement disagrees with embedding similarity."
        ),
    )


class RenderedTome(BaseModel):
    """Phase E.1 v0.3 structured-render schema. The renderer constrains
    the LLM to emit both the narrative tome AND the explicit list of
    triples it considers preserved in that tome. The bench treats
    `claimed_triples` as an LLM self-attestation, cross-checked
    against an independent re-extraction — divergence between the two
    is an adversarial signal (claimed-but-not-reextracted = LLM
    hallucinated preservation; reextracted-but-not-claimed = LLM
    encoded facts it didn't know it encoded)."""

    tome: str = Field(
        min_length=1,
        description=(
            "The rendered narrative prose. Should faithfully present "
            "every input fact while honouring the system prompt's axis "
            "directives (length, formality, audience, perspective)."
        ),
    )
    claimed_triples: List[SemanticTriplet] = Field(
        description=(
            "The (subject, predicate, object) triples explicitly preserved "
            "in the tome above. List every input fact that appears, even "
            "if rephrased. Use the same canonicalisation rules as extraction "
            "(lowercase, snake_case predicates)."
        ),
    )


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

    # ------------------------------------------------------------------
    # Entailment / NLI (v0.4 audit layer)
    # ------------------------------------------------------------------

    async def check_entailment(
        self,
        fact: Tuple[str, str, str],
        passage: str,
    ) -> bool:
        """Yes/no NLI judgement: does ``passage`` express the
        ``(subject, predicate, object)`` fact?

        Used by the slider bench's audit pass to distinguish real fact
        loss from embedding-similarity false negatives. Costs one LLM
        call per (fact, passage) pair; the bench scopes audits to
        cells where semantic preservation < threshold so cost stays
        bounded (~30 weak cells × ~6 facts × $0.001 ≈ $0.18 per
        audit run on the standard paragraph corpus).
        """
        s, p, o = fact
        sys_prompt = (
            "You are an NLI judge. Decide whether a passage supports a "
            "specific fact. Be strict: rephrasings that preserve meaning "
            "count as supported; loose topic-similarity does not."
        )
        user_prompt = (
            f"FACT: {s} {p} {o}\n\n"
            f"PASSAGE:\n{passage}\n\n"
            "Does the passage state or directly imply this fact?"
        )
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=EntailmentResponse,
            max_tokens=200,
        )
        parsed = response.choices[0].message.parsed
        return bool(parsed and parsed.is_supported)


class OpenAIChatClient:
    """Adapts ``LiveLLMAdapter`` to the ``LLMChatClient`` Protocol used by
    ``slider_renderer.render``. Kept here (not in slider_renderer.py) so
    the renderer module stays free of the ``openai`` dep."""

    def __init__(self, adapter: "LiveLLMAdapter"):
        self._adapter = adapter

    async def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
    ) -> str:
        response = await self._adapter.client.chat.completions.create(
            model=self._adapter.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def chat_completion_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
    ) -> Tuple[str, List[Tuple[str, str, str]]]:
        """v0.3 structured render path. Returns (tome, claimed_triples)
        where claimed_triples is the LLM's self-attestation of which
        source facts survived the round-trip. Cross-runtime guarantee:
        OpenAI's ``beta.chat.completions.parse`` enforces the
        RenderedTome schema, so the returned tuple is always the right
        shape — no parse errors possible."""
        response = await self._adapter.client.beta.chat.completions.parse(
            model=self._adapter.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=RenderedTome,
            max_tokens=max_tokens,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            return "", []
        triples: List[Tuple[str, str, str]] = [
            (t.subject.lower(), t.predicate.lower(), t.object_.lower())
            for t in parsed.claimed_triples
            if t.certainty != "speculative"
        ]
        return parsed.tome, triples
