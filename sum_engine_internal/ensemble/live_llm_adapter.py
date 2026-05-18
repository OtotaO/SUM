"""
Live LLM Adapter — The Reality Bridge

Replaces mock generators and extractors with real AI models.
Uses Pydantic structured outputs to enforce strict (subject, predicate,
object) schemas, and OpenAI's embedding endpoint for the Vector Bridge.

Author: ototao
License: Apache License 2.0
"""

import json
import logging
import os
from typing import Any, List, Optional, Tuple

from openai import AsyncOpenAI, LengthFinishReasonError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# v0.8 layered-defense cap. Prompt-side instruction; not enforced
# by OpenAI's structured-output schema validator (which doesn't honor
# `maxItems` per https://community.openai.com/t/min-maxitems-are-not-
# supported-in-structured-output/958567). LLM compliance with
# stated numerical caps is empirically high under structured output.
EXTRACTION_TRIPLE_CAP = 64
EXTRACTION_RETRY_CAP = 32


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


def salvage_partial_triplets(partial_json: str) -> List[SemanticTriplet]:
    """Best-effort recovery of complete triplet objects from a
    truncated structured-output response.

    OpenAI's `beta.chat.completions.parse` raises
    LengthFinishReasonError when the LLM's emission exceeds the
    completion-token ceiling mid-output. The exception's
    `e.completion.choices[0].message.content` carries the partial
    JSON string, truncated mid-token. This helper walks the partial,
    returns whatever complete triplet objects appear before the
    truncation point, and gives up cleanly if salvage finds nothing.

    Pure function. No I/O. Returns empty list when input is malformed
    or contains no closed triplet objects (caller decides whether to
    retry, fall back, or raise).
    """
    arr_pos = partial_json.find('"triplets"')
    if arr_pos == -1:
        return []
    bracket_start = partial_json.find('[', arr_pos)
    if bracket_start == -1:
        return []

    items: List[SemanticTriplet] = []
    depth = 0
    obj_start = -1
    in_string = False
    escape = False
    for i in range(bracket_start + 1, len(partial_json)):
        c = partial_json[i]
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and obj_start >= 0:
                obj_text = partial_json[obj_start:i + 1]
                try:
                    obj_data = json.loads(obj_text)
                    items.append(SemanticTriplet.model_validate(obj_data))
                except (json.JSONDecodeError, ValueError):
                    pass  # malformed object, skip
                obj_start = -1
            elif depth < 0:
                break  # array closed; done
    return items


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
        base_url: str | None = None,
    ):
        # base_url support lets this adapter front any OpenAI-compatible
        # endpoint — HF Inference Providers, Ollama, llama.cpp, Modal-
        # hosted vLLM, Fireworks.ai, etc. Default (base_url=None) keeps
        # the legacy OpenAI behaviour byte-identical.
        kwargs: dict[str, Any] = {"api_key": api_key or os.getenv("OPENAI_API_KEY")}
        if base_url is not None:
            kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**kwargs)
        self.model = model
        self.embedding_model = embedding_model
        self.base_url = base_url

    @classmethod
    def from_model(
        cls,
        model: str,
        *,
        api_key: str | None = None,
    ) -> "LiveLLMAdapter":
        """Route a model id through ``llm_dispatch``'s routing rules and
        return an ``LiveLLMAdapter`` pointed at the right endpoint.

        Supports nine provider patterns the substrate knows:

          * ``gpt-* / o1-* / o3-* / o4-*``     → OpenAI (`OPENAI_API_KEY`)
          * ``claude-*``                       → not yet supported here
              (the slider's LLM-axis Python path is OpenAI-SDK-shaped;
              Anthropic routing is the Worker's job — use /api/render
              with ``X-Render-LLM-Key-Anthropic`` instead)
          * ``ollama:<model>``                 → http://localhost:11434/v1
          * ``llamacpp:<model>``               → http://localhost:8080/v1
          * ``local:<model>``                  → $SUM_LOCAL_LLM_BASE
          * ``org/model`` (HF-namespaced)      → HF Inference Providers
              router at https://router.huggingface.co/v1 with `HF_TOKEN`
          * ``nim:<model>``                    → NVIDIA NIM
              (`NVIDIA_API_KEY`; 1000 free credits on signup, 80+ models)
          * ``groq:<model>``                   → Groq Cloud
              (`GROQ_API_KEY`; free daily quota, fastest TTFT)
          * ``cerebras:<model>``               → Cerebras Cloud
              (`CEREBRAS_API_KEY`; free daily quota, fastest tok/sec)

        Callers without OpenAI credits can use HF (any open-weights
        model), Ollama (local + free), NIM (1000 free credits), Groq /
        Cerebras (free daily quota), or a Modal-/Fireworks-hosted
        endpoint via the ``local:`` prefix. The receipt's ``provider``
        field reports the API shape; ``extra.llm_endpoint`` reports the
        actual routing target (model + base_url). See callers in
        ``sum_engine_internal/transforms/slider.py`` for how that
        propagates into ``sum.transform_receipt.v1``.

        Full cascade design + recipes per provider:
        ``docs/FALLBACK_PROVIDER_CASCADE_2026-05-18.md`` +
        ``docs/BYOK_AND_FREE_PROVIDERS.md``.
        """
        from sum_engine_internal.ensemble.llm_dispatch import (
            CEREBRAS_BASE_URL,
            CEREBRAS_ENV_VAR,
            GROQ_BASE_URL,
            GROQ_ENV_VAR,
            HF_ROUTER_BASE_URL,
            LLAMACPP_DEFAULT_BASE_URL,
            LOCAL_LLM_ENV_VAR,
            NIM_BASE_URL,
            NIM_ENV_VAR,
            OLLAMA_DEFAULT_BASE_URL,
        )

        m = model.lower()

        # Local-server prefixes (free, local)
        if m.startswith("ollama:"):
            return cls(
                api_key=api_key or "no-key-needed-for-local",
                model=model[len("ollama:"):],
                base_url=OLLAMA_DEFAULT_BASE_URL,
            )
        if m.startswith("llamacpp:"):
            return cls(
                api_key=api_key or "no-key-needed-for-local",
                model=model[len("llamacpp:"):],
                base_url=LLAMACPP_DEFAULT_BASE_URL,
            )
        if m.startswith("local:"):
            base = os.environ.get(LOCAL_LLM_ENV_VAR)
            if not base:
                raise ValueError(
                    f"slider LLM-axis: model {model!r} uses the "
                    f"`local:` prefix but {LOCAL_LLM_ENV_VAR} is not "
                    f"set. Point it at your OpenAI-API-compatible "
                    f"server's base URL (e.g. a Modal-hosted vLLM "
                    f"endpoint, Fireworks.ai's base, etc.)."
                )
            return cls(
                api_key=api_key or os.environ.get("OPENAI_API_KEY", "no-key"),
                model=model[len("local:"):],
                base_url=base,
            )

        # Free-tier hosted-inference prefixes (cascade tier 1 from
        # docs/FALLBACK_PROVIDER_CASCADE_2026-05-18.md). All three are
        # OpenAI-API-compatible — drop into the same adapter shape as
        # HF / Ollama / local.
        if m.startswith("nim:"):
            token = api_key or os.environ.get(NIM_ENV_VAR)
            if not token:
                raise ValueError(
                    f"slider LLM-axis: NVIDIA NIM model {model!r} "
                    f"needs {NIM_ENV_VAR} env var or explicit api_key. "
                    f"Sign up free at https://build.nvidia.com — 1000 "
                    f"credits on signup, up to 5000 with request, "
                    f"80+ models including some free of credit "
                    f"(Zhipu GLM-4 family)."
                )
            return cls(
                api_key=token,
                model=model[len("nim:"):],
                base_url=NIM_BASE_URL,
            )
        if m.startswith("groq:"):
            token = api_key or os.environ.get(GROQ_ENV_VAR)
            if not token:
                raise ValueError(
                    f"slider LLM-axis: Groq model {model!r} needs "
                    f"{GROQ_ENV_VAR} env var or explicit api_key. "
                    f"Sign up free at https://console.groq.com — "
                    f"daily token quota, fastest TTFT (<300ms), "
                    f"Llama 3.3 70B + Mixtral + Gemma2."
                )
            return cls(
                api_key=token,
                model=model[len("groq:"):],
                base_url=GROQ_BASE_URL,
            )
        if m.startswith("cerebras:"):
            token = api_key or os.environ.get(CEREBRAS_ENV_VAR)
            if not token:
                raise ValueError(
                    f"slider LLM-axis: Cerebras model {model!r} needs "
                    f"{CEREBRAS_ENV_VAR} env var or explicit api_key. "
                    f"Sign up free at https://cloud.cerebras.ai — "
                    f"daily token quota, 3000 tok/sec on gpt-oss-120B "
                    f"(fastest end-to-end as of 2026-05)."
                )
            return cls(
                api_key=token,
                model=model[len("cerebras:"):],
                base_url=CEREBRAS_BASE_URL,
            )

        # HF Inference Providers — namespaced id
        if "/" in model:
            token = api_key or os.environ.get("HF_TOKEN")
            if not token:
                raise ValueError(
                    f"slider LLM-axis: HF-namespaced model {model!r} "
                    f"needs HF_TOKEN env var or explicit api_key. "
                    f"Get a token at https://huggingface.co/settings/tokens "
                    f"(read scope is sufficient for Inference Providers)."
                )
            return cls(api_key=token, model=model, base_url=HF_ROUTER_BASE_URL)

        # Default OpenAI
        return cls(api_key=api_key, model=model)

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

        v0.8 layered defense against `LengthFinishReasonError` on
        long / triple-dense documents:
            1. Prompt-stated cap of 64 triplets (LLM compliance is
               empirically high; OpenAI's schema validator does NOT
               honor maxItems, so this is the only count knob).
            2. On `LengthFinishReasonError`, salvage complete
               triplets from `e.completion.choices[0].message.content`
               via `salvage_partial_triplets`. Free — same response.
            3. If salvage yields nothing, retry once with a tighter
               cap (32) and an explicit overshoot note.
            4. If retry also overflows, re-raise.
        """
        triplets = await self._extract_triplets_with_recovery(
            chunk, cap=EXTRACTION_TRIPLE_CAP, retry_attempted=False,
        )
        return [
            (t.subject.lower(), t.predicate.lower(), t.object_.lower())
            for t in triplets
            # Phase 19A: Skip speculative extractions (negations)
            if t.certainty != "speculative"
        ]

    async def _extract_triplets_with_recovery(
        self,
        chunk: str,
        cap: int,
        retry_attempted: bool,
    ) -> List[SemanticTriplet]:
        """Inner extraction with `LengthFinishReasonError` recovery.
        Returns SemanticTriplet objects pre-canonicalisation so the
        caller controls lowercasing and certainty filtering."""
        sys_prompt = (
            "Extract all distinct factual claims from the text "
            "as subject-predicate-object triplets.\n\n"
            f"Return at most {cap} triplets. If the source contains "
            "more, return the most semantically distinct, prioritising "
            "load-bearing facts (subject + predicate + object that "
            "are essential to the document's claims).\n\n"
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
        )
        user_prompt = chunk
        if retry_attempted:
            user_prompt = (
                f"{chunk}\n\n"
                "[NOTE: prior extraction attempt exceeded the response "
                f"budget. Return at most {cap} most-essential triplets.]"
            )
        try:
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=ExtractionResponse,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                return []
            return list(parsed.triplets)
        except LengthFinishReasonError as e:
            partial = e.completion.choices[0].message.content or ""
            salvaged = salvage_partial_triplets(partial)
            if salvaged:
                logger.warning(
                    "extract_triplets salvaged %d triplets from a "
                    "LengthFinishReasonError partial response (cap=%d, "
                    "completion_tokens=%s).",
                    len(salvaged), cap,
                    e.completion.usage.completion_tokens if e.completion.usage else "?",
                )
                return salvaged
            if retry_attempted:
                logger.error(
                    "extract_triplets: LengthFinishReasonError on retry; "
                    "no salvage. Re-raising.",
                )
                raise
            logger.warning(
                "extract_triplets: first attempt overflowed with no "
                "salvage. Retrying with tighter cap=%d.",
                EXTRACTION_RETRY_CAP,
            )
            return await self._extract_triplets_with_recovery(
                chunk, cap=EXTRACTION_RETRY_CAP, retry_attempted=True,
            )

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


class OpenAICompatibleChatClient:
    """Adapts ``LiveLLMAdapter`` to the ``LLMChatClient`` Protocol used
    by ``slider_renderer.render`` when the routing target is an
    OpenAI-API-compatible but NOT-OpenAI-proper provider.

    Used for HF Inference Providers, NVIDIA NIM, Groq, Cerebras, Ollama,
    llama.cpp, and any ``local:`` endpoint — providers that implement
    the standard ``chat.completions.create`` surface but NOT the
    ``beta.chat.completions.parse`` structured-output extension. Calling
    that beta endpoint against most compatible providers returns either
    a 404 or a degenerate parse (e.g. NIM Llama 3.3 70B returns a
    single-token tome — surfaced as F7 in DOGFOOD_FINDINGS_2026-05-17).

    ``slider_renderer.render`` uses ``hasattr(llm, 'chat_completion_structured')``
    to pick its path. This class deliberately does NOT define that
    method, so the renderer falls through to plain ``chat_completion``.
    The cross-provider cost is the loss of the schema-enforced
    self-attestation (``claimed_triples``); the per-axis NLI audit
    still validates fact preservation independently.
    """

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


class OpenAIChatClient(OpenAICompatibleChatClient):
    """OpenAI-proper extension. Adds ``chat_completion_structured`` via
    ``beta.chat.completions.parse`` for schema-enforced rendering.
    This endpoint is OpenAI-specific and not portable to compatible
    providers — see ``OpenAICompatibleChatClient`` docstring.

    Used only when the adapter routes to OpenAI's own API
    (``base_url`` is unset). The factory ``make_chat_client`` below
    is the canonical way to instantiate the right class.
    """

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


def make_chat_client(adapter: "LiveLLMAdapter") -> OpenAICompatibleChatClient:
    """Pick the right chat-client class for the adapter's routing target.

    Routing rule:
      - ``adapter.base_url is None`` → OpenAI proper → ``OpenAIChatClient``
        (includes structured outputs via ``beta.chat.completions.parse``)
      - ``adapter.base_url`` is set → OpenAI-compatible provider (HF,
        NIM, Groq, Cerebras, Ollama, llama.cpp, local) →
        ``OpenAICompatibleChatClient`` (plain chat only)

    The two-class split surfaces F7 (DOGFOOD_FINDINGS_2026-05-17): the
    ``beta.chat.completions.parse`` extension is OpenAI-specific and
    returns degenerate parses (or 404s) when called against compatible
    providers. The factory makes the right choice automatically; slider
    callers should use this instead of constructing ``OpenAIChatClient``
    directly.
    """
    if adapter.base_url is None:
        return OpenAIChatClient(adapter)
    return OpenAICompatibleChatClient(adapter)
