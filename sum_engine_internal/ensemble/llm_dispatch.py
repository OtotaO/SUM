"""Vendor-agnostic LLM adapter for §2.5 frontier-model benchmarks.

Two providers, one surface:

  - ``OpenAIAdapter``    — wraps ``AsyncOpenAI`` (gpt-* / o1-* / o3-* models)
  - ``AnthropicAdapter`` — wraps ``AsyncAnthropic`` (claude-* models)

Both expose:

  - ``parse_structured(*, system, user, schema, call_timeout_s)``
      Returns an instance of *schema* (a ``BaseModel`` subclass) or
      ``None`` if the model emitted no parseable output.
      OpenAI uses ``beta.chat.completions.parse`` with the Pydantic
      schema as ``response_format``. Anthropic uses tool-use: the
      Pydantic ``.model_json_schema()`` becomes the tool's
      ``input_schema``, ``tool_choice`` forces use, and the
      ``tool_use.input`` is fed back through ``model_validate``.

  - ``generate_text(*, system, user, call_timeout_s)``
      Plain text completion. OpenAI: ``chat.completions.create``.
      Anthropic: ``messages.create``.

Per-call timeouts are enforced at the adapter boundary via
``asyncio.wait_for`` and surface the project's standard
``S25CallTimeoutError`` so the s25 runner's per-doc skip path
catches both providers identically.

The ``get_adapter(model: str)`` dispatcher picks by model-id prefix:

  - ``claude-...``  → AnthropicAdapter
  - ``gpt-...``, ``o1-...``, ``o3-...``, ``o4-...`` → OpenAIAdapter
  - anything else   → ValueError (be explicit; refuse to guess)

Both providers run under their own optional extra:

  - ``pip install 'sum-engine[llm]'``       (OpenAI)
  - ``pip install 'sum-engine[anthropic]'`` (Anthropic)

Both extras can coexist; importing the dispatcher with neither
installed is fine — only ``get_adapter`` actually needs the SDK
that matches the chosen model.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Standard tagged error (mirrors scripts.bench.runners.s25_generator_side)
# ---------------------------------------------------------------------


class LLMCallTimeoutError(Exception):
    """Raised when a single LLM call exceeds the per-call budget.

    Tagged separately from generic exceptions so the runner can
    distinguish a hung-LLM doc-skip from a truly broken doc.
    """


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------


_OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-")
_ANTHROPIC_PREFIXES = ("claude-",)


def get_adapter(
    model: str,
    *,
    api_key: Optional[str] = None,
) -> "_BaseAdapter":
    """Construct the right adapter for *model*.

    *api_key* defaults to:
      - ``$OPENAI_API_KEY``     for openai models
      - ``$ANTHROPIC_API_KEY``  for anthropic models

    Raises ``ValueError`` for unrecognised model ids — explicit refusal
    beats silent fallthrough when a typo would route the wrong way.
    Raises ``ImportError`` if the matching SDK extra is not installed.
    """
    m = model.lower()
    if m.startswith(_ANTHROPIC_PREFIXES):
        return AnthropicAdapter(model=model, api_key=api_key)
    if m.startswith(_OPENAI_PREFIXES):
        return OpenAIAdapter(model=model, api_key=api_key)
    raise ValueError(
        f"llm_dispatch: cannot route model {model!r} — expected an id "
        f"starting with one of {sorted(_OPENAI_PREFIXES + _ANTHROPIC_PREFIXES)}. "
        f"Add the prefix to llm_dispatch._OPENAI_PREFIXES / "
        f"_ANTHROPIC_PREFIXES if a new family was just released."
    )


# ---------------------------------------------------------------------
# Base adapter (timeout helper)
# ---------------------------------------------------------------------


@dataclass
class _BaseAdapter:
    model: str

    @staticmethod
    async def _await_with_timeout(coro, timeout_s: float, what: str):
        try:
            return await asyncio.wait_for(coro, timeout=timeout_s)
        except asyncio.TimeoutError as e:
            raise LLMCallTimeoutError(
                f"{what}: exceeded per-call budget of {timeout_s:.1f}s"
            ) from e


# ---------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------


class OpenAIAdapter(_BaseAdapter):
    def __init__(self, *, model: str, api_key: Optional[str] = None):
        super().__init__(model=model)
        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "OpenAIAdapter requires the [llm] extra. "
                "Run: pip install 'sum-engine[llm]'"
            ) from e
        self._client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

    async def parse_structured(
        self,
        *,
        system: str,
        user: str,
        schema: Type["BaseModel"],
        call_timeout_s: float,
    ) -> Optional["BaseModel"]:
        coro = self._client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema,
        )
        response = await self._await_with_timeout(
            coro, call_timeout_s, what="openai.parse_structured",
        )
        return response.choices[0].message.parsed

    async def generate_text(
        self,
        *,
        system: str,
        user: str,
        call_timeout_s: float,
    ) -> str:
        coro = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        response = await self._await_with_timeout(
            coro, call_timeout_s, what="openai.generate_text",
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------


# Anthropic's messages.create requires max_tokens. Pick a generous
# but bounded ceiling — enough for a 50-doc benchmark's longest
# narrative without runaway cost.
_ANTHROPIC_MAX_TOKENS = 4096

# Tool name we register for structured extraction. Must match
# tool_choice; immaterial otherwise.
_STRUCTURED_TOOL_NAME = "emit_structured_output"


class AnthropicAdapter(_BaseAdapter):
    def __init__(self, *, model: str, api_key: Optional[str] = None):
        super().__init__(model=model)
        try:
            from anthropic import AsyncAnthropic  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "AnthropicAdapter requires the [anthropic] extra. "
                "Run: pip install 'sum-engine[anthropic]'"
            ) from e
        self._client = AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )

    async def parse_structured(
        self,
        *,
        system: str,
        user: str,
        schema: Type["BaseModel"],
        call_timeout_s: float,
    ) -> Optional["BaseModel"]:
        """Map Pydantic structured output → Anthropic tool-use.

        The Pydantic ``.model_json_schema()`` is fed verbatim as the
        tool's ``input_schema``. ``tool_choice`` forces the model to
        emit a tool_use block. The block's ``input`` is the
        structured payload, validated back through Pydantic.
        """
        input_schema = _pydantic_to_anthropic_input_schema(schema)
        tool = {
            "name": _STRUCTURED_TOOL_NAME,
            "description": f"Emit a {schema.__name__} object as the response.",
            "input_schema": input_schema,
        }
        coro = self._client.messages.create(
            model=self.model,
            max_tokens=_ANTHROPIC_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool],
            tool_choice={"type": "tool", "name": _STRUCTURED_TOOL_NAME},
        )
        response = await self._await_with_timeout(
            coro, call_timeout_s, what="anthropic.parse_structured",
        )

        # Locate the tool_use block. There MUST be exactly one because
        # tool_choice forced it; defensive scan in case of API changes.
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "tool_use":
                payload = getattr(block, "input", None)
                if not isinstance(payload, dict):
                    return None
                try:
                    return schema.model_validate(payload)
                except Exception:  # noqa: BLE001 — Pydantic ValidationError + family
                    logger.warning(
                        "anthropic.parse_structured: payload failed "
                        "%s.model_validate; payload=%s",
                        schema.__name__, json.dumps(payload)[:512],
                    )
                    return None
        return None

    async def generate_text(
        self,
        *,
        system: str,
        user: str,
        call_timeout_s: float,
    ) -> str:
        coro = self._client.messages.create(
            model=self.model,
            max_tokens=_ANTHROPIC_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        response = await self._await_with_timeout(
            coro, call_timeout_s, what="anthropic.generate_text",
        )
        # Concatenate any text blocks. Single block is the common case.
        chunks: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                chunks.append(getattr(block, "text", ""))
        return "".join(chunks)


# ---------------------------------------------------------------------
# Pydantic → Anthropic input_schema bridge
# ---------------------------------------------------------------------


def _pydantic_to_anthropic_input_schema(schema: Type["BaseModel"]) -> dict[str, Any]:
    """Convert a Pydantic v2 schema to an Anthropic ``input_schema``.

    Anthropic requires JSON Schema with ``type: "object"`` at the top
    level. Pydantic's ``model_json_schema()`` produces almost-
    compatible output; the differences we patch:

      - ``$defs`` → inline. Anthropic's tool input_schema accepts
        ``$defs`` per their docs, but inlining keeps the surface
        smaller. Pydantic emits ``$defs`` for nested models; we
        expand ``$ref`` references before passing to Anthropic.

      - Pydantic uses ``"title"`` keys liberally; Anthropic ignores
        them harmlessly. We leave them.

    The output is a fresh dict; the input schema is not mutated.
    """
    raw = schema.model_json_schema()
    return _inline_defs(raw)


def _inline_defs(node: Any, defs: Optional[dict] = None) -> Any:
    """Walk a JSON Schema node and inline any ``$ref: "#/$defs/X"``
    against the schema's top-level ``$defs`` map. Top-level call seeds
    *defs*; recursion uses the seeded value."""
    if defs is None and isinstance(node, dict):
        defs = node.get("$defs", {})
    if isinstance(node, dict):
        if "$ref" in node:
            ref = node["$ref"]
            # Only handle local $defs refs; pass through anything else.
            prefix = "#/$defs/"
            if ref.startswith(prefix) and defs is not None:
                key = ref[len(prefix):]
                target = defs.get(key, {})
                return _inline_defs(target, defs)
            return node
        out = {k: _inline_defs(v, defs) for k, v in node.items() if k != "$defs"}
        return out
    if isinstance(node, list):
        return [_inline_defs(item, defs) for item in node]
    return node
