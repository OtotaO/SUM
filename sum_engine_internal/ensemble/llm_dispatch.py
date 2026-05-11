"""Vendor-agnostic LLM adapter for §2.5 frontier-model benchmarks.

Three providers, one surface:

  - ``OpenAIAdapter``    — wraps ``AsyncOpenAI`` (gpt-* / o1-* / o3-* /
    o4-* models, also reused for HF Inference Providers and local
    OpenAI-API-compatible servers via base_url override)
  - ``AnthropicAdapter`` — wraps ``AsyncAnthropic`` (claude-* models)
  - ``LocalLLMAdapter``  — OpenAIAdapter pointed at a local OpenAI-API-
    compatible server: Ollama (`http://localhost:11434/v1`),
    llama.cpp server (`http://localhost:8080/v1`), or any other
    server that implements `/chat/completions`. No new SDK
    dependency — uses the OpenAI SDK as a drop-in client. This
    is the open-weights pathway named in the NLnet NGI Zero
    application's "vendor adapters for OpenAI and a local
    open-weights pathway" deliverable.

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

  - ``claude-...``         → AnthropicAdapter
  - ``gpt-...``, ``o1-...``, ``o3-...``, ``o4-...`` → OpenAIAdapter
  - ``ollama:<model>``     → LocalLLMAdapter (Ollama; default base
    ``http://localhost:11434/v1``)
  - ``llamacpp:<model>``   → LocalLLMAdapter (llama.cpp server;
    default base ``http://localhost:8080/v1``)
  - ``local:<model>``      → LocalLLMAdapter (generic; base resolved
    from the ``SUM_LOCAL_LLM_BASE`` env var, no built-in default
    because the path is "you brought your own server")
  - ``org/model`` (HF namespaced) → OpenAIAdapter pointed at HF
    Inference Providers router
  - anything else          → ValueError (be explicit; refuse to guess)

Providers run under their own optional extras:

  - ``pip install 'sum-engine[openai]'``    (OpenAI; covers the
    LocalLLMAdapter path too — Ollama / llama.cpp / generic local
    servers all use the OpenAI SDK as a drop-in client)
  - ``pip install 'sum-engine[llm]'``       (back-compat alias for
    [openai])
  - ``pip install 'sum-engine[anthropic]'`` (Anthropic)

Extras can coexist; importing the dispatcher with none installed is
fine — only ``get_adapter`` actually needs the SDK that matches the
chosen model.

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

# Hugging Face Inference Providers exposes an OpenAI-compatible router at
# `https://router.huggingface.co/v1`, so we reuse OpenAIAdapter with that
# base_url when the model id is a HF-style namespaced id (`org/model`).
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"

# Local-server defaults. The OpenAI-compatible surface exposed by both
# Ollama and llama.cpp's `server` binary lets us reuse OpenAIAdapter as
# the client — we just point base_url at the local server. The two
# defaults match each project's documented port (Ollama: 11434, llama.cpp
# server: 8080). The generic `local:` route has no built-in default; the
# caller MUST set SUM_LOCAL_LLM_BASE because the prefix is meant for
# bring-your-own-server scenarios where guessing the port would be
# wrong more often than right.
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
LLAMACPP_DEFAULT_BASE_URL = "http://localhost:8080/v1"
LOCAL_LLM_ENV_VAR = "SUM_LOCAL_LLM_BASE"

# Local-server prefix shapes. We strip the prefix before passing the
# model id to the API because local servers expect bare model names
# (e.g., `llama3`, `phi3:14b`), not the SUM-internal routing prefix.
_LOCAL_PREFIXES: tuple[tuple[str, str | None], ...] = (
    ("ollama:", OLLAMA_DEFAULT_BASE_URL),
    ("llamacpp:", LLAMACPP_DEFAULT_BASE_URL),
    ("local:", None),  # base resolved from env at call time
)


def get_adapter(
    model: str,
    *,
    api_key: Optional[str] = None,
) -> "_BaseAdapter":
    """Construct the right adapter for *model*.

    Routing is by model-id shape:
      - ``claude-...``               → AnthropicAdapter (`$ANTHROPIC_API_KEY`)
      - ``gpt-... / o1-* / o3-* / o4-*`` → OpenAIAdapter (`$OPENAI_API_KEY`)
      - ``ollama:<model>``           → LocalLLMAdapter pointed at
        ``http://localhost:11434/v1`` (Ollama default port)
      - ``llamacpp:<model>``         → LocalLLMAdapter pointed at
        ``http://localhost:8080/v1`` (llama.cpp server default port)
      - ``local:<model>``            → LocalLLMAdapter with base resolved
        from the ``SUM_LOCAL_LLM_BASE`` env var. No built-in default — a
        bring-your-own-server route raises ``ValueError`` if the env var
        is unset, because guessing the port would be wrong more often
        than right.
      - any id containing ``/``      → OpenAIAdapter pointed at the HF
        Inference Providers router (`$HF_TOKEN`). HF model ids are
        canonically namespaced (e.g. ``meta-llama/Llama-4-Maverick…``,
        ``Qwen/Qwen3.6-35B-A3B``), and the OpenAI Python SDK works as a
        drop-in client against the HF router because the HF router
        implements the `/chat/completions` surface.

    Raises ``ValueError`` for unrecognised model ids — explicit refusal
    beats silent fallthrough when a typo would route the wrong way.
    Raises ``ImportError`` if the matching SDK extra is not installed.
    """
    m = model.lower()
    # Local-server prefixes run first. Order matters because
    # `llamacpp:` and `ollama:` are unambiguous and we want them to
    # win before the HF `/` route picks up something like
    # `ollama:org/model`.
    for prefix, default_base in _LOCAL_PREFIXES:
        if m.startswith(prefix):
            bare_model = model[len(prefix):]
            base_url = default_base or os.environ.get(LOCAL_LLM_ENV_VAR)
            if base_url is None:
                raise ValueError(
                    f"llm_dispatch: model {model!r} uses the "
                    f"`local:` prefix but {LOCAL_LLM_ENV_VAR} is not "
                    f"set. Set the env var to your local OpenAI-API-"
                    f"compatible server's base URL, or use the "
                    f"`ollama:` / `llamacpp:` prefixes for their "
                    f"documented default ports."
                )
            return LocalLLMAdapter(
                model=bare_model, api_key=api_key, base_url=base_url,
            )
    # HF route detection runs after local prefixes because a few HF orgs
    # (e.g. ``openai`` for gpt-oss) start with "gpt-" but the namespace
    # prefix makes the routing target unambiguous: namespaced ids
    # belong to HF.
    if "/" in model:
        hf_token = api_key or os.environ.get("HF_TOKEN")
        return OpenAIAdapter(
            model=model, api_key=hf_token, base_url=HF_ROUTER_BASE_URL,
        )
    if m.startswith(_ANTHROPIC_PREFIXES):
        return AnthropicAdapter(model=model, api_key=api_key)
    if m.startswith(_OPENAI_PREFIXES):
        return OpenAIAdapter(model=model, api_key=api_key)
    raise ValueError(
        f"llm_dispatch: cannot route model {model!r} — expected an id "
        f"starting with one of {sorted(_OPENAI_PREFIXES + _ANTHROPIC_PREFIXES)}, "
        f"or a HF-namespaced id of the form ``org/model``, "
        f"or a local-prefixed id (``ollama:`` / ``llamacpp:`` / "
        f"``local:``). "
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
    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Construct an AsyncOpenAI client.

        ``base_url`` is forwarded verbatim to the SDK; ``None`` means
        the SDK uses its default (api.openai.com). Set it to
        ``HF_ROUTER_BASE_URL`` (``https://router.huggingface.co/v1``)
        to route through Hugging Face Inference Providers, in which
        case ``api_key`` should be an HF token rather than an OpenAI
        key.
        """
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
            base_url=base_url,
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
# Local OpenAI-API-compatible server (Ollama / llama.cpp / generic)
# ---------------------------------------------------------------------


class LocalLLMAdapter(OpenAIAdapter):
    """OpenAIAdapter pointed at a local OpenAI-API-compatible server.

    Ollama (default port 11434) and the llama.cpp `server` binary
    (default port 8080) both expose a `/chat/completions` endpoint
    compatible with the OpenAI Python SDK. So we don't need a new
    HTTP client — we use AsyncOpenAI with `base_url` set to the
    local server, and `api_key` defaulted to a placeholder (most
    local servers ignore the bearer token, but the SDK requires
    `api_key` to be non-empty).

    Constructed by ``get_adapter`` when the model id starts with
    one of: ``ollama:``, ``llamacpp:``, ``local:``. The prefix is
    stripped before this adapter is constructed; the underlying
    API receives the bare model name (e.g. ``llama3:8b``).

    Same call surface as OpenAIAdapter: ``parse_structured`` for
    Pydantic-typed outputs (works against any server that supports
    the OpenAI structured-outputs schema; Ollama added support in
    v0.4 via its `format: "json_schema"` field, with parity to
    OpenAI's `beta.chat.completions.parse` improving across
    versions), and ``generate_text`` for plain completions.

    The render-receipt `provider` field for local-served renders is
    not yet defined as a distinct value — the funder commitment
    (NLnet) is the Python-side pathway, not the Worker dispatch.
    A future ``cf-ai-gateway-local`` / ``local`` provider variant
    would be added if the Worker grows a local-LLM path; until
    then the receipt's `provider` field reflects the OpenAI-SDK
    surface in use.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: str,
    ):
        # Local servers typically ignore the bearer; the SDK requires
        # *something* truthy in `api_key`. "local" is a placeholder that
        # signals intent in any server-side log that records it.
        effective_key = api_key or os.environ.get("OPENAI_API_KEY") or "local"
        super().__init__(model=model, api_key=effective_key, base_url=base_url)
        self._base_url = base_url

    def __repr__(self) -> str:
        return f"LocalLLMAdapter(model={self.model!r}, base_url={self._base_url!r})"


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
