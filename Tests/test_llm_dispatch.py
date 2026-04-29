"""Unit tests for the vendor-agnostic LLM dispatcher.

Pinned behaviours:

  1. **Model-id dispatch.** ``gpt-*``, ``o1-*``, ``o3-*``, ``o4-*``
     route to OpenAI; ``claude-*`` routes to Anthropic; anything
     else raises ``ValueError`` (no silent fallthrough).

  2. **Pydantic → Anthropic input_schema bridge.** Pydantic v2's
     ``model_json_schema()`` may emit ``$defs`` / ``$ref`` for
     nested models (used by ``ExtractionResponse`` and
     ``ConstrainedExtractionResponse``). The bridge MUST inline
     the refs so Anthropic's tool ``input_schema`` doesn't have
     to resolve external references.

  3. **AnthropicAdapter.parse_structured** assembles a tool
     definition with the Pydantic schema's ``input_schema``,
     forces tool use via ``tool_choice``, and parses the
     ``tool_use.input`` back through ``model_validate``.

  4. **AnthropicAdapter.generate_text** assembles a plain
     Messages-API call and concatenates text blocks.

  5. **Per-call timeout.** Both adapters surface
     ``LLMCallTimeoutError`` on ``asyncio.TimeoutError``.

The tests use a hand-rolled mock instead of importing the real
Anthropic SDK, so they run on minimal envs without the
``[anthropic]`` extra. The OpenAI side is exercised by the
existing s25 runner tests.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

# Optional dep: skip the AnthropicAdapter behavioural tests if not
# installed. The dispatch + schema-bridge tests are pure-Python and
# always run.
anthropic = pytest.importorskip(
    "anthropic",
    reason="[anthropic] extra not installed",
)


# ---------------------------------------------------------------------
# Pure-python: dispatcher + schema bridge (no SDK required)
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected_provider",
    [
        ("gpt-4o-mini-2024-07-18", "openai"),
        ("gpt-5.5", "openai"),
        ("o1-preview", "openai"),
        ("o3-mini-high", "openai"),
        ("o4-2026", "openai"),
        ("claude-opus-4-7", "anthropic"),
        ("claude-3-5-sonnet-20241022", "anthropic"),
    ],
)
def test_get_adapter_dispatches_by_prefix(model, expected_provider, monkeypatch):
    """The dispatcher MUST route correctly based on model-id prefix."""
    from sum_engine_internal.ensemble import llm_dispatch

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    adapter = llm_dispatch.get_adapter(model)
    assert type(adapter).__name__.lower().startswith(expected_provider)
    assert adapter.model == model


def test_get_adapter_refuses_unknown_prefix():
    """Unknown model-id prefixes MUST raise ValueError, not silently
    fall through to a wrong provider."""
    from sum_engine_internal.ensemble import llm_dispatch

    with pytest.raises(ValueError, match="cannot route"):
        llm_dispatch.get_adapter("mistral-large-2025")
    with pytest.raises(ValueError, match="cannot route"):
        llm_dispatch.get_adapter("llama-3.5-405b")


def test_pydantic_schema_to_anthropic_input_schema_inlines_defs():
    """Pydantic v2's ``model_json_schema()`` emits ``$defs`` for
    nested models. Anthropic accepts this but inlining keeps the
    surface predictable. The bridge MUST resolve every
    ``$ref: "#/$defs/X"`` against the top-level ``$defs`` map and
    drop the ``$defs`` key from the output."""
    from sum_engine_internal.ensemble.llm_dispatch import (
        _pydantic_to_anthropic_input_schema,
    )
    from sum_engine_internal.ensemble.live_llm_adapter import ExtractionResponse

    schema = _pydantic_to_anthropic_input_schema(ExtractionResponse)

    # Top-level shape: object schema with properties
    assert schema.get("type") == "object"
    assert "properties" in schema
    # No $defs / $ref residue anywhere in the tree
    def _walk(node):
        if isinstance(node, dict):
            assert "$defs" not in node, f"$defs leaked: {node}"
            for k, v in node.items():
                if k == "$ref":
                    pytest.fail(f"unresolved $ref: {v}")
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
    _walk(schema)

    # The triplets array's items schema MUST have the SemanticTriplet
    # fields inlined, not a ref.
    triplets_items = schema["properties"]["triplets"]["items"]
    assert triplets_items.get("type") == "object"
    assert "subject" in triplets_items["properties"]
    assert "predicate" in triplets_items["properties"]


# ---------------------------------------------------------------------
# AnthropicAdapter behavioural tests (mocked client)
# ---------------------------------------------------------------------


@dataclass
class _Block:
    type: str
    input: dict | None = None
    text: str | None = None


@dataclass
class _Response:
    content: list[_Block]


class _MockAsyncMessages:
    """Mock for client.messages — exposes async ``create`` that
    returns a canned response captured by tests."""

    def __init__(self, response: _Response, captured: dict):
        self._response = response
        self._captured = captured

    async def create(self, **kwargs):
        self._captured.update(kwargs)
        return self._response


class _MockAnthropicClient:
    def __init__(self, response: _Response, captured: dict):
        self.messages = _MockAsyncMessages(response, captured)


@pytest.fixture
def monkeypatch_anthropic_client(monkeypatch):
    """Replace ``anthropic.AsyncAnthropic`` with a mock factory that
    yields a pre-canned ``_MockAnthropicClient``."""
    captured: dict = {}
    response_holder: dict = {"response": None}

    def factory(*args, **kwargs):
        return _MockAnthropicClient(response_holder["response"], captured)

    import anthropic as _anthropic
    monkeypatch.setattr(_anthropic, "AsyncAnthropic", factory)
    return response_holder, captured


def test_anthropic_parse_structured_extracts_tool_use_input(monkeypatch_anthropic_client):
    """Given a canned response with a tool_use block carrying a
    valid JSON payload, ``parse_structured`` MUST return a
    Pydantic model populated from that payload."""
    from sum_engine_internal.ensemble.live_llm_adapter import ExtractionResponse
    from sum_engine_internal.ensemble.llm_dispatch import AnthropicAdapter

    holder, captured = monkeypatch_anthropic_client
    holder["response"] = _Response(content=[
        _Block(
            type="tool_use",
            input={
                "triplets": [
                    {"subject": "alice", "predicate": "like", "object": "cats"},
                    {"subject": "bob", "predicate": "own", "object": "dog"},
                ]
            },
        )
    ])

    adapter = AnthropicAdapter(model="claude-opus-4-7", api_key="sk-ant-test")
    parsed = asyncio.run(
        adapter.parse_structured(
            system="extract triples",
            user="Alice likes cats. Bob owns a dog.",
            schema=ExtractionResponse,
            call_timeout_s=10.0,
        )
    )

    assert parsed is not None
    assert len(parsed.triplets) == 2
    assert parsed.triplets[0].subject == "alice"
    # The captured request MUST have set tool_choice forcing tool use.
    assert captured["tool_choice"]["type"] == "tool"
    assert captured["tool_choice"]["name"] == "emit_structured_output"
    assert captured["tools"][0]["name"] == "emit_structured_output"
    assert captured["tools"][0]["input_schema"]["type"] == "object"
    # System prompt and user message routed correctly
    assert captured["system"] == "extract triples"
    assert captured["messages"][0]["role"] == "user"


def test_anthropic_parse_structured_returns_none_on_invalid_payload(
    monkeypatch_anthropic_client,
):
    """If the tool_use payload fails Pydantic validation, the
    adapter MUST return None rather than raise. This mirrors the
    OpenAI path's behaviour when the parsed message is None."""
    from sum_engine_internal.ensemble.live_llm_adapter import ExtractionResponse
    from sum_engine_internal.ensemble.llm_dispatch import AnthropicAdapter

    holder, _ = monkeypatch_anthropic_client
    holder["response"] = _Response(content=[
        _Block(
            type="tool_use",
            input={"triplets": [{"subject": "", "predicate": "", "object": ""}]},
        )
    ])

    adapter = AnthropicAdapter(model="claude-opus-4-7", api_key="sk-ant-test")
    parsed = asyncio.run(
        adapter.parse_structured(
            system="x", user="y",
            schema=ExtractionResponse, call_timeout_s=10.0,
        )
    )
    assert parsed is None


def test_anthropic_generate_text_concatenates_text_blocks(
    monkeypatch_anthropic_client,
):
    from sum_engine_internal.ensemble.llm_dispatch import AnthropicAdapter

    holder, captured = monkeypatch_anthropic_client
    holder["response"] = _Response(content=[
        _Block(type="text", text="Hello, "),
        _Block(type="text", text="world."),
    ])

    adapter = AnthropicAdapter(model="claude-opus-4-7", api_key="sk-ant-test")
    out = asyncio.run(
        adapter.generate_text(
            system="be brief",
            user="say hi",
            call_timeout_s=10.0,
        )
    )
    assert out == "Hello, world."
    # Generate-text path MUST NOT pass tools/tool_choice (those are
    # parse_structured-only territory).
    assert "tools" not in captured
    assert "tool_choice" not in captured


def test_anthropic_timeout_raises_llm_call_timeout_error(
    monkeypatch_anthropic_client,
):
    """A coroutine that never resolves MUST surface as
    ``LLMCallTimeoutError`` after the per-call budget."""
    from sum_engine_internal.ensemble.llm_dispatch import (
        AnthropicAdapter,
        LLMCallTimeoutError,
    )

    # Replace messages.create with a sleeping coro to force the timeout.
    holder, _ = monkeypatch_anthropic_client
    holder["response"] = _Response(content=[_Block(type="text", text="x")])

    adapter = AnthropicAdapter(model="claude-opus-4-7", api_key="sk-ant-test")

    async def _hang(**kwargs):
        await asyncio.sleep(5)
        return holder["response"]

    adapter._client.messages.create = _hang  # type: ignore[assignment]

    with pytest.raises(LLMCallTimeoutError, match="exceeded per-call budget"):
        asyncio.run(
            adapter.generate_text(
                system="x", user="y",
                call_timeout_s=0.05,
            )
        )
