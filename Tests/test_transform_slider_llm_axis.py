"""T1c-followup — slider LLM-axis dispatch through the transform registry.

Pre-T1c-followup, ``SliderTransform.apply()`` raised
``NotImplementedError`` when any of length / formality / audience /
perspective was off the 0.5 bin centre, deferring to the legacy
``slider_renderer.render`` API. This PR wires the registry's
``slider`` transform to dispatch through that library when LLM axes
are off-centre, producing a ``sum.transform_receipt.v1`` envelope
with ``provider = openai`` and
``digital_source_type = trainedAlgorithmicMedia``.

These tests cover:

  - Off-centre LLM axis with a fake LLM client + extractor produces
    a ``TransformResult`` with LLM-path provenance.
  - No OPENAI_API_KEY → clean ValueError with the operator-actionable
    hint, NOT a NotImplementedError.
  - Canonical path (all LLM axes at 0.5) still produces the
    deterministic render — unchanged from T1a.
  - The transform_id / hashes still verify back through
    ``sum.transform_receipt.v1`` end-to-end (T4-style integration).

Fact-preservation claims that live downstream of this dispatch path
are ``empirical-benchmark`` per ``docs/PROOF_BOUNDARY.md`` §5 and
gated on the bench-hardening worktrail
(``docs/BENCH_HARDENING_FROM_QCVV.md``).
"""
from __future__ import annotations

from typing import Any, List, Tuple
from unittest.mock import patch

import pytest

from sum_engine_internal.transforms import get_transform
from sum_engine_internal.transforms._base import TransformEnv


# ─── Fakes ──────────────────────────────────────────────────────────


class _FakeLLMChatClient:
    """Satisfies the slider_renderer.LLMChatClient protocol with a
    deterministic, no-network tome generator."""

    def __init__(self, tome: str = "A fake tome."):
        self._tome = tome
        self.chat_completion_calls: list[tuple[str, str]] = []
        self.chat_completion_structured_calls: list[tuple[str, str]] = []

    async def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
    ) -> str:
        self.chat_completion_calls.append((system_prompt, user_prompt))
        return self._tome

    async def chat_completion_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
    ) -> tuple[str, list[Tuple[str, str, str]]]:
        self.chat_completion_structured_calls.append((system_prompt, user_prompt))
        # Return the fake tome plus an empty "claimed" set; the
        # renderer treats empty claimed as the standard fallback.
        return self._tome, []


class _FakeExtractor:
    """Triple extractor stub. Returns a pinned set so slider_renderer's
    drift measurement is deterministic across test runs."""

    def __init__(self, triples: list[Tuple[str, str, str]]):
        self._triples = list(triples)
        self.calls: list[str] = []

    async def __call__(self, text: str) -> List[Tuple[str, str, str]]:
        self.calls.append(text)
        return list(self._triples)


class _FakeLiveLLMAdapter:
    """Stands in for ``LiveLLMAdapter`` so the slider transform's
    ``_apply_llm_axis`` builds a no-network dispatch under test."""

    def __init__(self, *args, fake_extractor: _FakeExtractor, **kwargs):
        self.client = None  # not used by our fake chat client
        self.model = kwargs.get("model", "fake-test-model")
        self._fake_extractor = fake_extractor

    async def extract_triplets(self, chunk: str) -> List[Tuple[str, str, str]]:
        return await self._fake_extractor(chunk)


class _FakeOpenAIChatClient(_FakeLLMChatClient):
    """Drop-in stand-in for ``OpenAIChatClient`` — same chat surface,
    no openai dep, no network."""

    def __init__(self, adapter, tome: str = "A fake tome."):
        super().__init__(tome=tome)
        self._adapter = adapter


# ─── Tests ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_canonical_path_unchanged_by_t1c_followup():
    """All LLM axes at 0.5 → canonical path, provider = canonical-path,
    no LLM client built. Regression guard for T1a behaviour."""
    slider = get_transform("slider")
    env = TransformEnv()  # no LLM key — should still succeed
    result = await slider.apply(
        {"triples": [["alice", "likes", "cats"], ["bob", "owns", "dog"]]},
        {
            "density": 1.0,
            "length": 0.5, "formality": 0.5,
            "audience": 0.5, "perspective": 0.5,
        },
        env,
    )
    assert result.provider == "canonical-path"
    assert result.digital_source_type == "algorithmicMedia"
    assert result.model == "canonical-deterministic-v0"
    assert result.llm_calls_made == 0
    assert "alice" in result.output and "bob" in result.output


@pytest.mark.asyncio
async def test_llm_axis_without_openai_key_raises_clear_error():
    """LLM-axis render with no OPENAI_API_KEY in env raises ValueError
    (NOT NotImplementedError as in T1a). The error message names the
    operator-actionable fix."""
    slider = get_transform("slider")
    env = TransformEnv()  # openai_api_key is None
    with pytest.raises(ValueError) as exc_info:
        await slider.apply(
            {"triples": [["alice", "likes", "cats"]]},
            {
                "density": 1.0,
                "length": 0.9,  # off-centre — forces LLM path
                "formality": 0.5,
                "audience": 0.5,
                "perspective": 0.5,
            },
            env,
        )
    msg = str(exc_info.value).lower()
    assert "openai_api_key" in msg or "openai api key" in msg
    # Anthropic-via-Worker hint surfaces — operator-actionable.
    assert "worker" in msg or "anthropic" in msg


@pytest.mark.asyncio
async def test_llm_axis_dispatch_returns_llm_provenance():
    """Off-centre LLM axis with a fake LLM client produces a
    TransformResult whose provider = openai, digital_source_type =
    trainedAlgorithmicMedia, and llm_calls_made >= 1. The receipt's
    parameters_hash / input_hash / output_hash are computed from the
    same canonicalisers as the canonical path."""
    slider = get_transform("slider")
    env = TransformEnv(openai_api_key="test-sk-not-a-real-key")

    triples = [("alice", "likes", "cats")]
    fake_extractor = _FakeExtractor(triples)

    def fake_live_adapter_ctor(*args, **kwargs):
        return _FakeLiveLLMAdapter(*args, fake_extractor=fake_extractor, **kwargs)

    def fake_chat_client_ctor(adapter):
        return _FakeOpenAIChatClient(adapter, tome="alice likes cats indeed.")

    # Patch the two classes inside live_llm_adapter so the slider
    # transform's lazy import resolves to our fakes.
    with patch(
        "sum_engine_internal.ensemble.live_llm_adapter.LiveLLMAdapter",
        fake_live_adapter_ctor,
    ), patch(
        "sum_engine_internal.ensemble.live_llm_adapter.OpenAIChatClient",
        fake_chat_client_ctor,
    ):
        result = await slider.apply(
            {"triples": [["alice", "likes", "cats"]]},
            {
                "density": 1.0,
                "length": 0.9,  # off-centre
                "formality": 0.5,
                "audience": 0.5,
                "perspective": 0.5,
            },
            env,
        )

    assert result.provider == "openai"
    assert result.digital_source_type == "trainedAlgorithmicMedia"
    assert result.llm_calls_made >= 1
    assert "alice" in result.output  # fake tome contains source claim
    # Drift map surfaces in extras; LLM-axis renders ALWAYS measure
    # per-axis drift for downstream consumers / bench wiring.
    assert "drift" in result.extra
    assert "render_id" in result.extra
    # Extractor was called at least once (re-extraction pass).
    assert fake_extractor.calls, "fake extractor should have been called"


@pytest.mark.asyncio
async def test_llm_axis_receipt_round_trips_through_verifier():
    """End-to-end integration: LLM-axis render → sign receipt → verify.
    Locks that the transform-receipt verifier accepts the LLM-path
    envelope the same way it accepts the canonical-path one."""
    pytest.importorskip("joserfc", reason="install sum-engine[receipt-verify]")
    from joserfc.jwk import OKPKey

    from sum_engine_internal.infrastructure.jcs import canonicalize
    from sum_engine_internal.transform_receipt import (
        build_payload,
        canonical_hash,
        sign_transform_receipt,
        verify_transform_receipt,
    )

    slider = get_transform("slider")
    env = TransformEnv(openai_api_key="test-sk-not-a-real-key")

    triples = [("alice", "likes", "cats")]
    fake_extractor = _FakeExtractor(triples)

    def fake_live_adapter_ctor(*args, **kwargs):
        return _FakeLiveLLMAdapter(*args, fake_extractor=fake_extractor, **kwargs)

    def fake_chat_client_ctor(adapter):
        return _FakeOpenAIChatClient(adapter, tome="alice likes cats indeed.")

    with patch(
        "sum_engine_internal.ensemble.live_llm_adapter.LiveLLMAdapter",
        fake_live_adapter_ctor,
    ), patch(
        "sum_engine_internal.ensemble.live_llm_adapter.OpenAIChatClient",
        fake_chat_client_ctor,
    ):
        result = await slider.apply(
            {"triples": [["alice", "likes", "cats"]]},
            {
                "density": 1.0,
                "length": 0.9,
                "formality": 0.5,
                "audience": 0.5,
                "perspective": 0.5,
            },
            env,
        )

    params = {
        "density": 1.0, "length": 0.9, "formality": 0.5,
        "audience": 0.5, "perspective": 0.5,
    }
    raw_input = {"triples": [["alice", "likes", "cats"]]}

    parameters_hash = canonical_hash(slider.canonicalize_parameters(params))
    input_hash = canonical_hash(slider.canonicalize_input(raw_input))
    output_hash = canonical_hash(slider.canonicalize_output(result.output))

    kid = "t1c-followup-test-kid"
    key = OKPKey.generate_key("Ed25519")
    private = key.as_dict(private=True)

    payload = build_payload(
        transform="slider",
        parameters_hash=parameters_hash,
        input_hash=input_hash,
        output_hash=output_hash,
        model=result.model,
        provider=result.provider,
        digital_source_type=result.digital_source_type,
    )
    receipt = sign_transform_receipt(payload, private_jwk=private, kid=kid)

    public = key.as_dict(private=False)
    public["kid"] = kid
    public["alg"] = "EdDSA"
    public["use"] = "sig"
    verify_result = verify_transform_receipt(receipt, {"keys": [public]})
    assert verify_result.verified is True
    assert receipt["payload"]["provider"] == "openai"
    assert receipt["payload"]["digital_source_type"] == "trainedAlgorithmicMedia"
