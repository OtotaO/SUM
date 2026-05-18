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
    ``_apply_llm_axis`` builds a no-network dispatch under test.

    Surfaces ``model``, ``base_url``, and ``extract_triplets`` —
    the three attributes slider._apply_llm_axis reads off the
    adapter. Constructor accepts the from_model factory's signature
    AND the legacy direct-instantiation signature, so both old and
    new tests work without conditional branches."""

    def __init__(self, *args, fake_extractor: _FakeExtractor, **kwargs):
        self.client = None  # not used by our fake chat client
        self.model = kwargs.get("model", "fake-test-model")
        self.base_url = kwargs.get("base_url")  # honest routing surface
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
async def test_llm_axis_without_openai_key_raises_clear_error(monkeypatch):
    """LLM-axis render with default model (OpenAI-shaped) and no
    OPENAI_API_KEY in env raises ValueError that names the four
    free / BYO escape valves (HF / Ollama / local / Worker-Anthropic)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    slider = get_transform("slider")
    env = TransformEnv()  # openai_api_key is None, model is None
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
    # Error names OpenAI as the default, and lists at least three
    # alternatives: HF (HF_TOKEN), Ollama, local (Modal/Fireworks).
    assert "openai" in msg
    assert "hf_token" in msg or "hugging face" in msg
    assert "ollama" in msg
    assert "local" in msg or "modal" in msg


def test_live_llm_adapter_from_model_routing(monkeypatch):
    """LiveLLMAdapter.from_model dispatches to the right base URL
    based on the model-id shape. Locks the routing contract that
    docs/BYOK_AND_FREE_PROVIDERS.md cites."""
    from sum_engine_internal.ensemble.live_llm_adapter import LiveLLMAdapter

    # OpenAI default — no base_url
    a = LiveLLMAdapter.from_model("gpt-4o-mini", api_key="sk-test")
    assert a.base_url is None
    assert a.model == "gpt-4o-mini"

    # HF Inference Providers — namespaced id
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    a = LiveLLMAdapter.from_model("meta-llama/Llama-3.3-70B-Instruct")
    assert a.base_url == "https://router.huggingface.co/v1"
    assert a.model == "meta-llama/Llama-3.3-70B-Instruct"

    # Ollama prefix
    a = LiveLLMAdapter.from_model("ollama:llama3.1")
    assert a.base_url == "http://localhost:11434/v1"
    assert a.model == "llama3.1"

    # llama.cpp prefix
    a = LiveLLMAdapter.from_model("llamacpp:phi-3")
    assert a.base_url == "http://localhost:8080/v1"
    assert a.model == "phi-3"

    # local prefix needs $SUM_LOCAL_LLM_BASE
    monkeypatch.setenv("SUM_LOCAL_LLM_BASE", "https://my-modal.run/v1")
    a = LiveLLMAdapter.from_model("local:custom-model")
    assert a.base_url == "https://my-modal.run/v1"
    assert a.model == "custom-model"


def test_live_llm_adapter_from_model_hf_without_token_raises(monkeypatch):
    """HF-namespaced model id with no HF_TOKEN → clean error pointing
    at where to get one."""
    from sum_engine_internal.ensemble.live_llm_adapter import LiveLLMAdapter

    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValueError, match="HF_TOKEN"):
        LiveLLMAdapter.from_model("meta-llama/Llama-3.3-70B-Instruct")


def test_live_llm_adapter_from_model_local_without_base_raises(monkeypatch):
    """local: prefix with no $SUM_LOCAL_LLM_BASE → clean error."""
    from sum_engine_internal.ensemble.live_llm_adapter import LiveLLMAdapter

    monkeypatch.delenv("SUM_LOCAL_LLM_BASE", raising=False)
    with pytest.raises(ValueError, match="SUM_LOCAL_LLM_BASE"):
        LiveLLMAdapter.from_model("local:my-model")


def test_live_llm_adapter_from_model_nim_routing(monkeypatch):
    """`nim:<model>` routes to NVIDIA NIM base URL using NVIDIA_API_KEY."""
    from sum_engine_internal.ensemble.live_llm_adapter import LiveLLMAdapter

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-token")
    a = LiveLLMAdapter.from_model("nim:meta/llama-3.3-70b-instruct")
    assert a.base_url == "https://integrate.api.nvidia.com/v1"
    assert a.model == "meta/llama-3.3-70b-instruct"


def test_live_llm_adapter_from_model_nim_without_key_raises(monkeypatch):
    """`nim:` without NVIDIA_API_KEY → error pointing at build.nvidia.com."""
    from sum_engine_internal.ensemble.live_llm_adapter import LiveLLMAdapter

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        LiveLLMAdapter.from_model("nim:meta/llama-3.3-70b-instruct")


def test_live_llm_adapter_from_model_groq_routing(monkeypatch):
    """`groq:<model>` routes to Groq Cloud using GROQ_API_KEY."""
    from sum_engine_internal.ensemble.live_llm_adapter import LiveLLMAdapter

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_token")
    a = LiveLLMAdapter.from_model("groq:llama-3.3-70b-versatile")
    assert a.base_url == "https://api.groq.com/openai/v1"
    assert a.model == "llama-3.3-70b-versatile"


def test_live_llm_adapter_from_model_groq_without_key_raises(monkeypatch):
    """`groq:` without GROQ_API_KEY → error pointing at console.groq.com."""
    from sum_engine_internal.ensemble.live_llm_adapter import LiveLLMAdapter

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        LiveLLMAdapter.from_model("groq:llama-3.3-70b-versatile")


def test_live_llm_adapter_from_model_cerebras_routing(monkeypatch):
    """`cerebras:<model>` routes to Cerebras Cloud using CEREBRAS_API_KEY."""
    from sum_engine_internal.ensemble.live_llm_adapter import LiveLLMAdapter

    monkeypatch.setenv("CEREBRAS_API_KEY", "csk-test-token")
    a = LiveLLMAdapter.from_model("cerebras:llama3.1-8b")
    assert a.base_url == "https://api.cerebras.ai/v1"
    assert a.model == "llama3.1-8b"


def test_live_llm_adapter_from_model_cerebras_without_key_raises(monkeypatch):
    """`cerebras:` without CEREBRAS_API_KEY → error pointing at cloud.cerebras.ai."""
    from sum_engine_internal.ensemble.live_llm_adapter import LiveLLMAdapter

    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="CEREBRAS_API_KEY"):
        LiveLLMAdapter.from_model("cerebras:llama3.1-8b")


def test_make_chat_client_routes_by_base_url(monkeypatch):
    """F7 fix: make_chat_client picks OpenAIChatClient for OpenAI-proper
    adapters (no base_url) and OpenAICompatibleChatClient for everything
    else. The compatible client deliberately lacks
    chat_completion_structured so slider_renderer.render falls through to
    plain chat completion — beta.chat.completions.parse is OpenAI-only
    and returns degenerate parses on HF / NIM / Groq / Cerebras."""
    from sum_engine_internal.ensemble.live_llm_adapter import (
        LiveLLMAdapter,
        OpenAIChatClient,
        OpenAICompatibleChatClient,
        make_chat_client,
    )

    # OpenAI proper → has structured method
    openai_adapter = LiveLLMAdapter.from_model("gpt-4o-mini", api_key="sk-test")
    client = make_chat_client(openai_adapter)
    assert isinstance(client, OpenAIChatClient)
    assert hasattr(client, "chat_completion_structured")
    assert hasattr(client, "chat_completion")

    # NIM → compatible client, NO structured method
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    nim_adapter = LiveLLMAdapter.from_model("nim:meta/llama-3.3-70b-instruct")
    client = make_chat_client(nim_adapter)
    assert isinstance(client, OpenAICompatibleChatClient)
    assert not isinstance(client, OpenAIChatClient)
    assert not hasattr(client, "chat_completion_structured"), (
        "compatible client must NOT expose chat_completion_structured — "
        "slider_renderer.render uses hasattr() to pick the path; exposing "
        "it would route NIM/HF/Groq/Cerebras into the degenerate "
        "beta.chat.completions.parse path (F7)"
    )
    assert hasattr(client, "chat_completion")

    # HF Inference Providers → same compatible behaviour
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    hf_adapter = LiveLLMAdapter.from_model("meta-llama/Llama-3.3-70B-Instruct")
    client = make_chat_client(hf_adapter)
    assert isinstance(client, OpenAICompatibleChatClient)
    assert not hasattr(client, "chat_completion_structured")

    # Ollama → same
    ollama_adapter = LiveLLMAdapter.from_model("ollama:llama3.1")
    client = make_chat_client(ollama_adapter)
    assert isinstance(client, OpenAICompatibleChatClient)
    assert not hasattr(client, "chat_completion_structured")


@pytest.mark.asyncio
async def test_llm_axis_routes_to_hf_when_model_is_namespaced(monkeypatch):
    """env.model = 'org/model' + HF_TOKEN → adapter base_url is HF
    router. Receipt's extra.llm_endpoint reports the HF endpoint
    honestly. Mocks the LLM call so no network."""
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    slider = get_transform("slider")
    env = TransformEnv(model="meta-llama/Llama-3.3-70B-Instruct")

    triples = [("alice", "likes", "cats")]
    fake_extractor = _FakeExtractor(triples)

    def fake_from_model(model, api_key=None):
        # Inline the routing decision: HF-namespaced → HF router URL.
        # Avoid calling Real.from_model (which the patch has replaced
        # with this very function → recursion).
        a = _FakeLiveLLMAdapter(
            fake_extractor=fake_extractor,
            model=model,
            base_url="https://router.huggingface.co/v1",
        )
        return a

    def fake_chat_client_ctor(adapter):
        return _FakeOpenAIChatClient(adapter, tome="alice likes cats indeed.")

    with patch(
        "sum_engine_internal.ensemble.live_llm_adapter.LiveLLMAdapter.from_model",
        fake_from_model,
    ), patch(
        "sum_engine_internal.ensemble.live_llm_adapter.make_chat_client",
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

    # The slider succeeded via the HF route. Receipt extras name the
    # actual endpoint so consumers can audit which provider served.
    assert "llm_endpoint" in result.extra
    endpoint = result.extra["llm_endpoint"]
    assert endpoint["model"] == "meta-llama/Llama-3.3-70B-Instruct"
    assert "huggingface.co" in endpoint["base_url"]


@pytest.mark.asyncio
async def test_llm_axis_routes_to_ollama_when_model_is_prefixed(monkeypatch):
    """env.model = 'ollama:llama3.1' → adapter base_url is localhost
    Ollama. No API key needed."""
    slider = get_transform("slider")
    env = TransformEnv(model="ollama:llama3.1")

    triples = [("alice", "likes", "cats")]
    fake_extractor = _FakeExtractor(triples)

    def fake_from_model(model, api_key=None):
        # Ollama prefix → strip prefix, base = localhost:11434.
        bare = model[len("ollama:"):] if model.startswith("ollama:") else model
        a = _FakeLiveLLMAdapter(
            fake_extractor=fake_extractor,
            model=bare,
            base_url="http://localhost:11434/v1",
        )
        return a

    def fake_chat_client_ctor(adapter):
        return _FakeOpenAIChatClient(adapter, tome="alice likes cats indeed.")

    with patch(
        "sum_engine_internal.ensemble.live_llm_adapter.LiveLLMAdapter.from_model",
        fake_from_model,
    ), patch(
        "sum_engine_internal.ensemble.live_llm_adapter.make_chat_client",
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

    endpoint = result.extra["llm_endpoint"]
    assert endpoint["model"] == "llama3.1"  # stripped prefix
    assert endpoint["base_url"] == "http://localhost:11434/v1"


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

    def fake_from_model(model, api_key=None):
        return _FakeLiveLLMAdapter(
            fake_extractor=fake_extractor,
            api_key=api_key,
            model=model,
        )

    def fake_chat_client_ctor(adapter):
        return _FakeOpenAIChatClient(adapter, tome="alice likes cats indeed.")

    # Patch the classmethod factory + OpenAIChatClient inside
    # live_llm_adapter so the slider transform's lazy import resolves
    # to our fakes.
    with patch(
        "sum_engine_internal.ensemble.live_llm_adapter.LiveLLMAdapter.from_model",
        fake_from_model,
    ), patch(
        "sum_engine_internal.ensemble.live_llm_adapter.make_chat_client",
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

    def fake_from_model(model, api_key=None):
        return _FakeLiveLLMAdapter(
            fake_extractor=fake_extractor,
            api_key=api_key,
            model=model,
        )

    def fake_chat_client_ctor(adapter):
        return _FakeOpenAIChatClient(adapter, tome="alice likes cats indeed.")

    with patch(
        "sum_engine_internal.ensemble.live_llm_adapter.LiveLLMAdapter.from_model",
        fake_from_model,
    ), patch(
        "sum_engine_internal.ensemble.live_llm_adapter.make_chat_client",
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
