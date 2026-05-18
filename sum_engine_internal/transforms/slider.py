"""Slider — the first registered transform.

The slider product runs on the transform registry. Existing
``sum_engine_internal.ensemble.slider_renderer.render`` stays as the
implementation library; this module is the protocol adapter that
exposes that library as ``Transform(name="slider")``.

What the slider transform does:

  - Accepts triples + slider position parameters.
  - For canonical-path renders (all LLM axes at 0.5 bin centre):
    applies the deterministic density prefix and composes a tome
    via a pure-algorithmic generator. Provider = ``canonical-path``,
    digital_source_type = ``algorithmicMedia``, llm_calls_made = 0.
  - For LLM-axis renders (any of length / formality / audience /
    perspective off-centre): dispatches through
    ``slider_renderer.render`` via ``LiveLLMAdapter`` (OpenAI). The
    transform receipt's provider field reflects what actually served
    (``openai`` today; Anthropic dispatch through the Python registry
    is a sibling PR — the Worker's TS path already routes both).

LLM-axis fact preservation is `empirical-benchmark` per
``docs/PROOF_BOUNDARY.md`` §5 — measured against the slider bench,
not absolutely guaranteed. The bench-hardening worktrail
(``docs/BENCH_HARDENING_FROM_QCVV.md`` T1-T3) extends the measurement
with iteration stability + worst-case DKW bounds before any release
cites the LLM-axis path as load-bearing.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from sum_engine_internal.infrastructure.jcs import canonicalize
from sum_engine_internal.transforms._base import (
    DigitalSourceType,
    Transform,
    TransformEnv,
    TransformResult,
)


# Slider axis bin-centres mirror worker/src/routes/render.ts +
# ensemble/slider_renderer.py: the four LLM axes snap to one of five
# bin centres {0.1, 0.3, 0.5, 0.7, 0.9}; density passes through
# unchanged. This canonicalisation matters because the receipt's
# parameters_hash is computed AFTER quantization, so two slider
# positions that quantize to the same bin produce byte-identical
# parameter hashes (and therefore share the cache).
_LLM_AXES = ("length", "formality", "audience", "perspective")
_BIN_COUNT = 5


def _snap_to_bin(value: float, bins: int = _BIN_COUNT) -> float:
    """Map [0, 1] → bin centre. Mirrors worker/src/routes/render.ts."""
    if value < 0.0 or value > 1.0:
        raise ValueError(f"slider value out of [0, 1]: {value}")
    idx = min(int(value * bins), bins - 1)
    return (idx + 0.5) / bins


def _quantize(params: dict[str, Any]) -> dict[str, float]:
    """Quantize parameters to the bin grid. Density passes through;
    LLM axes snap to bin centres. Raises ValueError on any out-of-
    range or missing axis."""
    out: dict[str, float] = {}
    density = params.get("density")
    if density is None or not isinstance(density, (int, float)):
        raise ValueError("slider parameters: missing or non-numeric 'density'")
    if density < 0.0 or density > 1.0:
        raise ValueError(f"density out of [0, 1]: {density}")
    out["density"] = float(density)
    for axis in _LLM_AXES:
        v = params.get(axis)
        if v is None or not isinstance(v, (int, float)):
            raise ValueError(f"slider parameters: missing or non-numeric {axis!r}")
        out[axis] = _snap_to_bin(float(v))
    return out


def _requires_llm(quantized: dict[str, float]) -> bool:
    """LLM dispatch is required iff any non-density axis is off-centre.
    Mirrors worker/src/render/axis_prompts.ts::requiresExtrapolator."""
    return any(quantized[axis] != 0.5 for axis in _LLM_AXES)


def _apply_density(triples: list[tuple[str, str, str]], density: float) -> list[tuple[str, str, str]]:
    """Keep the leading ``ceil(density * len)`` triples by SHA-256
    deterministic order. Mirrors worker/src/render/axis_prompts.ts.

    Empty input yields empty output regardless of density. Density 0
    yields one triple (the prefix); density 1 yields all triples.
    """
    if not triples:
        return []
    sorted_by_hash = sorted(
        triples,
        key=lambda t: hashlib.sha256(
            ("|".join(t)).encode("utf-8")
        ).hexdigest(),
    )
    if density <= 0.0:
        return sorted_by_hash[:1]
    keep = max(1, int(len(sorted_by_hash) * density + 0.9999))
    return sorted_by_hash[:keep]


def _deterministic_tome(triples: list[tuple[str, str, str]]) -> str:
    """Pure-algorithmic prose composition. Same shape as
    worker/src/render/axis_prompts.ts::deterministicTome."""
    if not triples:
        return ""
    sentences = []
    for s, p, o in triples:
        article = "The" if not s[:1].isupper() else ""
        pred = p.replace("_", " ")
        sentences.append(
            f"{article} {s} {pred} {o}.".strip()
            .replace("  ", " ")
        )
    return " ".join(sentences)


def _sort_triples_componentwise(
    triples: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Component-wise tuple-lex sort. Matches Python's default
    ``sorted(tuple-list)`` AND TypeScript's
    ``canonicalize(sortedArray)`` byte-for-byte — this is what
    keeps ``input_hash`` cross-runtime stable."""
    return sorted(triples)


@dataclass
class SliderTransform:
    """The slider transform — first entry in the registry.

    Concrete instance held in ``SLIDER_TRANSFORM`` below; the registry
    stores instances (not classes) so per-deployment configuration
    can be parametrised in a future revision without subclassing.
    """
    name: str = "slider"
    requires_llm: bool = True  # may require; canonical-path doesn't.
    digital_source_type: DigitalSourceType = "trainedAlgorithmicMedia"

    def canonicalize_parameters(self, params: dict[str, Any]) -> bytes:
        """JCS-canonical bytes of the QUANTIZED parameters. Quantization
        runs before hashing so a slider drag of length=0.43 vs 0.45
        produces the same parameters_hash (both snap to 0.5)."""
        quantized = _quantize(params)
        return canonicalize(quantized)

    def canonicalize_input(self, raw_input: Any) -> bytes:
        """Slider input is a triple set. Accepted shape:
            {"triples": [["s","p","o"], ...]}
        Canonicalisation: JCS of the component-wise-sorted triple list."""
        if not isinstance(raw_input, dict) or "triples" not in raw_input:
            raise ValueError(
                "slider input: expected dict with 'triples' key; got "
                f"{type(raw_input).__name__}"
            )
        triples_raw = raw_input["triples"]
        triples = [
            (str(t[0]), str(t[1]), str(t[2]))
            for t in triples_raw
        ]
        return canonicalize(_sort_triples_componentwise(triples))

    def canonicalize_output(self, output: Any) -> bytes:
        """Slider output is a tome string. Canonical bytes = UTF-8."""
        if not isinstance(output, str):
            raise ValueError(
                f"slider output: expected str (tome), got {type(output).__name__}"
            )
        return output.encode("utf-8")

    async def apply(
        self,
        input: Any,
        parameters: dict[str, Any],
        env: TransformEnv,
    ) -> TransformResult:
        """Run the slider transform.

        Canonical path (all LLM axes at bin centre 0.5) composes a
        deterministic tome and returns it unmodified.

        LLM-axis path (any of length / formality / audience /
        perspective off-centre) dispatches through ``_apply_llm_axis``
        — builds an OpenAI-backed chat client + extractor from ``env``
        and delegates to ``slider_renderer.render``. Provider on the
        signed receipt becomes ``openai`` and
        ``digital_source_type`` becomes ``trainedAlgorithmicMedia``.
        Missing ``env.openai_api_key`` raises ``ValueError`` with the
        operator-actionable fix (NOT ``NotImplementedError``).
        """
        if not isinstance(input, dict) or "triples" not in input:
            raise ValueError(
                "slider input: expected dict with 'triples' key"
            )
        triples_raw = input["triples"]
        triples: list[tuple[str, str, str]] = [
            (str(t[0]), str(t[1]), str(t[2])) for t in triples_raw
        ]

        quantized = _quantize(parameters)

        if _requires_llm(quantized):
            return await self._apply_llm_axis(triples, quantized, env)

        # ---- Canonical path (all LLM axes at bin centre 0.5) ----
        kept = _apply_density(triples, quantized["density"])
        tome = _deterministic_tome(kept)
        return TransformResult(
            output=tome,
            model="canonical-deterministic-v0",
            provider="canonical-path",
            digital_source_type="algorithmicMedia",
            llm_calls_made=0,
            extra={
                "quantized_parameters": quantized,
                "triples_used": kept,
            },
        )

    async def _apply_llm_axis(
        self,
        triples: list[tuple[str, str, str]],
        quantized: dict[str, float],
        env: TransformEnv,
    ) -> TransformResult:
        """LLM-axis render — route through slider_renderer.render.

        Builds an OpenAI-backed ``LLMChatClient`` from ``env.openai_api_key``
        and a triple-extractor from the same adapter, then delegates the
        actual render pipeline (cache, density, prompt construction,
        drift measurement) to the existing slider_renderer library so we
        don't duplicate the LLM-path code.

        Today's Python adapter is OpenAI-only. Anthropic dispatch
        through the transform registry is Worker-only (the TS path in
        ``worker/src/transforms/`` will route Anthropic when the
        sibling PR for the Worker's slider LLM path lands). Raising a
        clear ValueError when no OpenAI key is configured matches the
        registry contract: ``requires_llm = True`` transforms raise if
        no key is available.
        """
        # Lazy imports keep the canonical-path import-fast and avoid
        # pulling openai into environments that never call the LLM
        # axis. Both modules are part of the sum_engine_internal
        # package, no optional-extras dance required.
        import os

        from sum_engine_internal.ensemble.live_llm_adapter import (
            LiveLLMAdapter,
            make_chat_client,
        )
        from sum_engine_internal.ensemble.slider_renderer import (
            TomeSliders,
            render as slider_render,
        )

        # Model resolution: explicit env.model wins; else the slider
        # falls back to OpenAI's gpt-4o-mini. The resolved model id
        # determines provider routing via LiveLLMAdapter.from_model:
        # OpenAI ids → openai.com, HF-namespaced (`org/model`) → HF
        # Inference Providers ($HF_TOKEN), ollama:/llamacpp:/local:
        # → matching local endpoint. See docs/BYOK_AND_FREE_PROVIDERS.md.
        model = env.model or env.default_openai_model
        m = model.lower()
        needs_openai_key = (
            "/" not in model
            and not m.startswith(("ollama:", "llamacpp:", "local:"))
        )
        if needs_openai_key and not env.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "slider transform: LLM-axis render with model "
                f"{model!r} routes to OpenAI but no API key is set. "
                "Options:\n"
                "  - export OPENAI_API_KEY=sk-... (pay-as-you-go)\n"
                "  - set env.model='meta-llama/Llama-3.3-70B-Instruct' "
                "(or any HF model id) + export HF_TOKEN=hf_... to use "
                "Hugging Face Inference Providers credits\n"
                "  - set env.model='ollama:llama3.1' for a local "
                "Ollama install (free, no key)\n"
                "  - set env.model='local:<your-model>' + "
                "export SUM_LOCAL_LLM_BASE=https://...modal.run/v1 "
                "for a Modal-hosted OpenAI-compatible endpoint\n"
                "  - for the Worker path with Anthropic, use POST "
                "/api/transform with the X-Render-LLM-Key-Anthropic "
                "header.\n"
                "Full matrix at docs/BYOK_AND_FREE_PROVIDERS.md."
            )

        # Pick the api_key arg to pass into from_model. The factory
        # falls back to env vars (HF_TOKEN, OPENAI_API_KEY) when None;
        # we only forward an explicit key when the caller's env names
        # it as such, otherwise we let the factory's env lookup decide.
        api_key_for_factory: str | None = None
        if needs_openai_key:
            api_key_for_factory = env.openai_api_key
        # For HF / local prefixes, LiveLLMAdapter.from_model will pick
        # up HF_TOKEN / SUM_LOCAL_LLM_BASE from the environment.

        adapter = LiveLLMAdapter.from_model(model, api_key=api_key_for_factory)
        # F7 fix (DOGFOOD_FINDINGS_2026-05-17): pick the right chat
        # client for the routing target. OpenAI proper gets the
        # structured-output path; all other OpenAI-compatible providers
        # (HF / NIM / Groq / Cerebras / Ollama / llama.cpp / local) get
        # the plain-chat path because beta.chat.completions.parse is
        # OpenAI-specific and returns degenerate parses elsewhere.
        llm_client = make_chat_client(adapter)

        sliders = TomeSliders(
            density=quantized["density"],
            length=quantized["length"],
            formality=quantized["formality"],
            audience=quantized["audience"],
            perspective=quantized["perspective"],
        )

        # slider_renderer.render applies density internally — pass the
        # full triple list, not the post-density subset.
        render_result = await slider_render(
            triples=tuple(triples),
            sliders=sliders,
            llm=llm_client,
            extractor=adapter.extract_triplets,
        )

        # Drift map: per-axis drift value as reported by
        # slider_renderer.measure_drift. Surface as `extra` so the
        # response body carries the operationally-meaningful
        # signal without being baked into the receipt payload.
        # Per docs/BENCH_HARDENING_FROM_QCVV.md, this is the
        # empirical-benchmark surface; iteration stability + DKW
        # worst-case bounds extend it in T1-T3.
        drift_by_axis = {
            axis_drift.axis.value: {
                "value": axis_drift.value,
                "threshold": axis_drift.threshold,
                "classification": axis_drift.classification,
            }
            for axis_drift in render_result.drift
        }

        # Routing-aware extra: surfaces which OpenAI-API-compatible
        # endpoint actually served the call. The receipt's `provider`
        # field stays as one of the schema literals ("openai") because
        # the API SHAPE was OpenAI's; the routing-target (HF / Ollama
        # / local / OpenAI proper) is honest provenance and goes into
        # `extra.llm_endpoint`. Future schema v2 may add per-routing
        # provider literals; until then, this is the proof-boundary-
        # safe way to be honest about what served without lying about
        # what the receipt attests.
        endpoint_info: dict[str, Any] = {
            "model": adapter.model,
            "base_url": adapter.base_url or "https://api.openai.com/v1 (default)",
        }

        return TransformResult(
            output=render_result.tome,
            model=model,
            provider="openai",
            digital_source_type="trainedAlgorithmicMedia",
            llm_calls_made=render_result.llm_calls_made,
            extra={
                "quantized_parameters": quantized,
                "triples_used": [list(t) for t in render_result.triples_used],
                "drift": drift_by_axis,
                "render_id": render_result.render_id,
                "cache_status": render_result.cache_status.value,
                "llm_endpoint": endpoint_info,
            },
        )


# Module-level instance — auto-registered in transforms/__init__.py.
SLIDER_TRANSFORM = SliderTransform()
