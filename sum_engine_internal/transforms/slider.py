"""Slider — the first registered transform.

The slider product migrates onto the transform registry. Existing
``sum_engine_internal.ensemble.slider_renderer.render`` stays as the
implementation library; this module is the protocol adapter that
exposes that library as ``Transform(name="slider")``.

What this v0 of the slider transform does:

  - Accepts triples + slider position parameters.
  - Applies the deterministic density prefix (canonical-path).
  - Returns a tome + receipt for the canonical-path case.
  - Does NOT yet wire the LLM-axis dispatch (length / formality /
    audience / perspective off-centre). That's deferred to T1b
    (Worker-side migration + Python LLM-axis adapter). For now,
    LLM-axis renders raise ``NotImplementedError`` from the transform
    adapter so the registry contract is honoured but the legacy
    ``slider_renderer.render`` remains the supported path for
    LLM renders until T1b lands.

Why this scoping: T1's goal is to get the registry / receipt /
verifier surfaces shipping cleanly. The slider's LLM-path code is
non-trivial (cache, drift measurement, prompt construction); pulling
it into the transform abstraction without breaking the existing
``/api/render`` callers needs its own focused PR. Pre-T1b, the
recommended path for LLM-axis renders is the legacy ``slider_renderer
.render`` API or the live Worker; for canonical-path renders, the
transform registry is the cleaner surface.
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

        v0 (this PR — T1a) supports the canonical-path render. If
        the quantized slider position requires LLM extrapolation,
        raises NotImplementedError pointing at the legacy
        slider_renderer.render API. T1b will close that gap by
        wiring the LLM-axis dispatch into the transform adapter.
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
        kept = _apply_density(triples, quantized["density"])

        if _requires_llm(quantized):
            raise NotImplementedError(
                "slider transform v0 (T1a) supports canonical-path renders "
                "only. For LLM-axis renders (any of length/formality/"
                "audience/perspective ≠ 0.5), use "
                "sum_engine_internal.ensemble.slider_renderer.render OR "
                "the live Worker's /api/render endpoint. LLM-axis "
                "dispatch lands in T1b."
            )

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


# Module-level instance — auto-registered in transforms/__init__.py.
SLIDER_TRANSFORM = SliderTransform()
