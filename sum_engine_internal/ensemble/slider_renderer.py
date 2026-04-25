"""Slider Renderer — Tags → Tomes with axis-conditioned LLM rendering.

The Phase E genesis-vision module. Takes a set of canonicalised triples
plus a TomeSliders position; returns a rendered tome with per-axis
round-trip drift measured against the source triples. Caches by
quantized slider position so adjacent drag positions hit the same cell.

Architecture (filled by STATE 4):

    render(triples, sliders, llm) ─┬─→ check cache (quantize sliders)
                                    │     hit  → return cached RenderResult
                                    │     miss → continue
                                    │
                                    ├─→ apply density (deterministic prefix)
                                    │
                                    ├─→ build_system_prompt(quantized_sliders)
                                    │
                                    ├─→ llm.chat.completions.create(...)
                                    │     → tome (string)
                                    │
                                    ├─→ re-extract triples from tome
                                    │   (sieve or LLM extractor)
                                    │
                                    ├─→ compute drift = |source △ reextracted|
                                    │   per axis (drift_density, drift_length,
                                    │   drift_formality, drift_audience,
                                    │   drift_perspective)
                                    │
                                    ├─→ store in cache under quantize(sliders)
                                    │
                                    └─→ return RenderResult

Why drift is computed PER AXIS rather than as a single scalar:
each axis has different acceptable tolerance. Density at 0.3 should
preserve 30% of source triples (drift "expected"). Audience at 0.9
should preserve 100% (any drift is a hallucination). The UI uses
per-axis drift to color individual sliders red when they violate
their axis-specific contract.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Optional, Protocol, Sequence

from sum_engine_internal.ensemble.tome_sliders import (
    SLIDER_BINS_PER_AXIS,
    TomeSliders,
    quantize,
)


# ─── Type contracts ───────────────────────────────────────────────────


# A canonicalised triple in (subject, predicate, object) form. Strings
# are already lowercased + underscore-joined per the sieve's
# canonicalisation rules. The renderer never re-canonicalises; the
# caller owns input shape.
Triple = tuple[str, str, str]


class CacheStatus(str, Enum):
    """Where the rendered output came from."""

    HIT = "hit"
    MISS = "miss"
    BYPASS = "bypass"   # caller passed cache=None; never even checked


class DriftAxis(str, Enum):
    """Names match TomeSliders fields."""

    DENSITY = "density"
    LENGTH = "length"
    FORMALITY = "formality"
    AUDIENCE = "audience"
    PERSPECTIVE = "perspective"


@dataclass(frozen=True)
class AxisDrift:
    """Per-axis drift measurement for one render.

    `value` is the drift metric IN THE UNITS THE AXIS DEFINES — they are
    not all the same scale. Density drift is (1 - retained_fraction);
    LLM-axis drifts are |source_set △ reextracted_set| / |union|.
    The threshold is what the SLIDER_CONTRACT.md says counts as
    'unacceptable' for that axis at the slider's current position.
    """

    axis: DriftAxis
    value: float                      # measured drift, axis-defined units
    threshold: float                  # contract limit at this slider position
    classification: str               # "ok" | "warn" | "fail"
    explanation: Optional[str] = None # human-readable why-it-failed, if classification != "ok"


@dataclass(frozen=True)
class RenderResult:
    """Output of slider_renderer.render().

    `tome`            — the generated narrative text. May be the canonical
                        deterministic rendering (when all LLM axes are at
                        neutral) or an LLM-conditioned rendering.
    `triples_used`    — the post-density triples actually fed to the LLM.
    `drift`           — per-axis measurement against source triples.
    `cache_status`    — provenance of the result.
    `llm_calls_made`  — 0 on cache hit; 1 on miss (one LLM call per render
                        — the axes condition the system prompt rather than
                        firing N parallel calls).
    `wall_clock_ms`   — total time including cache lookup, LLM call, and
                        drift measurement.
    `quantized_sliders`— what the cache key actually used (post-snap).
    `render_id`       — content-addressed identifier of this render
                        (sha256 of triples_used + quantized_sliders + tome).
    """

    tome: str
    triples_used: tuple[Triple, ...]
    drift: tuple[AxisDrift, ...]
    cache_status: CacheStatus
    llm_calls_made: int
    wall_clock_ms: int
    quantized_sliders: TomeSliders
    render_id: str


# ─── Cache protocol ───────────────────────────────────────────────────


class SliderCache(Protocol):
    """Protocol any cache backend must satisfy. STATE 4 ships an
    in-memory implementation (single-process); the Worker provides
    a KV-backed implementation in worker/src/cache/bin_cache.ts that
    matches the same key shape.
    """

    async def get(self, key: str) -> Optional[RenderResult]: ...
    async def put(self, key: str, value: RenderResult, ttl_seconds: int) -> None: ...
    async def stats(self) -> dict[str, int]: ...


# ─── LLM client protocol ──────────────────────────────────────────────


class LLMChatClient(Protocol):
    """Subset of the OpenAI-compatible API the renderer requires.

    Renderer never imports `LiveLLMAdapter` directly — it depends on
    this protocol so tests can inject a deterministic fake without
    pulling in the real LLM call path.
    """

    async def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
    ) -> str: ...


# ─── Extractor protocol ───────────────────────────────────────────────


# Re-extraction is needed to compute drift. The renderer doesn't know
# whether to use the sieve (deterministic) or the LLM (high recall);
# caller decides by injecting an extractor function.
TripleExtractor = Callable[[str], Awaitable[list[Triple]]]


# ─── Public API ───────────────────────────────────────────────────────


def cache_key(triples: Sequence[Triple], quantized: TomeSliders) -> str:
    """Derive the cache key from canonicalised triples + quantized
    sliders. Pure function — same input always returns same key.

    Format: sha256_hex(json.dumps({sorted_triples, quantized_sliders}))
    truncated to 32 chars (16 bytes) for compact key strings.

    The sort is critical for cache-key stability: callers may pass
    the same triple set in any order; we want them to hit the same
    cache cell. The post-density triples are what gets keyed (so a
    slider position with density=0.3 caches a different cell than
    density=1.0 even with identical input).
    """
    sorted_triples = sorted(tuple(t) for t in triples)
    payload = {
        "triples": sorted_triples,
        "sliders": {
            "density": quantized.density,
            "length": quantized.length,
            "formality": quantized.formality,
            "audience": quantized.audience,
            "perspective": quantized.perspective,
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


async def render(
    triples: Sequence[Triple],
    sliders: TomeSliders,
    llm: LLMChatClient,
    extractor: TripleExtractor,
    cache: Optional[SliderCache] = None,
    cache_ttl_seconds: int = 24 * 60 * 60,
) -> RenderResult:
    """Render a tome from triples at the requested slider position.

    Pipeline (STATE 4):
        1. Quantize sliders → cache key.
        2. cache.get(key) — return on hit.
        3. apply_density(triples, sliders.density) → kept_triples.
        4. build_system_prompt(quantized_sliders) → system_prompt.
        5. format triples into user_prompt.
        6. llm.chat_completion(system_prompt, user_prompt) → tome.
        7. extractor(tome) → reextracted_triples.
        8. measure_drift(triples, reextracted_triples, sliders) → AxisDrift list.
        9. Build RenderResult, cache.put, return.

    STATE 2 stub: raises NotImplementedError. Type signature is the
    deliverable; logic ships in STATE 4.
    """
    raise NotImplementedError(
        "slider_renderer.render — STATE 4 deliverable. "
        "Type signature stable; pipeline implementation pending."
    )


def measure_drift(
    source_triples: Sequence[Triple],
    reextracted_triples: Sequence[Triple],
    sliders: TomeSliders,
) -> tuple[AxisDrift, ...]:
    """Per-axis drift between source and re-extracted triples.

    Pure function. Deterministic per input. STATE 4 fills the actual
    per-axis arithmetic per docs/SLIDER_CONTRACT.md.
    """
    raise NotImplementedError(
        "slider_renderer.measure_drift — STATE 4 deliverable. "
        "Per-axis drift formulas are in docs/SLIDER_CONTRACT.md."
    )


# ─── In-memory cache (default for tests + single-process use) ─────────


@dataclass
class InMemorySliderCache:
    """Simple dict-backed cache. Not for production multi-process use.
    The Worker uses worker/src/cache/bin_cache.ts (KV-backed) for that.
    """

    _data: dict[str, RenderResult] = field(default_factory=dict)
    _hits: int = 0
    _misses: int = 0

    async def get(self, key: str) -> Optional[RenderResult]:
        result = self._data.get(key)
        if result is None:
            self._misses += 1
        else:
            self._hits += 1
        return result

    async def put(self, key: str, value: RenderResult, ttl_seconds: int) -> None:
        # In-memory: no TTL enforcement (process is the lifetime).
        # The Worker KV cache enforces TTL on its side.
        self._data[key] = value

    async def stats(self) -> dict[str, int]:
        return {
            "size": len(self._data),
            "hits": self._hits,
            "misses": self._misses,
        }
