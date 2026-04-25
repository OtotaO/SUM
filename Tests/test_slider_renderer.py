"""Phase E.1 unit tests for the slider renderer.

SCAFFOLD STATE: every test below is marked xfail with a reason.
EXECUTE STATE will remove the xfails as the implementation lands.
The test bodies are NOT stubs — they are the actual contracts the
implementation must satisfy. Treat them as the spec.

Author: ototao
License: Apache License 2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

import pytest

from sum_engine_internal.ensemble.slider_renderer import (
    AxisDrift,
    CacheStatus,
    DriftAxis,
    InMemorySliderCache,
    RenderResult,
    Triple,
    cache_key,
    measure_drift,
    render,
)
from sum_engine_internal.ensemble.tome_sliders import (
    SLIDER_BINS_PER_AXIS,
    TomeSliders,
    quantize,
    snap_to_bin,
)


# ─── Snap + quantize (already implemented in tome_sliders) ───────────


class TestSnapAndQuantize:

    @pytest.mark.parametrize("value,expected", [
        (0.0, 0.1),
        (0.05, 0.1),
        (0.19, 0.1),
        (0.20, 0.3),
        (0.50, 0.5),
        (0.99, 0.9),
        (1.0, 0.9),
    ])
    def test_snap_to_bin_5(self, value, expected):
        assert snap_to_bin(value, bins=5) == pytest.approx(expected)

    def test_snap_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            snap_to_bin(-0.01)
        with pytest.raises(ValueError):
            snap_to_bin(1.01)

    def test_quantize_returns_frozen_with_all_axes_snapped(self):
        sliders = TomeSliders(
            density=0.42, length=0.42, formality=0.42,
            audience=0.42, perspective=0.42,
        )
        q = quantize(sliders)
        assert q.density == pytest.approx(0.5)
        assert q.length == pytest.approx(0.5)
        assert q.formality == pytest.approx(0.5)
        assert q.audience == pytest.approx(0.5)
        assert q.perspective == pytest.approx(0.5)


# ─── Cache key derivation ────────────────────────────────────────────


class TestCacheKey:

    def test_cache_key_pure_function(self):
        triples = [("alice", "like", "cat"), ("bob", "own", "dog")]
        sliders = TomeSliders(density=0.5, length=0.5, formality=0.5,
                              audience=0.5, perspective=0.5)
        k1 = cache_key(triples, sliders)
        k2 = cache_key(triples, sliders)
        assert k1 == k2

    def test_cache_key_order_independent(self):
        triples_a = [("alice", "like", "cat"), ("bob", "own", "dog")]
        triples_b = [("bob", "own", "dog"), ("alice", "like", "cat")]
        sliders = TomeSliders()
        assert cache_key(triples_a, sliders) == cache_key(triples_b, sliders)

    def test_cache_key_changes_with_sliders(self):
        triples = [("alice", "like", "cat")]
        s1 = TomeSliders(density=1.0, length=0.5, formality=0.5, audience=0.5, perspective=0.5)
        s2 = TomeSliders(density=1.0, length=0.5, formality=0.5, audience=0.5, perspective=0.9)
        # Quantize first; near-identical sliders MUST collide post-quantize
        # but DIFFERENT bins MUST diverge.
        assert cache_key(triples, quantize(s1)) != cache_key(triples, quantize(s2))

    def test_cache_key_changes_with_triples(self):
        sliders = TomeSliders()
        t1 = [("alice", "like", "cat")]
        t2 = [("alice", "like", "dog")]
        assert cache_key(t1, sliders) != cache_key(t2, sliders)

    def test_cache_key_length(self):
        triples = [("a", "b", "c")]
        sliders = TomeSliders()
        assert len(cache_key(triples, sliders)) == 32


# ─── In-memory cache ─────────────────────────────────────────────────


class TestInMemoryCache:

    @pytest.mark.asyncio
    async def test_miss_returns_none(self):
        c = InMemorySliderCache()
        assert await c.get("nonexistent") is None
        s = await c.stats()
        assert s["misses"] == 1
        assert s["hits"] == 0

    @pytest.mark.asyncio
    async def test_put_then_get(self):
        c = InMemorySliderCache()
        rr = _fake_render_result()
        await c.put("k1", rr, ttl_seconds=60)
        got = await c.get("k1")
        assert got == rr
        s = await c.stats()
        assert s["hits"] == 1
        assert s["misses"] == 0
        assert s["size"] == 1


# ─── Renderer pipeline (xfail until EXECUTE state) ───────────────────


class _FakeLLM:
    """Deterministic LLM stub. Returns a tome built from the input
    facts joined with the system prompt — guarantees identity for
    correctness tests without requiring a real API call."""
    last_system_prompt: str = ""
    last_user_prompt: str = ""
    call_count: int = 0

    async def chat_completion(self, system_prompt, user_prompt, max_tokens=2048):
        self.call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        # Simple deterministic stub: echo the facts as sentences.
        # Real STATE-4 tests inject deterministic templates per axis.
        return user_prompt


async def _fake_extractor(_text: str) -> list[Triple]:
    """Returns empty list for the stub — STATE 4 swaps in a real
    extractor (sieve) so drift can be measured."""
    return []


def _fake_render_result() -> RenderResult:
    return RenderResult(
        tome="The alice like cat.",
        triples_used=(("alice", "like", "cat"),),
        drift=(AxisDrift(axis=DriftAxis.DENSITY, value=0.0,
                         threshold=0.001, classification="ok"),),
        cache_status=CacheStatus.MISS,
        llm_calls_made=1,
        wall_clock_ms=42,
        quantized_sliders=quantize(TomeSliders()),
        render_id="abc123",
    )


@pytest.mark.xfail(
    reason="STATE 4 — render() raises NotImplementedError until the "
           "pipeline is filled. Test bodies are spec, not stub.",
    strict=True,
)
class TestRenderPipeline:

    @pytest.mark.asyncio
    async def test_density_only_is_deterministic(self):
        """At density=1.0 with all other axes neutral, render() must
        return the canonical deterministic tome (no LLM call needed,
        canonical-tome-generator path)."""
        triples: list[Triple] = [("alice", "like", "cat"), ("bob", "own", "dog")]
        sliders = TomeSliders(density=1.0, length=0.5, formality=0.5,
                              audience=0.5, perspective=0.5)
        llm = _FakeLLM()
        result = await render(triples, sliders, llm, _fake_extractor)
        assert llm.call_count == 0  # neutral axes ⇒ canonical path, no LLM
        assert result.cache_status == CacheStatus.BYPASS  # no cache passed
        # Density drift must be ≤ 0.001 per SLIDER_CONTRACT.md.
        density_drift = next(d for d in result.drift if d.axis == DriftAxis.DENSITY)
        assert density_drift.value <= 0.001

    @pytest.mark.asyncio
    async def test_cache_hit_skips_llm(self):
        triples: list[Triple] = [("alice", "like", "cat")]
        sliders = TomeSliders(density=1.0, length=0.7, formality=0.5,
                              audience=0.5, perspective=0.5)
        llm = _FakeLLM()
        cache = InMemorySliderCache()
        # First call — miss + LLM call.
        r1 = await render(triples, sliders, llm, _fake_extractor, cache=cache)
        assert r1.cache_status == CacheStatus.MISS
        assert llm.call_count == 1
        # Second call with same args — hit, no additional LLM call.
        r2 = await render(triples, sliders, llm, _fake_extractor, cache=cache)
        assert r2.cache_status == CacheStatus.HIT
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_quantize_collapses_near_identical_sliders(self):
        """Two calls with sliders 0.51 and 0.52 (same bin) must hit the
        same cache cell."""
        triples: list[Triple] = [("a", "b", "c")]
        cache = InMemorySliderCache()
        llm = _FakeLLM()
        await render(triples, TomeSliders(length=0.51), llm, _fake_extractor, cache=cache)
        await render(triples, TomeSliders(length=0.52), llm, _fake_extractor, cache=cache)
        # Both 0.51 and 0.52 snap to bin 0.5.
        assert llm.call_count == 1


# ─── measure_drift unit (xfail until EXECUTE) ─────────────────────────


@pytest.mark.xfail(reason="STATE 4 — measure_drift raises NotImplementedError.", strict=True)
class TestMeasureDrift:

    def test_density_drift_zero_when_canonical(self):
        triples = [("a", "b", "c"), ("d", "e", "f")]
        sliders = TomeSliders(density=1.0)
        drift = measure_drift(triples, triples, sliders)
        density_d = next(d for d in drift if d.axis == DriftAxis.DENSITY)
        assert density_d.value == pytest.approx(0.0, abs=1e-6)

    def test_density_drift_at_half(self):
        triples = [("a", "b", "c"), ("d", "e", "f")]
        # density=0.5 means we expect ~1 of 2 triples retained.
        # If 1 was retained, drift = 0.
        retained = [("a", "b", "c")]
        sliders = TomeSliders(density=0.5)
        drift = measure_drift(triples, retained, sliders)
        density_d = next(d for d in drift if d.axis == DriftAxis.DENSITY)
        assert density_d.value <= 0.5  # within tolerance per SLIDER_CONTRACT.md
