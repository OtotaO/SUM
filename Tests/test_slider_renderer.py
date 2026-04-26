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
    _normalize_predicate,
    _normalize_triple,
    cache_key,
    fact_preservation,
    fact_preservation_normalized,
    measure_drift,
    order_preservation,
    render,
    semantic_fact_preservation,
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

    def test_quantize_snaps_llm_axes_and_passes_density_through(self):
        sliders = TomeSliders(
            density=0.42, length=0.42, formality=0.42,
            audience=0.42, perspective=0.42,
        )
        q = quantize(sliders)
        # density is deterministic ⇒ exempt from binning so 1.0 stays 1.0.
        assert q.density == pytest.approx(0.42)
        # The four LLM axes snap to their bin centres.
        assert q.length == pytest.approx(0.5)
        assert q.formality == pytest.approx(0.5)
        assert q.audience == pytest.approx(0.5)
        assert q.perspective == pytest.approx(0.5)

    def test_quantize_preserves_density_endpoints(self):
        """Density 1.0 must stay 1.0 so users can request full coverage."""
        s = TomeSliders(density=1.0, length=0.5, formality=0.5, audience=0.5, perspective=0.5)
        assert quantize(s).density == 1.0
        s0 = TomeSliders(density=0.0, length=0.5, formality=0.5, audience=0.5, perspective=0.5)
        assert quantize(s0).density == 0.0


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
    correctness tests without requiring a real API call.

    v0.3: gained `chat_completion_structured` so tests can exercise
    the constrained-decoding path. Returns the same echo-tome plus an
    empty claimed-triples list (tests that care about claimed_triples
    use a richer fake)."""
    last_system_prompt: str = ""
    last_user_prompt: str = ""
    call_count: int = 0
    structured_calls: int = 0
    claimed_triples_to_return: list = []

    async def chat_completion(self, system_prompt, user_prompt, max_tokens=2048):
        self.call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return user_prompt

    async def chat_completion_structured(self, system_prompt, user_prompt, max_tokens=2048):
        self.call_count += 1
        self.structured_calls += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return user_prompt, list(self.claimed_triples_to_return)


async def _fake_extractor(_text: str) -> list[Triple]:
    """Returns empty list for the stub — STATE 4 swaps in a real
    extractor (sieve) so drift can be measured."""
    return []


def _fake_render_result() -> RenderResult:
    return RenderResult(
        tome="The alice like cat.",
        triples_used=(("alice", "like", "cat"),),
        reextracted_triples=(("alice", "like", "cat"),),
        claimed_triples=(("alice", "like", "cat"),),
        drift=(AxisDrift(axis=DriftAxis.DENSITY, value=0.0,
                         threshold=0.001, classification="ok"),),
        cache_status=CacheStatus.MISS,
        llm_calls_made=1,
        wall_clock_ms=42,
        quantized_sliders=quantize(TomeSliders()),
        render_id="abc123",
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


class TestStructuredRenderPath:
    """v0.3 — render() prefers chat_completion_structured when the
    LLM client provides it; falls back to chat_completion otherwise."""

    @pytest.mark.asyncio
    async def test_structured_path_called_on_llm_axis(self):
        triples: list[Triple] = [("alice", "like", "cat")]
        sliders = TomeSliders(density=1.0, length=0.7, formality=0.5,
                              audience=0.5, perspective=0.5)
        llm = _FakeLLM()
        llm.claimed_triples_to_return = [("alice", "like", "cat")]
        result = await render(triples, sliders, llm, _fake_extractor)
        assert llm.structured_calls == 1
        # claimed_triples surfaced in RenderResult.
        assert result.claimed_triples == (("alice", "like", "cat"),)

    @pytest.mark.asyncio
    async def test_canonical_path_no_llm_claimed_equals_kept(self):
        """Canonical path (all neutral) bypasses LLM but still populates
        claimed_triples = kept_triples so downstream readers can rely
        on the field never being empty for a successful render."""
        triples: list[Triple] = [("alice", "like", "cat"), ("bob", "own", "dog")]
        sliders = TomeSliders()  # all neutral, density=1.0
        llm = _FakeLLM()
        result = await render(triples, sliders, llm, _fake_extractor)
        assert llm.structured_calls == 0  # canonical path
        assert result.claimed_triples == result.triples_used

    @pytest.mark.asyncio
    async def test_legacy_chat_completion_only_client_falls_back(self):
        """A bare-minimum LLM client without chat_completion_structured
        still works — render() falls back to plain chat_completion and
        leaves claimed_triples empty."""
        class LegacyLLM:
            call_count = 0
            async def chat_completion(self, system_prompt, user_prompt, max_tokens=2048):
                self.call_count += 1
                return user_prompt
        triples: list[Triple] = [("alice", "like", "cat")]
        sliders = TomeSliders(density=1.0, length=0.7, formality=0.5,
                              audience=0.5, perspective=0.5)
        llm = LegacyLLM()
        result = await render(triples, sliders, llm, _fake_extractor)
        assert llm.call_count == 1
        assert result.claimed_triples == ()


class TestFactAndOrderPreservation:
    """v0.2 round-trip metrics. fact_preservation is set-based and
    MontageLie-vulnerable; order_preservation defends against the
    permutation attack."""

    def test_fact_preservation_full_when_identical(self):
        triples = [("a", "b", "c"), ("d", "e", "f"), ("g", "h", "i")]
        assert fact_preservation(triples, triples) == 1.0

    def test_fact_preservation_half_when_half_extracted(self):
        src = [("a", "b", "c"), ("d", "e", "f"), ("g", "h", "i"), ("j", "k", "l")]
        reext = [("a", "b", "c"), ("d", "e", "f")]
        assert fact_preservation(src, reext) == 0.5

    def test_fact_preservation_empty_source(self):
        assert fact_preservation([], []) == 1.0
        assert fact_preservation([], [("a", "b", "c")]) == 1.0

    def test_order_preservation_perfect_when_same_order(self):
        triples = [("a", "b", "c"), ("d", "e", "f"), ("g", "h", "i")]
        assert order_preservation(triples, triples) == 1.0

    def test_order_preservation_zero_when_fully_reversed(self):
        src = [("a", "b", "c"), ("d", "e", "f"), ("g", "h", "i")]
        reversed_ = list(reversed(src))
        assert order_preservation(src, reversed_) == 0.0

    def test_order_preservation_montagelie_attack(self):
        """The headline regression test: a render that preserves every
        source triple as a SET but rearranges them into a deceptive
        order. Set-based fact_preservation is fooled (1.0); pairwise
        order_preservation correctly detects the attack."""
        src = [
            ("alice", "born_in", "1990"),
            ("alice", "graduated", "2012"),
            ("alice", "married", "2015"),
            ("alice", "had_child", "2018"),
        ]
        # Adversary preserves every triple but reorders the timeline:
        # child before marriage before graduation before birth.
        montagelie = list(reversed(src))
        assert fact_preservation(src, montagelie) == 1.0  # SET fooled
        assert order_preservation(src, montagelie) == 0.0  # ORDER catches it

    def test_order_preservation_nan_with_under_two_preserved(self):
        src = [("a", "b", "c"), ("d", "e", "f")]
        reext = [("a", "b", "c")]  # only 1 preserved → no pairs
        import math
        assert math.isnan(order_preservation(src, reext))

    def test_order_preservation_partial(self):
        """3 triples preserved, 1 swap: 2 of 3 pairs correct → 0.667."""
        src = [("a", "b", "c"), ("d", "e", "f"), ("g", "h", "i")]
        # Swap d and g positions
        reext = [("a", "b", "c"), ("g", "h", "i"), ("d", "e", "f")]
        # Pairs in source order: (a,d), (a,g), (d,g)
        # In reext: (a,g): a before g ✓ ; (a,d): a before d ✓ ; (d,g): d after g ✗
        assert order_preservation(src, reext) == pytest.approx(2/3, abs=1e-6)


class TestNormalizationLayer:
    """A3 — surface-form normalization. Catches preposition / auxiliary-
    verb / article drift without LLM calls."""

    def test_preposition_suffix_stripped(self):
        assert _normalize_predicate("graduated_in") == "graduated"
        assert _normalize_predicate("born_on") == "born"
        assert _normalize_predicate("traveled_through") == "traveled"

    def test_auxiliary_prefix_stripped(self):
        assert _normalize_predicate("was_born") == "born"
        assert _normalize_predicate("has_written") == "written"
        assert _normalize_predicate("are_located") == "located"

    def test_combined_prefix_and_suffix(self):
        assert _normalize_predicate("was_born_in") == "born"
        assert _normalize_predicate("has_traveled_to") == "traveled"

    def test_short_predicates_not_overstripped(self):
        # 'in' shouldn't strip down to empty
        assert _normalize_predicate("in") == "in"
        # 'is' alone shouldn't strip down to empty
        assert _normalize_predicate("is") == "is"

    def test_articles_stripped_from_entities(self):
        t = ("the_alice", "graduated_in", "the_year_2012")
        n = _normalize_triple(t)
        assert n[0] == "alice"
        assert n[1] == "graduated"
        assert n[2] == "year_2012"

    def test_normalization_preserves_meaning_difference(self):
        """Different predicates must NOT collapse — even after stripping."""
        a = _normalize_triple(("alice", "loves", "bob"))
        b = _normalize_triple(("alice", "hates", "bob"))
        assert a != b

    def test_fact_preservation_normalized_catches_preposition_drift(self):
        """The exact failure mode found in the bench: source has
        'graduated', re-extraction emits 'graduated_in'. Strict scores
        0; normalized scores 1."""
        source = [("alice", "graduated", "2012")]
        reextracted = [("alice", "graduated_in", "2012")]
        assert fact_preservation(source, reextracted) == 0.0
        assert fact_preservation_normalized(source, reextracted) == 1.0

    def test_fact_preservation_normalized_catches_auxiliary_drift(self):
        source = [("alice", "born", "1990")]
        reextracted = [("alice", "was_born_in", "1990")]
        assert fact_preservation(source, reextracted) == 0.0
        assert fact_preservation_normalized(source, reextracted) == 1.0


# ── A1 semantic layer ────────────────────────────────────────────────


async def _identity_embed(text: str) -> list[float]:
    """Deterministic test embedder: same text → same vector. Built so
    identical strings cosine-sim to 1.0 and different strings to ~0.0."""
    import hashlib
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # 32-byte hash → 8-dim float vector. Identical inputs ⇒ identical
    # vectors ⇒ cosine sim = 1.0. Different inputs ⇒ nearly-orthogonal.
    return [b / 255.0 - 0.5 for b in h[:8]]


class TestSemanticPreservation:

    @pytest.mark.asyncio
    async def test_identical_triples_score_one(self):
        triples = [("alice", "loves", "bob"), ("carol", "owns", "house")]
        score = await semantic_fact_preservation(triples, triples, _identity_embed)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_disjoint_triples_score_zero(self):
        src = [("alice", "loves", "bob")]
        reext = [("xyz", "abc", "def")]
        score = await semantic_fact_preservation(src, reext, _identity_embed,
                                                 threshold=0.85)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_empty_source_returns_one(self):
        score = await semantic_fact_preservation([], [], _identity_embed)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_empty_reextracted_with_nonempty_source_returns_zero(self):
        score = await semantic_fact_preservation(
            [("a", "b", "c")], [], _identity_embed,
        )
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_greedy_one_to_one_assignment(self):
        """Two source triples, one re-extracted that matches both
        identically — only ONE should claim it."""
        src = [("alice", "loves", "bob"), ("alice", "loves", "bob")]
        reext = [("alice", "loves", "bob")]
        score = await semantic_fact_preservation(src, reext, _identity_embed)
        # One matched out of two source triples ⇒ 0.5
        assert score == 0.5


class TestMeasureDrift:

    def test_density_drift_zero_when_canonical(self):
        triples = [("a", "b", "c"), ("d", "e", "f")]
        sliders = TomeSliders(density=1.0)
        # Tome content irrelevant to density drift; pass any string.
        drift = measure_drift(triples, triples, "the a b c. the d e f.", sliders)
        density_d = next(d for d in drift if d.axis == DriftAxis.DENSITY)
        assert density_d.value == pytest.approx(0.0, abs=1e-6)

    def test_density_drift_at_half(self):
        triples = [("a", "b", "c"), ("d", "e", "f")]
        # density=0.5 means we expect ~1 of 2 triples retained.
        # If 1 was retained, drift = 0.
        retained = [("a", "b", "c")]
        sliders = TomeSliders(density=0.5)
        drift = measure_drift(triples, retained, "the a b c.", sliders)
        density_d = next(d for d in drift if d.axis == DriftAxis.DENSITY)
        assert density_d.value <= 0.5  # within tolerance per SLIDER_CONTRACT.md
