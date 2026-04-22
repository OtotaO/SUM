"""
Deterministic Arbiter Tests

Verifies that the DeterministicArbiter resolves contradictions
identically across all invocations — no LLM, pure SHA-256 ordering.

Author: ototao
License: Apache License 2.0
"""

import asyncio
import pytest

from sum_engine_internal.ensemble.epistemic_arbiter import DeterministicArbiter


@pytest.fixture
def arbiter():
    return DeterministicArbiter()


class TestDeterministicResolution:

    @pytest.mark.asyncio
    async def test_single_conflict_resolves(self, arbiter):
        """A single conflict produces exactly one resolution."""
        conflicts = [("earth", "orbits", "sun", "moon")]
        result = await arbiter.collapse_wave_function(conflicts)
        assert ("earth", "orbits") in result
        assert result[("earth", "orbits")] in ("sun", "moon")

    @pytest.mark.asyncio
    async def test_resolution_is_deterministic(self, arbiter):
        """Same conflict resolves identically every time."""
        conflicts = [("earth", "orbits", "sun", "moon")]
        r1 = await arbiter.collapse_wave_function(conflicts)
        r2 = await arbiter.collapse_wave_function(conflicts)
        assert r1 == r2

    @pytest.mark.asyncio
    async def test_order_independent(self, arbiter):
        """Swapping obj_a and obj_b produces the same winner."""
        c1 = [("earth", "orbits", "sun", "moon")]
        c2 = [("earth", "orbits", "moon", "sun")]
        r1 = await arbiter.collapse_wave_function(c1)
        r2 = await arbiter.collapse_wave_function(c2)
        assert r1[("earth", "orbits")] == r2[("earth", "orbits")]

    @pytest.mark.asyncio
    async def test_different_conflicts_different_winners(self, arbiter):
        """Different subject/predicates can produce different winners."""
        conflicts = [
            ("earth", "orbits", "sun", "moon"),
            ("water", "boils_at", "100c", "212f"),
        ]
        result = await arbiter.collapse_wave_function(conflicts)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_identical_objects_picks_first(self, arbiter):
        """If both objects are identical, result is that object."""
        conflicts = [("earth", "orbits", "sun", "sun")]
        result = await arbiter.collapse_wave_function(conflicts)
        assert result[("earth", "orbits")] == "sun"

    @pytest.mark.asyncio
    async def test_canonical_hash_differs(self, arbiter):
        """Two different objects produce different canonical hashes."""
        h1 = arbiter._canonical_hash("earth", "orbits", "sun")
        h2 = arbiter._canonical_hash("earth", "orbits", "moon")
        assert h1 != h2
        assert len(h1) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_cross_instance_agreement(self):
        """Two independent arbiter instances agree on resolution."""
        a1 = DeterministicArbiter()
        a2 = DeterministicArbiter()
        conflicts = [("temperature", "unit", "celsius", "fahrenheit")]
        r1 = await a1.collapse_wave_function(conflicts)
        r2 = await a2.collapse_wave_function(conflicts)
        assert r1 == r2
