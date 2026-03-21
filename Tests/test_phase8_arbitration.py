"""
Phase 8 Tests — Epistemic Superposition & Wave Function Collapse

Validates:
    - Level 3 Curvature detection for contradictory facts
    - Wave function collapse via mock LLM judge
    - Non-conflicting facts merge cleanly (Level 1)
    - Arbiter fallback when judge hallucinates
    - EventBroadcaster delivers messages to subscribers
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import pytest
import networkx as nx

from internal.ensemble.gauge_orchestrator import (
    GaugeTheoreticOrchestrator,
    CommutativityLevel,
)
from internal.ensemble.epistemic_arbiter import (
    EpistemicArbiter,
    EventBroadcaster,
)


# ─── Mock helpers ─────────────────────────────────────────────────────

async def mock_judge_london(prompt: str) -> str:
    """Judge that always picks London."""
    return "london"


async def mock_judge_hallucinate(prompt: str) -> str:
    """Judge that returns an answer not in the options."""
    return "mars"


# ─── 1. Curvature Detection ──────────────────────────────────────────

class TestCurvatureDetection:

    def test_level3_contradiction_detected(self):
        """Same subject + predicate with different objects → Level 3."""
        orch = GaugeTheoreticOrchestrator()

        base = nx.MultiDiGraph()
        base.add_edge("alice", "new_york", relation="lives_in")

        new = nx.MultiDiGraph()
        new.add_edge("alice", "london", relation="lives_in")

        level, conflicts = orch.detect_curvature(base, new)

        assert level == CommutativityLevel.L3_CURVATURE
        assert len(conflicts) == 1
        assert conflicts[0] == ("alice", "lives_in", "new_york", "london")

    def test_level1_no_conflict(self):
        """Completely independent facts → Level 1."""
        orch = GaugeTheoreticOrchestrator()

        base = nx.MultiDiGraph()
        base.add_edge("alice", "30", relation="age")

        new = nx.MultiDiGraph()
        new.add_edge("bob", "engineer", relation="role")

        level, conflicts = orch.detect_curvature(base, new)

        assert level == CommutativityLevel.L1_COMMUTATIVE
        assert len(conflicts) == 0

    def test_identical_fact_no_conflict(self):
        """Same fact in both graphs → no conflict."""
        orch = GaugeTheoreticOrchestrator()

        base = nx.MultiDiGraph()
        base.add_edge("alice", "30", relation="age")

        new = nx.MultiDiGraph()
        new.add_edge("alice", "30", relation="age")

        level, conflicts = orch.detect_curvature(base, new)

        assert level == CommutativityLevel.L1_COMMUTATIVE
        assert len(conflicts) == 0


# ─── 2. Wave Function Collapse ───────────────────────────────────────

class TestWaveFunctionCollapse:

    @pytest.mark.asyncio
    async def test_arbiter_resolves_contradiction(self):
        """Arbiter picks London, old NY edge removed."""
        arbiter = EpistemicArbiter(mock_judge_london)
        orch = GaugeTheoreticOrchestrator(arbiter.collapse_wave_function)

        base = nx.MultiDiGraph()
        base.add_edge("alice", "new_york", relation="lives_in")

        new = nx.MultiDiGraph()
        new.add_edge("alice", "london", relation="lives_in")

        merged = await orch.merge_extractions(base, [new])

        edges = list(merged.edges(data=True))
        assert len(edges) == 1
        assert edges[0][0] == "alice"
        assert edges[0][1] == "london"
        assert edges[0][2]["relation"] == "lives_in"
        assert edges[0][2].get("verified_curvature") is True

    @pytest.mark.asyncio
    async def test_arbiter_fallback_on_hallucination(self):
        """If judge hallucinates, fallback to obj_a."""
        arbiter = EpistemicArbiter(mock_judge_hallucinate)

        resolutions = await arbiter.collapse_wave_function(
            [("alice", "lives_in", "new_york", "london")]
        )

        # "mars" is not a valid option → falls back to "new_york"
        assert resolutions[("alice", "lives_in")] == "new_york"

    @pytest.mark.asyncio
    async def test_merge_preserves_non_conflicting(self):
        """Non-conflicting facts from new graph are preserved in merge."""
        arbiter = EpistemicArbiter(mock_judge_london)
        orch = GaugeTheoreticOrchestrator(arbiter.collapse_wave_function)

        base = nx.MultiDiGraph()
        base.add_edge("alice", "new_york", relation="lives_in")

        new = nx.MultiDiGraph()
        new.add_edge("alice", "london", relation="lives_in")  # conflict
        new.add_edge("alice", "30", relation="age")            # no conflict

        merged = await orch.merge_extractions(base, [new])

        edges = list(merged.edges(data=True))
        relations = {d["relation"]: (u, v) for u, v, d in edges}

        assert "lives_in" in relations
        assert relations["lives_in"] == ("alice", "london")
        assert "age" in relations
        assert relations["age"] == ("alice", "30")


# ─── 3. Event Broadcaster ────────────────────────────────────────────

class TestEventBroadcaster:

    @pytest.mark.asyncio
    async def test_broadcast_delivers_to_subscribers(self):
        """Messages are delivered to all subscribed queues."""
        broadcaster = EventBroadcaster()

        q1 = broadcaster.subscribe()
        q2 = broadcaster.subscribe()

        await broadcaster.broadcast("test message")

        msg1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        msg2 = await asyncio.wait_for(q2.get(), timeout=1.0)

        assert msg1 == "test message"
        assert msg2 == "test message"

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_queue(self):
        """Unsubscribed queues stop receiving messages."""
        broadcaster = EventBroadcaster()

        q1 = broadcaster.subscribe()
        broadcaster.unsubscribe(q1)

        await broadcaster.broadcast("after unsub")

        assert q1.empty()
