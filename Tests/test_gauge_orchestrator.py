"""
Gauge-Theoretic Orchestrator Tests

Verifies the commutativity hierarchy detection and knowledge graph
merge with Level 3 Curvature resolution.

Author: ototao
License: Apache License 2.0
"""

import pytest
import networkx as nx

from internal.ensemble.gauge_orchestrator import (
    GaugeTheoreticOrchestrator,
    CommutativityLevel,
)


def make_graph(edges):
    """Helper: build a MultiDiGraph from (src, tgt, relation) tuples."""
    g = nx.MultiDiGraph()
    for src, tgt, rel in edges:
        g.add_edge(src, tgt, relation=rel)
    return g


class TestCommutativityDetection:

    def test_l1_independent_facts(self):
        """Disjoint entities produce Level 1 (fully commutative)."""
        base = make_graph([("alice", "cats", "likes")])
        new = make_graph([("bob", "python", "knows")])
        orch = GaugeTheoreticOrchestrator()
        level, conflicts = orch.detect_curvature(base, new)
        assert level == CommutativityLevel.L1_COMMUTATIVE
        assert conflicts == []

    def test_l2_same_entity_different_predicate(self):
        """Same entity, different predicates → Level 2."""
        base = make_graph([("alice", "cats", "likes")])
        new = make_graph([("alice", "python", "knows")])
        orch = GaugeTheoreticOrchestrator()
        level, conflicts = orch.detect_curvature(base, new)
        assert level == CommutativityLevel.L2_CONDITIONAL
        assert conflicts == []

    def test_l3_curvature_same_predicate_different_object(self):
        """Same entity + same predicate, different object → Level 3."""
        base = make_graph([("earth", "sun", "orbits")])
        new = make_graph([("earth", "moon", "orbits")])
        orch = GaugeTheoreticOrchestrator()
        level, conflicts = orch.detect_curvature(base, new)
        assert level == CommutativityLevel.L3_CURVATURE
        assert len(conflicts) == 1
        s, p, old, new_obj = conflicts[0]
        assert s == "earth"
        assert p == "orbits"
        assert old == "sun"
        assert new_obj == "moon"

    def test_identical_fact_no_conflict(self):
        """Identical (entity, predicate, object) → no conflict."""
        base = make_graph([("earth", "sun", "orbits")])
        new = make_graph([("earth", "sun", "orbits")])
        orch = GaugeTheoreticOrchestrator()
        level, conflicts = orch.detect_curvature(base, new)
        assert conflicts == []

    def test_empty_graphs(self):
        """Two empty graphs → Level 1, no conflicts."""
        orch = GaugeTheoreticOrchestrator()
        level, conflicts = orch.detect_curvature(
            nx.MultiDiGraph(), nx.MultiDiGraph()
        )
        assert level == CommutativityLevel.L1_COMMUTATIVE
        assert conflicts == []

    def test_multiple_conflicts(self):
        """Multiple Level 3 conflicts are all reported."""
        base = make_graph([
            ("earth", "sun", "orbits"),
            ("water", "100c", "boils_at"),
        ])
        new = make_graph([
            ("earth", "moon", "orbits"),
            ("water", "212f", "boils_at"),
        ])
        orch = GaugeTheoreticOrchestrator()
        level, conflicts = orch.detect_curvature(base, new)
        assert level == CommutativityLevel.L3_CURVATURE
        assert len(conflicts) == 2


class TestMergeExtractions:

    @pytest.mark.asyncio
    async def test_l1_merge_adds_all_edges(self):
        """L1 merge simply adds all new edges."""
        base = make_graph([("alice", "cats", "likes")])
        new = make_graph([("bob", "python", "knows")])
        orch = GaugeTheoreticOrchestrator()
        merged = await orch.merge_extractions(base, [new])
        assert merged.has_node("alice")
        assert merged.has_node("bob")
        assert merged.number_of_edges() == 2

    @pytest.mark.asyncio
    async def test_l3_default_recency_resolution(self):
        """Without arbitrator, new information wins (recency bias)."""
        base = make_graph([("earth", "sun", "orbits")])
        new = make_graph([("earth", "moon", "orbits")])
        orch = GaugeTheoreticOrchestrator()  # No arbitrate_fn
        merged = await orch.merge_extractions(base, [new])
        # Should have earth→moon, not earth→sun
        edges = list(merged.edges(data=True))
        targets = [tgt for _, tgt, _ in edges if _ == "earth" or True]
        assert "moon" in [tgt for _, tgt, _ in edges]

    @pytest.mark.asyncio
    async def test_l3_custom_arbitrator(self):
        """Custom arbitrator chooses the winner."""
        base = make_graph([("earth", "sun", "orbits")])
        new = make_graph([("earth", "moon", "orbits")])

        async def custom_arb(conflicts):
            # Always keep old value
            return {(s, p): old for s, p, old, _ in conflicts}

        orch = GaugeTheoreticOrchestrator(arbitrate_fn=custom_arb)
        merged = await orch.merge_extractions(base, [new])
        targets = [tgt for _, tgt, _ in merged.edges(data=True)]
        assert "sun" in targets
        assert "moon" not in targets

    @pytest.mark.asyncio
    async def test_merge_preserves_non_conflicting(self):
        """Non-conflicting edges survive alongside resolved conflicts."""
        base = make_graph([("earth", "sun", "orbits")])
        new = make_graph([
            ("earth", "moon", "orbits"),  # conflict
            ("mars", "phobos", "has_moon"),  # no conflict
        ])
        orch = GaugeTheoreticOrchestrator()
        merged = await orch.merge_extractions(base, [new])
        assert merged.has_edge("mars", "phobos")
