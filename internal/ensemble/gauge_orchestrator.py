"""
Gauge-Theoretic Orchestrator — Commutativity Hierarchy Engine

Implements Yaroslavtsev's three-level commutativity detection:
  - Level 1 (Commutative): Independent facts, safe to merge in any order.
  - Level 2 (Conditionally commutative): Same entity, different predicates.
  - Level 3 (Curvature): Same entity, same predicate, different objects.
    These MUST be serialized — triggers the EpistemicArbiter.

Author: ototao
License: Apache License 2.0
"""

import logging
from enum import IntEnum
from typing import Callable, List, Tuple, Optional

import networkx as nx

logger = logging.getLogger(__name__)


class CommutativityLevel(IntEnum):
    """Yaroslavtsev's Commutativity Hierarchy."""

    L1_COMMUTATIVE = 1  # Independent facts — parallel merge safe
    L2_CONDITIONAL = 2  # Same entity, different predicates
    L3_CURVATURE = 3    # Same entity, same predicate, different objects


class GaugeTheoreticOrchestrator:
    """
    Manages the merge of knowledge graph extractions according to
    the Commutativity Hierarchy.

    When Level 3 Curvature is detected, the ``arbitrate`` callback
    (an ``EpistemicArbiter.collapse_wave_function``) is invoked to
    select the winning fact.
    """

    def __init__(self, arbitrate_fn: Optional[Callable] = None):
        self.arbitrate = arbitrate_fn  # async func(conflicts) -> resolutions

    def detect_curvature(
        self,
        base_graph: nx.MultiDiGraph,
        new_graph: nx.MultiDiGraph,
    ) -> Tuple[CommutativityLevel, List[Tuple[str, str, str, str]]]:
        """
        Detect the commutativity level between two knowledge graphs.

        Args:
            base_graph: Existing knowledge graph.
            new_graph:  Newly extracted graph.

        Returns:
            (level, conflicts) where conflicts is a list of
            (subject, predicate, old_object, new_object) tuples.
        """
        conflicts: List[Tuple[str, str, str, str]] = []
        max_level = CommutativityLevel.L1_COMMUTATIVE

        # Build an index of (source, relation) -> target from the base graph
        base_index: dict[Tuple[str, str], str] = {}
        for src, tgt, data in base_graph.edges(data=True):
            rel = data.get("relation", "related_to")
            base_index[(src, rel)] = tgt

        # Check new graph edges against the base index
        for src, tgt, data in new_graph.edges(data=True):
            rel = data.get("relation", "related_to")

            if (src, rel) in base_index:
                old_tgt = base_index[(src, rel)]
                if old_tgt != tgt:
                    # Level 3: Same subject + predicate, different object
                    conflicts.append((src, rel, old_tgt, tgt))
                    max_level = CommutativityLevel.L3_CURVATURE
                # else: identical fact, no conflict
            elif src in {s for s, _, _ in base_graph.edges(data=True)}:
                # Level 2: Same entity, different predicates
                if max_level < CommutativityLevel.L2_CONDITIONAL:
                    max_level = CommutativityLevel.L2_CONDITIONAL

        return max_level, conflicts

    async def merge_extractions(
        self,
        base_graph: nx.MultiDiGraph,
        new_graphs: List[nx.MultiDiGraph],
    ) -> nx.MultiDiGraph:
        """
        Merge multiple extraction graphs into the base graph,
        resolving Level 3 Curvature via arbitration.

        Args:
            base_graph: The canonical knowledge graph.
            new_graphs: List of newly extracted graphs to merge.

        Returns:
            Merged graph with contradictions resolved.
        """
        merged = base_graph.copy()

        for new_graph in new_graphs:
            level, conflicts = self.detect_curvature(merged, new_graph)

            if level == CommutativityLevel.L3_CURVATURE and conflicts:
                if self.arbitrate:
                    resolutions = await self.arbitrate(conflicts)
                else:
                    # Default: new information wins (recency bias)
                    resolutions = {
                        (subj, pred): new_obj
                        for subj, pred, _old, new_obj in conflicts
                    }

                # Apply resolutions
                for (subj, pred), winner in resolutions.items():
                    # Remove conflicting edges
                    edges_to_remove = []
                    for u, v, key, data in merged.edges(
                        keys=True, data=True
                    ):
                        if u == subj and data.get("relation") == pred:
                            edges_to_remove.append((u, v, key))

                    for u, v, key in edges_to_remove:
                        merged.remove_edge(u, v, key=key)

                    # Add the winner
                    merged.add_edge(
                        subj,
                        winner,
                        relation=pred,
                        verified_curvature=True,
                    )

                    logger.info(
                        "Curvature resolved: %s %s → %s", subj, pred, winner
                    )

            # Add all non-conflicting edges from new_graph
            conflict_keys = {(s, p) for s, p, _, _ in conflicts}
            for src, tgt, data in new_graph.edges(data=True):
                rel = data.get("relation", "related_to")
                if (src, rel) not in conflict_keys:
                    merged.add_edge(src, tgt, **data)

        return merged
