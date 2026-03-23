"""
Causal Discovery Engine — Topological Inference

Sweeps active axioms for transitive topological links (causality,
implication, inhibition) and computes transitive closures to
synthesize novel knowledge that the system was never explicitly taught.

If the algebra contains:
    chemical_x → inhibits → enzyme_y
    enzyme_y   → causes   → disease_z

The engine will deduce:
    chemical_x → treats → disease_z

Author: ototao
License: Apache License 2.0
"""

from typing import List, Tuple
import networkx as nx


class CausalDiscoveryEngine:
    """
    Horizon V: The Automated Scientist.

    Sweeps active axioms for transitive topological links
    and synthesizes novel knowledge via strict predicate rules.
    """

    TRANSITIVE_PREDICATES = {"causes", "implies", "leads_to", "requires", "is_a"}
    INVERSE_PREDICATES = {"inhibits": "treats", "prevents": "solves"}

    def __init__(self, algebra):
        self.algebra = algebra

    def sweep_for_discoveries(
        self, current_state: int
    ) -> List[Tuple[str, str, str]]:
        """
        Extract directed graph from active axioms, compute transitive
        closures, and return novel triplets not already in the state.

        Args:
            current_state: The Gödel integer to analyze.

        Returns:
            List of (subject, predicate, object) triplets that are
            logically entailed but not yet in the state.
        """
        active_axioms = self.algebra.get_active_axioms(current_state)

        # Build directed graph from causal/transitive predicates
        G = nx.DiGraph()
        for ax in active_axioms:
            parts = ax.split("||")
            if len(parts) == 3:
                s, p, o = parts
                if p in self.TRANSITIVE_PREDICATES or p in self.INVERSE_PREDICATES:
                    G.add_edge(s, o, predicate=p)

        # Compute 2-hop transitive closures
        discoveries = set()

        for node in list(G.nodes()):
            for neighbor in list(G.successors(node)):
                p1 = G.edges[node, neighbor]["predicate"]
                for target in list(G.successors(neighbor)):
                    if node == target:
                        continue  # Skip self-loops

                    p2 = G.edges[neighbor, target]["predicate"]

                    # Same transitive predicate: A→B→C  ⟹  A→C
                    if (
                        p1 in self.TRANSITIVE_PREDICATES
                        and p2 in self.TRANSITIVE_PREDICATES
                    ):
                        discoveries.add((node, p1, target))

                    # Inverse + transitive: A inhibits B, B causes C  ⟹  A treats C
                    elif (
                        p1 in self.INVERSE_PREDICATES
                        and p2 in self.TRANSITIVE_PREDICATES
                    ):
                        inferred_pred = self.INVERSE_PREDICATES[p1]
                        discoveries.add((node, inferred_pred, target))

        # Filter to only truly novel discoveries
        novel_discoveries = []
        for s, p, o in discoveries:
            prime = self.algebra.get_or_mint_prime(s, p, o)
            if current_state % prime != 0:
                novel_discoveries.append((s, p, o))

        return novel_discoveries
