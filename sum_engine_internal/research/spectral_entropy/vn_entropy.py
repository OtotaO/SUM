"""Von Neumann entropy of a graph Laplacian.

Pipeline (one call to ``graph_entropy(triples)`` does all of it):

  triples ──build_axiom_graph──▶ (nodes, edges)
          ──normalized_laplacian──▶ L_norm  (symmetric, PSD)
          ──density_matrix──▶ ρ = L_norm / Tr(L_norm)
          ──von_neumann_entropy──▶ S(ρ) = -Σ λ_i log λ_i

The entropy is a single scalar in [0, log(N-1)] for an N-node
graph. Empty / single-node / disconnected graphs are handled
explicitly; an empty graph returns S=0.

Why this shape:
  - Combinatorial Laplacian L = D − A is symmetric PSD with
    smallest eigenvalue 0 (the all-ones eigenvector); using L /
    Tr(L) as the density matrix is the De Domenico-Biamonte
    "thermal at infinite temperature" choice — simpler than
    ρ = exp(-βL)/Z and just as discriminative for our drift-monitor
    use case.
  - Substrate triples are ``(subject, predicate, object)`` strings;
    we project to a simple undirected graph by treating subjects
    and objects as nodes and adding an edge between any pair that
    appears in *any* triple. Multiplicity / predicate-types are
    intentionally collapsed; if Phase C ever needs predicate-typed
    edges, switch to a multigraph Laplacian.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from sum_engine_internal.graph_store import Triple


def build_axiom_graph(
    triples: Iterable[Triple],
) -> tuple[list[str], np.ndarray]:
    """Project triples to a simple undirected graph.

    Returns:
        (sorted_node_list, adjacency_matrix). The node list is
        sorted lexicographically so two callers seeing the same
        triple set get the same node ordering — important for
        deterministic eigenvalue computation across processes.
    """
    nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()
    for t in triples:
        nodes.add(t.subject); nodes.add(t.object)
        # Self-loops on (s, p, s) are dropped — they don't add
        # structural information for entropy purposes.
        if t.subject == t.object:
            continue
        a, b = sorted([t.subject, t.object])
        edges.add((a, b))
    sorted_nodes = sorted(nodes)
    n = len(sorted_nodes)
    A = np.zeros((n, n), dtype=np.float64)
    idx = {node: i for i, node in enumerate(sorted_nodes)}
    for a, b in edges:
        i, j = idx[a], idx[b]
        A[i, j] = 1.0
        A[j, i] = 1.0
    return sorted_nodes, A


def normalized_laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Combinatorial Laplacian L = D − A.

    Despite the name, we return the COMBINATORIAL form (not the
    symmetric-normalized I − D^{-1/2} A D^{-1/2}). The De
    Domenico-Biamonte density matrix uses the combinatorial form
    directly — name kept for callers who think in spectral terms.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    degrees = adjacency.sum(axis=1)
    return np.diag(degrees) - adjacency


def density_matrix(laplacian: np.ndarray) -> np.ndarray:
    """ρ = L / Tr(L). Trace-1 PSD matrix; legitimate density.

    Tr(L) = sum of degrees = 2 |E|, so for a graph with no edges
    Tr(L)=0; we return a zero matrix in that case (and
    ``von_neumann_entropy`` returns S=0 for it).
    """
    trace = float(np.trace(laplacian))
    if trace <= 0:
        return np.zeros_like(laplacian)
    return laplacian / trace


def von_neumann_entropy(rho: np.ndarray, *, eps: float = 1e-12) -> float:
    """S(ρ) = -Σ λ_i log λ_i (natural log; in nats).

    Eigenvalues below ``eps`` are treated as zero (numerical-noise
    floor). The entropy is in [0, log(N-1)] for the graph density
    matrices we build; bounded above by log(rank-1) where rank is
    the number of non-zero eigenvalues.
    """
    if rho.size == 0:
        return 0.0
    # rho is symmetric PSD by construction; eigvalsh is correct +
    # faster than the general eig
    eigs = np.linalg.eigvalsh(rho)
    # Numerical noise can produce tiny negatives in PSD matrices;
    # clip and threshold
    eigs = np.clip(eigs, 0.0, None)
    nonzero = eigs[eigs > eps]
    if nonzero.size == 0:
        return 0.0
    return float(-np.sum(nonzero * np.log(nonzero)))


def graph_entropy(
    triples: Iterable[Triple],
    *,
    return_intermediates: bool = False,
) -> float | dict:
    """One-shot pipeline. Pass triples, get a single scalar.

    If ``return_intermediates`` is True, returns a dict with
    ``{nodes, edges, n_nodes, n_edges, entropy}`` for diagnostics.
    """
    triples_list = list(triples)
    nodes, A = build_axiom_graph(triples_list)
    L = normalized_laplacian(A)
    rho = density_matrix(L)
    S = von_neumann_entropy(rho)
    if return_intermediates:
        return {
            "n_nodes": len(nodes),
            "n_edges": int(A.sum() / 2),  # symmetric matrix, half-count
            "entropy": S,
            "nodes": nodes,
        }
    return S
