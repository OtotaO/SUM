"""Von Neumann graph entropy contract tests.

Layers:
  1. **Bounds & extremes** — entropy ∈ [0, log(N-1)]; K_n hits the
     upper bound; empty graph and isolated nodes hit 0.
  2. **Determinism** — same triple set → same entropy across
     re-orderings; stable across processes.
  3. **Drift sensitivity** — adding off-corpus triples changes S
     in a measurable direction.
  4. **Pipeline correctness** — graph builder collapses self-loops,
     deduplicates parallel edges, sorts nodes lexicographically.
"""
from __future__ import annotations

import numpy as np
import pytest

from sum_engine_internal.graph_store import Triple
from sum_engine_internal.research.spectral_entropy import (
    build_axiom_graph,
    density_matrix,
    graph_entropy,
    normalized_laplacian,
    von_neumann_entropy,
)


def _t(s, p, o):
    return Triple(s, p, o)


# -- Bounds and extremes -----------------------------------------------


def test_empty_graph_returns_zero_entropy():
    assert graph_entropy([]) == 0.0


def test_single_self_loop_returns_zero_entropy():
    """Self-loops add no structural information; the graph builder
    drops them. A graph with only self-loops has zero edges and
    therefore zero entropy."""
    assert graph_entropy([_t("a", "knows", "a")]) == 0.0


def test_complete_graph_K_n_hits_upper_bound():
    """K_n (every node connected to every other) maximises the
    entropy; theoretical max is log(N-1) for the De Domenico-
    Biamonte density matrix."""
    for n in [3, 5, 8]:
        nodes = [f"v{i}" for i in range(n)]
        triples = [
            _t(a, "p", b) for i, a in enumerate(nodes) for b in nodes[i + 1:]
        ]
        S = graph_entropy(triples)
        max_S = float(np.log(n - 1))
        # K_n hits the upper bound (modulo tiny float error)
        assert abs(S - max_S) < 1e-10, (
            f"K_{n} entropy {S:.10f} should equal log({n - 1}) = {max_S:.10f}"
        )


def test_path_graph_entropy_is_strictly_below_K_n_upper_bound():
    """A path on N nodes has lower entropy than K_N — fewer edges,
    less spectral mixing."""
    n = 5
    path = [_t(f"v{i}", "p", f"v{i+1}") for i in range(n - 1)]
    S_path = graph_entropy(path)
    assert 0 < S_path < float(np.log(n - 1))


def test_disconnected_pair_has_low_entropy():
    """Two disjoint edges: a-b and c-d. Lower entropy than a single
    connected component of 4 nodes."""
    triples = [_t("a", "p", "b"), _t("c", "p", "d")]
    S = graph_entropy(triples)
    assert S >= 0
    assert S < float(np.log(3))  # < log(N-1) for N=4


# -- Determinism -------------------------------------------------------


def test_entropy_is_invariant_under_triple_order():
    triples = [
        _t("alice", "owns", "rex"),
        _t("bob", "likes", "cats"),
        _t("carol", "writes", "code"),
        _t("dave", "drives", "tesla"),
    ]
    S1 = graph_entropy(triples)
    S2 = graph_entropy(list(reversed(triples)))
    assert S1 == S2


def test_entropy_is_invariant_under_duplicate_triples():
    """Adding the same (s, p, o) twice doesn't change the graph
    (the builder dedups edges)."""
    triples = [_t("a", "p", "b"), _t("c", "q", "d")]
    S1 = graph_entropy(triples)
    S2 = graph_entropy(triples + triples)
    assert S1 == S2


def test_entropy_is_invariant_under_predicate_relabelling():
    """The graph builder collapses predicate types — only edges
    matter for entropy. (s, "owns", o) and (s, "knows", o) produce
    the same graph."""
    A = [_t("a", "owns", "b")]
    B = [_t("a", "knows", "b")]
    assert graph_entropy(A) == graph_entropy(B)


def test_node_ordering_is_lexicographic():
    """Build the graph from triples in mixed order, verify the
    returned node list is sorted lex (deterministic across
    callers)."""
    triples = [_t("c", "p", "a"), _t("b", "p", "d")]
    nodes, _ = build_axiom_graph(triples)
    assert nodes == ["a", "b", "c", "d"]


# -- Drift sensitivity -------------------------------------------------


def test_adding_off_corpus_triple_changes_entropy():
    """The headline use case: structural drift detection. Adding
    a triple that introduces new disconnected nodes changes S."""
    clean = [_t(f"e{i}", "rel", f"e{(i+1)%5}") for i in range(5)]  # 5-cycle
    S_clean = graph_entropy(clean)
    drift = clean + [_t("off_corpus_x", "rel", "off_corpus_y")]
    S_drift = graph_entropy(drift)
    assert S_drift != S_clean


def test_adding_redundant_edge_to_complete_graph_does_not_increase_entropy():
    """Adding (a, q, b) when (a, p, b) already exists doesn't add
    a new edge → entropy unchanged."""
    triples = [_t("a", "p", "b")]
    S1 = graph_entropy(triples)
    S2 = graph_entropy(triples + [_t("a", "different_predicate", "b")])
    assert S1 == S2


# -- Pipeline correctness ----------------------------------------------


def test_build_axiom_graph_drops_self_loops():
    nodes, A = build_axiom_graph([_t("a", "p", "a"), _t("a", "p", "b")])
    assert nodes == ["a", "b"]
    # Diagonal stays zero (no self-loops)
    assert A[0, 0] == 0.0
    assert A[1, 1] == 0.0
    # The non-self-loop edge is present
    assert A[0, 1] == 1.0
    assert A[1, 0] == 1.0


def test_normalized_laplacian_row_sums_to_zero():
    """L = D − A has row sums of zero (the all-ones vector is in
    the kernel)."""
    _, A = build_axiom_graph([_t("a", "p", "b"), _t("b", "p", "c")])
    L = normalized_laplacian(A)
    assert np.allclose(L.sum(axis=1), 0.0)


def test_density_matrix_has_unit_trace():
    _, A = build_axiom_graph([_t("a", "p", "b"), _t("b", "p", "c")])
    L = normalized_laplacian(A)
    rho = density_matrix(L)
    assert abs(np.trace(rho) - 1.0) < 1e-12


def test_density_matrix_of_empty_returns_zero_matrix():
    _, A = build_axiom_graph([])
    L = normalized_laplacian(A)
    rho = density_matrix(L)
    assert rho.shape == (0, 0)


def test_density_matrix_of_isolated_nodes_only_returns_zeros():
    """Triples with no s≠o pairs (only self-loops) produce a graph
    of isolated nodes — zero edges, zero trace."""
    _, A = build_axiom_graph([_t("a", "p", "a"), _t("b", "p", "b")])
    L = normalized_laplacian(A)
    assert np.allclose(L, 0.0)


def test_graph_entropy_with_intermediates_returns_dict():
    triples = [_t("a", "p", "b"), _t("b", "p", "c")]
    out = graph_entropy(triples, return_intermediates=True)
    assert isinstance(out, dict)
    assert out["n_nodes"] == 3
    assert out["n_edges"] == 2
    assert out["entropy"] > 0
    assert out["nodes"] == ["a", "b", "c"]


# -- Numerical robustness ---------------------------------------------


def test_entropy_is_nonnegative():
    """von Neumann entropy is provably non-negative; numerical
    noise can produce tiny negatives if mishandled."""
    rng = np.random.RandomState(0)
    triples = [_t(f"e{rng.randint(0, 20)}", "p", f"e{rng.randint(0, 20)}")
               for _ in range(50)]
    assert graph_entropy(triples) >= 0


def test_entropy_handles_disconnected_components():
    """Graph with 3 disjoint K_2's. Should produce a valid entropy
    (the Laplacian is block-diagonal)."""
    triples = [_t("a", "p", "b"), _t("c", "p", "d"), _t("e", "p", "f")]
    S = graph_entropy(triples)
    assert S >= 0
    assert np.isfinite(S)
