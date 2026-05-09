"""Robust-PCA contract tests.

Layers:
  1. PCP/ADMM math: convergence + exact recovery on synthetic ground
     truth (the Candès et al. 2011 provable kernel).
  2. Corruption-score wrapper: per-row L1 norm of S, monotone with
     injected corruption magnitude.
  3. Axiom embedding: deterministic + correct shape.
  4. Edge cases: empty input, all-zero input, non-finite input.

The corpus-application precision/recall numbers live in the spike
receipt (sum.robust_pca_axiom_spike.v1) — those depend on workload
and embedding choice and don't belong in CI gates.
"""
from __future__ import annotations

import numpy as np
import pytest

from sum_engine_internal.research.robust_pca import (
    PCPConvergenceError,
    PCPResult,
    corruption_score,
    embed_triple,
    embed_triples,
    pcp,
)


# -- Core math: PCP recovery -------------------------------------------


@pytest.fixture
def low_rank_plus_sparse():
    """Build M = L + S with known L (rank 5) and S (5% sparse)."""
    rng = np.random.RandomState(42)
    n, d, r = 200, 200, 5
    U = rng.randn(n, r); V = rng.randn(r, d)
    L_true = U @ V
    mask = rng.rand(n, d) < 0.05
    S_true = np.zeros((n, d))
    S_true[mask] = rng.choice([-5, 5], size=int(mask.sum()))
    return L_true + S_true, L_true, S_true


def test_pcp_returns_PCPResult(low_rank_plus_sparse):
    M, _, _ = low_rank_plus_sparse
    r = pcp(M)
    assert isinstance(r, PCPResult)
    assert r.L.shape == M.shape and r.S.shape == M.shape


def test_pcp_recovers_exact_rank(low_rank_plus_sparse):
    M, L_true, _ = low_rank_plus_sparse
    r = pcp(M)
    expected_rank = int(np.linalg.matrix_rank(L_true))
    assert r.rank_estimate == expected_rank, (
        f"recovered rank {r.rank_estimate} != true rank {expected_rank}"
    )


def test_pcp_recovers_L_to_high_precision(low_rank_plus_sparse):
    M, L_true, _ = low_rank_plus_sparse
    r = pcp(M)
    rel_err = np.linalg.norm(r.L - L_true, "fro") / np.linalg.norm(L_true, "fro")
    assert rel_err < 1e-6, f"L recovery error {rel_err:.2e} above tolerance"


def test_pcp_recovers_S_to_high_precision(low_rank_plus_sparse):
    M, _, S_true = low_rank_plus_sparse
    r = pcp(M)
    rel_err = np.linalg.norm(r.S - S_true, "fro") / np.linalg.norm(S_true, "fro")
    assert rel_err < 1e-6, f"S recovery error {rel_err:.2e} above tolerance"


def test_pcp_residual_below_tol(low_rank_plus_sparse):
    M, _, _ = low_rank_plus_sparse
    r = pcp(M, tol=1e-7)
    assert r.residual_norm < 1e-7


def test_pcp_default_lambda_matches_candes_2011():
    """λ = 1/√(max(n, d)) is the Candès et al. 2011 default. Pin it
    so a refactor doesn't silently change the regulariser."""
    M = np.zeros((30, 50))
    M[0, 0] = 1.0  # avoid all-zero short-circuit
    r = pcp(M, max_iter=20)
    assert abs(r.lam - 1.0 / np.sqrt(50)) < 1e-12


# -- Edge cases --------------------------------------------------------


def test_pcp_rejects_empty_matrix():
    with pytest.raises(ValueError, match="empty"):
        pcp(np.zeros((0, 5)))


def test_pcp_rejects_non_finite_input():
    M = np.array([[1.0, np.inf], [0.0, 0.0]])
    with pytest.raises(ValueError, match="non-finite"):
        pcp(M)


def test_pcp_handles_all_zero_input():
    M = np.zeros((5, 5))
    r = pcp(M)
    assert r.n_iter == 0
    assert r.residual_norm == 0
    assert r.rank_estimate == 0
    assert r.sparsity_estimate == 0


def test_pcp_raises_when_max_iter_exhausted():
    """Use a hard-to-solve setup + tiny max_iter to force the error
    path. Confirms the error is informative."""
    rng = np.random.RandomState(0)
    M = rng.randn(20, 30)
    with pytest.raises(PCPConvergenceError, match="did not converge"):
        pcp(M, max_iter=1, tol=1e-15)


# -- corruption_score --------------------------------------------------


def test_corruption_score_returns_per_row_l1_of_S(low_rank_plus_sparse):
    M, _, _ = low_rank_plus_sparse
    scores = corruption_score(M)
    assert scores.shape == (M.shape[0],)
    # All non-negative (L1 norm)
    assert (scores >= 0).all()


def test_corruption_score_higher_for_corrupted_rows():
    """The substrate's load-bearing claim: rows with injected
    corruption score higher than rows without."""
    rng = np.random.RandomState(42)
    n, d, r = 100, 100, 3
    U = rng.randn(n, r); V = rng.randn(r, d)
    L_true = U @ V
    mask = rng.rand(n, d) < 0.05
    S_true = np.zeros((n, d))
    S_true[mask] = rng.choice([-5, 5], size=int(mask.sum()))
    M = L_true + S_true
    scores = corruption_score(M)
    corrupt = mask.any(axis=1)
    # Strict: corrupt-row mean must exceed clean-row mean by 10×
    ratio = scores[corrupt].mean() / max(scores[~corrupt].mean(), 1e-9)
    assert ratio > 10, f"corrupt/clean score ratio {ratio:.1f} should be >> 1"


# -- Axiom embedding ---------------------------------------------------


def _t(s, p, o):
    from sum_engine_internal.graph_store import Triple
    return Triple(s, p, o)


def test_embed_triple_returns_correct_shape():
    v = embed_triple(_t("alice", "likes", "cats"), n_buckets=64)
    assert v.shape == (3 * 64,)


def test_embed_triple_has_exactly_three_ones():
    """One per role (subject, predicate, object). Hash collisions
    within a single triple are essentially zero at n_buckets=64."""
    v = embed_triple(_t("alice", "likes", "cats"))
    assert v.sum() == 3.0
    assert int((v == 1).sum()) == 3


def test_embed_triple_is_deterministic():
    """Same input → same vector across calls (sha256-based, not
    Python's randomised hash)."""
    v1 = embed_triple(_t("alice", "likes", "cats"))
    v2 = embed_triple(_t("alice", "likes", "cats"))
    assert np.array_equal(v1, v2)


def test_embed_triples_returns_correct_shape():
    M = embed_triples([_t("alice", "likes", "cats"), _t("bob", "owns", "rex")])
    assert M.shape == (2, 3 * 64)


def test_embed_triples_handles_empty_input():
    M = embed_triples([], n_buckets=64)
    assert M.shape == (0, 3 * 64)


def test_embed_triples_each_row_is_individual_embed():
    triples = [_t("alice", "likes", "cats"), _t("bob", "owns", "rex")]
    M = embed_triples(triples)
    assert np.array_equal(M[0], embed_triple(triples[0]))
    assert np.array_equal(M[1], embed_triple(triples[1]))
