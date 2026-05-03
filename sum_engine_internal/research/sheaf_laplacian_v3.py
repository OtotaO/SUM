"""v3 sheaf-Laplacian: receipt-weighted edges.

Extends the v2 detector with per-edge weights derived from the
trust loop's own outputs — Ed25519-signed render receipts. The
weighted sheaf Laplacian (Hansen-Ghrist 2019, §3.2 weighted
generalization) is

    L_F^w = δ^T W δ

where W is a diagonal |E| × |E| matrix of non-negative edge
weights. The quadratic form becomes

    x^T L_F^w x = Σ_e w_e · ‖F_h^{(r)} x_u − F_t^{(r)} x_v‖²

Edges with higher weight contribute more to the consistency score.
The mathematical claim from §3.2 carries through with one line of
algebra: ``W^{1/2} δ`` is still a coboundary of a sheaf (the same
combinatorics, scaled stalks); ``L_F^w = (W^{1/2} δ)^T (W^{1/2} δ)``
remains symmetric PSD.

Why this is the natural v3 (and "fractal" in the architectural
sense the project calls out): the edge weights come from the
system's own trust artifacts. Signed render receipts (PR #102 trust
substrate) attest that a particular triple was rendered through a
known-issuer Worker; their signatures, verified against a JWKS, can
be machine-checked by the same verifier the cross-runtime trust
triangle (K1-K4) already exercises. v3 takes those receipts and
weighs the detector's confidence by them. Higher trust → higher
weight → sharper detection signal in regions the system already
believes; unsigned regions get a lower-weight floor that doesn't
silence them entirely (a hallucination in an unsigned region is
still detectable; just weighted less).

This module is pure math. The mapping from "receipts in hand" to
"per-edge weights" lives in :func:`weights_from_receipts`; the
mapping from "JWKS verification result" to "trusted set" is the
caller's responsibility — this module trusts the caller's
trusted-edge set and weights accordingly.

Falsifiable predictions (pinned in tests):

  H1. With non-trivial weights, the quadratic form scales linearly
      in the weighted edge contributions: doubling all weights
      doubles V; setting one edge's weight to 0 zeros that edge's
      contribution exactly.

  H2. When weights are uniform (w_e = c for all e), v3's quadratic
      form reduces to c · v2's quadratic form. (Sanity: v3 is a
      strict generalization of v2.)

  H3. Tampering a trusted (high-weight) edge produces a sharper V
      jump than tampering an unsigned (low-weight) edge — that is,
      the per-edge contribution scales with weight. This is what
      makes receipt-weighting *useful*, not just well-defined.

  H4. weights_from_receipts produces a deterministic, parallel-to-
      sheaf.edges weight vector with the contract: trusted edges
      get trusted_weight; revoked-key edges get revoked_weight;
      everything else gets default_weight.

Behind the same ``[research]`` extras flag as v1/v2.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from sum_engine_internal.research.sheaf_laplacian_v2 import (
    KnowledgeSheafV2,
    Triple,
    per_edge_residual_v2,
    sheaf_laplacian_v2,
)


# ─── Per-edge weights ────────────────────────────────────────────────


def weights_from_receipts(
    sheaf: KnowledgeSheafV2,
    trusted_edges: Iterable[Triple] = (),
    revoked_edges: Iterable[Triple] = (),
    *,
    trusted_weight: float = 1.0,
    default_weight: float = 0.1,
    revoked_weight: float = 0.0,
) -> np.ndarray:
    """Build a per-edge weight vector from a trusted/revoked partition.

    The caller decides which edges are "trusted" (verified Ed25519
    signature against a known-issuer JWKS) and which are "revoked"
    (signed by a key now on the revoked-kids list). Everything else
    falls into the default bucket.

    Returns an ``(|E|,)`` ndarray indexed parallel to ``sheaf.edges``.
    Weight contract:
        revoked_weight ≤ default_weight ≤ trusted_weight

    The default ordering (0.0 ≤ 0.1 ≤ 1.0) makes:
      - revoked edges contribute zero (they vanish from the
        consistency check, since a revoked signature is not
        something we want to weigh against an honest cochain).
      - default edges contribute 1/10 of trusted edges.

    A trusted edge that ALSO appears in revoked_edges resolves to
    revoked (revocation overrides trust — once a key is revoked,
    its prior signatures are no longer load-bearing).

    Raises ValueError if any weight is negative — Hansen-Ghrist §3.2's
    weighted Laplacian requires non-negative weights for the PSD
    claim to carry.
    """
    if revoked_weight < 0 or default_weight < 0 or trusted_weight < 0:
        raise ValueError(
            f"weights must be non-negative; got "
            f"revoked={revoked_weight}, default={default_weight}, "
            f"trusted={trusted_weight}"
        )

    trusted_set = set(trusted_edges)
    revoked_set = set(revoked_edges)

    n_e = len(sheaf.edges)
    weights = np.full(n_e, default_weight, dtype=np.float64)
    for i, edge in enumerate(sheaf.edges):
        if edge in revoked_set:
            weights[i] = revoked_weight
        elif edge in trusted_set:
            weights[i] = trusted_weight
    return weights


# ─── Weighted Laplacian quadratic form ────────────────────────────────


def weighted_laplacian_quadratic_form_v3(
    sheaf: KnowledgeSheafV2,
    x: np.ndarray,
    weights: np.ndarray,
) -> float:
    """x^T L_F^w x = Σ_e w_e · ‖F_h^{(r)} x_u − F_t^{(r)} x_v‖².

    Factored form: avoids materializing L_F^w. Cost: O(|E| · d²).
    """
    if weights.shape != (len(sheaf.edges),):
        raise ValueError(
            f"weights shape must be (|E|={len(sheaf.edges)},); "
            f"got {weights.shape}"
        )
    residuals = per_edge_residual_v2(sheaf, x)        # (|E|, d)
    per_edge = np.sum(residuals * residuals, axis=1)   # (|E|,)
    return float(np.sum(weights * per_edge))


def weighted_sheaf_laplacian_v3(
    sheaf: KnowledgeSheafV2,
    weights: np.ndarray,
) -> np.ndarray:
    """Materialized weighted Laplacian L_F^w = δ^T W δ.

    Use only for math sanity (symmetry, PSD). For scoring, prefer
    :func:`weighted_laplacian_quadratic_form_v3` — it avoids the
    |V|·d × |V|·d matrix construction.
    """
    if weights.shape != (len(sheaf.edges),):
        raise ValueError(
            f"weights shape must be (|E|={len(sheaf.edges)},); "
            f"got {weights.shape}"
        )
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative for PSD")

    # Build δ (|E|·d × |V|·d) from v2's machinery, then apply W^{1/2}
    # row-scaling. This is equivalent to W^{1/2} δ and lets us reuse
    # sheaf_laplacian_v2's coboundary construction without code dup.
    L_unweighted = sheaf_laplacian_v2(sheaf)
    if np.allclose(weights, weights[0]):
        # Uniform weights: L^w = c · L (saves a full rebuild).
        return float(weights[0]) * L_unweighted

    # Non-uniform: rebuild via the explicit coboundary path.
    # We compute δ row-by-edge, scale each block by sqrt(w_e),
    # then form δ^T (W^{1/2} δ) = δ^T W δ.
    n_v = len(sheaf.vertices)
    n_e = len(sheaf.edges)
    d = sheaf.stalk_dim

    delta = np.zeros((n_e * d, n_v * d), dtype=np.float64)
    for i, (s, _, o) in enumerate(sheaf.edges):
        r_idx = sheaf.edge_relation[i]
        u = sheaf.vertex_index[s]
        v = sheaf.vertex_index[o]
        delta[i * d:(i + 1) * d, u * d:(u + 1) * d] = sheaf.F_h[r_idx]
        delta[i * d:(i + 1) * d, v * d:(v + 1) * d] = -sheaf.F_t[r_idx]

    # W^{1/2} ⊗ I_d applied to δ: each edge-row block gets scaled by sqrt(w_e).
    sqrt_w = np.sqrt(weights)
    scaled_delta = delta * np.repeat(sqrt_w, d)[:, None]

    return scaled_delta.T @ scaled_delta


# ─── Per-edge weighted contributions (localization) ──────────────────


def weighted_per_edge_discrepancy_v3(
    sheaf: KnowledgeSheafV2,
    x: np.ndarray,
    weights: np.ndarray,
) -> list[tuple[Triple, float]]:
    """Per-edge weighted contribution ``w_e · ‖residual_e‖²``, sorted desc.

    Localization analogue of v2's per_edge_discrepancy_v2 with
    weights folded in. The top-ranked edge is the one whose
    *weighted* contribution dominates V — useful for surfacing
    "the high-trust edge that's misbehaving" rather than "the
    low-weight edge that has high raw discrepancy but isn't
    trusted enough to matter."
    """
    if weights.shape != (len(sheaf.edges),):
        raise ValueError(
            f"weights shape must be (|E|={len(sheaf.edges)},); "
            f"got {weights.shape}"
        )
    residuals = per_edge_residual_v2(sheaf, x)
    contribs = [
        (sheaf.edges[i], float(weights[i] * np.sum(residuals[i] * residuals[i])))
        for i in range(len(sheaf.edges))
    ]
    contribs.sort(key=lambda kv: kv[1], reverse=True)
    return contribs


# ─── v3 combined detector ────────────────────────────────────────────


def combined_detector_score_v3(
    sheaf: KnowledgeSheafV2,
    embeddings: np.ndarray,
    render_triples: list[Triple],
    weights: np.ndarray,
    lambda_deficit: float = 0.05,
) -> dict:
    """v3 combined detector: weighted Laplacian + presence deficit.

    Mirrors v2.2's ``combined_detector_score`` shape (so downstream
    consumers can swap v2.2 → v3 by changing the import) but uses
    the weighted Laplacian on the cochain-residual side. The
    presence-deficit term is unchanged — receipts inform the
    *consistency* term of the score, not the entity-presence term.

    Returns a dict with the same keys as v2.2 plus ``v_laplacian_w``
    (weighted Laplacian) and ``edge_weights`` for transparency.
    """
    from sum_engine_internal.research.sheaf_laplacian_v2 import (
        cochain_one_hot_v2,
        combined_detector_score,
    )

    # Reuse v2.2's deficit + cochain logic (no need to duplicate).
    # v2.2's kwarg is named ``presence_weight``; v3 keeps the more
    # mathematically descriptive ``lambda_deficit`` at its own boundary.
    base = combined_detector_score(
        sheaf, embeddings, render_triples, presence_weight=lambda_deficit,
    )
    # Override v_laplacian with the weighted form computed on the same cochain.
    x_cochain = cochain_one_hot_v2(sheaf, render_triples, embedding=embeddings)
    v_laplacian_w = weighted_laplacian_quadratic_form_v3(sheaf, x_cochain, weights)

    return {
        **base,
        "v_laplacian_w": v_laplacian_w,
        # Combined v3 score: weighted Laplacian + the v2.2 deficit term.
        # v2.2's ``v_deficit`` field is already ``presence_weight ·
        # deficit²`` (the deficit term, post-λ-weighting), so v3 just
        # adds it once. Earlier formulation
        # ``v_laplacian_w + lambda_deficit * base["v_deficit"]``
        # double-counted λ — caught by the audit-tightening pass on
        # 2026-05-02 with `test_combined_v3_lambda_wiring_with_nonzero_deficit`.
        "v_combined_v3": v_laplacian_w + base["v_deficit"],
        "edge_weights": weights.tolist(),
    }


# ─── v3.1: harmonic extension over (boundary, interior) ──────────────
#
# Hansen-Ghrist 2019, Proposition 4.1 / Theorem 4.5: given a sheaf F
# on a graph G, a partition V = B ∪ I (boundary ∪ interior), and a
# cochain x_B specified on the boundary, the harmonic extension is
# the unique cochain x ∈ C^0(G; F) that
#
#   (i)  agrees with x_B on B
#   (ii) minimizes ‖δx‖² over the interior I
#
# Block-decompose the Laplacian by the (B, I) partition:
#
#   L_F = [L_BB  L_BI]
#         [L_IB  L_II]
#
# Setting ∂‖δx‖²/∂x_I = 0 gives x_I^* = -L_II^{-1} L_IB x_B (a
# closed-form solution when L_II is invertible).
#
# Why this matters for SUM: trusted-receipt-backed vertices form the
# boundary B (we trust the sheaf's structure there); untrusted/
# unsigned vertices form the interior I. The harmonic extension is
# the most-consistent interpolation given the boundary constraints.
# A render whose interior cochain diverges from the harmonic
# extension is flagged — the system is using its own trust artifacts
# (the boundary) to score consistency on the parts it doesn't already
# trust (the interior).
#
# This is the Hansen-Ghrist machinery v3 named as out-of-scope (v3.1
# candidate); the weighted-Laplacian primitive v3 shipped is the
# prerequisite this module builds on.


def _block_indices(vertex_indices: list[int], stalk_dim: int) -> list[int]:
    """Expand vertex indices to flat |V|·d row/col indices of L_F.

    A vertex v in the sheaf occupies rows/cols [v·d, (v+1)·d) of
    the materialized Laplacian. ``np.ix_`` then takes the
    rectangular submatrix.
    """
    return [v * stalk_dim + j for v in vertex_indices for j in range(stalk_dim)]


def harmonic_extension(
    sheaf: KnowledgeSheafV2,
    boundary_indices: list[int],
    x_B: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Compute the harmonic extension of x_B over the interior I = V \\ B.

    Returns ``(x_I_star, interior_indices)`` where ``x_I_star`` has
    shape ``(|I|, d)`` and is the unique minimizer of ‖δx‖² subject
    to ``x[B] = x_B``.

    ``boundary_indices`` is a list of vertex indices (not flat
    matrix indices). ``x_B`` has shape ``(|B|, d)``, indexed
    parallel to ``boundary_indices``.

    Optional ``weights`` (per-edge, shape (|E|,)) computes the
    *weighted* harmonic extension under L_F^w. Without weights,
    falls back to the unweighted v2 Laplacian.

    Numerical stability: uses ``np.linalg.lstsq`` rather than a
    direct inverse, so a rank-deficient ``L_II`` (disconnected
    interior, or interior with a global section) returns the
    minimum-norm solution rather than crashing. The caller can
    surface degenerate cases by checking the residual.
    """
    n_v = len(sheaf.vertices)
    d = sheaf.stalk_dim

    if not all(0 <= v < n_v for v in boundary_indices):
        raise ValueError(
            f"boundary_indices must be in [0, {n_v}); got {boundary_indices}"
        )
    if x_B.shape != (len(boundary_indices), d):
        raise ValueError(
            f"x_B shape must be (|B|={len(boundary_indices)}, d={d}); "
            f"got {x_B.shape}"
        )

    interior_indices = [v for v in range(n_v) if v not in set(boundary_indices)]
    if not interior_indices:
        # Degenerate: every vertex is on the boundary. Harmonic
        # extension is empty; return a (0, d) array for shape
        # consistency.
        return np.zeros((0, d), dtype=np.float64), []

    if weights is None:
        L = sheaf_laplacian_v2(sheaf)
    else:
        L = weighted_sheaf_laplacian_v3(sheaf, weights)

    B_flat = _block_indices(boundary_indices, d)
    I_flat = _block_indices(interior_indices, d)

    L_II = L[np.ix_(I_flat, I_flat)]
    L_IB = L[np.ix_(I_flat, B_flat)]

    x_B_flat = x_B.reshape(-1)
    rhs = -L_IB @ x_B_flat

    # lstsq returns the minimum-norm solution when L_II is rank-
    # deficient (which can happen when the interior has a global
    # section — e.g. disconnected from the boundary, or contains
    # only a constant cochain). Robust to numerical edge cases.
    x_I_flat, _residuals, _rank, _singvals = np.linalg.lstsq(L_II, rhs, rcond=None)
    return x_I_flat.reshape(len(interior_indices), d), interior_indices


def boundary_deviation(
    sheaf: KnowledgeSheafV2,
    x_full: np.ndarray,
    boundary_indices: list[int],
    *,
    weights: np.ndarray | None = None,
) -> dict:
    """Distance from a full cochain to the harmonic extension of its boundary.

    A render whose interior cochain matches the harmonic extension
    of its boundary cochain is "consistent with the trust frame";
    one that diverges is flagged.

    Returns:
      ``deviation`` — ‖x_I_actual − x_I^*‖² (squared L2 distance,
        the natural metric in the d-stalk inner product).
      ``v_at_actual`` — x^T L x at the actual cochain.
      ``v_at_extension`` — x^T L x at the harmonic-extended cochain
        (boundary held to x_B; interior to x_I^*). By the
        minimization property, this is ≤ v_at_actual; equality iff
        the actual interior already matches the harmonic extension.
      ``boundary_size`` — |B|; ``interior_size`` — |I|.
    """
    n_v = len(sheaf.vertices)
    d = sheaf.stalk_dim
    if x_full.shape != (n_v, d):
        raise ValueError(
            f"x_full shape must be (|V|={n_v}, d={d}); got {x_full.shape}"
        )

    interior_indices = [v for v in range(n_v) if v not in set(boundary_indices)]
    x_B = x_full[boundary_indices]

    x_I_star, _ = harmonic_extension(
        sheaf, boundary_indices, x_B, weights=weights,
    )

    if not interior_indices:
        # Degenerate: full boundary, nothing to deviate over. Devation
        # is 0 by convention.
        if weights is None:
            v_actual = float(np.sum(per_edge_residual_v2(sheaf, x_full) ** 2))
        else:
            v_actual = weighted_laplacian_quadratic_form_v3(sheaf, x_full, weights)
        return {
            "deviation": 0.0,
            "v_at_actual": v_actual,
            "v_at_extension": v_actual,
            "boundary_size": len(boundary_indices),
            "interior_size": 0,
        }

    x_I_actual = x_full[interior_indices]
    deviation = float(np.sum((x_I_actual - x_I_star) ** 2))

    # Build the harmonic-extended full cochain to compute v_at_extension
    x_extended = x_full.copy()
    for k, v_idx in enumerate(interior_indices):
        x_extended[v_idx] = x_I_star[k]

    if weights is None:
        v_actual = float(np.sum(per_edge_residual_v2(sheaf, x_full) ** 2))
        v_ext = float(np.sum(per_edge_residual_v2(sheaf, x_extended) ** 2))
    else:
        v_actual = weighted_laplacian_quadratic_form_v3(sheaf, x_full, weights)
        v_ext = weighted_laplacian_quadratic_form_v3(sheaf, x_extended, weights)

    return {
        "deviation": deviation,
        "v_at_actual": v_actual,
        "v_at_extension": v_ext,
        "boundary_size": len(boundary_indices),
        "interior_size": len(interior_indices),
    }


def boundary_from_weights(
    sheaf: KnowledgeSheafV2,
    weights: np.ndarray,
    *,
    threshold: float = 0.5,
) -> list[int]:
    """Derive a boundary vertex set from per-edge weights.

    A vertex is on the boundary iff *all* of its incident edges have
    weight ≥ ``threshold``. The intuition: a vertex is "trusted"
    when every edge connecting it to its neighbours is backed by a
    verified receipt. Vertices with even one untrusted incident edge
    fall to the interior.

    With the default ``trusted_weight=1.0`` / ``default_weight=0.1``
    from :func:`weights_from_receipts`, threshold=0.5 partitions
    cleanly: trusted-only neighbourhoods → boundary; anything else →
    interior.

    Returns a list of vertex indices (sorted ascending).
    """
    n_v = len(sheaf.vertices)
    incident_edge_weights: list[list[float]] = [[] for _ in range(n_v)]
    for i, (s, _, o) in enumerate(sheaf.edges):
        incident_edge_weights[sheaf.vertex_index[s]].append(float(weights[i]))
        incident_edge_weights[sheaf.vertex_index[o]].append(float(weights[i]))

    boundary: list[int] = []
    for v in range(n_v):
        ws = incident_edge_weights[v]
        if not ws:
            # Isolated vertex (no incident edges): not on the
            # boundary — it carries no trust signal.
            continue
        if min(ws) >= threshold:
            boundary.append(v)
    return boundary
