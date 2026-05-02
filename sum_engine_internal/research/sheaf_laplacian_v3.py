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
        # Combined v3 score: weighted Laplacian + λ · deficit² (same
        # algebraic shape as v2.2).
        "v_combined_v3": v_laplacian_w + lambda_deficit * (base["v_deficit"]),
        "edge_weights": weights.tolist(),
    }
