"""v2.1 sheaf-Laplacian hallucination detector — learned restriction maps.

Implements ``docs/SHEAF_HALLUCINATION_DETECTOR.md`` §3.3 v2.1: d-dim
stalks with per-relation **learned** restriction maps trained via the
contrastive sheaf-embedding loss (Gebhart, Hansen & Schrater 2023,
AISTATS, arXiv:2110.03789, Definition 11 / Equation 4). Math grounded
in Hansen & Ghrist (2019), *Toward a Spectral Theory of Cellular
Sheaves* (arXiv:1808.01513, JACT 2019) §3.2 — the d-dim sheaf
Laplacian formula and the block-Laplacian factorization.

Why v2.1, not v2.0:

  Reading Hansen-Ghrist §3.2 surfaced that d-dim stalks with
  *identity* restriction maps don't address v1's blindspots — the
  global-section condition becomes "x_v = x_u for every edge,"
  which is wrong (alice's embedding shouldn't equal MIT's just
  because they're connected by `graduated`). Per-relation
  *learned* restriction maps are what make d > 1 meaningful.
  See spec §3.3 for the full v2.0 / v2.1 / v2.2 split.

Scope:

  - Pure numpy. No torch / pytorch. CPU-only. No external API spend.
  - Default d ∈ {8, 32, 64}. Sparse storage isn't necessary at this
    scale; dense block matrices are simpler and faster for the
    naturalistic-prose corpus size (≤ 100 vertices / 100 edges).
  - Training: γ-gapped margin ranking loss with LCWA negative
    sampling (Gebhart Def. 11). Adam-style optimization on the
    restriction-map parameters; entity embeddings frozen at
    one-hot for v2.1 (lifted to context-window embeddings in v2.2).

Behind the same ``[research]`` extras flag as v1 — production
install path is unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


Triple = tuple[str, str, str]


# ─── Data structures ──────────────────────────────────────────────────


@dataclass(frozen=True)
class KnowledgeSheafV2:
    """Cellular sheaf with d-dim stalks and per-relation restriction maps.

    Hansen-Ghrist §3.2: the degree-0 Laplacian is the symmetric block
    matrix with diagonal block at vertex v equal to the sum over
    incident edges of F*_{v⊵e} F_{v⊵e}, and off-diagonal block at
    (u, v) equal to -F*_{u⊵e} F_{v⊵e} for the edge e between them.

    For our knowledge-graph schema (single-type vertices, multiple
    relation types), each relation r has two restriction maps:
    ``F_h⊵r`` (head-side) and ``F_t⊵r`` (tail-side), both
    d × d matrices.
    """
    vertices: tuple[str, ...]                                     # entity names
    edges: tuple[Triple, ...]                                     # (subject, predicate, object) tuples
    relations: tuple[str, ...]                                    # distinct predicate names
    vertex_index: dict[str, int]                                  # entity name → vertex idx
    relation_index: dict[str, int]                                # predicate → relation idx
    edge_relation: tuple[int, ...]                                # for each edge, its relation idx
    stalk_dim: int                                                # d
    F_h: np.ndarray                                               # shape (|R|, d, d) — head restriction maps
    F_t: np.ndarray                                               # shape (|R|, d, d) — tail restriction maps

    def __post_init__(self) -> None:
        if self.stalk_dim < 1:
            raise ValueError(f"stalk_dim must be ≥ 1; got {self.stalk_dim}")
        n_r = len(self.relations)
        d = self.stalk_dim
        expected = (n_r, d, d)
        if self.F_h.shape != expected:
            raise ValueError(
                f"F_h shape {self.F_h.shape} != expected {expected}"
            )
        if self.F_t.shape != expected:
            raise ValueError(
                f"F_t shape {self.F_t.shape} != expected {expected}"
            )
        if len(self.edge_relation) != len(self.edges):
            raise ValueError(
                f"edge_relation has {len(self.edge_relation)} entries "
                f"but {len(self.edges)} edges"
            )

    @classmethod
    def from_triples(
        cls,
        triples: list[Triple],
        stalk_dim: int = 32,
        F_h: np.ndarray | None = None,
        F_t: np.ndarray | None = None,
        seed: int = 0,
    ) -> "KnowledgeSheafV2":
        """Construct a sheaf from a triple list. Restriction maps default
        to identity if F_h / F_t are not supplied; pass trained
        matrices to evaluate a post-training cochain.
        """
        seen_v: dict[str, None] = {}
        seen_r: dict[str, None] = {}
        for s, p, o in triples:
            seen_v.setdefault(s, None)
            seen_v.setdefault(o, None)
            seen_r.setdefault(p, None)
        vertices = tuple(seen_v)
        relations = tuple(seen_r)
        vertex_index = {v: i for i, v in enumerate(vertices)}
        relation_index = {r: i for i, r in enumerate(relations)}
        edges = tuple(triples)
        edge_relation = tuple(relation_index[p] for (_, p, _) in triples)

        n_r = len(relations)
        d = stalk_dim
        if F_h is None:
            F_h = np.tile(np.eye(d, dtype=np.float64)[None, :, :], (n_r, 1, 1))
        if F_t is None:
            F_t = np.tile(np.eye(d, dtype=np.float64)[None, :, :], (n_r, 1, 1))

        return cls(
            vertices=vertices,
            edges=edges,
            relations=relations,
            vertex_index=vertex_index,
            relation_index=relation_index,
            edge_relation=edge_relation,
            stalk_dim=d,
            F_h=F_h,
            F_t=F_t,
        )


# ─── Math primitives (Hansen-Ghrist §3.2) ─────────────────────────────


def per_edge_residual_v2(sheaf: KnowledgeSheafV2, x: np.ndarray) -> np.ndarray:
    """Per-edge δ residual: (F_h⊵e x_u − F_t⊵e x_v) ∈ R^d for each edge.

    x has shape (|V|, d). Output shape: (|E|, d).

    For an edge e: u → v carrying relation r, the residual is
    F_h^{(r)} x_u − F_t^{(r)} x_v. The sheaf is consistent at edge e
    iff this residual is zero.
    """
    n_e = len(sheaf.edges)
    d = sheaf.stalk_dim
    out = np.zeros((n_e, d), dtype=np.float64)
    for i, (s, _, o) in enumerate(sheaf.edges):
        r_idx = sheaf.edge_relation[i]
        x_u = x[sheaf.vertex_index[s]]
        x_v = x[sheaf.vertex_index[o]]
        out[i] = sheaf.F_h[r_idx] @ x_u - sheaf.F_t[r_idx] @ x_v
    return out


def laplacian_quadratic_form_v2(sheaf: KnowledgeSheafV2, x: np.ndarray) -> float:
    """x^T L_F x = ‖δx‖² = Σ_e ‖F_h^{(r)} x_u − F_t^{(r)} x_v‖²

    Computed without materializing L. Cost: O(|E| · d²)
    matrix-vector multiplies. For our scale (|E| ≤ 100, d ≤ 64) this
    is microseconds.
    """
    residuals = per_edge_residual_v2(sheaf, x)
    return float(np.sum(residuals * residuals))


def per_edge_discrepancy_v2(sheaf: KnowledgeSheafV2, x: np.ndarray) -> list[tuple[Triple, float]]:
    """Per-edge contribution ‖per-edge residual‖², sorted descending.

    Localization analogue of v1's per_edge_discrepancy at d > 1.
    """
    residuals = per_edge_residual_v2(sheaf, x)
    contribs = [
        (sheaf.edges[i], float(np.sum(residuals[i] * residuals[i])))
        for i in range(len(sheaf.edges))
    ]
    contribs.sort(key=lambda kv: kv[1], reverse=True)
    return contribs


def sheaf_laplacian_v2(sheaf: KnowledgeSheafV2) -> np.ndarray:
    """Full block Laplacian L_F = δ^T δ, materialized as a dense
    matrix of shape (|V|·d, |V|·d).

    Use only for spectral analysis; for the quadratic form, prefer
    ``laplacian_quadratic_form_v2`` which avoids the O(|V|²·d²)
    materialization cost.
    """
    n_v = len(sheaf.vertices)
    d = sheaf.stalk_dim
    n_e = len(sheaf.edges)
    delta = np.zeros((n_e * d, n_v * d), dtype=np.float64)
    for i, (s, _, o) in enumerate(sheaf.edges):
        r_idx = sheaf.edge_relation[i]
        u_col = sheaf.vertex_index[s] * d
        v_col = sheaf.vertex_index[o] * d
        row = i * d
        delta[row:row + d, u_col:u_col + d] = sheaf.F_h[r_idx]
        delta[row:row + d, v_col:v_col + d] = -sheaf.F_t[r_idx]
    return delta.T @ delta


# ─── Cochain construction ─────────────────────────────────────────────


def cochain_one_hot_v2(
    sheaf: KnowledgeSheafV2,
    extracted_triples: list[Triple],
    embedding: np.ndarray | None = None,
) -> np.ndarray:
    """v2.1 cochain construction: per-vertex one-hot in R^d (where
    d ≥ |V|) for entities mentioned in the render, else zero.

    Spec §3.3 v2.1. v2.2 will replace this with sentence-transformer
    context-window embeddings.

    If ``embedding`` is supplied, it must have shape (|V|, d); the
    cochain assigns ``embedding[v_idx]`` if entity v is mentioned,
    else zero. This lets callers pass *trained* entity embeddings
    (e.g. from a contrastive sheaf-embedding pre-training pass)
    rather than the default one-hot.
    """
    n_v = len(sheaf.vertices)
    d = sheaf.stalk_dim
    if embedding is None:
        if d < n_v:
            raise ValueError(
                f"one-hot cochain requires stalk_dim ≥ |V|; "
                f"got d={d} but |V|={n_v}. Pass an explicit "
                f"`embedding` of shape (|V|, d) for d < |V|."
            )
        embedding = np.eye(n_v, d, dtype=np.float64)
    if embedding.shape != (n_v, d):
        raise ValueError(
            f"embedding shape {embedding.shape} != expected ({n_v}, {d})"
        )

    mentioned: set[str] = set()
    for s, _, o in extracted_triples:
        mentioned.add(s)
        mentioned.add(o)

    x = np.zeros((n_v, d), dtype=np.float64)
    for v in mentioned:
        if v in sheaf.vertex_index:
            idx = sheaf.vertex_index[v]
            x[idx] = embedding[idx]
    return x


# ─── Contrastive training (Gebhart Def. 11 / Eq. 4) ───────────────────


def _sample_negative_triples(
    triples: list[Triple],
    n_negatives_per_positive: int,
    rng: np.random.Generator,
) -> list[Triple]:
    """Local-closed-world-assumption negative sampling: for each
    positive (h, r, t), produce ``n_negatives_per_positive`` synthetic
    negatives (h, r, t') where t' is a different entity from the
    same vertex set and the resulting triple is not in the positive
    set.
    """
    pos_set = {(h, r, t) for (h, r, t) in triples}
    all_entities = sorted({e for (h, _, t) in triples for e in (h, t)})
    if len(all_entities) < 2:
        return []
    negatives: list[Triple] = []
    for (h, r, t) in triples:
        for _ in range(n_negatives_per_positive):
            for _attempt in range(10):
                t_prime = all_entities[int(rng.integers(0, len(all_entities)))]
                if t_prime != t and (h, r, t_prime) not in pos_set:
                    negatives.append((h, r, t_prime))
                    break
    return negatives


def train_restriction_maps(
    triples: list[Triple],
    stalk_dim: int = 32,
    epochs: int = 200,
    learning_rate: float = 0.01,
    margin: float = 1.0,
    n_negatives_per_positive: int = 5,
    seed: int = 0,
) -> tuple[KnowledgeSheafV2, np.ndarray, list[float]]:
    """Train per-relation restriction maps F_h, F_t via the γ-gapped
    contrastive loss from Gebhart 2023 Def. 11 / Eq. 4.

    Loss per positive triple p, negative triple n:
        max(0, V_p + γ - V_n)
    where V is the per-edge Laplacian quadratic-form contribution.

    Returns:
      sheaf: KnowledgeSheafV2 with trained F_h, F_t.
      embeddings: shape (|V|, d) — learned entity embeddings.
      loss_history: per-epoch mean margin-ranking loss (training
        diagnostic; should decrease monotonically once the LR is
        appropriate).

    v2.1 starts F_h, F_t at identity and entity embeddings at
    one-hot (each entity at its own canonical basis vector). Adam-
    style updates on F_h, F_t and embeddings; embeddings can be
    frozen by setting ``learning_rate_embed = 0`` in a future API
    extension. Training is CPU-only; tested fast at our corpus size.
    """
    rng = np.random.default_rng(seed)
    sheaf = KnowledgeSheafV2.from_triples(triples, stalk_dim=stalk_dim)
    n_v = len(sheaf.vertices)
    n_r = len(sheaf.relations)
    d = stalk_dim

    # Initialize entity embeddings as one-hot (with d ≥ |V|), or
    # random unit vectors if d < |V|.
    if d >= n_v:
        embeddings = np.eye(n_v, d, dtype=np.float64)
    else:
        embeddings = rng.standard_normal((n_v, d))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9

    # Mutable copies for SGD updates.
    F_h = sheaf.F_h.copy()
    F_t = sheaf.F_t.copy()

    loss_history: list[float] = []
    for epoch in range(epochs):
        negatives = _sample_negative_triples(
            list(triples), n_negatives_per_positive, rng,
        )
        epoch_loss = 0.0
        n_terms = 0

        # Pair each positive with k negatives sharing the same head
        # and relation. (Simpler: pair each positive with its k
        # consecutive negatives in the sample.)
        for i, (h, r, t) in enumerate(triples):
            r_idx = sheaf.relation_index[r]
            h_idx = sheaf.vertex_index[h]
            t_idx = sheaf.vertex_index[t]

            # Per-edge residual for the positive triple
            res_pos = F_h[r_idx] @ embeddings[h_idx] - F_t[r_idx] @ embeddings[t_idx]
            v_pos = float(np.dot(res_pos, res_pos))

            # k negatives following this positive in the sample
            neg_slice = negatives[
                i * n_negatives_per_positive
                : (i + 1) * n_negatives_per_positive
            ]
            for (h_n, r_n, t_n) in neg_slice:
                rn_idx = sheaf.relation_index[r_n]
                hn_idx = sheaf.vertex_index[h_n]
                tn_idx = sheaf.vertex_index[t_n]
                res_neg = F_h[rn_idx] @ embeddings[hn_idx] - F_t[rn_idx] @ embeddings[tn_idx]
                v_neg = float(np.dot(res_neg, res_neg))

                gap = v_pos + margin - v_neg
                if gap > 0:
                    epoch_loss += gap
                    n_terms += 1
                    # Gradient of (v_pos + γ - v_neg):
                    #   ∂v_pos/∂F_h[r] = 2 res_pos x_h^T
                    #   ∂v_pos/∂F_t[r] = -2 res_pos x_t^T
                    #   ∂v_neg/∂F_h[r_n] = 2 res_neg x_{h_n}^T
                    #   ∂v_neg/∂F_t[r_n] = -2 res_neg x_{t_n}^T
                    F_h[r_idx]   -= learning_rate * 2.0 * np.outer(res_pos, embeddings[h_idx])
                    F_t[r_idx]   -= learning_rate * (-2.0) * np.outer(res_pos, embeddings[t_idx])
                    F_h[rn_idx]  -= learning_rate * (-2.0) * np.outer(res_neg, embeddings[hn_idx])
                    F_t[rn_idx]  -= learning_rate * 2.0 * np.outer(res_neg, embeddings[tn_idx])

        loss_history.append(epoch_loss / max(n_terms, 1))

    trained_sheaf = KnowledgeSheafV2.from_triples(
        list(triples), stalk_dim=d, F_h=F_h, F_t=F_t,
    )
    return trained_sheaf, embeddings, loss_history


# ─── Consistency profile (v2.1 envelope) ──────────────────────────────


def combined_detector_score(
    sheaf: KnowledgeSheafV2,
    embeddings: np.ndarray,
    rendered_triples: list[Triple],
    presence_weight: float = 0.05,
) -> dict:
    """v2.2 combined detector: Laplacian quadratic form + presence-
    deficit regularizer.

    Reasoning (added 2026-05-01 after PR #111's analytical
    realization that the Laplacian quadratic form fundamentally
    cannot detect presence-pattern issues, only cross-edge
    disagreement under the trained restriction maps):

      V_total = ‖δx‖² + λ · (presence_deficit)²

    where ``presence_deficit`` is the number of source vertices
    NOT mentioned in the render. The Laplacian term carries the
    relation-aware signal (catches A2 predicate-flip / A3 off-graph
    fabrication after training); the deficit term carries the
    presence-pattern signal (catches density-dropout, including
    on disconnected source graphs which the Laplacian alone
    cannot detect by design).

    The two terms are orthogonal signals — combining them is the
    publishable v2.2 artifact, not a workaround. ``presence_weight``
    (λ) is the relative weighting; default 0.05 was calibrated by
    inspection on the v2.1-falsification 4-fact disconnected-graph
    data (clean = 0.438 Laplacian-only; dropout = 0.327 Laplacian
    plus 4 deficit² × 0.05 = 0.527; combined detector flips sign
    correctly).
    """
    x = cochain_one_hot_v2(sheaf, rendered_triples, embedding=embeddings)
    laplacian_term = laplacian_quadratic_form_v2(sheaf, x)

    mentioned: set[str] = set()
    for s, _, o in rendered_triples:
        mentioned.add(s)
        mentioned.add(o)
    source_vertices = set(sheaf.vertices)
    missing = source_vertices - mentioned
    presence_deficit = len(missing)
    deficit_term = presence_weight * (presence_deficit ** 2)

    return {
        "v_combined": float(laplacian_term + deficit_term),
        "v_laplacian": float(laplacian_term),
        "v_deficit": float(deficit_term),
        "presence_deficit_count": presence_deficit,
        "missing_entities": sorted(missing),
        "presence_weight": presence_weight,
    }


def consistency_profile_v2(
    sheaf: KnowledgeSheafV2,
    embeddings: np.ndarray,
    rendered_extractions: list[list[Triple]],
) -> dict:
    """Mean & std of the v2.1 Laplacian quadratic form across a
    render manifold. Spec §3.3 v2.1 envelope.

    Returns ``None`` for scalar fields when the manifold is empty,
    matching the v1 ``consistency_profile`` honesty pattern (PR #109
    foot-gun fix).
    """
    if not rendered_extractions:
        return {
            "render_count": 0,
            "stalk_dim": sheaf.stalk_dim,
            "version": "v2.1-learned-restriction-maps",
            "mean_laplacian": None,
            "std_laplacian": None,
            "max_per_render": None,
            "argmax_render_idx": None,
            "per_render_v": [],
            "per_edge_top3_argmax_render": [],
        }

    per_render_v: list[float] = []
    per_render_localization: list[list[tuple[Triple, float]]] = []
    for triples_n in rendered_extractions:
        x_n = cochain_one_hot_v2(sheaf, triples_n, embedding=embeddings)
        v_n = laplacian_quadratic_form_v2(sheaf, x_n)
        per_render_v.append(v_n)
        per_render_localization.append(per_edge_discrepancy_v2(sheaf, x_n))

    arr = np.array(per_render_v)
    argmax_idx = int(arr.argmax())
    return {
        "render_count": len(rendered_extractions),
        "stalk_dim": sheaf.stalk_dim,
        "version": "v2.1-learned-restriction-maps",
        "mean_laplacian": float(arr.mean()),
        "std_laplacian": float(arr.std()),
        "max_per_render": float(arr.max()),
        "argmax_render_idx": argmax_idx,
        "per_render_v": per_render_v,
        "per_edge_top3_argmax_render": per_render_localization[argmax_idx][:3],
    }
