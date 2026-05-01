"""v1 sheaf-Laplacian hallucination detector.

Faithful to docs/SHEAF_HALLUCINATION_DETECTOR.md §3.2 (v1, 1-dim
presence stalks). Math grounded in Gebhart, Hansen & Schrater
(2023, AISTATS, arXiv:2110.03789) Equation 1 and the sheaf-Laplacian
theory of Hansen & Ghrist (2019).

Every function below references the spec equation it implements.
The v1 detector lives behind the ``[research]`` extras flag in
``pyproject.toml`` — it is not on the production install path.

v1 known blindspots (verified empirically by the synthetic
micro-benchmark in scripts/research/sheaf_microbench.py and pinned
by Tests/research/test_sheaf_laplacian.py):

  - Predicate-flip perturbations (A2): invisible. Presence stalks
    do not carry predicate information. v2 (learned-embedding stalks)
    is required for predicate-sensitive detection.
  - Off-graph fact-fabrication (A3): invisible. Entities not in the
    source vertex set are silently ignored by ``cochain_from_extracted``.
    v2 is required to flag fabricated entities.
  - Empty-render false negative: a render that extracts zero triples
    yields x = 0, hence x^T L x = 0 — the same score as a perfectly
    consistent render. Callers should treat n_extracted == 0 as a
    separate signal, not rely on the Laplacian alone.

v1 verified-positive detection classes (6/6 detect rate, 100%
top-1 localization accuracy on the synthetic micro-benchmark):

  - Entity-swap (A1): one source entity replaced.
  - Triple-drop (A4): one triple omitted; isolated endpoints vanish.
  - Consistent-entity-swap (A5): the SAME swap applied across the
    full render manifold. Caught by the *mean* Laplacian even though
    per-render variance is zero. (The spec originally
    mischaracterized this as a v1 blindspot; corrected in §6 to
    distinguish A5-via-swap (caught) from A5-via-predicate-flip
    (missed).)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


Triple = tuple[str, str, str]


@dataclass(frozen=True)
class KnowledgeSheaf:
    """A cellular sheaf on a knowledge graph (Gebhart et al. 2023, Def. 4).

    For v1 (1-dim presence stalks):
      F(v) = R for every vertex v
      F(e) = R for every edge e
      F_h⊵_h r = F_t⊵_t r = 1  (identity on R)

    The vertex order is fixed at construction so cochains are
    indexable as plain numpy arrays.
    """
    vertices: tuple[str, ...]
    edges: tuple[Triple, ...]            # (subject, predicate, object)
    vertex_index: dict[str, int]         # name → index into cochain vector
    stalk_dim: int                       # = 1 for v1

    @classmethod
    def from_triples(cls, triples: list[Triple], stalk_dim: int = 1) -> "KnowledgeSheaf":
        seen: dict[str, None] = {}
        for s, _, o in triples:
            seen.setdefault(s, None)
            seen.setdefault(o, None)
        vertices = tuple(seen)
        return cls(
            vertices=vertices,
            edges=tuple(triples),
            vertex_index={v: i for i, v in enumerate(vertices)},
            stalk_dim=stalk_dim,
        )


def coboundary_matrix(sheaf: KnowledgeSheaf) -> np.ndarray:
    """δ : C^0 → C^1 as a matrix.

    Spec §2.1 / Gebhart Def. 4: for an edge e: u → v with identity
    restriction maps, (δx)_e = x_v - x_u. Each row of δ has
    exactly two non-zeros: -1 at u's column, +1 at v's column.

    Shape: (|E|, |V|) for stalk_dim=1; (|E|*d, |V|*d) when d>1
    using a Kronecker product. v1 uses d=1.
    """
    if sheaf.stalk_dim != 1:
        raise NotImplementedError("v1 supports stalk_dim=1 only")
    n_v = len(sheaf.vertices)
    n_e = len(sheaf.edges)
    delta = np.zeros((n_e, n_v), dtype=np.float64)
    for i, (s, _, o) in enumerate(sheaf.edges):
        delta[i, sheaf.vertex_index[s]] = -1.0   # F_u⊵e applied to head
        delta[i, sheaf.vertex_index[o]] = +1.0   # F_v⊵e applied to tail
    return delta


def sheaf_laplacian(sheaf: KnowledgeSheaf) -> np.ndarray:
    """L_F = δ^T δ (Gebhart Eq. 1; Hansen-Ghrist 2019)."""
    delta = coboundary_matrix(sheaf)
    return delta.T @ delta


def laplacian_quadratic_form(sheaf: KnowledgeSheaf, x: np.ndarray) -> float:
    """x^T L_F x. Zero iff x is a global section.

    For v1 1-dim stalks, x ∈ R^|V|; this measures sum over edges of
    (x_v - x_u)^2.
    """
    L = sheaf_laplacian(sheaf)
    return float(x @ L @ x)


def per_edge_discrepancy(sheaf: KnowledgeSheaf, x: np.ndarray) -> list[tuple[Triple, float]]:
    """Per-edge contribution to the Laplacian quadratic form.

    Used for the localization claim (P2 in the spec). Returns
    [(edge, |F_v⊵e x_v - F_u⊵e x_u|^2), ...] sorted descending.
    """
    delta = coboundary_matrix(sheaf)
    edge_residual = delta @ x          # shape (|E|,)
    contribs = [
        (sheaf.edges[i], float(edge_residual[i] ** 2))
        for i in range(len(sheaf.edges))
    ]
    contribs.sort(key=lambda kv: kv[1], reverse=True)
    return contribs


def cochain_from_extracted(
    sheaf: KnowledgeSheaf,
    extracted_triples: list[Triple],
) -> np.ndarray:
    """Build the 0-cochain x ∈ C^0(G; F_1d) from a render's
    re-extracted triples (spec §3.2 step 3b).

    x_v = 1 if v appears as subject or object in extracted_triples,
    else 0. This is the 1-dim presence indicator.
    """
    mentioned: set[str] = set()
    for s, _, o in extracted_triples:
        mentioned.add(s)
        mentioned.add(o)
    x = np.zeros(len(sheaf.vertices), dtype=np.float64)
    for v in mentioned:
        if v in sheaf.vertex_index:
            x[sheaf.vertex_index[v]] = 1.0
    return x


def consistency_profile(
    source_triples: list[Triple],
    rendered_extractions: list[list[Triple]],
) -> dict:
    """Mean & std of the Laplacian quadratic form across a render
    manifold. Spec §3.2 step 5.

    rendered_extractions: a list, one entry per rendering, each a
    list of (s, p, o) triples extracted from that rendering.

    Returns the consistency profile envelope shape from spec §3.5
    (without the receipt-binding parts, which are v3).
    """
    sheaf = KnowledgeSheaf.from_triples(source_triples, stalk_dim=1)

    per_render_v: list[float] = []
    per_render_localization: list[list[tuple[Triple, float]]] = []
    for triples_n in rendered_extractions:
        x_n = cochain_from_extracted(sheaf, triples_n)
        v_n = laplacian_quadratic_form(sheaf, x_n)
        per_render_v.append(v_n)
        per_render_localization.append(per_edge_discrepancy(sheaf, x_n))

    arr = np.array(per_render_v) if per_render_v else np.array([0.0])
    return {
        "render_count": len(rendered_extractions),
        "stalk_dim": sheaf.stalk_dim,
        "version": "v1-presence-stalks",
        "mean_laplacian": float(arr.mean()),
        "std_laplacian": float(arr.std()),
        "max_per_render": float(arr.max()),
        "argmax_render_idx": int(arr.argmax()),
        "per_render_v": per_render_v,
        "per_edge_top3_argmax_render": per_render_localization[
            int(arr.argmax())
        ][:3],
    }
