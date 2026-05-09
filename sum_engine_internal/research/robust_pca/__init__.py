"""Robust PCA via Principal Component Pursuit — Phase A (per the
deep-research article §9.1).

Decomposes an axiom-feature matrix M ∈ ℝ^{n × d} into a low-rank
consensus L₀ and a sparse corruption residual S₀:

    min_{L,S}  ‖L‖_∗ + λ ‖S‖_1    s.t.   M = L + S

with λ = 1/√(max(n,d)) per Candès, Li, Ma & Wright (JACM 58(3):11,
2011). Under incoherence on L₀ and a uniform-random support model
on S₀, the convex program **exactly recovers** (L₀, S₀) with high
probability — that's the **[provable]** core kernel.

Application to SUM: rows with high ‖S₀_i‖_1 are corruption-flagged
axioms (the `seed_v2 doc_015` failure class). Substrate-shaped
output: a `corruption_score` per axiom row that's directly usable
as a signal for downstream gating, plus the recovered low-rank L₀
for follow-on analysis (e.g., column-similarity for
duplicate-predicate detection — `seed_long solar_system` class).
"""
from sum_engine_internal.research.robust_pca.pcp import (
    pcp,
    corruption_score,
    PCPResult,
    PCPConvergenceError,
)
from sum_engine_internal.research.robust_pca.axiom_embedding import (
    embed_triple,
    embed_triples,
)

__all__ = [
    "pcp", "corruption_score", "PCPResult", "PCPConvergenceError",
    "embed_triple", "embed_triples",
]
