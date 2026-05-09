"""Von Neumann graph entropy — single-scalar drift detector for the
axiom graph.

Defines a density matrix ρ over the normalized graph Laplacian
and computes its von Neumann entropy

    S(ρ) = -Σ λ_i log λ_i

over the eigenvalues λ_i of ρ. This is the spectral analogue of
Shannon entropy for the graph's structural complexity. Drift in
S(ρ) over bundle versions is a structural-shift detector
*orthogonal* to the existing sheaf-Laplacian hallucination check
— same spectral substrate, different summary statistic.

Provable kernel: De Domenico & Biamonte, *Phys. Rev. X* 6:041062
(2016), "Spectral entropies as information-theoretic tools for
complex network comparison." Builds on the von Neumann entropy
(Phys. Math. der Quantenmechanik 1932) interpretation of
graph-Laplacian density matrices.

The entropy is a single scalar in [0, log(N-1)] for an N-node
graph; cheaper to compute, ship in receipts, alert on, than any
multi-eigenvalue summary. Substrate use: monitor S across nightly
bundle snapshots; alert on |ΔS| > 2σ as a tripwire for topical /
structural drift.
"""
from sum_engine_internal.research.spectral_entropy.vn_entropy import (
    build_axiom_graph,
    normalized_laplacian,
    density_matrix,
    von_neumann_entropy,
    graph_entropy,
)

__all__ = [
    "build_axiom_graph",
    "normalized_laplacian",
    "density_matrix",
    "von_neumann_entropy",
    "graph_entropy",
]
