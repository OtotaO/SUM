"""v3.2 sheaf-Laplacian: combined detector closing the F3 STRUCTURAL FAIL.

PR #124's corpus-scale ROC bench surfaced F3 FAIL on v3.1's boundary
deviation — the standalone ``deviation`` field gave trusted-mean AUC
≈ 0.50 across the seed_long_paragraphs corpus. PR #125's 8-cell
diagnostic settled this as *structural*, not parametric: when the
per-doc graph has ``L_IB = 0`` (no edges crossing the trust frame's
boundary↔interior partition), the harmonic extension is independent
of ``x_B`` by linear algebra, so ``deviation`` is exactly invariant
under boundary-only perturbations. See ``docs/SHEAF_HALLUCINATION_
DETECTOR.md`` §3.4.3 for the full settling.

v3.2's response is a *strict generalization* of v3 that adds the
deviation signal as a complementary term:

    v_combined_v32 = v_laplacian_w + γ · deviation_w + λ · v_deficit

The two cochain-side terms catch complementary things:

  - **v_laplacian_w** (from v3): sums ``w_e · ‖F_h^(r) x_u −
    F_t^(r) x_v‖²`` over every edge. Catches cochain changes anywhere
    on the graph, including boundary perturbations regardless of
    ``L_IB`` topology.
  - **deviation_w** (from v3.1): ``‖x_I_actual − x_I^*‖²`` where
    ``x_I^* = -L_II^{-1} L_IB x_B`` is the harmonic extension of the
    boundary cochain into the interior. Catches *interior-vs-trust-
    frame inconsistency* — the system using its own trust artifacts
    (the boundary) to score consistency on parts it doesn't already
    trust (the interior). Only informative when ``L_IB ≠ 0``.

When ``γ = 0``, v3.2 reduces to v3 numerically (subsumption — the
H16 contract). When ``γ > 0``, deviation contributes additively where
it has signal; falls back to a constant where it's structurally
blind, so v_laplacian_w still carries the perturbation signal. The
combined score is informative either way — that's the F3 fall-back
guarantee (H18).

Falsifiable predictions (pinned in ``Tests/research/test_sheaf_laplacian_v32.py``):

  H16. **Subsumption.** ``γ_deviation = 0`` → v3.2 numerically equals
       v3. Strict generalization, not a different detector.

  H17. **L_IB ≠ 0 visibility.** On a graph with cross-partition
       edges, deviation_w changes under boundary perturbation.

  H18. **F3 fall-back.** On a graph with L_IB = 0, v_laplacian_w
       still surfaces the perturbation; the combined score is
       informative even when deviation is structurally blind.

  H19. **No λ double-counting.** The audit-tightening pattern
       (caught at the v3 layer in PR #123) doesn't reappear at
       the v3.2 wrapper.

  H20. **Degenerate-boundary fall-back.** Empty B or full B →
       deviation_w is 0 by convention; combined score reduces to v3.

Behind the same ``[research]`` extras flag as v1/v2/v3.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from sum_engine_internal.research.sheaf_laplacian_v2 import (
    KnowledgeSheafV2,
    Triple,
    cochain_one_hot_v2,
)
from sum_engine_internal.research.sheaf_laplacian_v3 import (
    boundary_deviation,
    combined_detector_score_v3,
)


def combined_detector_score_v32(
    sheaf: KnowledgeSheafV2,
    embeddings: np.ndarray,
    render_triples: list[Triple],
    weights: np.ndarray,
    *,
    lambda_deficit: float = 0.05,
    gamma_deviation: float = 1.0,
    boundary_indices: Iterable[int] | None = None,
) -> dict:
    """v3.2 combined detector: weighted Laplacian + deviation + deficit.

    Strict generalization of ``combined_detector_score_v3``. With
    ``gamma_deviation = 0``, output["v_combined_v32"] equals
    ``output["v_combined_v3"]`` (the H16 subsumption contract).

    Parameters
    ----------
    sheaf, embeddings, render_triples, weights, lambda_deficit
        Same semantics as v3's ``combined_detector_score_v3``.
    gamma_deviation
        Coefficient on the harmonic-extension deviation term.
        Default 1.0; pass 0.0 to recover v3 exactly. Calibration
        strategy is the caller's responsibility — bench scripts
        typically set it to mean(v_laplacian_w_clean) /
        mean(deviation_w_clean) so the two terms contribute
        comparably.
    boundary_indices
        Vertex indices forming the boundary B. If None, an empty
        boundary is used (deviation_w = 0, behaviour identical to
        v3 at that input). Bench scripts derive this from
        :func:`sheaf_laplacian_v3.boundary_from_weights`.

    Returns
    -------
    dict with v3's keys plus:
        - deviation_w: ‖x_I_actual − x_I^*‖² (the harmonic-extension
          deviation under the weighted Laplacian)
        - v_combined_v32: v_laplacian_w + γ · deviation_w + v_deficit
        - boundary_indices: list[int] of vertex indices on B
        - interior_size: |I|

    The deviation term uses the *weighted* harmonic extension
    (passing ``weights`` through to ``boundary_deviation``), so the
    signal lives in the same metric as v_laplacian_w.
    """
    base_v3 = combined_detector_score_v3(
        sheaf, embeddings, render_triples, weights,
        lambda_deficit=lambda_deficit,
    )

    # Boundary: caller-supplied or empty (graceful fall-back).
    if boundary_indices is None:
        boundary_list: list[int] = []
    else:
        boundary_list = list(boundary_indices)

    # Construct the same cochain v3 used (cochain_one_hot_v2 with
    # the trained embeddings). The deviation primitive operates on
    # the full |V| × d cochain.
    x_full = cochain_one_hot_v2(sheaf, render_triples, embedding=embeddings)

    if not boundary_list or len(boundary_list) == len(sheaf.vertices):
        # Degenerate partition: deviation undefined or trivial. Fall
        # back to deviation_w = 0 (H20 contract). v_combined_v32 then
        # equals v_laplacian_w + v_deficit (= v3) by construction.
        deviation_w = 0.0
        interior_size = (
            len(sheaf.vertices) if not boundary_list else 0
        )
    else:
        dev_result = boundary_deviation(
            sheaf, x_full, boundary_list, weights=weights,
        )
        deviation_w = float(dev_result["deviation"])
        interior_size = int(dev_result["interior_size"])

    # H19: combined score is the additive sum. v_deficit already
    # carries λ baked in by v2.2 (same audit-tightened convention as
    # v3 — see PR #123). Adding gamma_deviation · deviation_w is the
    # ONLY new arithmetic introduced by v3.2.
    v_combined_v32 = (
        base_v3["v_laplacian_w"]
        + gamma_deviation * deviation_w
        + base_v3["v_deficit"]
    )

    return {
        **base_v3,
        "deviation_w": deviation_w,
        "v_combined_v32": v_combined_v32,
        "boundary_indices": list(boundary_list),
        "interior_size": interior_size,
        "gamma_deviation": gamma_deviation,
    }
