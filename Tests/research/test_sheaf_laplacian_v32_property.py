"""Property-based (Hypothesis) tests for v3.2 H17 / H18.

The companion file ``test_sheaf_laplacian_v32.py`` pins H16-H20 with
specific synthetic graphs (one example per claim). That's adequate
contract-fixing but weak for *universal* claims: H17 says "**for any**
sheaf with cross-partition edges and a non-trivial boundary change,
deviation_w differs"; H18 says "**for any** sheaf with L_IB = 0 and a
non-trivial boundary perturbation, v_laplacian_w differs."

Hypothesis lets us upgrade those from "tested with one example" to
"tested with ≥30 randomly generated graph + cochain combinations,
falsified once a counterexample appears." This is the falsifiability
discipline the substrate values: a property-based test is the
strongest version of the H17/H18 contract we can express in code.

The strategies are deliberately *narrow* — small stalk dim, small
graph, bounded float ranges. We're not benchmarking; we're proving
the *math claim* holds for the family of graphs the bench actually
encounters. Wider graphs would test the same property at higher
LAPACK noise and more numerical edge cases without a corresponding
information gain about the math.
"""
from __future__ import annotations

import pytest

# Skip if [research] extras (numpy) or hypothesis missing.
np = pytest.importorskip("numpy")
hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings, strategies as st

from sum_engine_internal.research.sheaf_laplacian_v2 import (
    KnowledgeSheafV2,
)
from sum_engine_internal.research.sheaf_laplacian_v3 import (
    boundary_deviation,
    weighted_laplacian_quadratic_form_v3,
)


# ─── Strategy helpers ─────────────────────────────────────────────────


def _gaussian_array(shape: tuple[int, ...], seed: int) -> "np.ndarray":
    """Deterministic Gaussian draws — Hypothesis controls `seed`."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape)


# Stalk dimensions in [2, 4] cover the math claim without inviting
# LAPACK threading variation that would obscure failures.
_stalk_dim = st.integers(min_value=2, max_value=4)
_seed = st.integers(min_value=0, max_value=2**31 - 1)
_weight = st.floats(min_value=0.1, max_value=2.0,
                    allow_nan=False, allow_infinity=False)


# ─── H17 universal: L_IB ≠ 0 → deviation_w changes under boundary perturbation ─


@given(d=_stalk_dim, x_seed=_seed, perturb_seed=_seed,
       w_b_to_i=_weight, w_i_to_i=_weight)
@settings(max_examples=30, deadline=None)
def test_h17_universal_l_ib_nonzero_deviation_visible(
    d, x_seed, perturb_seed, w_b_to_i, w_i_to_i,
):
    """For *any* 3-vertex chain sheaf with v0 on the boundary and edges
    (v0→v1), (v1→v2) crossing the partition, a non-trivial change to
    v0's stalk produces a non-zero Δdeviation_w.

    Sheaf topology pinned (chain v0-v1-v2 with v0 ∈ B, {v1,v2} ⊂ I)
    so L_IB ≠ 0 by construction. Embeddings, cochain values, and
    perturbation are all Hypothesis-generated.

    H17's universal-quantifier upgrade: instead of "tested on one
    8-dim example" (the H17 example test), we now claim "the math
    holds across 30 randomized stalk dims / cochains / perturbations
    / weights." Hypothesis falsifies if any case violates.
    """
    triples = [("v0", "r", "v1"), ("v1", "r", "v2")]
    sheaf = KnowledgeSheafV2.from_triples(triples, stalk_dim=d)
    boundary = [sheaf.vertex_index["v0"]]

    x_clean = _gaussian_array((3, d), x_seed)
    new_v0 = _gaussian_array((d,), perturb_seed)
    # Reject vanishingly improbable case where the random perturb
    # equals the clean stalk numerically (would be a vacuous test).
    if np.allclose(new_v0, x_clean[boundary[0]]):
        return

    x_perturbed = x_clean.copy()
    x_perturbed[boundary[0]] = new_v0

    weights = np.array([w_b_to_i, w_i_to_i], dtype=np.float64)

    dev_clean = boundary_deviation(sheaf, x_clean, boundary, weights=weights)
    dev_perturbed = boundary_deviation(
        sheaf, x_perturbed, boundary, weights=weights,
    )

    delta = abs(dev_perturbed["deviation"] - dev_clean["deviation"])
    assert delta > 1e-9, (
        f"H17 universal violation: deviation_w should change under "
        f"boundary perturbation when L_IB ≠ 0. Got "
        f"clean={dev_clean['deviation']}, "
        f"perturbed={dev_perturbed['deviation']}, "
        f"|Δ|={delta}. Hypothesis seed combo "
        f"(d={d}, x_seed={x_seed}, perturb_seed={perturb_seed})."
    )


# ─── H18 universal: L_IB = 0 → v_laplacian_w changes under boundary perturbation ─


@given(d=_stalk_dim, x_seed=_seed, perturb_seed=_seed,
       w_within_b=_weight, w_within_i=_weight)
@settings(max_examples=30, deadline=None)
def test_h18_universal_l_ib_zero_v_laplacian_w_visible(
    d, x_seed, perturb_seed, w_within_b, w_within_i,
):
    """For *any* 4-vertex sheaf with B={v0,v1}, I={v2,v3} and edges
    (v0→v1) ⊂ B and (v2→v3) ⊂ I (so L_IB = 0 by construction), a
    non-trivial change to v0's stalk produces a non-zero
    Δv_laplacian_w.

    This is the F3 fall-back guarantee at universal-quantifier
    strength: even when deviation is structurally blind (L_IB = 0),
    v_laplacian_w surfaces the perturbation by summing residuals
    over every edge — including the (v0→v1) edge that lives entirely
    within B.
    """
    triples = [("v0", "r", "v1"), ("v2", "r", "v3")]
    sheaf = KnowledgeSheafV2.from_triples(triples, stalk_dim=d)
    v0_idx = sheaf.vertex_index["v0"]

    x_clean = _gaussian_array((4, d), x_seed)
    new_v0 = _gaussian_array((d,), perturb_seed)
    if np.allclose(new_v0, x_clean[v0_idx]):
        return
    x_perturbed = x_clean.copy()
    x_perturbed[v0_idx] = new_v0

    weights = np.array([w_within_b, w_within_i], dtype=np.float64)

    v_clean = weighted_laplacian_quadratic_form_v3(sheaf, x_clean, weights)
    v_perturbed = weighted_laplacian_quadratic_form_v3(
        sheaf, x_perturbed, weights,
    )
    delta = abs(v_perturbed - v_clean)
    assert delta > 1e-9, (
        f"H18 universal violation: v_laplacian_w should change when "
        f"v0's stalk is perturbed (its incident edge (v0→v1) is in "
        f"the sum even with L_IB=0). Got clean={v_clean}, "
        f"perturbed={v_perturbed}, |Δ|={delta}."
    )


# ─── H16 universal: γ=0 ≡ v3 across random graph + cochain ─────────────


@given(d=_stalk_dim, x_seed=_seed,
       w0=_weight, w1=_weight, w2=_weight)
@settings(max_examples=20, deadline=None)
def test_h16_universal_gamma_zero_equals_v3_across_random_inputs(
    d, x_seed, w0, w1, w2,
):
    """v3.2 with γ=0 must equal v3 *for any* legitimate input. The
    example test pins this on three hand-picked graphs; this lifts
    it to the universal claim across randomly weighted toy sheaves.
    """
    from sum_engine_internal.research.sheaf_laplacian_v32 import (
        combined_detector_score_v32,
    )
    from sum_engine_internal.research.sheaf_laplacian_v3 import (
        combined_detector_score_v3,
    )

    triples = [
        ("alice", "knows", "bob"),
        ("bob", "owns", "dog"),
        ("carol", "writes", "python"),
    ]
    sheaf = KnowledgeSheafV2.from_triples(triples, stalk_dim=d)
    embeddings = _gaussian_array((len(sheaf.vertices), d), x_seed)
    weights = np.array([w0, w1, w2], dtype=np.float64)
    boundary = [0]  # arbitrary — γ=0 makes deviation contribution null

    v3_score = combined_detector_score_v3(
        sheaf, embeddings, triples, weights, lambda_deficit=0.05,
    )
    v32_score = combined_detector_score_v32(
        sheaf, embeddings, triples, weights,
        lambda_deficit=0.05, gamma_deviation=0.0,
        boundary_indices=boundary,
    )
    assert np.isclose(
        v32_score["v_combined_v32"], v3_score["v_combined_v3"],
    ), (
        f"H16 universal violation: v3.2(γ=0) must equal v3 for any "
        f"input. Got v3.2={v32_score['v_combined_v32']}, "
        f"v3={v3_score['v_combined_v3']}."
    )
