"""Math + falsifiability tests for v3 receipt-weighted detector.

Three blocks:

  Block 1 (math sanity): weighted Laplacian is symmetric, PSD,
    reduces to v2 when weights are uniform, and the factored form
    matches the materialized form numerically.

  Block 2 (weights_from_receipts contract): trusted/revoked/default
    partitioning produces the documented weight vector.

  Block 3 (falsifiable predictions H1-H4 from the v3 module
    docstring): linearity in weights, zero-weight kills contribution,
    revocation overrides trust, weighted localization.

  Block 4 (utility): tampering a trusted edge produces a sharper V
    jump than tampering an unsigned edge — this is what makes
    receipt-weighting *useful*, not just well-defined.
"""
from __future__ import annotations

import pytest

# Skip entire file if [research] extras aren't installed.
np = pytest.importorskip("numpy")

from sum_engine_internal.research.sheaf_laplacian_v2 import (
    KnowledgeSheafV2,
    cochain_one_hot_v2,
    laplacian_quadratic_form_v2,
    sheaf_laplacian_v2,
    train_restriction_maps,
)
from sum_engine_internal.research.sheaf_laplacian_v3 import (
    combined_detector_score_v3,
    weighted_laplacian_quadratic_form_v3,
    weighted_per_edge_discrepancy_v3,
    weighted_sheaf_laplacian_v3,
    weights_from_receipts,
)


# ─── Block 1: math sanity ────────────────────────────────────────────


def _toy_sheaf_v3(d: int = 8):
    triples = [
        ("alice", "knows", "bob"),
        ("bob", "owns", "dog"),
        ("carol", "writes", "python"),
    ]
    return KnowledgeSheafV2.from_triples(triples, stalk_dim=d), triples


def test_weighted_laplacian_is_symmetric():
    sheaf, _ = _toy_sheaf_v3()
    weights = np.array([1.0, 0.5, 0.1])
    L = weighted_sheaf_laplacian_v3(sheaf, weights)
    assert np.allclose(L, L.T), "L_F^w must be symmetric (still δ^T W δ)"


def test_weighted_laplacian_is_positive_semidefinite():
    sheaf, _ = _toy_sheaf_v3()
    weights = np.array([1.0, 0.5, 0.1])
    L = weighted_sheaf_laplacian_v3(sheaf, weights)
    eigvals = np.linalg.eigvalsh(L)
    assert np.all(eigvals >= -1e-9), f"L_F^w must be PSD; got eigvals={eigvals}"


def test_uniform_weights_reduce_to_v2_scaled():
    """Sanity: with uniform weights w_e = c, L^w = c · L_v2 (and hence
    x^T L^w x = c · x^T L_v2 x). v3 is a strict generalization of v2."""
    sheaf, _ = _toy_sheaf_v3()
    L_v2 = sheaf_laplacian_v2(sheaf)
    for c in [0.5, 1.0, 2.0]:
        weights = np.full(len(sheaf.edges), c)
        L_w = weighted_sheaf_laplacian_v3(sheaf, weights)
        assert np.allclose(L_w, c * L_v2), (
            f"uniform weight {c} should give L^w = {c} · L_v2"
        )


def test_factored_form_matches_full_laplacian():
    """The factored ``Σ_e w_e · ‖residual_e‖²`` form must equal
    ``x^T L^w x`` to floating-point precision. The implementation
    avoids materializing L^w for performance; this test pins the
    equivalence."""
    sheaf, _ = _toy_sheaf_v3()
    rng = np.random.default_rng(42)
    weights = np.array([0.3, 1.7, 0.5])
    L_w = weighted_sheaf_laplacian_v3(sheaf, weights)
    for _ in range(10):
        x = rng.standard_normal((len(sheaf.vertices), sheaf.stalk_dim))
        v_factored = weighted_laplacian_quadratic_form_v3(sheaf, x, weights)
        v_full = float(x.flatten() @ L_w @ x.flatten())
        assert np.isclose(v_factored, v_full), (
            f"factored {v_factored} != full {v_full}"
        )


def test_negative_weights_rejected():
    """Hansen-Ghrist §3.2's PSD claim requires non-negative weights.
    Construction must reject negative weights at the boundary."""
    sheaf, _ = _toy_sheaf_v3()
    bad = np.array([1.0, -0.5, 0.1])
    with pytest.raises(ValueError, match="must be non-negative"):
        weighted_sheaf_laplacian_v3(sheaf, bad)


def test_wrong_weights_shape_rejected():
    """Defensive: weights vector must be exactly (|E|,). Wrong
    shape is an early-failure error so downstream V isn't silently
    wrong."""
    sheaf, _ = _toy_sheaf_v3()
    bad = np.array([1.0, 0.5])  # only 2 weights for 3 edges
    with pytest.raises(ValueError, match="weights shape must be"):
        weighted_laplacian_quadratic_form_v3(
            sheaf, np.zeros((len(sheaf.vertices), sheaf.stalk_dim)), bad,
        )


# ─── Block 2: weights_from_receipts contract ─────────────────────────


def test_weights_from_receipts_default_when_no_partition():
    """No trusted/revoked partition → every edge gets default_weight."""
    sheaf, triples = _toy_sheaf_v3()
    w = weights_from_receipts(sheaf)
    assert w.shape == (len(sheaf.edges),)
    assert np.all(w == 0.1)  # default_weight default


def test_weights_from_receipts_trusted_set_lifted():
    sheaf, triples = _toy_sheaf_v3()
    w = weights_from_receipts(sheaf, trusted_edges=[triples[0], triples[2]])
    # triples[0] and triples[2] trusted → 1.0; triples[1] default → 0.1
    assert w[0] == 1.0
    assert w[1] == 0.1
    assert w[2] == 1.0


def test_weights_from_receipts_revoked_set_zeroed():
    sheaf, triples = _toy_sheaf_v3()
    w = weights_from_receipts(sheaf, revoked_edges=[triples[1]])
    assert w[0] == 0.1
    assert w[1] == 0.0   # revoked
    assert w[2] == 0.1


def test_weights_from_receipts_revocation_overrides_trust():
    """An edge in BOTH trusted and revoked sets must resolve to
    revoked. Once a key is revoked, its prior signatures are no
    longer load-bearing — the trust loop's own discipline is that
    revocation is a kill-switch, not a downgrade."""
    sheaf, triples = _toy_sheaf_v3()
    w = weights_from_receipts(
        sheaf,
        trusted_edges=[triples[0]],
        revoked_edges=[triples[0]],
    )
    assert w[0] == 0.0


def test_weights_from_receipts_rejects_negative_weights():
    sheaf, _ = _toy_sheaf_v3()
    with pytest.raises(ValueError, match="non-negative"):
        weights_from_receipts(sheaf, default_weight=-0.5)


def test_weights_from_receipts_indexed_parallel_to_edges():
    """weights[i] must correspond to sheaf.edges[i]. Pinning the
    parallel-indexing contract so a future refactor that reorders
    edges-vs-weights breaks loudly."""
    sheaf, triples = _toy_sheaf_v3()
    # Pick the second edge as trusted; the weight at index 1 must be 1.0.
    w = weights_from_receipts(sheaf, trusted_edges=[sheaf.edges[1]])
    assert w[1] == 1.0
    assert w[0] == 0.1
    assert w[2] == 0.1


# ─── Block 3: falsifiable predictions H1-H4 ──────────────────────────


def test_h1_doubling_weights_doubles_quadratic_form():
    """H1: x^T L^w x is linear in the weights. Doubling all weights
    doubles V."""
    sheaf, triples = _toy_sheaf_v3()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((len(sheaf.vertices), sheaf.stalk_dim))
    w = np.array([0.3, 1.7, 0.5])
    v1 = weighted_laplacian_quadratic_form_v3(sheaf, x, w)
    v2 = weighted_laplacian_quadratic_form_v3(sheaf, x, 2 * w)
    assert np.isclose(v2, 2 * v1), (
        f"V should be linear in weights; v(w)={v1}, v(2w)={v2}"
    )


def test_h1_zero_weight_kills_edge_contribution():
    """H1: setting one edge's weight to 0 zeros that edge's
    contribution exactly. Pin: V(weights with edge i zeroed) =
    V(full weights) − w_i_original · ‖residual_i‖²."""
    sheaf, triples = _toy_sheaf_v3()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((len(sheaf.vertices), sheaf.stalk_dim))
    w_full = np.array([1.0, 1.0, 1.0])

    from sum_engine_internal.research.sheaf_laplacian_v2 import per_edge_residual_v2
    residuals = per_edge_residual_v2(sheaf, x)
    edge_idx = 1
    expected_drop = float(np.sum(residuals[edge_idx] * residuals[edge_idx]))

    w_zeroed = w_full.copy()
    w_zeroed[edge_idx] = 0.0

    v_full = weighted_laplacian_quadratic_form_v3(sheaf, x, w_full)
    v_zeroed = weighted_laplacian_quadratic_form_v3(sheaf, x, w_zeroed)
    assert np.isclose(v_full - v_zeroed, expected_drop), (
        f"zeroing edge {edge_idx}'s weight should drop V by "
        f"its residual contribution {expected_drop}; "
        f"actual drop = {v_full - v_zeroed}"
    )


def test_h2_uniform_weights_v3_equals_scaled_v2():
    """H2: with uniform weights c, V_v3(x; w=c) = c · V_v2(x).
    Already proven at the matrix level (test_uniform_weights_reduce_
    to_v2_scaled); this test pins it at the quadratic-form level."""
    sheaf, _ = _toy_sheaf_v3()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((len(sheaf.vertices), sheaf.stalk_dim))
    v_v2 = laplacian_quadratic_form_v2(sheaf, x)
    for c in [0.5, 1.0, 2.0, 7.3]:
        weights = np.full(len(sheaf.edges), c)
        v_v3 = weighted_laplacian_quadratic_form_v3(sheaf, x, weights)
        assert np.isclose(v_v3, c * v_v2), (
            f"v3 with uniform c={c} should give c·v2; "
            f"v_v2={v_v2}, v_v3={v_v3}"
        )


def test_h3_per_edge_contribution_scales_with_weight():
    """H3: per-edge weighted contribution = w_e · ‖residual_e‖² in
    the v3 ranker. Pin so a future refactor that drops the weight
    factor in localization gets caught."""
    sheaf, triples = _toy_sheaf_v3()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((len(sheaf.vertices), sheaf.stalk_dim))

    from sum_engine_internal.research.sheaf_laplacian_v2 import per_edge_residual_v2
    residuals = per_edge_residual_v2(sheaf, x)
    raw_contribs = [float(np.sum(r * r)) for r in residuals]

    weights = np.array([0.1, 1.0, 0.5])
    contribs = weighted_per_edge_discrepancy_v3(sheaf, x, weights)

    # Build expected: edge_i → w_i · raw_contribs_i, then sort desc by score
    expected = sorted(
        [(sheaf.edges[i], weights[i] * raw_contribs[i]) for i in range(3)],
        key=lambda kv: kv[1],
        reverse=True,
    )
    for (e_got, v_got), (e_exp, v_exp) in zip(contribs, expected):
        assert e_got == e_exp
        assert np.isclose(v_got, v_exp)


def test_h4_weights_from_receipts_deterministic_and_parallel():
    """H4: weights_from_receipts is deterministic (no random
    sampling) and produces a vector parallel-indexed to
    sheaf.edges. Two calls with the same inputs must produce
    byte-identical outputs."""
    sheaf, triples = _toy_sheaf_v3()
    w1 = weights_from_receipts(sheaf, trusted_edges=[triples[0], triples[2]])
    w2 = weights_from_receipts(sheaf, trusted_edges=[triples[0], triples[2]])
    assert np.array_equal(w1, w2)


# ─── Block 4: utility — receipt-weighting actually helps ─────────────


def test_tampering_trusted_edge_yields_sharper_v_jump_than_untrusted():
    """The headline utility claim: tampering a trusted (high-weight)
    edge produces a sharper V jump than tampering an untrusted
    (low-weight) edge — that is the *point* of receipt-weighting.

    Setup: 3-triple sheaf with 2 trusted and 1 untrusted edge. Train
    embeddings; build the clean cochain. Then tamper one trusted
    edge's contribution (perturb its head vertex's embedding) and
    one untrusted edge's contribution. Compare ΔV.

    If ΔV(trusted-tampered) > ΔV(untrusted-tampered), receipt-
    weighting amplifies signal where the system already trusts;
    that's the legendary-status-utility claim. If the inequality
    inverts, v3 is mathematically well-defined but useless — file
    a falsification and rethink.
    """
    triples = [
        ("alice", "knows", "bob"),    # trusted (will be tampered)
        ("bob", "knows", "carol"),    # trusted (control)
        ("carol", "owns", "dog"),     # untrusted (will be tampered)
    ]
    trained, embeddings, _ = train_restriction_maps(
        triples, stalk_dim=8, epochs=200, seed=0,
    )

    # Receipt-derived weights: first two edges trusted, third unsigned.
    weights = weights_from_receipts(
        trained,
        trusted_edges=[triples[0], triples[1]],
    )
    # weights = [1.0, 1.0, 0.1]

    x_clean = cochain_one_hot_v2(trained, triples, embedding=embeddings)
    v_clean = weighted_laplacian_quadratic_form_v3(trained, x_clean, weights)

    rng = np.random.default_rng(0)
    perturbation = rng.standard_normal(trained.stalk_dim) * 0.5

    # Tamper the trusted edge: perturb alice's embedding.
    x_trusted_tampered = x_clean.copy()
    x_trusted_tampered[trained.vertex_index["alice"]] += perturbation
    v_trusted_tampered = weighted_laplacian_quadratic_form_v3(
        trained, x_trusted_tampered, weights,
    )

    # Tamper the untrusted edge: perturb dog's embedding by the SAME magnitude.
    x_untrusted_tampered = x_clean.copy()
    x_untrusted_tampered[trained.vertex_index["dog"]] += perturbation
    v_untrusted_tampered = weighted_laplacian_quadratic_form_v3(
        trained, x_untrusted_tampered, weights,
    )

    delta_trusted = v_trusted_tampered - v_clean
    delta_untrusted = v_untrusted_tampered - v_clean

    assert delta_trusted > delta_untrusted, (
        f"v3 receipt-weighting must amplify signal at trusted edges; "
        f"got Δ_trusted={delta_trusted:.4f}, Δ_untrusted={delta_untrusted:.4f}. "
        f"If Δ_trusted ≤ Δ_untrusted, v3 is mathematically well-defined "
        f"but provides no utility benefit over v2 — file a falsification "
        f"PR and reconsider the weight design."
    )


# ─── Block 5: combined detector v3 shape ─────────────────────────────


def test_combined_detector_v3_shape_matches_v2_2_plus_extras():
    """Combined v3 returns the same keys as v2.2's combined detector
    plus ``v_laplacian_w`` and ``edge_weights``. Future consumers
    can swap v2.2 → v3 by changing the import."""
    triples = [("alice", "knows", "bob"), ("bob", "owns", "dog")]
    trained, embeddings, _ = train_restriction_maps(
        triples, stalk_dim=8, epochs=100, seed=0,
    )
    weights = weights_from_receipts(trained, trusted_edges=[triples[0]])
    score = combined_detector_score_v3(trained, embeddings, triples, weights)

    # v2.2's keys all carried through
    for key in ("v_laplacian", "v_deficit", "v_combined",
                "presence_deficit_count", "missing_entities"):
        assert key in score, f"v3 must carry v2.2 key {key!r}; got {sorted(score)}"

    # v3-specific keys
    assert "v_laplacian_w" in score
    assert "v_combined_v3" in score
    assert "edge_weights" in score
    assert score["edge_weights"] == weights.tolist()


def test_combined_v3_clean_render_no_signal():
    """A render that mentions every source entity must score
    deficit=0; the combined v3 V then equals the weighted Laplacian
    term alone. Negative-control mirroring v2.2's
    test_v2_2_combined_detector_no_signal_on_clean_render."""
    triples = [("alice", "knows", "bob"), ("bob", "owns", "dog")]
    trained, embeddings, _ = train_restriction_maps(
        triples, stalk_dim=8, epochs=200, seed=0,
    )
    weights = weights_from_receipts(trained, trusted_edges=list(triples))
    score = combined_detector_score_v3(trained, embeddings, triples, weights)
    assert score["presence_deficit_count"] == 0
    assert score["v_deficit"] == 0.0
    # combined_v3 = v_laplacian_w + λ · 0 = v_laplacian_w
    assert np.isclose(score["v_combined_v3"], score["v_laplacian_w"])
