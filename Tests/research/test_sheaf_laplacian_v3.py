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


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_h1_quadratic_form_linear_in_weights_with_known_ratios(seed):
    """H1: x^T L^w x is linear in the weights. Pinned across four
    independent properties of linearity, on multiple seeds:

      (a) doubling weights doubles V (homogeneity);
      (b) all-zero weights give V = 0 (no contribution);
      (c) selecting only edge i (weight vector = e_i) gives
          V = ‖residual_i‖² exactly;
      (d) sum-of-singletons equals sum-of-V (additivity).

    A buggy implementation that ignored per-edge structure (e.g.
    `np.sum(weights) * constant`) would pass (a) but fail (c)
    and (d). Multiple seeds prevent the test passing for
    fixture-specific numeric coincidence at seed=0.
    """
    sheaf, _ = _toy_sheaf_v3()
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((len(sheaf.vertices), sheaf.stalk_dim))

    from sum_engine_internal.research.sheaf_laplacian_v2 import per_edge_residual_v2
    residuals = per_edge_residual_v2(sheaf, x)
    raw_per_edge = np.array([float(np.sum(r * r)) for r in residuals])

    # (a) homogeneity
    w = np.array([0.3, 1.7, 0.5])
    v1 = weighted_laplacian_quadratic_form_v3(sheaf, x, w)
    v2 = weighted_laplacian_quadratic_form_v3(sheaf, x, 2 * w)
    assert np.isclose(v2, 2 * v1), f"homogeneity: v(2w) != 2 v(w)"

    # (b) zero weights
    v_zero = weighted_laplacian_quadratic_form_v3(
        sheaf, x, np.zeros(len(sheaf.edges)),
    )
    assert v_zero == 0.0, f"zero weights must give V=0; got {v_zero}"

    # (c) singleton weight = e_i picks out edge i exactly
    for i in range(len(sheaf.edges)):
        e_i = np.zeros(len(sheaf.edges))
        e_i[i] = 1.0
        v_single = weighted_laplacian_quadratic_form_v3(sheaf, x, e_i)
        assert np.isclose(v_single, raw_per_edge[i]), (
            f"singleton w=e_{i} should give V = ‖residual_{i}‖² "
            f"= {raw_per_edge[i]}; got {v_single}"
        )

    # (d) additivity: V(w1 + w2) = V(w1) + V(w2)
    w1 = np.array([1.0, 0.0, 0.5])
    w2 = np.array([0.2, 0.8, 0.0])
    v_sum = weighted_laplacian_quadratic_form_v3(sheaf, x, w1 + w2)
    v1_only = weighted_laplacian_quadratic_form_v3(sheaf, x, w1)
    v2_only = weighted_laplacian_quadratic_form_v3(sheaf, x, w2)
    assert np.isclose(v_sum, v1_only + v2_only), (
        f"additivity: V(w1+w2)={v_sum} != V(w1)+V(w2)={v1_only + v2_only}"
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
    factor in localization gets caught.

    Tightened 2026-05-02 to avoid the audit-flagged tautology:
    instead of comparing to ``w_i * raw_contrib_i`` (same formula
    the implementation uses → any wrong implementation that mults
    in the same order passes), pin against a SENTINEL weight
    vector where the expected value is hand-known regardless of
    the implementation's order of operations:

      - With weights = [0, 1, 0]: contribs[0] = (edges[1],
        raw_contribs_1) exactly, contribs[1] = (edges[0], 0),
        contribs[2] = (edges[2], 0). Any localization that
        respects "weight zero kills contribution; weight one
        preserves it raw" passes.
    """
    sheaf, _ = _toy_sheaf_v3()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((len(sheaf.vertices), sheaf.stalk_dim))

    from sum_engine_internal.research.sheaf_laplacian_v2 import per_edge_residual_v2
    residuals = per_edge_residual_v2(sheaf, x)
    raw_contribs = [float(np.sum(r * r)) for r in residuals]

    # Sentinel: only edge 1 contributes
    weights = np.array([0.0, 1.0, 0.0])
    contribs = weighted_per_edge_discrepancy_v3(sheaf, x, weights)

    # Top contrib must be edge 1 with score = raw_contrib_1 (the
    # only non-zero edge); the other two must score exactly 0.
    assert contribs[0][0] == sheaf.edges[1]
    assert np.isclose(contribs[0][1], raw_contribs[1])
    # The remaining two edges must score exactly zero
    zero_edges = {contribs[1][0], contribs[2][0]}
    assert zero_edges == {sheaf.edges[0], sheaf.edges[2]}
    assert contribs[1][1] == 0.0
    assert contribs[2][1] == 0.0


def test_h4_weights_from_receipts_deterministic():
    """H4 (determinism only): weights_from_receipts has no random
    sampling. Two calls with the same inputs produce byte-identical
    outputs. The parallel-indexing claim is pinned separately at
    :func:`test_weights_from_receipts_indexed_parallel_to_edges`."""
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

    **Tightened 2026-05-02 after audit:** loop over multiple
    perturbation seeds. n=1 was too thin for a headline utility
    claim; one seed could pass by luck. Pin that the inequality
    holds in *at least* 8/10 seeded perturbations, with the mean
    Δ_trusted strictly greater than mean Δ_untrusted. If the
    inequality holds <80%, v3 is mathematically well-defined but
    its utility is fixture-dependent rather than robust — file
    a falsification.
    """
    triples = [
        ("alice", "knows", "bob"),    # trusted (will be tampered)
        ("bob", "knows", "carol"),    # trusted (control)
        ("carol", "owns", "dog"),     # untrusted (will be tampered)
    ]
    trained, embeddings, _ = train_restriction_maps(
        triples, stalk_dim=8, epochs=200, seed=0,
    )
    weights = weights_from_receipts(
        trained,
        trusted_edges=[triples[0], triples[1]],
    )
    # weights = [1.0, 1.0, 0.1]

    x_clean = cochain_one_hot_v2(trained, triples, embedding=embeddings)
    v_clean = weighted_laplacian_quadratic_form_v3(trained, x_clean, weights)

    deltas_trusted: list[float] = []
    deltas_untrusted: list[float] = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        perturbation = rng.standard_normal(trained.stalk_dim) * 0.5

        x_trusted_tampered = x_clean.copy()
        x_trusted_tampered[trained.vertex_index["alice"]] += perturbation
        v_t = weighted_laplacian_quadratic_form_v3(
            trained, x_trusted_tampered, weights,
        )

        x_untrusted_tampered = x_clean.copy()
        x_untrusted_tampered[trained.vertex_index["dog"]] += perturbation
        v_u = weighted_laplacian_quadratic_form_v3(
            trained, x_untrusted_tampered, weights,
        )

        deltas_trusted.append(v_t - v_clean)
        deltas_untrusted.append(v_u - v_clean)

    wins = sum(1 for t, u in zip(deltas_trusted, deltas_untrusted) if t > u)
    mean_trusted = float(np.mean(deltas_trusted))
    mean_untrusted = float(np.mean(deltas_untrusted))
    assert wins >= 8, (
        f"v3 receipt-weighting must amplify signal at trusted edges in "
        f"≥ 8/10 seeded perturbations; got {wins}/10. "
        f"deltas_trusted={deltas_trusted}, "
        f"deltas_untrusted={deltas_untrusted}"
    )
    assert mean_trusted > mean_untrusted, (
        f"mean Δ_trusted ({mean_trusted:.4f}) should exceed "
        f"mean Δ_untrusted ({mean_untrusted:.4f})"
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


def test_combined_v3_lambda_wiring_with_nonzero_deficit():
    """**Surfaced and fixed 2026-05-02 after audit.** The clean-
    render test above can't pin λ — when ``v_deficit == 0``,
    ``v_combined = v_laplacian + 0`` is independent of λ. This
    partner test drives a render that drops a source entity
    (deficit > 0) and pins the literal arithmetic.

    The audit caught a real bug: v3's earlier formula
    ``v_laplacian_w + lambda_deficit * base["v_deficit"]`` double-
    counted λ, because v2.2's ``v_deficit`` field is already
    ``presence_weight · deficit²`` (post-λ-weighting). Corrected
    formula:

        v_combined_v3 = v_laplacian_w + v_deficit

    where ``v_deficit`` already carries the λ baked in.

    A regression in either side of this contract surfaces here.
    """
    triples = [
        ("alice", "knows", "bob"),
        ("bob", "owns", "dog"),
        ("carol", "writes", "python"),
    ]
    trained, embeddings, _ = train_restriction_maps(
        triples, stalk_dim=8, epochs=200, seed=0,
    )
    weights = weights_from_receipts(trained, trusted_edges=list(triples))

    # Render drops carol entirely: deficit ≥ 1
    dropout_render = [
        ("alice", "knows", "bob"),
        ("bob", "owns", "dog"),
    ]
    score = combined_detector_score_v3(
        trained, embeddings, dropout_render, weights,
        lambda_deficit=0.05,
    )
    assert score["presence_deficit_count"] >= 1
    assert score["v_deficit"] > 0.0

    # Pin the post-fix algebra: combined = laplacian + deficit (deficit
    # already carries λ via v2.2's combined_detector_score)
    expected = score["v_laplacian_w"] + score["v_deficit"]
    assert np.isclose(score["v_combined_v3"], expected), (
        f"v_combined_v3 must equal v_laplacian_w + v_deficit (deficit "
        f"already λ-weighted); got {score['v_combined_v3']}, "
        f"expected {expected}"
    )

    # Pin that v_deficit ITSELF scales linearly with λ — this is the
    # property that catches the original double-counting bug. Doubling
    # λ should double v_deficit (since deficit_count² is unchanged).
    score_2x = combined_detector_score_v3(
        trained, embeddings, dropout_render, weights,
        lambda_deficit=0.10,
    )
    assert np.isclose(score_2x["v_deficit"], 2 * score["v_deficit"]), (
        f"doubling λ must double v_deficit; got "
        f"v_deficit(λ=0.05)={score['v_deficit']}, "
        f"v_deficit(λ=0.10)={score_2x['v_deficit']}"
    )

    # And v_combined_v3 changes by exactly Δ(v_deficit) (laplacian unchanged)
    delta_combined = score_2x["v_combined_v3"] - score["v_combined_v3"]
    delta_deficit = score_2x["v_deficit"] - score["v_deficit"]
    assert np.isclose(delta_combined, delta_deficit), (
        f"changing λ should change combined by exactly Δ(v_deficit) "
        f"(laplacian term is independent of λ); got "
        f"Δ_combined={delta_combined}, Δ_deficit={delta_deficit}"
    )


# ─── Block 6: v3.1 — harmonic extension ──────────────────────────────


from sum_engine_internal.research.sheaf_laplacian_v3 import (
    boundary_deviation,
    boundary_from_weights,
    harmonic_extension,
)


def _toy_sheaf_v3_1(d: int = 4):
    """5-vertex toy sheaf with a clear boundary/interior split.

    Vertices: alice, bob, carol, dave, eve.
    Edges:
        (alice, knows, bob)        — trusted
        (bob, knows, carol)        — trusted
        (carol, owns, dog)         — but dog isn't here; replace
    Concretely: trusted edges form a "spine" through alice/bob/
    carol; dave + eve hang off as the untrusted interior.
    """
    triples = [
        ("alice", "knows", "bob"),    # trusted
        ("bob", "knows", "carol"),    # trusted
        ("carol", "knows", "dave"),   # untrusted
        ("dave", "knows", "eve"),     # untrusted
    ]
    sheaf = KnowledgeSheafV2.from_triples(triples, stalk_dim=d)
    return sheaf, triples


def test_harmonic_extension_agrees_with_x_B_on_boundary_by_construction():
    """H6: harmonic_extension returns ONLY the interior; the
    boundary x_B is the input. Pin the contract that boundary
    indices are returned in the partition info, and reconstructing
    the full cochain (boundary + interior) preserves x_B byte-
    identically on B."""
    sheaf, _ = _toy_sheaf_v3_1()
    n_v = len(sheaf.vertices)
    d = sheaf.stalk_dim
    rng = np.random.default_rng(0)
    boundary = [0, 1, 2]
    x_B = rng.standard_normal((len(boundary), d))
    x_I_star, interior = harmonic_extension(sheaf, boundary, x_B)
    # Interior indices are the complement
    assert sorted(interior) == [3, 4]
    # Reconstructing the full cochain must restore x_B exactly on B
    x_full = np.zeros((n_v, d))
    for k, v in enumerate(boundary):
        x_full[v] = x_B[k]
    for k, v in enumerate(interior):
        x_full[v] = x_I_star[k]
    for k, v in enumerate(boundary):
        assert np.allclose(x_full[v], x_B[k])


def test_harmonic_extension_minimizes_v_subject_to_boundary_constraint():
    """H7 (the *defining* property): the harmonic extension is the
    cochain that minimizes ‖δx‖² subject to x[B] = x_B. Pin: any
    perturbation of the interior cochain (off the harmonic extension)
    yields a STRICTLY LARGER V."""
    sheaf, _ = _toy_sheaf_v3_1()
    n_v = len(sheaf.vertices)
    d = sheaf.stalk_dim
    rng = np.random.default_rng(7)
    boundary = [0, 1, 2]
    x_B = rng.standard_normal((len(boundary), d))

    x_I_star, interior = harmonic_extension(sheaf, boundary, x_B)
    x_full_optimal = np.zeros((n_v, d))
    for k, v in enumerate(boundary):
        x_full_optimal[v] = x_B[k]
    for k, v in enumerate(interior):
        x_full_optimal[v] = x_I_star[k]

    from sum_engine_internal.research.sheaf_laplacian_v2 import (
        laplacian_quadratic_form_v2,
    )
    v_optimal = laplacian_quadratic_form_v2(sheaf, x_full_optimal)

    # Perturb the interior in 5 directions; each must give V ≥ v_optimal.
    for seed in range(5):
        rng2 = np.random.default_rng(seed + 100)
        perturbation = rng2.standard_normal((len(interior), d)) * 0.5
        x_full_perturbed = x_full_optimal.copy()
        for k, v in enumerate(interior):
            x_full_perturbed[v] = x_I_star[k] + perturbation[k]
        v_perturbed = laplacian_quadratic_form_v2(sheaf, x_full_perturbed)
        assert v_perturbed >= v_optimal - 1e-9, (
            f"harmonic extension is the minimum; perturbation gave "
            f"V={v_perturbed:.6f} < optimal V={v_optimal:.6f}"
        )


def test_harmonic_extension_unique_when_L_II_invertible():
    """H8: when L_II has full rank (typical case for a connected
    interior with non-trivial restriction maps), the harmonic
    extension is unique. Two calls with the same x_B must give
    byte-identical x_I_star."""
    sheaf, _ = _toy_sheaf_v3_1()
    rng = np.random.default_rng(42)
    boundary = [0, 1, 2]
    x_B = rng.standard_normal((len(boundary), sheaf.stalk_dim))
    x_I_a, _ = harmonic_extension(sheaf, boundary, x_B)
    x_I_b, _ = harmonic_extension(sheaf, boundary, x_B)
    assert np.array_equal(x_I_a, x_I_b)


def test_harmonic_extension_full_boundary_returns_empty_interior():
    """H9 (degenerate): if every vertex is on the boundary, the
    interior is empty; the function returns a (0, d) array. No
    crash on the degenerate edge case (linear-algebra would
    otherwise try to invert a 0×0 matrix)."""
    sheaf, _ = _toy_sheaf_v3_1()
    boundary = list(range(len(sheaf.vertices)))
    x_B = np.zeros((len(boundary), sheaf.stalk_dim))
    x_I_star, interior = harmonic_extension(sheaf, boundary, x_B)
    assert x_I_star.shape == (0, sheaf.stalk_dim)
    assert interior == []


def test_harmonic_extension_rejects_invalid_boundary_indices():
    """H10: defensive — boundary indices outside [0, |V|) must
    raise ValueError so a mis-indexed caller fails loudly."""
    sheaf, _ = _toy_sheaf_v3_1()
    n_v = len(sheaf.vertices)
    with pytest.raises(ValueError, match="boundary_indices"):
        harmonic_extension(
            sheaf, [0, n_v + 1], np.zeros((2, sheaf.stalk_dim)),
        )


def test_harmonic_extension_rejects_wrong_x_B_shape():
    sheaf, _ = _toy_sheaf_v3_1()
    bad = np.zeros((3, sheaf.stalk_dim + 1))   # wrong d
    with pytest.raises(ValueError, match="x_B shape"):
        harmonic_extension(sheaf, [0, 1, 2], bad)


# ─── Block 7: v3.1 — boundary_deviation utility ──────────────────────


def test_boundary_deviation_zero_on_self_extended_cochain():
    """H11: a cochain whose interior IS the harmonic extension of
    its own boundary has deviation = 0. Pin the round-trip."""
    sheaf, _ = _toy_sheaf_v3_1()
    n_v = len(sheaf.vertices)
    d = sheaf.stalk_dim
    rng = np.random.default_rng(0)
    boundary = [0, 1, 2]
    x_B = rng.standard_normal((len(boundary), d))

    x_I_star, interior = harmonic_extension(sheaf, boundary, x_B)
    x_full = np.zeros((n_v, d))
    for k, v in enumerate(boundary):
        x_full[v] = x_B[k]
    for k, v in enumerate(interior):
        x_full[v] = x_I_star[k]

    result = boundary_deviation(sheaf, x_full, boundary)
    assert np.isclose(result["deviation"], 0.0, atol=1e-9), (
        f"self-extended cochain must have zero deviation; "
        f"got {result['deviation']}"
    )


def test_boundary_deviation_detects_interior_tampering():
    """H12 (utility): tampering an interior vertex (while holding
    the boundary fixed) produces a non-zero deviation. This is the
    hallucination-detection use case — the boundary establishes the
    "trusted frame," and the deviation flags renders that drift on
    the untrusted parts."""
    sheaf, _ = _toy_sheaf_v3_1()
    n_v = len(sheaf.vertices)
    d = sheaf.stalk_dim
    rng = np.random.default_rng(0)
    boundary = [0, 1, 2]
    x_B = rng.standard_normal((len(boundary), d))

    x_I_star, interior = harmonic_extension(sheaf, boundary, x_B)
    x_full_clean = np.zeros((n_v, d))
    for k, v in enumerate(boundary):
        x_full_clean[v] = x_B[k]
    for k, v in enumerate(interior):
        x_full_clean[v] = x_I_star[k]

    # Tamper interior vertex 3 (dave)
    x_full_tampered = x_full_clean.copy()
    x_full_tampered[3] += rng.standard_normal(d) * 0.7

    clean = boundary_deviation(sheaf, x_full_clean, boundary)
    tampered = boundary_deviation(sheaf, x_full_tampered, boundary)
    assert tampered["deviation"] > clean["deviation"] + 0.01, (
        f"tampering interior must increase deviation; "
        f"clean={clean['deviation']:.4f}, tampered={tampered['deviation']:.4f}"
    )


def test_boundary_deviation_v_at_extension_is_minimum():
    """H13: by the harmonic extension's defining property,
    v_at_extension ≤ v_at_actual for every cochain whose interior
    is not already the extension. Pin the inequality."""
    sheaf, _ = _toy_sheaf_v3_1()
    n_v = len(sheaf.vertices)
    d = sheaf.stalk_dim
    rng = np.random.default_rng(2)
    boundary = [0, 1, 2]
    x_full = rng.standard_normal((n_v, d))   # arbitrary cochain
    result = boundary_deviation(sheaf, x_full, boundary)
    assert result["v_at_extension"] <= result["v_at_actual"] + 1e-9, (
        f"v_at_extension must be ≤ v_at_actual; "
        f"v_at_extension={result['v_at_extension']:.4f}, "
        f"v_at_actual={result['v_at_actual']:.4f}"
    )


def test_boundary_deviation_with_identity_maps_is_weight_invariant_on_chain_graphs():
    """**Surfaced empirically 2026-05-02.** With identity restriction
    maps + a chain topology (alice—bob—carol—dave—eve, single path),
    the harmonic extension is weight-invariant for any all-positive
    weight vector.

    Algebra: ∂V/∂x_dave = 0 with identity maps gives
    (w2 + w3) x_dave = w2 x_carol + w3 x_eve, and ∂V/∂x_eve = 0
    gives x_eve = x_dave. Substituting → x_dave = x_carol
    independent of w2 / w3 (so long as w2 ≠ 0). The harmonic
    extension is the boundary-extended *constant* cochain on each
    interior chain segment.

    This is a *property* of identity-maps-chain sheaves, not a bug
    in the weights kwarg — pinned in code so a future test author
    doesn't try to "fix" it. The weight effect IS visible on
    trained sheaves (see :func:`test_boundary_deviation_with_weights_visible_on_trained_sheaf`).
    """
    sheaf, _ = _toy_sheaf_v3_1()
    n_v = len(sheaf.vertices)
    d = sheaf.stalk_dim
    rng = np.random.default_rng(0)
    boundary = [0, 1, 2]
    x_full = rng.standard_normal((n_v, d))

    unweighted = boundary_deviation(sheaf, x_full, boundary)
    weights = np.array([1.0, 1.0, 0.001, 1.0])
    weighted = boundary_deviation(sheaf, x_full, boundary, weights=weights)
    # PIN the math property: identity maps + chain → weight-invariant.
    assert np.isclose(unweighted["deviation"], weighted["deviation"]), (
        f"With identity restriction maps + chain topology, the "
        f"harmonic extension should be weight-invariant. If this "
        f"assertion ever fires, EITHER the implementation changed "
        f"(weights now route differently) OR the toy graph is no "
        f"longer a chain. Investigate before 'fixing' the test."
    )


def test_boundary_deviation_with_weights_visible_with_multiple_bridge_edges():
    """**The actual utility pin** for v3.1's weights kwarg.

    Surfaced empirically 2026-05-02: on a sheaf with a *single*
    bridge edge connecting boundary to interior (a chain), the
    harmonic extension is weight-invariant — the analytic
    factorization

        x_I(r) = -r · M(r)^{-1} (B x_B), r = w_bridge / w_interior

    cancels into an r-independent quantity when B has rank 1 (i.e.,
    one bridge column). This is true even on a trained sheaf with
    distinct restriction maps. See
    test_boundary_deviation_with_identity_maps_is_weight_invariant_on_chain_graphs
    for the chain-topology pin.

    The weight effect IS observable when the boundary-interior
    interface has *multiple* bridge edges. This test pins that
    case so the weights kwarg has a measurable contract.

    Topology:
        alice ─knows─ bob       (boundary)
        bob ─owns─ carol        (bridge 1)
        alice ─likes─ carol     (bridge 2)
        carol ─knows─ dave      (interior)

    With two bridges (bob→carol, alice→carol), down-weighting one
    bridge shifts the harmonic extension toward the other.
    """
    triples = [
        ("alice", "knows", "bob"),
        ("bob", "owns", "carol"),
        ("alice", "likes", "carol"),
        ("carol", "knows", "dave"),
    ]
    trained, embeddings, _ = train_restriction_maps(
        triples, stalk_dim=8, epochs=200, seed=0,
    )
    n_v = len(trained.vertices)
    d = trained.stalk_dim
    rng = np.random.default_rng(0)
    boundary = [trained.vertex_index[v] for v in ("alice", "bob")]
    x_full = rng.standard_normal((n_v, d))

    unweighted = boundary_deviation(trained, x_full, boundary)
    # Heavily down-weight bridge 1 (bob→carol): deviation should change
    weights = np.array([1.0, 0.01, 1.0, 1.0])
    weighted = boundary_deviation(trained, x_full, boundary, weights=weights)
    assert not np.isclose(unweighted["deviation"], weighted["deviation"]), (
        f"With multiple bridge edges, weights must change the harmonic "
        f"extension; got identical {unweighted['deviation']:.6f} in "
        f"both cases — weights kwarg is silently ignored."
    )


# ─── Block 8: v3.1 — boundary_from_weights helper ────────────────────


def test_boundary_from_weights_picks_only_fully_trusted_vertices():
    """A vertex is on the boundary iff EVERY incident edge has
    weight ≥ threshold. Pin: in a graph where vertex bob has both
    a trusted (alice→bob) and untrusted (bob→carol) incident edge,
    bob is NOT on the boundary."""
    sheaf, triples = _toy_sheaf_v3_1()
    weights = weights_from_receipts(
        sheaf,
        trusted_edges=[triples[0], triples[1]],   # alice-bob, bob-carol trusted
        # carol-dave, dave-eve are default (0.1)
    )
    boundary = boundary_from_weights(sheaf, weights, threshold=0.5)
    # alice has only the trusted alice-bob edge → boundary
    # bob has trusted alice-bob AND trusted bob-carol → boundary
    # carol has trusted bob-carol AND untrusted carol-dave → NOT boundary
    # dave has untrusted carol-dave AND untrusted dave-eve → NOT boundary
    # eve has only untrusted dave-eve → NOT boundary
    assert sheaf.vertex_index["alice"] in boundary
    assert sheaf.vertex_index["bob"] in boundary
    assert sheaf.vertex_index["carol"] not in boundary
    assert sheaf.vertex_index["dave"] not in boundary
    assert sheaf.vertex_index["eve"] not in boundary


def test_boundary_from_weights_empty_when_nothing_trusted():
    sheaf, _ = _toy_sheaf_v3_1()
    weights = weights_from_receipts(sheaf)   # all default (0.1)
    boundary = boundary_from_weights(sheaf, weights, threshold=0.5)
    assert boundary == []


def test_boundary_from_weights_full_when_everything_trusted():
    sheaf, triples = _toy_sheaf_v3_1()
    weights = weights_from_receipts(sheaf, trusted_edges=list(triples))
    boundary = boundary_from_weights(sheaf, weights, threshold=0.5)
    assert sorted(boundary) == list(range(len(sheaf.vertices)))
