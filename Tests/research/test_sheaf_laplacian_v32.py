"""Math + falsifiability tests for v3.2 — the F3 STRUCTURAL FAIL closer.

v3.1's ``boundary_deviation.deviation`` is a useful primitive but a
poor *standalone* score. F3 corpus bench (PR #124) and the F3
diagnostic (PR #125) showed why: when the per-doc graph topology
gives ``L_IB = 0`` (boundary disjoint from interior), the harmonic
extension is independent of ``x_B``, so the deviation field is
exactly invariant under any boundary-only perturbation. The
diagnostic's 8-cell sweep settled this as structural, not parametric
(see ``docs/SHEAF_HALLUCINATION_DETECTOR.md`` §3.4.3).

v3.2's response: a strict generalization of v3 that combines the
weighted Laplacian energy (which catches *any* cochain change anywhere
on the graph) with the harmonic-extension deviation (which catches
interior-vs-trust-frame inconsistency where ``L_IB ≠ 0``):

    v_combined_v32 = v_laplacian_w + γ · deviation_w + λ · v_deficit

With ``γ = 0`` v3.2 reduces to v3 (subsumption); with ``γ > 0`` it
adds the deviation signal. The two terms are complementary: when
``L_IB = 0``, deviation contributes a constant (clean and perturbed
agree), so v3.2 falls back gracefully to v3's signal. When
``L_IB ≠ 0``, deviation responds to boundary-changes via the
harmonic extension; v3.2 captures both.

Falsifiable predictions pinned in this file:

  H16. **Subsumption.** With ``γ_deviation = 0`` and the same other
       inputs, v3.2's combined score is numerically equal to v3's
       ``v_combined_v3``. v3.2 is a strict generalization, not a
       different detector.

  H17. **Boundary-perturbation visibility under non-zero L_IB.**
       On a synthetic graph with cross-partition edges (so the
       harmonic extension genuinely depends on ``x_B``), v3.2's
       ``deviation_w`` differs between clean and a boundary-only
       perturbation. The F3 structural-fail mechanism does not apply
       when the graph topology has informative boundary↔interior
       coupling.

  H18. **F3 structural fall-back.** On a synthetic graph with
       ``L_IB = 0`` (the F3-failure topology), v3.2 still surfaces
       a non-zero perturbation signal — via ``v_laplacian_w``, not
       ``deviation_w``. The combined score is informative even when
       deviation is structurally blind.

  H19. **No double-counting of λ.** When ``v_deficit > 0``,
       ``v_combined_v32 = v_laplacian_w + γ_deviation · deviation_w
       + v_deficit`` (deficit term already carries λ baked in by
       v2.2's ``combined_detector_score``). Doubling λ doubles
       ``v_deficit`` only — does not double-multiply on the v3.2
       wrapper. (Same audit-tightening pattern as v3 PR #123.)

  H20. **Empty-deviation graceful fall-back.** When the boundary
       partition is degenerate (every vertex on B, or every vertex
       in I), ``deviation_w`` is zero by convention; the combined
       score reduces to ``v_laplacian_w + λ · v_deficit`` (= v3
       exactly). No crashes, no NaN.
"""
from __future__ import annotations

import pytest

# Skip entire file if [research] extras aren't installed.
np = pytest.importorskip("numpy")

from sum_engine_internal.research.sheaf_laplacian_v2 import (
    KnowledgeSheafV2,
    cochain_one_hot_v2,
    train_restriction_maps,
)
from sum_engine_internal.research.sheaf_laplacian_v3 import (
    boundary_from_weights,
    weights_from_receipts,
)
from sum_engine_internal.research.sheaf_laplacian_v32 import (
    combined_detector_score_v32,
)


# ─── Block 1: H16 subsumption (γ = 0 reduces to v3) ───────────────────


def _toy_v32_graph():
    """A small graph with non-trivial L_IB structure.

    Edges:
      (alice, graduated_from, mit)   - boundary↔boundary if alice & mit boundary
      (alice, knows, bob)             - boundary↔interior (cross-partition)
      (bob, owns, dog)                - interior↔interior

    With trusted = {graduated_from edges}, weights = [1, 0.1, 0.1] under
    default trust contract. boundary_from_weights(threshold=0.5) requires
    every incident edge ≥ 0.5; only mit qualifies (its sole edge is
    weight 1.0). alice is incident to a 0.1 edge so falls to interior.

    For H17/H18 we'll need to construct boundary explicitly to control
    L_IB structure, not rely on the default boundary derivation.
    """
    triples = [
        ("alice", "graduated_from", "mit"),
        ("alice", "knows", "bob"),
        ("bob", "owns", "dog"),
    ]
    trained, embeddings, _ = train_restriction_maps(
        triples, stalk_dim=8, epochs=200, learning_rate=0.005, margin=0.5,
        n_negatives_per_positive=3, seed=0,
    )
    return trained, embeddings, triples


def test_h16_subsumption_gamma_zero_equals_v3():
    """γ_deviation = 0 → v_combined_v32 == v_combined_v3 (numerically).

    v3.2 must be a strict generalization, not a different detector.
    Without this, the API surfaces two scores that drift under any
    edit to either path; with it, v3.2 always at least preserves v3's
    answer.
    """
    from sum_engine_internal.research.sheaf_laplacian_v3 import (
        combined_detector_score_v3,
    )

    trained, embeddings, triples = _toy_v32_graph()
    weights = weights_from_receipts(trained, trusted_edges=triples[:1])  # only graduated_from trusted
    boundary = [trained.vertex_index["mit"]]  # explicit B

    score_v3 = combined_detector_score_v3(
        trained, embeddings, triples, weights, lambda_deficit=0.05,
    )
    score_v32 = combined_detector_score_v32(
        trained, embeddings, triples, weights,
        lambda_deficit=0.05, gamma_deviation=0.0,
        boundary_indices=boundary,
    )
    assert np.isclose(score_v32["v_combined_v32"], score_v3["v_combined_v3"]), (
        f"H16 subsumption broke: v3.2(γ=0) = {score_v32['v_combined_v32']}, "
        f"v3 = {score_v3['v_combined_v3']}. v3.2 must reduce to v3 at γ=0."
    )


def test_h16_subsumption_carries_v3_keys():
    """v3.2 carries every v3 key — downstream consumers can read both
    v_laplacian_w and v_combined_v3 from a v3.2 score dict.
    """
    trained, embeddings, triples = _toy_v32_graph()
    weights = weights_from_receipts(trained, trusted_edges=triples[:1])
    boundary = [trained.vertex_index["mit"]]
    score = combined_detector_score_v32(
        trained, embeddings, triples, weights,
        boundary_indices=boundary,
    )
    for key in ("v_laplacian", "v_deficit", "v_combined", "v_laplacian_w",
                "v_combined_v3", "edge_weights"):
        assert key in score, f"v3.2 must carry v3 key {key!r}; got {sorted(score)}"
    # v3.2-specific keys
    for key in ("deviation_w", "v_combined_v32", "boundary_indices",
                "interior_size"):
        assert key in score, f"v3.2 must add key {key!r}; got {sorted(score)}"


# ─── Block 2: H17 boundary-perturbation visibility (L_IB ≠ 0) ─────────


def _v32_graph_with_cross_partition_edge():
    """Construct a synthetic where L_IB ≠ 0 by design.

    Vertices: alice, bob, carol, dave (in this order)
    Edges:
      (alice, knows, bob)       - alice→bob, weight 1.0 (trusted)
      (bob, knows, carol)       - bob→carol, weight 0.1 (untrusted)
      (carol, knows, dave)      - carol→dave, weight 1.0 (trusted)

    boundary_indices = [alice, dave] (forced by hand)
    interior_indices = [bob, carol]

    The "alice-knows-bob" edge crosses B↔I (alice on B, bob on I).
    The "carol-knows-dave" edge crosses I↔B (carol on I, dave on B).
    The "bob-knows-carol" edge is purely interior.

    Therefore L_IB ≠ 0 — the deviation field is informative.
    """
    triples = [
        ("alice", "knows", "bob"),
        ("bob", "knows", "carol"),
        ("carol", "knows", "dave"),
    ]
    trained, embeddings, _ = train_restriction_maps(
        triples, stalk_dim=8, epochs=200, learning_rate=0.005, margin=0.5,
        n_negatives_per_positive=3, seed=0,
    )
    return trained, embeddings, triples


def test_h17_deviation_visible_when_l_ib_nonzero():
    """Boundary perturbation on a graph with cross-partition edges
    produces non-zero Δdeviation_w. The F3 structural-fail mechanism
    requires L_IB = 0; this graph defeats it by construction.
    """
    trained, embeddings, triples = _v32_graph_with_cross_partition_edge()
    weights = weights_from_receipts(
        trained, trusted_edges=[triples[0], triples[2]],
    )
    boundary = [trained.vertex_index["alice"], trained.vertex_index["dave"]]

    # Clean render uses all four entities. Perturbed: swap alice → bob in the
    # first triple, so alice no longer mentioned, but bob still mentioned via
    # edge 1+2. Net cochain change: alice's stalk drops from embedding to 0;
    # bob's stalk unchanged (still mentioned). The change is at the boundary.
    perturbed = [
        ("bob", "knows", "bob"),  # alice → bob (in-vocab swap, A1)
        ("bob", "knows", "carol"),
        ("carol", "knows", "dave"),
    ]

    clean_score = combined_detector_score_v32(
        trained, embeddings, triples, weights,
        gamma_deviation=1.0, boundary_indices=boundary,
    )
    perturbed_score = combined_detector_score_v32(
        trained, embeddings, perturbed, weights,
        gamma_deviation=1.0, boundary_indices=boundary,
    )
    delta_dev = perturbed_score["deviation_w"] - clean_score["deviation_w"]
    assert abs(delta_dev) > 1e-6, (
        f"H17 broke: with L_IB ≠ 0, a boundary perturbation must produce "
        f"non-zero Δdeviation_w. Got clean={clean_score['deviation_w']}, "
        f"perturbed={perturbed_score['deviation_w']}, delta={delta_dev}. "
        f"This means either the cross-partition edge construction is broken "
        f"or the deviation primitive regressed."
    )


def test_h17_combined_score_strictly_increases_under_perturbation():
    """The combined v3.2 score on a cross-partition graph must
    strictly *increase* (not stay flat) under a boundary perturbation.
    Both v_laplacian_w and γ · deviation_w should contribute.
    """
    trained, embeddings, triples = _v32_graph_with_cross_partition_edge()
    weights = weights_from_receipts(
        trained, trusted_edges=[triples[0], triples[2]],
    )
    boundary = [trained.vertex_index["alice"], trained.vertex_index["dave"]]
    perturbed = [
        ("bob", "knows", "bob"),
        ("bob", "knows", "carol"),
        ("carol", "knows", "dave"),
    ]
    clean = combined_detector_score_v32(
        trained, embeddings, triples, weights,
        gamma_deviation=1.0, boundary_indices=boundary,
    )
    perturbed_score = combined_detector_score_v32(
        trained, embeddings, perturbed, weights,
        gamma_deviation=1.0, boundary_indices=boundary,
    )
    assert perturbed_score["v_combined_v32"] > clean["v_combined_v32"], (
        f"H17 utility broke: perturbation should raise combined score; "
        f"got clean={clean['v_combined_v32']}, "
        f"perturbed={perturbed_score['v_combined_v32']}"
    )


# ─── Block 3: H18 F3-topology fall-back (L_IB = 0) ────────────────────


def _v32_graph_l_ib_zero():
    """Construct the F3 failure topology: edges live entirely within
    boundary or entirely within interior. L_IB = 0.

    Vertices: alice, mit, bob, google
    Edges:
      (alice, graduated_from, mit)  - both endpoints on B (trusted)
      (bob, works_at, google)       - both endpoints on I (untrusted)
    """
    triples = [
        ("alice", "graduated_from", "mit"),
        ("bob", "works_at", "google"),
    ]
    trained, embeddings, _ = train_restriction_maps(
        triples, stalk_dim=8, epochs=200, learning_rate=0.005, margin=0.5,
        n_negatives_per_positive=3, seed=0,
    )
    return trained, embeddings, triples


def test_h18_v_laplacian_w_visible_when_l_ib_zero():
    """In the F3-failure topology (L_IB = 0), deviation_w is structurally
    blind to boundary perturbations — but v_laplacian_w isn't, because
    it sums residuals over all edges including boundary ones.

    v3.2's combined score must still surface the perturbation, via
    v_laplacian_w. This is the "structural fall-back" guarantee.
    """
    trained, embeddings, triples = _v32_graph_l_ib_zero()
    weights = weights_from_receipts(trained, trusted_edges=triples[:1])
    boundary = [
        trained.vertex_index["alice"], trained.vertex_index["mit"],
    ]

    # A1 perturbation hits the trusted edge: swap alice → bob (in-vocab,
    # so cochain at alice drops to 0; bob's stalk unchanged because
    # bob is also already mentioned in the works_at edge).
    perturbed = [
        ("bob", "graduated_from", "mit"),
        ("bob", "works_at", "google"),
    ]
    clean = combined_detector_score_v32(
        trained, embeddings, triples, weights,
        gamma_deviation=1.0, boundary_indices=boundary,
    )
    perturbed_score = combined_detector_score_v32(
        trained, embeddings, perturbed, weights,
        gamma_deviation=1.0, boundary_indices=boundary,
    )

    # deviation_w may be ~zero (structural blind spot), but v_laplacian_w
    # MUST change because the alice-graduated-mit residual changed.
    delta_lap = perturbed_score["v_laplacian_w"] - clean["v_laplacian_w"]
    assert abs(delta_lap) > 1e-6, (
        f"H18 broke: v_laplacian_w should change under boundary perturbation "
        f"even when L_IB=0; got delta={delta_lap}"
    )

    # Combined score therefore changes too — graceful fall-back from
    # blind deviation to the still-informative Laplacian term.
    delta_combined = (
        perturbed_score["v_combined_v32"] - clean["v_combined_v32"]
    )
    assert abs(delta_combined) > 1e-6, (
        f"H18 fall-back broke: v_combined_v32 should change even when "
        f"deviation is blind; got delta={delta_combined}"
    )


# ─── Block 4: H19 λ wiring (no double-counting) ───────────────────────


def test_h19_no_lambda_double_counting_via_v32_wrapper():
    """v3.2 wraps v3, which wraps v2.2. The audit-tightening pass on
    2026-05-02 (PR #123) caught a λ double-counting bug at the v3
    layer; this test pins that v3.2 doesn't reintroduce it at its
    own layer.

    Doubling λ_deficit must double v_deficit only, NOT double-multiply
    via the v3.2 wrapper.
    """
    triples = [
        ("alice", "knows", "bob"),
        ("bob", "owns", "dog"),
        ("carol", "writes", "python"),
    ]
    trained, embeddings, _ = train_restriction_maps(
        triples, stalk_dim=8, epochs=200, seed=0,
    )
    weights = weights_from_receipts(trained, trusted_edges=triples)
    boundary = [trained.vertex_index["alice"]]

    # Drop carol entirely (deficit ≥ 1)
    dropout_render = [
        ("alice", "knows", "bob"),
        ("bob", "owns", "dog"),
    ]
    score_lo = combined_detector_score_v32(
        trained, embeddings, dropout_render, weights,
        lambda_deficit=0.05, gamma_deviation=1.0,
        boundary_indices=boundary,
    )
    score_hi = combined_detector_score_v32(
        trained, embeddings, dropout_render, weights,
        lambda_deficit=0.10, gamma_deviation=1.0,
        boundary_indices=boundary,
    )
    assert score_lo["v_deficit"] > 0.0
    assert np.isclose(score_hi["v_deficit"], 2 * score_lo["v_deficit"]), (
        f"H19 broke: doubling λ should double v_deficit; got "
        f"v_deficit(0.05)={score_lo['v_deficit']}, "
        f"v_deficit(0.10)={score_hi['v_deficit']}"
    )

    # Pin the literal arithmetic of the combined score
    expected_lo = (
        score_lo["v_laplacian_w"]
        + 1.0 * score_lo["deviation_w"]
        + score_lo["v_deficit"]
    )
    assert np.isclose(score_lo["v_combined_v32"], expected_lo), (
        f"v_combined_v32 must equal v_laplacian_w + γ·deviation_w + "
        f"v_deficit (deficit pre-λ-weighted); got "
        f"{score_lo['v_combined_v32']}, expected {expected_lo}"
    )


# ─── Block 5: H20 graceful fall-back on degenerate boundary ───────────


def test_h20_empty_boundary_degenerate_partition_no_crash():
    """When the boundary partition is degenerate (empty B, or B = V),
    deviation_w is 0 by convention and v_combined_v32 reduces to
    v_laplacian_w + λ · v_deficit (== v3 exactly). No NaN, no crash.
    """
    trained, embeddings, triples = _toy_v32_graph()
    weights = weights_from_receipts(trained, trusted_edges=triples)
    score_empty_b = combined_detector_score_v32(
        trained, embeddings, triples, weights,
        gamma_deviation=1.0, boundary_indices=[],
    )
    assert score_empty_b["deviation_w"] == 0.0
    assert score_empty_b["interior_size"] == len(trained.vertices)
    # With empty B, v_combined_v32 = v_laplacian_w + 0 + v_deficit == v3
    expected = score_empty_b["v_laplacian_w"] + score_empty_b["v_deficit"]
    assert np.isclose(score_empty_b["v_combined_v32"], expected)

    # Full B (no interior): also degenerate
    full_b = list(range(len(trained.vertices)))
    score_full_b = combined_detector_score_v32(
        trained, embeddings, triples, weights,
        gamma_deviation=1.0, boundary_indices=full_b,
    )
    assert score_full_b["deviation_w"] == 0.0
    assert score_full_b["interior_size"] == 0


# ─── Block 6: utility tests — H17 + H18 together ──────────────────────


def test_v32_strictly_dominates_or_equals_v3_when_gamma_zero():
    """For any inputs, v3.2(γ=0) == v3 numerically. This is a strong
    promise: deploying v3.2 in place of v3 with γ=0 is a no-op."""
    from sum_engine_internal.research.sheaf_laplacian_v3 import (
        combined_detector_score_v3,
    )

    # Run on three distinct graph configurations
    configs = [
        _toy_v32_graph(),
        _v32_graph_with_cross_partition_edge(),
        _v32_graph_l_ib_zero(),
    ]
    for trained, embeddings, triples in configs:
        weights = weights_from_receipts(trained, trusted_edges=triples[:1])
        boundary = [0]  # arbitrary — γ=0 makes it irrelevant
        v3_score = combined_detector_score_v3(
            trained, embeddings, triples, weights, lambda_deficit=0.05,
        )
        v32_score = combined_detector_score_v32(
            trained, embeddings, triples, weights,
            lambda_deficit=0.05, gamma_deviation=0.0,
            boundary_indices=boundary,
        )
        assert np.isclose(v32_score["v_combined_v32"], v3_score["v_combined_v3"])


def test_v32_documents_default_gamma():
    """The default γ is documented (currently 1.0). If we ever
    auto-calibrate, this test must update to pin the new contract.
    """
    import inspect
    sig = inspect.signature(combined_detector_score_v32)
    gamma_param = sig.parameters["gamma_deviation"]
    assert gamma_param.default == 1.0, (
        f"v3.2 default γ_deviation contract changed; was 1.0, "
        f"now {gamma_param.default}. Update this test if calibration "
        f"strategy changed."
    )
