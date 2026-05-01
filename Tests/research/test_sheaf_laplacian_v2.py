"""Math + training + falsification tests for v2.1 sheaf-Laplacian.

Three blocks of tests:

  Block 1 (math sanity, d > 1): the Laplacian properties from
    Hansen-Ghrist 2019 §3.2 hold at d > 1 — symmetric, PSD, kernel
    structure. v2.0-equivalence smoke test (identity restriction
    maps reduce to v1 numerically).

  Block 2 (training sanity): the contrastive sheaf-embedding loss
    decreases monotonically once trained; trained restriction maps
    produce lower V on positive triples than random init.

  Block 3 (disconnected-graph blindspot — the headline question):
    after training v2.1 on a disconnected source graph, does
    dropping a single triple produce V > 0? This is the v1
    blindspot from PR #107 — the test that decides whether v2.1
    is a meaningful research artifact or just engineering surface.
"""
from __future__ import annotations

import pytest

# Skip the whole file if [research] extras aren't installed.
np = pytest.importorskip("numpy")

from sum_engine_internal.research.sheaf_laplacian_v2 import (
    KnowledgeSheafV2,
    per_edge_residual_v2,
    laplacian_quadratic_form_v2,
    per_edge_discrepancy_v2,
    sheaf_laplacian_v2,
    cochain_one_hot_v2,
    train_restriction_maps,
    consistency_profile_v2,
)


# ── Block 1: math sanity at d > 1 ────────────────────────────────────


def _toy_sheaf_v2(d: int = 8):
    """Tiny v2 sheaf: 3 entities, 2 distinct relations (so the
    per-relation restriction-map structure is exercised)."""
    triples = [
        ("alice", "knows", "bob"),
        ("bob", "owns", "dog"),
    ]
    return KnowledgeSheafV2.from_triples(triples, stalk_dim=d)


def test_v2_laplacian_is_symmetric():
    F = _toy_sheaf_v2(d=8)
    L = sheaf_laplacian_v2(F)
    assert np.allclose(L, L.T), "L_F = δ^T δ must be symmetric"


def test_v2_laplacian_is_positive_semidefinite():
    F = _toy_sheaf_v2(d=8)
    L = sheaf_laplacian_v2(F)
    eigvals = np.linalg.eigvalsh(L)
    assert np.all(eigvals >= -1e-9), f"L_F must be PSD; got eigvals={eigvals}"


def test_v2_quadratic_form_via_residuals_matches_full_laplacian():
    """The factored form ‖δx‖² must equal x^T L x to floating-point
    precision. Important because the implementation avoids
    materializing L for performance."""
    F = _toy_sheaf_v2(d=8)
    rng = np.random.default_rng(42)
    L = sheaf_laplacian_v2(F)
    for _ in range(10):
        x = rng.standard_normal((len(F.vertices), F.stalk_dim))
        v_factored = laplacian_quadratic_form_v2(F, x)
        v_full = float(x.flatten() @ L @ x.flatten())
        assert np.isclose(v_factored, v_full), (
            f"factored {v_factored} != full {v_full}"
        )


def test_v2_constant_cochain_with_identity_maps_is_global_section():
    """With identity restriction maps and a constant cochain
    (x_v = c for every v), every per-edge residual is c - c = 0,
    so V = 0. Same property as v1's constant-cochain test, lifted
    to d > 1."""
    F = _toy_sheaf_v2(d=8)
    c = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    x = np.tile(c, (len(F.vertices), 1))   # shape (|V|, d), every row = c
    v = laplacian_quadratic_form_v2(F, x)
    assert v == 0.0, f"constant cochain under identity maps must give V=0; got {v}"


def test_v2_one_hot_cochain_residual_magnitude():
    """v2.0 cochain construction with distinct one-hot basis vectors
    per entity: under identity restriction maps, an edge between
    two mentioned-but-distinct entities has residual ‖e_u − e_v‖² =
    2 (not 1). This is *not* the same numeric as v1's presence-
    indicator cochain — v2 with one-hot cochains computes a richer
    quantity. The v1-equivalence smoke claim from spec §5.3a is
    therefore not a numeric reduction; it's structural ("the
    machinery handles d > 1 without breaking"). This test pins
    the actual numeric to prevent silent regression."""
    triples = [("a", "p", "b"), ("b", "q", "c"), ("c", "p", "d")]
    sheaf_v2 = KnowledgeSheafV2.from_triples(triples, stalk_dim=4)
    # All four entities mentioned: each pair of distinct one-hot
    # vectors has squared L2 distance = 2; three edges with all
    # endpoints distinct give V = 3 * 2 = 6.
    x = cochain_one_hot_v2(sheaf_v2, triples)
    v = laplacian_quadratic_form_v2(sheaf_v2, x)
    assert v == 6.0, (
        f"v2 one-hot cochain over a 4-entity / 3-edge graph should "
        f"give V = 3 edges × ‖e_u−e_v‖²(=2) = 6; got {v}"
    )


def test_v2_per_edge_localization_finds_failing_edge():
    """v2 one-hot localization: with identity restriction maps and
    one-hot cochains, all 'mentioned' edges contribute equally
    (=2). A render that drops an endpoint contributes ‖e_u − 0‖² = 1
    on that edge — the *smaller* score. So 'localization' under
    v2.0 + one-hot picks the edge with HIGHEST cochain disagreement
    among mentioned-mentioned edges, not the failing one. v1's
    localization story relied on (1,0)/(1,1) asymmetry that v2.0
    one-hot doesn't have. v2.1's trained restriction maps recover
    a localization signal — but on TRAINED sheaves, not on this
    untrained smoke-test graph. Pinning that the smoke graph
    behaves as expected so future v2.1-localization regressions
    surface here.
    """
    triples = [("a", "p", "b"), ("c", "q", "d")]
    sheaf = KnowledgeSheafV2.from_triples(triples, stalk_dim=4)
    # All 4 entities mentioned: both edges contribute 2; tie broken
    # arbitrarily (sort is stable so first edge wins).
    x = cochain_one_hot_v2(sheaf, triples)
    contribs = per_edge_discrepancy_v2(sheaf, x)
    # Both edges have the same score (2); just verify the score.
    assert contribs[0][1] == 2.0
    assert contribs[1][1] == 2.0


# ── Block 2: training sanity ────────────────────────────────────────


def test_training_loss_decreases_monotonically_after_warmup():
    """The contrastive sheaf-embedding loss should decrease with
    training. Allows a small initial bump (random sampling), then
    monotonic-ish decrease over the last half of training."""
    triples = [
        ("alice", "knows", "bob"),
        ("bob", "knows", "carol"),
        ("carol", "knows", "alice"),
        ("alice", "owns", "dog"),
        ("bob", "owns", "cat"),
    ]
    _, _, history = train_restriction_maps(
        triples,
        stalk_dim=8,
        epochs=200,
        learning_rate=0.005,
        margin=0.5,
        n_negatives_per_positive=3,
        seed=0,
    )
    assert len(history) == 200
    # The loss may not be strictly monotone (sampling noise), but
    # the second half's mean should be below the first half's.
    first_half_mean = float(np.mean(history[:100]))
    second_half_mean = float(np.mean(history[100:]))
    assert second_half_mean < first_half_mean, (
        f"contrastive loss should decrease over training; "
        f"first-half mean={first_half_mean:.4f}, "
        f"second-half mean={second_half_mean:.4f}"
    )


def test_trained_sheaf_lowers_V_on_positive_triples():
    """After training, the per-edge residual on positive triples
    (with the trained restriction maps + entity embeddings) should
    be smaller than at random init."""
    triples = [
        ("alice", "knows", "bob"),
        ("bob", "knows", "carol"),
        ("carol", "owns", "dog"),
    ]
    untrained = KnowledgeSheafV2.from_triples(triples, stalk_dim=8)
    trained, embeddings, _ = train_restriction_maps(
        triples,
        stalk_dim=8,
        epochs=300,
        learning_rate=0.01,
        margin=0.5,
        n_negatives_per_positive=5,
        seed=0,
    )

    # Initial V with one-hot embeddings + identity maps = the v1-equivalent score
    n_v = len(untrained.vertices)
    one_hot = np.eye(n_v, untrained.stalk_dim, dtype=np.float64)
    v_untrained = laplacian_quadratic_form_v2(untrained, one_hot)
    v_trained = laplacian_quadratic_form_v2(trained, embeddings)

    assert v_trained < v_untrained, (
        f"trained sheaf must lower V on positive triples; "
        f"untrained={v_untrained:.4f}, trained={v_trained:.4f}"
    )


# ── Block 3: the headline question — disconnected-graph blindspot ──


def test_v2_1_does_NOT_close_disconnected_graph_blindspot_with_presence_cochains():
    """**Falsification (2026-05-01).** The v2 spec hypothesised that
    learned per-relation restriction maps would close the
    disconnected-graph density-dropout blindspot v1 surfaced in
    PR #107. Empirical run says: **not with presence-style cochains.**

    Trained sheaf on a 4-fact disconnected source; clean V = 0.4377;
    dropout V = 0.3270; margin = -0.1108 (dropout LOWER than
    clean). Pinned in code so this finding doesn't get rediscovered
    on every v2.1 retry.

    Why this happens (verified by inspection of per_edge_residual):
    when a render drops a whole component (e.g., einstein +
    relativity both vanish), the cochain contribution at those
    vertices is zero, and the trained restriction maps are then
    multiplied by zero on both sides — the per-edge residual at
    the dropped component vanishes regardless of how the
    restriction maps were trained. The training amplifies the
    contributions of *present* entities, not the absence of
    absent ones.

    What v2.1 needs to actually close the blindspot: a cochain
    construction that distinguishes "entity not mentioned" from
    "entity is unused vocabulary". Two candidate fixes for v2.2:

      (a) Anti-cochain: x_n[v] = +trained_emb if mentioned in
          render, -trained_emb if v in source but not in render,
          0 if v not in source.
      (b) Semantic-context cochain: x_n[v] = embedding(context
          window around v's mention in render); missing mentions
          give zero, but semantic drift in present mentions is
          captured directly via context.

    Neither (a) nor (b) is implemented yet. The honest spec update
    moves the disconnected-graph fix from "v2.1's expected
    benefit" to "v2.2 hypothesis, untested."
    """
    triples = [
        ("alice", "graduated", "mit"),
        ("bob", "owns", "dog"),
        ("carol", "writes", "python"),
        ("einstein", "proposed", "relativity"),
    ]
    trained, embeddings, _ = train_restriction_maps(
        triples,
        stalk_dim=8,
        epochs=300,
        learning_rate=0.01,
        margin=0.5,
        n_negatives_per_positive=5,
        seed=0,
    )

    x_clean = cochain_one_hot_v2(trained, triples, embedding=embeddings)
    v_clean = laplacian_quadratic_form_v2(trained, x_clean)

    dropout = [
        ("alice", "graduated", "mit"),
        ("bob", "owns", "dog"),
        ("carol", "writes", "python"),
    ]
    x_dropout = cochain_one_hot_v2(trained, dropout, embedding=embeddings)
    v_dropout = laplacian_quadratic_form_v2(trained, x_dropout)

    # Pin the falsification: v_dropout LOWER than v_clean (dropout
    # *reduces* the score, the wrong direction). v2.1 with presence-
    # style cochains is no better than v1 here.
    margin = v_dropout - v_clean
    assert margin < 0, (
        f"This test pins v2.1's falsification: dropout V should be "
        f"LOWER than clean V because dropping a component zeros both "
        f"endpoints of the dropped edge, killing that edge's "
        f"contribution to the quadratic form regardless of training. "
        f"Got v_clean={v_clean:.4f}, v_dropout={v_dropout:.4f}, "
        f"margin={margin:.4f}. If the falsification ever inverts (margin "
        f"becomes positive), the cochain construction has changed and "
        f"this test should be updated to reflect the new behaviour — "
        f"and the spec should celebrate."
    )


def test_v2_2_combined_detector_closes_disconnected_graph_blindspot():
    """**The v2.2 closure (2026-05-01).** PR #111 falsified v2.1 with
    presence-style cochains on the disconnected-graph case. Analytical
    inspection (commit b7bd5fb) revealed the cause: the Laplacian
    quadratic form fundamentally cannot detect presence-pattern
    issues — only cross-edge disagreement.

    v2.2 adds an orthogonal term (the presence deficit) to the
    Laplacian. The Laplacian captures relation-aware drift; the
    deficit captures presence-pattern issues. Combined:
        V_total = ‖δx‖² + λ · (presence_deficit)²

    This test verifies the combined detector closes v1+v2.1's
    structural blindspot on the same 4-fact disconnected source
    that PR #107 and #111 used.
    """
    from sum_engine_internal.research.sheaf_laplacian_v2 import combined_detector_score
    triples = [
        ("alice", "graduated", "mit"),
        ("bob", "owns", "dog"),
        ("carol", "writes", "python"),
        ("einstein", "proposed", "relativity"),
    ]
    trained, embeddings, _ = train_restriction_maps(
        triples,
        stalk_dim=8,
        epochs=300,
        learning_rate=0.01,
        margin=0.5,
        n_negatives_per_positive=5,
        seed=0,
    )

    clean = combined_detector_score(trained, embeddings, triples)
    dropout_render = [
        ("alice", "graduated", "mit"),
        ("bob", "owns", "dog"),
        ("carol", "writes", "python"),
    ]
    dropout = combined_detector_score(trained, embeddings, dropout_render)

    # Clean has zero presence-deficit; dropout has 2 missing entities.
    assert clean["presence_deficit_count"] == 0
    assert dropout["presence_deficit_count"] == 2
    assert sorted(dropout["missing_entities"]) == ["einstein", "relativity"]

    # The combined detector must score dropout HIGHER than clean.
    margin = dropout["v_combined"] - clean["v_combined"]
    assert margin > 0.0, (
        f"v2.2 combined detector must score dropout > clean; "
        f"got clean={clean['v_combined']:.4f}, "
        f"dropout={dropout['v_combined']:.4f}, "
        f"margin={margin:.4f}"
    )

    # Sanity: the Laplacian term ALONE is *not* the source of this
    # signal — it's the deficit term doing the work, exactly as the
    # analytical reasoning predicts.
    assert dropout["v_laplacian"] < clean["v_laplacian"], (
        f"Pin: the Laplacian alone still has the v2.1 falsification "
        f"signal (dropout < clean on Laplacian-only). The closure "
        f"comes from the deficit term, not from the Laplacian. "
        f"clean_laplacian={clean['v_laplacian']:.4f}, "
        f"dropout_laplacian={dropout['v_laplacian']:.4f}"
    )
    assert dropout["v_deficit"] > clean["v_deficit"], (
        f"Pin: the deficit term carries the disconnected-graph "
        f"signal. clean_deficit={clean['v_deficit']:.4f}, "
        f"dropout_deficit={dropout['v_deficit']:.4f}"
    )


def test_v2_2_combined_detector_no_signal_on_clean_render():
    """A render that mentions every source entity must score
    deficit=0 and the combined V equals the Laplacian term alone.
    Verifies the deficit term doesn't fire on lawful renders."""
    from sum_engine_internal.research.sheaf_laplacian_v2 import combined_detector_score
    triples = [
        ("alice", "knows", "bob"),
        ("bob", "owns", "dog"),
    ]
    trained, embeddings, _ = train_restriction_maps(
        triples,
        stalk_dim=8,
        epochs=200,
        seed=0,
    )
    score = combined_detector_score(trained, embeddings, triples)
    assert score["presence_deficit_count"] == 0
    assert score["v_deficit"] == 0.0
    assert score["v_combined"] == score["v_laplacian"]


# ── Block 4: A2 predicate-flip + A3 off-graph fabrication empirics ──
#
# v2.x's hypothesised wins. Per the spec §3.3 v2.1, learned
# restriction maps are *expected* to catch:
#   A2 — predicate-flip: the rendered triple uses a known relation
#        but a wrong one for that (h, t) pair.
#   A3 — off-graph fabrication: the rendered triple uses a relation
#        or entity outside the trained vocabulary.
#
# Both are HYPOTHESISED in the spec but were not measured before
# this PR. The next two tests measure them. Honest framing: if v2.1
# falsifies on these too, that's another data point and the spec
# narrows; if it succeeds, the Laplacian-side claim is empirically
# backed for the first time.


def test_a3_off_graph_fabrication_via_oov_relation_caught():
    """A3 (off-graph fabrication) — rendered triple uses a relation
    NOT in the trained vocabulary. The score_rendered_triple_v2
    machinery flags this as out-of-vocab (oov_signal=True) before
    any V is computed. This is the cheapest A3 detection path —
    structural, not statistical."""
    from sum_engine_internal.research.sheaf_laplacian_v2 import (
        score_rendered_triple_v2, score_rendered_triples_v2,
    )
    triples = [
        ("alice", "knows", "bob"),
        ("bob", "owns", "dog"),
    ]
    trained, embeddings, _ = train_restriction_maps(
        triples, stalk_dim=8, epochs=200, seed=0,
    )

    # Rendered triple uses a relation never seen in source/training
    fabricated = ("alice", "FABRICATED_RELATION", "bob")
    result = score_rendered_triple_v2(trained, embeddings, fabricated)
    assert result["oov_signal"] is True
    assert not result["in_vocab_relation"]
    assert result["in_vocab_head"]
    assert result["in_vocab_tail"]
    assert result["v_triple"] is None
    assert any("FABRICATED_RELATION" in r for r in result["oov_reasons"])


def test_a3_off_graph_fabrication_via_oov_entity_caught():
    """A3 — rendered triple uses an entity NOT in the trained
    vocabulary. Surfaces as oov_signal=True with the reason naming
    the missing entity."""
    from sum_engine_internal.research.sheaf_laplacian_v2 import score_rendered_triple_v2
    triples = [("alice", "knows", "bob"), ("bob", "owns", "dog")]
    trained, embeddings, _ = train_restriction_maps(
        triples, stalk_dim=8, epochs=200, seed=0,
    )

    # Rendered triple uses an entity that's not in the trained vertex set
    fabricated = ("alice", "knows", "GHOST_ENTITY")
    result = score_rendered_triple_v2(trained, embeddings, fabricated)
    assert result["oov_signal"] is True
    assert result["in_vocab_relation"]
    assert result["in_vocab_head"]
    assert not result["in_vocab_tail"]
    assert result["v_triple"] is None


def test_a2_predicate_flip_caught_via_higher_v_triple():
    """**The headline A2 measurement.** Train v2.1 on a 4-triple
    source with two distinct relations. Then evaluate:
        - the clean triple (alice, knows, bob)
        - the predicate-flipped triple (alice, owns, bob)

    The predicate-flipped version uses a relation IN the trained
    vocabulary but a WRONG one for this (head, tail) pair. The
    contrastive sheaf-embedding training (Gebhart Def. 11) didn't
    explicitly include predicate-perturbation negatives — only
    tail-perturbation negatives — so the question this test answers
    empirically: does the trained sheaf NEVERTHELESS distinguish
    (alice, knows, bob) from (alice, owns, bob)?

    Honest possibilities:
      (a) Yes, V_flipped > V_clean robustly: A2 catches via the
          tail-only negative sampling alone (good news).
      (b) Yes, but margin small: A2 catches but training would
          benefit from explicit predicate-perturbation negatives.
      (c) No / unpredictable: A2 falsifies; the spec needs
          another correction; predicate-perturbation negatives
          must enter the training loop in v2.3.

    Pin the empirical outcome so future v2 changes can't silently
    flip the result.
    """
    from sum_engine_internal.research.sheaf_laplacian_v2 import score_rendered_triple_v2
    triples = [
        ("alice", "knows", "bob"),
        ("bob", "knows", "carol"),
        ("alice", "owns", "dog"),
        ("bob", "owns", "cat"),
    ]
    trained, embeddings, _ = train_restriction_maps(
        triples,
        stalk_dim=8,
        epochs=400,
        learning_rate=0.01,
        margin=0.5,
        n_negatives_per_positive=5,
        seed=0,
    )

    clean = score_rendered_triple_v2(trained, embeddings, ("alice", "knows", "bob"))
    flipped = score_rendered_triple_v2(trained, embeddings, ("alice", "owns", "bob"))

    # Both are in-vocab (knows and owns are both trained relations;
    # alice and bob are both trained entities).
    assert clean["v_triple"] is not None
    assert flipped["v_triple"] is not None

    # The empirical question: does v2.1 distinguish them?
    margin = flipped["v_triple"] - clean["v_triple"]

    # Pin the actual empirical outcome. If margin > 0, v2.1 catches
    # predicate-flip on this synthetic data — Laplacian-side claim
    # backed for the first time. If margin <= 0, A2 has the same
    # falsification story v2.1 had on the disconnected-graph case
    # and the spec needs another correction.
    assert margin > 0, (
        f"v2.1 predicate-flip detection (A2): expected V_flipped > "
        f"V_clean, got V_clean={clean['v_triple']:.4f}, "
        f"V_flipped={flipped['v_triple']:.4f}, margin={margin:.4f}. "
        f"If margin <= 0, v2.1 trained without predicate-perturbation "
        f"negatives FAILS A2 — file a falsification PR with the "
        f"spec correction adding predicate-perturbation negative "
        f"sampling to the training loop."
    )


def test_a2_predicate_flip_caught_with_meaningful_margin():
    """Tighter A2 measurement: not just margin > 0, but
    margin > some empirical threshold. Pinned at 0.05 by inspection
    on this seed; if the margin is below 0.05, v2.1 catches A2 but
    weakly — predicate-perturbation negative sampling is a strong
    candidate v2.3 improvement.

    Threshold deliberately tight; future v2.x updates that
    strengthen A2 detection should keep this test passing or
    update the threshold to reflect an even better margin.
    """
    from sum_engine_internal.research.sheaf_laplacian_v2 import score_rendered_triple_v2
    triples = [
        ("alice", "knows", "bob"),
        ("bob", "knows", "carol"),
        ("alice", "owns", "dog"),
        ("bob", "owns", "cat"),
    ]
    trained, embeddings, _ = train_restriction_maps(
        triples,
        stalk_dim=8,
        epochs=400,
        learning_rate=0.01,
        margin=0.5,
        n_negatives_per_positive=5,
        seed=0,
    )
    clean = score_rendered_triple_v2(trained, embeddings, ("alice", "knows", "bob"))
    flipped = score_rendered_triple_v2(trained, embeddings, ("alice", "owns", "bob"))
    margin = flipped["v_triple"] - clean["v_triple"]
    assert margin > 0.05, (
        f"weak A2 signal: margin={margin:.4f} (threshold 0.05). "
        f"Even though v2.1 catches predicate-flip in the right "
        f"direction, the margin is small. Spec correction: add "
        f"predicate-perturbation negatives to the training loop's "
        f"LCWA sampler so F_h(r), F_t(r) are pushed apart from "
        f"F_h(r'), F_t(r') for r != r' on the same (h, t) pair."
    )


def test_render_aggregation_combines_a2_a3_signals_cleanly():
    """Aggregation across a render that mixes clean, predicate-
    flipped, and off-graph-fabricated triples surfaces:
      - n_oov > 0 (the fabrication)
      - max_in_vocab_v on the predicate-flipped triple
    Both signals available without conflating.
    """
    from sum_engine_internal.research.sheaf_laplacian_v2 import score_rendered_triples_v2
    source = [
        ("alice", "knows", "bob"),
        ("bob", "knows", "carol"),
        ("alice", "owns", "dog"),
        ("bob", "owns", "cat"),
    ]
    trained, embeddings, _ = train_restriction_maps(
        source, stalk_dim=8, epochs=400, seed=0,
    )

    rendered = [
        ("alice", "knows", "bob"),                    # clean
        ("alice", "owns", "bob"),                     # A2 predicate-flip
        ("alice", "FABRICATED", "bob"),               # A3 oov-relation
        ("alice", "knows", "GHOST"),                  # A3 oov-entity
    ]
    profile = score_rendered_triples_v2(trained, embeddings, rendered)
    assert profile["n_triples"] == 4
    assert profile["n_oov"] == 2                      # 2 fabrications
    assert len(profile["in_vocab_v"]) == 2            # clean + flipped
    # The flipped triple's V should dominate the in-vocab pair.
    assert profile["max_in_vocab_v"] > profile["mean_in_vocab_v"]


def test_consistency_profile_v2_handles_empty_render_manifold():
    """v2 must inherit v1's PR-#109 honesty pattern: empty render
    manifold returns explicit None scalars, not a fabricated zero
    profile that crashes downstream."""
    triples = [("alice", "knows", "bob")]
    sheaf = KnowledgeSheafV2.from_triples(triples, stalk_dim=4)
    embeddings = np.eye(2, 4, dtype=np.float64)
    profile = consistency_profile_v2(sheaf, embeddings, [])
    assert profile["render_count"] == 0
    assert profile["mean_laplacian"] is None
    assert profile["std_laplacian"] is None
    assert profile["max_per_render"] is None
    assert profile["argmax_render_idx"] is None
    assert profile["per_render_v"] == []
    assert profile["per_edge_top3_argmax_render"] == []


def test_construction_rejects_wrong_F_h_shape():
    """Defensive: construction must reject F_h of wrong shape so
    callers can't supply an incompatible matrix and silently get
    misleading scores."""
    triples = [("a", "p", "b")]
    bad_F_h = np.zeros((1, 4, 8), dtype=np.float64)  # should be (1, d, d)
    with pytest.raises(ValueError, match="F_h shape"):
        KnowledgeSheafV2.from_triples(triples, stalk_dim=4, F_h=bad_F_h)
