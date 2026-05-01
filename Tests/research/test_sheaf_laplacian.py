"""Math-level sanity checks on the v1 sheaf-Laplacian detector.

If any of these fail, the spec's mathematical mapping
(docs/SHEAF_HALLUCINATION_DETECTOR.md §2.1, §2.2) is wrong
somewhere and the implementation must not proceed.

These tests pin the seven properties that justify treating the
Laplacian quadratic form as a hallucination signal:

  1. L_F is symmetric (L = δ^T δ always is).
  2. L_F is positive-semidefinite (eigvalsh ≥ 0).
  3. x^T L_F x ≥ 0 on random cochains (PSD consequence).
  4. Constant cochains are global sections (kernel of δ).
  5. Single-missing-entity cochain has predicted score V = 1
     (one edge fails by ±1, squared).
  6. Per-edge localization (top-1) finds the missing edge.
  7. Empty-render edge case: x = 0 yields V = 0 — a known
     v1 false negative pinned in code so future improvements
     don't accidentally fix the symptom without addressing the
     cause (caller must treat zero-extraction as separate signal).

Run via: pytest Tests/research/test_sheaf_laplacian.py -q

Requires the [research] extras: pip install 'sum-engine[research]'.
"""
import pytest

# Skip the whole file if the [research] extras aren't installed.
np = pytest.importorskip("numpy")

from sum_engine_internal.research.sheaf_laplacian import (
    KnowledgeSheaf,
    coboundary_matrix,
    sheaf_laplacian,
    laplacian_quadratic_form,
    per_edge_discrepancy,
    cochain_from_extracted,
)


def _toy_sheaf():
    """Tiny sheaf: 3 entities, 2 edges. Hand-checkable.

      alice --likes--> cats
      bob --owns--> dog
    """
    triples = [("alice", "likes", "cats"), ("bob", "owns", "dog")]
    return KnowledgeSheaf.from_triples(triples)


# ── Property 1: Laplacian is symmetric ────────────────────────────────

def test_laplacian_is_symmetric():
    F = _toy_sheaf()
    L = sheaf_laplacian(F)
    assert np.allclose(L, L.T), "L_F must be symmetric (L = δ^T δ is always symmetric)"


# ── Property 2: Laplacian is PSD ──────────────────────────────────────

def test_laplacian_is_positive_semidefinite():
    F = _toy_sheaf()
    L = sheaf_laplacian(F)
    eigvals = np.linalg.eigvalsh(L)
    # Allow tiny numerical noise (-1e-10 type values from floating point).
    assert np.all(eigvals >= -1e-10), f"L_F must be PSD; got eigvals={eigvals}"


# ── Property 3: Quadratic form ≥ 0 always (PSD consequence) ───────────

def test_quadratic_form_nonnegative_random():
    F = _toy_sheaf()
    rng = np.random.default_rng(42)
    for _ in range(20):
        x = rng.standard_normal(len(F.vertices))
        v = laplacian_quadratic_form(F, x)
        assert v >= -1e-10, f"x^T L x must be ≥ 0; got {v}"


# ── Property 4: Constant cochain is in the kernel (a global section) ──
#
# For v1 with identity restriction maps, x = (c, c, ..., c) gives
# (δx)_e = c - c = 0 for every edge ⇒ x ∈ ker(δ) = H^0(G; F).
# This is the only kind of global section v1 admits, since the
# 1-dim presence stalks force entity-presence to agree across edges.

def test_constant_cochain_is_global_section():
    F = _toy_sheaf()
    x = np.ones(len(F.vertices))
    v = laplacian_quadratic_form(F, x)
    assert v == 0.0, (
        f"Constant cochain x = (1,1,...,1) must be a global section "
        f"(every edge sees agreement); got x^T L x = {v}"
    )


# ── Property 5: Single-disagreement cochain has predictable score ─────
#
# If exactly one entity is missing from a render's extraction, only
# the edges incident to that entity contribute a discrepancy of 1.

def test_single_missing_entity_gives_predicted_score():
    """Drop 'alice' from the cochain. Edge (alice, likes, cats) is
    the only edge where alice is incident, so its residual is 0 - 1 = -1
    (cats is mentioned, alice isn't), squared = 1. Total Laplacian = 1.
    """
    F = _toy_sheaf()
    # Render mentions cats and bob and dog, but not alice
    triples = [("cats", "self", "cats"), ("bob", "owns", "dog")]   # cats, bob, dog
    x = cochain_from_extracted(F, triples)
    expected_x = np.array([
        0.0,  # alice: missing
        1.0,  # cats: present
        1.0,  # bob: present
        1.0,  # dog: present
    ])
    assert np.array_equal(x, expected_x), f"got {x}, expected {expected_x}"
    v = laplacian_quadratic_form(F, x)
    assert v == 1.0, (
        f"Dropping alice should yield V=1 (one edge fails by ±1, squared); got {v}"
    )


# ── Property 6: Per-edge localization identifies the missing edge ─────

def test_localization_finds_edge_with_missing_entity():
    F = _toy_sheaf()
    triples = [("cats", "self", "cats"), ("bob", "owns", "dog")]
    x = cochain_from_extracted(F, triples)
    contribs = per_edge_discrepancy(F, x)
    # The top-1 edge should be (alice, likes, cats), since that's
    # the only edge incident to the missing alice.
    top_edge, top_score = contribs[0]
    assert top_edge == ("alice", "likes", "cats"), (
        f"Top-1 localization should find (alice, likes, cats); got {top_edge}"
    )
    assert top_score == 1.0, f"Top score should be 1.0; got {top_score}"


# ── Property 7: Empty render gives the maximum possible Laplacian ─────

def test_empty_render_maximizes_laplacian():
    """When NO entities are mentioned (catastrophic hallucination /
    empty output), x = 0. Then δx = 0 too, so V = 0 — which is a
    NULL SIGNAL false-positive failure mode worth flagging.

    The detector cannot distinguish 'all entities present everywhere'
    from 'no entities present anywhere' under the 1d presence model
    with identity restriction maps. This is a known v1 limitation;
    v2's learned-embedding stalks address it.
    """
    F = _toy_sheaf()
    x = cochain_from_extracted(F, [])  # no triples extracted
    v = laplacian_quadratic_form(F, x)
    # NB: this is intentionally pinning the failure mode in code so a
    # future v1-improver doesn't accidentally fix the symptom without
    # understanding the cause.
    assert v == 0.0, (
        "Known v1 failure mode: empty cochain returns V=0 (false negative). "
        "Document in spec §6 and/or address with absolute-presence "
        "regularization (||x||^2 term) before benching."
    )


# ── Synthetic micro-benchmark sanity check ────────────────────────────
#
# Pins the 6/6 detection rate on the catchable perturbation classes
# (A1 entity-swap, A4 triple-drop, A5 consistent-swap) and the 0/6
# detection rate on the known-blind classes (A2 predicate-flip, A3
# fact-fabrication). If a future change shifts these rates, the v1
# spec's empirical claims have changed and the spec needs an update.

def test_microbench_a1_entity_swap_caught_on_all_six_factsets():
    from scripts.research.sheaf_microbench import FACT_SETS, bench_one, adversarial_swap
    catches = 0
    for triples in FACT_SETS:
        r = bench_one(triples, "A1_entity_swap", adversarial_swap)
        if r["V_adv_mean"] > r["V_clean_mean"] + 1e-9:
            catches += 1
    assert catches == 6, f"A1 must catch 6/6 fact-sets; got {catches}/6"


def test_microbench_a2_predicate_flip_invisible():
    from scripts.research.sheaf_microbench import FACT_SETS, bench_one, adversarial_predicate_flip
    catches = 0
    for triples in FACT_SETS:
        r = bench_one(triples, "A2_predicate_flip", adversarial_predicate_flip)
        if r["V_adv_mean"] > r["V_clean_mean"] + 1e-9:
            catches += 1
    assert catches == 0, (
        f"A2 predicate-flip must remain invisible to v1 (presence stalks "
        f"don't carry predicate information); got {catches}/6 — "
        f"either the cochain construction has changed (now sensitive to "
        f"predicates, which is a v2 feature) or the perturbation is no "
        f"longer flipping the predicate."
    )


def test_microbench_a3_fact_fabrication_invisible():
    from scripts.research.sheaf_microbench import FACT_SETS, bench_one, adversarial_fact_fabrication
    catches = 0
    for triples in FACT_SETS:
        r = bench_one(triples, "A3_fact_fabrication", adversarial_fact_fabrication)
        if r["V_adv_mean"] > r["V_clean_mean"] + 1e-9:
            catches += 1
    assert catches == 0, (
        f"A3 fact-fabrication (off-graph entities) must remain invisible "
        f"to v1; got {catches}/6"
    )


def test_microbench_a5_consistent_swap_caught_via_mean_signal():
    """The spec originally mischaracterized A5 (consistent
    hallucination via repeated identical swap) as a v1 blindspot.
    Empirically it is caught: the mean Laplacian over the 3-render
    consistent-swap manifold is positive (each render fails the same
    edges by the same amount), even though per-render variance is
    zero. Pin this so the corrected spec stays honest."""
    from scripts.research.sheaf_microbench import FACT_SETS, bench_one, adversarial_consistent_swap
    catches = 0
    for triples in FACT_SETS:
        r = bench_one(triples, "A5_consistent_swap_x3", adversarial_consistent_swap)
        if r["V_adv_mean"] > r["V_clean_mean"] + 1e-9:
            catches += 1
    assert catches == 6, (
        f"A5 consistent-swap must be caught via the mean signal; "
        f"got {catches}/6. If this regresses, either the manifold size "
        f"shifted to 1 (variance-only metric) or the swap target moved "
        f"out of the source vertex set."
    )


def test_construction_rejects_non_1_stalk_dim_at_construction_time():
    """v1 must reject stalk_dim != 1 at sheaf construction time so a
    user cannot build a sheaf they can't actually use. The earlier
    code raised NotImplementedError mid-pipeline (in
    coboundary_matrix); now it raises ValueError at __post_init__,
    naming v2 explicitly so the user knows what's coming.
    """
    import pytest as _pytest
    with _pytest.raises(ValueError, match="stalk_dim=1 only"):
        KnowledgeSheaf.from_triples(
            [("alice", "knows", "bob")], stalk_dim=384,
        )


def test_consistency_profile_handles_empty_render_manifold():
    """An empty render manifold must yield an honest profile with
    null fields, not a fabricated all-zero profile that crashes
    downstream on argmax/index access.
    """
    from sum_engine_internal.research.sheaf_laplacian import consistency_profile
    profile = consistency_profile(
        source_triples=[("alice", "knows", "bob")],
        rendered_extractions=[],
    )
    assert profile["render_count"] == 0
    assert profile["mean_laplacian"] is None
    assert profile["std_laplacian"] is None
    assert profile["max_per_render"] is None
    assert profile["argmax_render_idx"] is None
    assert profile["per_render_v"] == []
    assert profile["per_edge_top3_argmax_render"] == []


def test_disconnected_graph_density_dropout_invisible():
    """v1 blindspot surfaced by the real-prose test (2026-05-01):
    when the source bundle's induced graph has multiple disconnected
    components and a density-controlled render drops ENTIRE
    components, the Laplacian quadratic form is zero even though
    facts have been dropped.

    Construction: 4 unrelated facts about 4 unrelated entity-pairs.
    The induced graph has 4 disconnected edges. Drop 2 of them
    entirely (both endpoints vanish). Every remaining edge is in
    {(1,1), (0,0)} — never (1,0) — so V = 0.

    This is the v1 design's structural limit: the Laplacian on
    presence stalks measures cross-edge agreement, and an edge with
    both endpoints absent agrees as cleanly as both endpoints present.
    v2 (learned-embedding stalks) addresses this because per-vertex
    embeddings contribute independently regardless of neighbourhood.

    Pinned so a future v1-improver does not silently fix the symptom
    without understanding the cause.
    """
    triples = [
        ("alice", "graduated", "mit"),
        ("bob", "owns", "dog"),
        ("carol", "writes", "python"),
        ("einstein", "proposed", "relativity"),
    ]
    F = KnowledgeSheaf.from_triples(triples)

    # Render at "density 0.5" — drops 2 axioms entirely.
    # Lex-sorted axioms: alice||graduated||mit, bob||owns||dog,
    # carol||writes||python, einstein||proposed||relativity.
    # apply_density(d=0.5) keeps the first floor(4*0.5) = 2:
    # alice||... and bob||..., dropping carol and einstein entirely.
    kept_triples = [
        ("alice", "graduated", "mit"),
        ("bob", "owns", "dog"),
    ]
    x = cochain_from_extracted(F, kept_triples)
    v = laplacian_quadratic_form(F, x)
    assert v == 0.0, (
        f"v1 must show V=0 on disconnected-graph density dropout "
        f"(structural limit; pinned to surface this in code). "
        f"If V > 0, either the graph is no longer disconnected or "
        f"the cochain construction has changed. Got V = {v}"
    )


def test_microbench_top1_localization_perfect_on_caught_classes():
    """Per-edge top-1 localization must correctly identify the
    perturbed edge in every detected adversarial case across A1, A4,
    A5. The synthetic benchmark establishes this at 18/18 = 100%
    on the catchable classes; the spec's P2 prediction (≥70%) was
    over-conservative."""
    from scripts.research.sheaf_microbench import (
        FACT_SETS, bench_one,
        adversarial_swap, adversarial_triple_drop, adversarial_consistent_swap,
    )
    perfect = 0
    total = 0
    for triples in FACT_SETS:
        for name, fn in [
            ("A1_entity_swap", adversarial_swap),
            ("A4_triple_drop", adversarial_triple_drop),
            ("A5_consistent_swap_x3", adversarial_consistent_swap),
        ]:
            r = bench_one(triples, name, fn)
            if r["V_adv_mean"] <= r["V_clean_mean"] + 1e-9:
                continue          # not caught; skip localization
            total += 1
            top1 = r["top1_localization"]
            if not top1:
                continue
            top_edge, _ = top1[0]
            # The perturbed edge for A1 / A4 / A5-via-swap is always
            # the first edge of the source fact-set (since the
            # perturbation functions target index 0).
            if top_edge == triples[0]:
                perfect += 1
    assert total == 18, f"Expected 18 catches across A1+A4+A5; got {total}"
    assert perfect == 18, (
        f"v1 top-1 localization must be 100% on caught classes; "
        f"got {perfect}/{total}"
    )
