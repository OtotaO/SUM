"""Bundle metadata: axiom_corruption_score (Wire #6).

Pins the Robust-PCA corruption-score field shipped on
CanonicalBundle. Same architectural discipline as wires #1-#5:

  - Field present + correct shape on bundles with ≥4 axioms
  - OUTSIDE the signed payload — does not change `signature`
  - None for empty bundles AND bundles with < 4 axioms (PCP
    decomposition is trivial below that)
  - Failures NEVER block attestation (defense-in-depth)
  - Round-trip import succeeds with the field present
"""
from __future__ import annotations

import pytest


@pytest.fixture
def codec_and_algebra():
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec
    algebra = GodelStateAlgebra()
    gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(algebra, gen, signing_key="contract_key")
    return codec, algebra


def _state_from(algebra, triples):
    return algebra.encode_chunk_state(triples)


_FIVE_TRIPLES = [
    ("alice", "build", "house"),
    ("bob", "write", "book"),
    ("carol", "discover", "fact"),
    ("dave", "ship", "code"),
    ("eve", "teach", "class"),
]


# -- Field shape -------------------------------------------------------


def test_bundle_includes_corruption_score_field(codec_and_algebra):
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, _FIVE_TRIPLES)
    bundle = codec.export_bundle(state, branch="t")
    assert "axiom_corruption_score" in bundle
    cs = bundle["axiom_corruption_score"]
    assert isinstance(cs, dict)
    for k in ("max_score", "mean_score", "median_score", "rank_estimate",
              "sparsity_estimate", "n_axioms", "lam"):
        assert k in cs, f"missing key: {k}"
    assert cs["max_score"] >= 0
    assert cs["mean_score"] >= 0
    assert cs["median_score"] >= 0
    assert cs["rank_estimate"] >= 0
    assert 0.0 <= cs["sparsity_estimate"] <= 1.0
    assert cs["n_axioms"] == 5
    assert cs["lam"] > 0


def test_max_score_at_least_mean_score(codec_and_algebra):
    """Per-row max ≥ per-row mean is a definitional invariant."""
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, _FIVE_TRIPLES)
    bundle = codec.export_bundle(state, branch="t")
    cs = bundle["axiom_corruption_score"]
    assert cs["max_score"] >= cs["mean_score"]


# -- Architectural discipline ------------------------------------------


def test_empty_bundle_omits_corruption_score_field(codec_and_algebra):
    codec, _ = codec_and_algebra
    bundle = codec.export_bundle(1, branch="empty")
    assert "axiom_corruption_score" not in bundle


def test_bundle_below_minimum_axioms_omits_corruption_score(codec_and_algebra):
    """Wire #6 is gated at n_axioms ≥ 4: PCP decomposition is
    trivial below that, so we report None rather than a
    misleading scalar."""
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("alice", "build", "house"),
        ("bob", "write", "book"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    # Field stripped from output when None
    assert "axiom_corruption_score" not in bundle


def test_signature_unchanged_by_corruption_score_field(codec_and_algebra):
    """Outside-signed-payload invariant — Wire #6 must not affect
    `signature`. Signed payload is canonical_tome|state_integer|timestamp."""
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, _FIVE_TRIPLES)
    bundle = codec.export_bundle(state, branch="t")
    sig_check = codec._sign(
        bundle["canonical_tome"],
        bundle["state_integer"],
        bundle["timestamp"],
    )
    assert sig_check == bundle["signature"]


def test_round_trip_import_succeeds_with_corruption_score(codec_and_algebra):
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, _FIVE_TRIPLES)
    bundle = codec.export_bundle(state, branch="t")
    assert codec.import_bundle(bundle) == state


def test_failed_corruption_score_does_not_block_attestation(monkeypatch):
    """Defense-in-depth canary: a forced helper exception must
    leave the bundle attestable, just without the field."""
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec
    from sum_engine_internal.infrastructure import canonical_codec as cc

    def _broken(*a, **kw):
        raise RuntimeError("forced failure for test")
    monkeypatch.setattr(cc, "_compute_axiom_corruption_score", _broken)

    algebra = GodelStateAlgebra()
    gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(algebra, gen, signing_key="k")
    state = algebra.encode_chunk_state(_FIVE_TRIPLES)
    bundle = codec.export_bundle(state, branch="t")
    assert bundle["axiom_count"] == 5
    assert "axiom_corruption_score" not in bundle
    assert codec.import_bundle(bundle) == state


# -- Wire-pattern coexistence -----------------------------------------


def test_corruption_score_coexists_with_other_metadata(codec_and_algebra):
    """All applicable wire fields populate together on a bundle
    large enough to exercise all of them. Z3 wire is gated on
    z3-solver being installed."""
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, _FIVE_TRIPLES)
    bundle = codec.export_bundle(state, branch="t")
    assert "axiom_graph_entropy" in bundle
    assert "axiom_distribution_mmd" in bundle
    assert "axiom_corruption_score" in bundle
    try:
        import z3  # noqa: F401
        assert "axiom_consistency_check" in bundle
    except ImportError:
        pass
