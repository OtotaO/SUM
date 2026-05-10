"""Bundle metadata: axiom_distribution_mmd field.

Pins wire #4 — Maximum Mean Discrepancy² between the bundle's
axiom-set distribution and the substrate's calibration baseline.
Same architectural discipline as wires #1, #2, #3:

  - Field present + correct shape on non-empty bundles
  - OUTSIDE the signed payload — does not change `signature`
  - None for empty bundles (stripped from output)
  - Failures NEVER block attestation (defense-in-depth at
    helper + call site)
  - Independent from the other three metadata fields
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


def test_bundle_includes_axiom_distribution_mmd_field(codec_and_algebra):
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("alice", "build", "house"),
        ("bob", "write", "book"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    assert "axiom_distribution_mmd" in bundle
    mmd = bundle["axiom_distribution_mmd"]
    assert isinstance(mmd, dict)
    for k in ("mmd_squared", "bandwidth", "n_baseline_samples", "n_bundle_samples"):
        assert k in mmd
    assert mmd["mmd_squared"] >= 0
    assert mmd["bandwidth"] > 0
    assert mmd["n_baseline_samples"] >= 100  # baseline calibrates on seed corpora
    assert mmd["n_bundle_samples"] == 2


def test_empty_bundle_omits_distribution_mmd_field(codec_and_algebra):
    codec, _ = codec_and_algebra
    bundle = codec.export_bundle(1, branch="empty")
    assert bundle.get("axiom_count") == 0
    assert "axiom_distribution_mmd" not in bundle


def test_signature_unchanged_by_distribution_mmd_field(codec_and_algebra):
    """The MMD field MUST NOT alter `signature` — it's outside the
    signed payload (`canonical_tome | state_integer | timestamp`)."""
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [("alice", "build", "house")])
    bundle = codec.export_bundle(state, branch="t")
    sig_check = codec._sign(
        bundle["canonical_tome"],
        bundle["state_integer"],
        bundle["timestamp"],
    )
    assert sig_check == bundle["signature"]


def test_round_trip_import_succeeds_with_mmd_field(codec_and_algebra):
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("alice", "build", "house"),
        ("bob", "write", "book"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    assert codec.import_bundle(bundle) == state


def test_failed_mmd_computation_does_not_block_attestation(monkeypatch):
    """Defense-in-depth canary. If the helper raises, attestation
    MUST still succeed."""
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec
    from sum_engine_internal.infrastructure import canonical_codec as cc

    def _broken(*a, **kw):
        raise RuntimeError("forced failure for test")
    monkeypatch.setattr(cc, "_compute_axiom_distribution_mmd", _broken)

    algebra = GodelStateAlgebra()
    gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(algebra, gen, signing_key="k")
    state = algebra.encode_chunk_state([("a", "build", "b")])
    bundle = codec.export_bundle(state, branch="t")
    assert bundle["axiom_count"] == 1
    assert "axiom_distribution_mmd" not in bundle  # None → stripped
    assert codec.import_bundle(bundle) == state


def test_all_four_metadata_fields_independent(codec_and_algebra):
    """Pin that the four bundle-metadata fields (entropy, entropy_ci,
    consistency_check, distribution_mmd) coexist without
    interfering with each other."""
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("alice", "build", "house"),
        ("bob", "write", "book"),
        ("carol", "discover", "fact"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    assert "axiom_graph_entropy" in bundle
    assert "axiom_graph_entropy_ci" in bundle
    assert "axiom_consistency_check" in bundle
    assert "axiom_distribution_mmd" in bundle
    # Each is its own type
    assert isinstance(bundle["axiom_graph_entropy"], float)
    assert isinstance(bundle["axiom_graph_entropy_ci"], list)
    assert isinstance(bundle["axiom_consistency_check"], dict)
    assert isinstance(bundle["axiom_distribution_mmd"], dict)


def test_in_distribution_bundle_yields_finite_mmd(codec_and_algebra):
    """A bundle with seed-corpus-style triples should produce a
    finite, reasonably-small MMD against the baseline (because it
    *is* drawn from the calibration distribution). Sanity check
    that we're not getting NaN / inf."""
    import math
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("alice", "build", "house"),
        ("bob", "write", "book"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    mmd_val = bundle["axiom_distribution_mmd"]["mmd_squared"]
    assert math.isfinite(mmd_val)
    assert mmd_val >= 0
    # Loose upper bound — RBF MMD² is bounded by 2 in absolute terms
    assert mmd_val < 2.0
