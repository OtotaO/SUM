"""Bundle metadata: axiom_distribution_mmd_threshold (K3).

Pins the conformal-style size-stratified threshold field.
Same architectural discipline as wires #1-#4:

  - Field present + correct shape on non-empty bundles
  - Size-stratified calibration eliminates the
    smaller-sample-larger-MMD² confounder
  - OUTSIDE the signed payload — does not change `signature`
  - None for empty bundles (stripped from output)
  - Failures NEVER block attestation (defense-in-depth)
  - Threshold field independent from but consistent with
    upstream `axiom_distribution_mmd` field
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


# -- Field shape -------------------------------------------------------


def test_bundle_includes_axiom_distribution_mmd_threshold_field(codec_and_algebra):
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("alice", "build", "house"),
        ("bob", "write", "book"),
        ("carol", "discover", "fact"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    assert "axiom_distribution_mmd_threshold" in bundle
    th = bundle["axiom_distribution_mmd_threshold"]
    assert isinstance(th, dict)
    for k in ("threshold_alpha", "threshold_value", "exceeds_threshold",
              "n_calibration_samples", "calibration_size_used"):
        assert k in th, f"missing key: {k}"
    assert 0 < th["threshold_alpha"] < 1
    assert th["threshold_value"] >= 0
    assert isinstance(th["exceeds_threshold"], bool)
    assert th["n_calibration_samples"] > 0
    assert th["calibration_size_used"] > 0


def test_threshold_calibration_size_matches_bundle_size_when_available(codec_and_algebra):
    """Size-stratified discipline: a bundle of N triples should
    use the calibration-size closest to N. With sizes
    (1, 2, 3, 5, 10, 20, 50) and bundle=3, expect calibration_size=3."""
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("alice", "build", "house"),
        ("bob", "write", "book"),
        ("carol", "discover", "fact"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    th = bundle["axiom_distribution_mmd_threshold"]
    # Bundle has 3 axioms; the closest stratified size is 3
    assert th["calibration_size_used"] == 3


def test_in_distribution_bundle_does_not_exceed_threshold(codec_and_algebra):
    """K3 substrate-level claim: an in-distribution bundle's
    MMD² should NOT exceed the size-matched threshold. If this
    fails after size-stratification was meant to fix it, we've
    reintroduced the smaller-sample confounder."""
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("alice", "build", "house"),
        ("bob", "write", "book"),
        ("carol", "discover", "fact"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    assert bundle["axiom_distribution_mmd_threshold"]["exceeds_threshold"] is False


# -- Architectural discipline ------------------------------------------


def test_empty_bundle_omits_threshold_field(codec_and_algebra):
    codec, _ = codec_and_algebra
    bundle = codec.export_bundle(1, branch="empty")
    assert "axiom_distribution_mmd_threshold" not in bundle


def test_signature_unchanged_by_threshold_field(codec_and_algebra):
    """Outside-signed-payload invariant. K3 must not affect
    `signature` — the signed payload is
    canonical_tome|state_integer|timestamp."""
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [("alice", "build", "house")])
    bundle = codec.export_bundle(state, branch="t")
    sig_check = codec._sign(
        bundle["canonical_tome"],
        bundle["state_integer"],
        bundle["timestamp"],
    )
    assert sig_check == bundle["signature"]


def test_round_trip_import_succeeds_with_threshold_field(codec_and_algebra):
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("alice", "build", "house"),
        ("bob", "write", "book"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    assert codec.import_bundle(bundle) == state


def test_failed_threshold_computation_does_not_block_attestation(monkeypatch):
    """Defense-in-depth canary."""
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec
    from sum_engine_internal.infrastructure import canonical_codec as cc

    def _broken(*a, **kw):
        raise RuntimeError("forced failure for test")
    monkeypatch.setattr(cc, "_compute_axiom_distribution_mmd_threshold", _broken)

    algebra = GodelStateAlgebra()
    gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(algebra, gen, signing_key="k")
    state = algebra.encode_chunk_state([("a", "build", "b")])
    bundle = codec.export_bundle(state, branch="t")
    assert bundle["axiom_count"] == 1
    assert "axiom_distribution_mmd_threshold" not in bundle
    assert codec.import_bundle(bundle) == state


def test_threshold_value_consistent_with_calibration_distribution(codec_and_algebra):
    """The threshold must come from the (1-α)-quantile of the
    calibration MMD² distribution at the chosen size. Verify by
    cross-checking against the computer's predict_threshold
    directly."""
    from sum_engine_internal.research.mmd import get_default_mmd_computer
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("alice", "build", "house"),
        ("bob", "write", "book"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    th_bundle = bundle["axiom_distribution_mmd_threshold"]
    mmd_bundle = bundle["axiom_distribution_mmd"]

    # Independent recomputation via the computer's API
    c = get_default_mmd_computer()
    th_direct = c.predict_threshold(
        observed_mmd_squared=mmd_bundle["mmd_squared"],
        bundle_size=mmd_bundle["n_bundle_samples"],
        alpha=0.10,
    )
    # Threshold value should agree exactly (deterministic given
    # the same calibration state)
    assert th_bundle["threshold_value"] == th_direct["threshold_value"]
    assert th_bundle["calibration_size_used"] == th_direct["calibration_size_used"]
