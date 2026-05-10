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
    # K2 expanded shape: K1's four fields + permutation_p_value + n_permutations
    for k in ("mmd_squared", "permutation_p_value", "n_permutations",
              "bandwidth", "n_baseline_samples", "n_bundle_samples"):
        assert k in mmd, f"missing key: {k}"
    assert mmd["mmd_squared"] >= 0
    assert mmd["bandwidth"] > 0
    assert mmd["n_baseline_samples"] >= 100  # baseline calibrates on seed corpora
    assert mmd["n_bundle_samples"] == 2
    # K2: p-value is in (0, 1] when computed (finite-sample correction
    # ensures > 0 even when zero permutations exceed observed)
    assert mmd["n_permutations"] > 0
    assert 0 < mmd["permutation_p_value"] <= 1


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


def test_all_metadata_fields_coexist(codec_and_algebra):
    """Pin that the bundle-metadata fields coexist without
    interfering with each other. Wires #1 (entropy), #2
    (entropy_ci) and #4 (distribution_mmd) are always present
    on non-empty bundles. Wire #3 (consistency_check) requires
    z3-solver — checked conditionally so the test passes in CI
    environments without the optional dep."""
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("alice", "build", "house"),
        ("bob", "write", "book"),
        ("carol", "discover", "fact"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    # Always present
    assert "axiom_graph_entropy" in bundle
    assert "axiom_graph_entropy_ci" in bundle
    assert "axiom_distribution_mmd" in bundle
    assert isinstance(bundle["axiom_graph_entropy"], float)
    assert isinstance(bundle["axiom_graph_entropy_ci"], list)
    assert isinstance(bundle["axiom_distribution_mmd"], dict)
    # Conditional: only present when z3-solver is installed
    try:
        import z3  # noqa: F401
        z3_available = True
    except ImportError:
        z3_available = False
    if z3_available:
        assert "axiom_consistency_check" in bundle
        assert isinstance(bundle["axiom_consistency_check"], dict)
    else:
        assert "axiom_consistency_check" not in bundle


def test_in_distribution_bundle_yields_non_significant_pvalue(codec_and_algebra):
    """K2 substrate-level claim: a bundle drawn from the same
    corpora as the calibration baseline should NOT produce a
    significant permutation p-value (p > 0.05). An in-distribution
    bundle that flags p < 0.05 means either: (a) the baseline has
    drifted, (b) the bundle is genuinely atypical for in-corpus
    content, or (c) the test infrastructure regressed."""
    codec, algebra = codec_and_algebra
    # These triples come from sieve-extractable seed-corpus prose
    state = _state_from(algebra, [
        ("alice", "build", "house"),
        ("bob", "write", "book"),
        ("carol", "discover", "fact"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    p = bundle["axiom_distribution_mmd"]["permutation_p_value"]
    # Loose threshold — p > 0.10 is "clearly not significant"
    # (tighter than 0.05 to give Monte-Carlo flake tolerance)
    assert p > 0.10, (
        f"in-distribution bundle flagged with p={p:.3f} (< 0.10); "
        f"either baseline drifted or this corpus shape is genuinely "
        f"atypical"
    )


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
