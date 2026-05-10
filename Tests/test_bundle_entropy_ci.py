"""Bundle metadata: axiom_graph_entropy_ci field.

Pins the substrate-integration contract for the calibrated
entropy-CI wrapping (PR #183 conformal kernel + PR #184 vN
entropy + the calibration baseline). The field:

  - Is a two-element ``[lower, upper]`` list at α=0.10
  - Is computed automatically at every `export_bundle` call when
    the calibration baseline is available
  - Is OUTSIDE the signed payload — its presence does not change
    `signature` / `public_signature`
  - Is None for empty bundles (and stripped from the JSON output)
  - Failures NEVER block attestation
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


def test_bundle_includes_axiom_graph_entropy_ci_field(codec_and_algebra):
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("alice", "knows", "bob"),
        ("bob", "owns", "rex"),
        ("carol", "writes", "code"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    assert "axiom_graph_entropy_ci" in bundle
    ci = bundle["axiom_graph_entropy_ci"]
    assert isinstance(ci, list)
    assert len(ci) == 2
    lower, upper = ci
    assert isinstance(lower, float) and isinstance(upper, float)
    assert lower <= upper


def test_actual_entropy_typically_falls_within_calibrated_ci(codec_and_algebra):
    """The bundle's own entropy is calibrated to be in-distribution
    (the corpora used for calibration are the SAME corpora these
    triples come from). For at least one realistic-sized bundle,
    the actual entropy should fall inside the CI."""
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("alice", "knows", "bob"),
        ("bob", "owns", "rex"),
        ("carol", "writes", "code"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    actual = bundle["axiom_graph_entropy"]
    lower, upper = bundle["axiom_graph_entropy_ci"]
    assert lower <= actual <= upper, (
        f"actual entropy {actual:.4f} outside calibrated CI "
        f"[{lower:.4f}, {upper:.4f}] — calibration may have drifted "
        f"or this triple set is genuinely atypical"
    )


def test_empty_bundle_omits_entropy_ci_field(codec_and_algebra):
    """state=1 means no axioms; CI is None → stripped from output."""
    codec, _ = codec_and_algebra
    bundle = codec.export_bundle(1, branch="empty")
    assert bundle.get("axiom_count") == 0
    assert "axiom_graph_entropy_ci" not in bundle


def test_signature_unchanged_by_entropy_ci_field(codec_and_algebra):
    """The CI field MUST NOT alter `signature` — it's outside the
    signed payload (`canonical_tome | state_integer | timestamp`).
    If this fails, the wiring has sneaked the CI into the signed
    payload (a wire-format break)."""
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [("alice", "knows", "bob")])
    bundle = codec.export_bundle(state, branch="t")
    sig_check = codec._sign(
        bundle["canonical_tome"],
        bundle["state_integer"],
        bundle["timestamp"],
    )
    assert sig_check == bundle["signature"]


def test_round_trip_import_succeeds_with_ci_field(codec_and_algebra):
    """import_bundle must accept bundles with the new field —
    invisible to the importer."""
    codec, algebra = codec_and_algebra
    state = _state_from(algebra, [
        ("alice", "knows", "bob"), ("bob", "owns", "rex"),
    ])
    bundle = codec.export_bundle(state, branch="t")
    assert codec.import_bundle(bundle) == state


def test_failed_ci_computation_does_not_block_attestation(monkeypatch):
    """If the CI helper raises, attestation MUST still succeed —
    the CI is non-load-bearing metadata. Defense-in-depth."""
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec
    from sum_engine_internal.infrastructure import canonical_codec as cc

    def _broken(*a, **kw):
        raise RuntimeError("forced failure for test")
    monkeypatch.setattr(cc, "_compute_axiom_graph_entropy_ci", _broken)

    algebra = GodelStateAlgebra()
    gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(algebra, gen, signing_key="k")
    state = algebra.encode_chunk_state([("a", "p", "b")])
    # Should NOT raise — bundle still attests, CI is None
    bundle = codec.export_bundle(state, branch="t")
    assert bundle["axiom_count"] == 1
    assert "axiom_graph_entropy_ci" not in bundle
    # Round-trip still works
    assert codec.import_bundle(bundle) == state


# -- Predictor-level tests ---------------------------------------------


def test_predictor_calibrates_from_baseline_file():
    from sum_engine_internal.research.conformal import BaselineEntropyPredictor
    p = BaselineEntropyPredictor()
    ok = p.calibrate_from_baseline()
    assert ok is True
    assert p.is_calibrated
    assert p.n_calibration_pairs > 0
    assert p.n_training_pairs > 0


def test_predictor_returns_none_for_nonpositive_axiom_count():
    from sum_engine_internal.research.conformal import get_default_predictor
    p = get_default_predictor()
    assert p.predict_ci(0) is None
    assert p.predict_ci(-1) is None


def test_predictor_ci_grows_with_axiom_count_in_log_scale():
    """Predicted entropy is a + b · log(1 + n); larger n →
    larger predicted point. CI width is constant (absolute-score
    conformal) so just check the point monotonically increases."""
    from sum_engine_internal.research.conformal import get_default_predictor
    p = get_default_predictor()
    iv_small = p.predict_ci(2)
    iv_large = p.predict_ci(50)
    assert iv_small.point < iv_large.point


def test_uncalibrated_predictor_returns_none(tmp_path):
    """Cold start: if the baseline file is missing,
    is_calibrated stays False and predict_ci returns None."""
    from sum_engine_internal.research.conformal import BaselineEntropyPredictor
    p = BaselineEntropyPredictor()
    # Point at a path that doesn't exist
    ok = p.calibrate_from_baseline(tmp_path / "nonexistent.json")
    assert ok is False
    assert not p.is_calibrated
    assert p.predict_ci(5) is None
