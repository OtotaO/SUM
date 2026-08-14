"""Unit tests for the SUM→GEPA meaning-metric adapter (examples/gepa_meaning_metric.py).

Deterministic and dependency-free: uses a fixed token-overlap stand-in judge, so
these run in any job without dspy/gepa or a model download. They pin the contract
GEPA relies on — a higher score for a more faithful transform, and feedback that
names what was dropped/added — plus the field-extraction and no-explain fallbacks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# examples/ is not a package; put it on the path so the adapter is importable.
_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
sys.path.insert(0, str(_EXAMPLES))

from gepa_meaning_metric import (  # noqa: E402
    MeaningSignal,
    make_gepa_metric,
    meaning_signal,
)

from sum_engine_internal.research.meaning.meaning_loss import (  # noqa: E402
    EntailmentScorer,
    LexicalCoverageScorer,
    _content_units,
)

SOURCE = "Alpha happened. Beta happened. Gamma happened."
FAITHFUL = "Alpha happened. Beta happened. Gamma happened."
LOSSY = "Alpha happened. Delta happened."  # drops Beta+Gamma, adds Delta


def _entails(premise: str, hypothesis: str, *, threshold: float = 0.6) -> bool:
    hyp = set(_content_units(hypothesis))
    if not hyp:
        return True
    return len(hyp & set(_content_units(premise))) / len(hyp) >= threshold


def _scorer() -> EntailmentScorer:
    return EntailmentScorer(entails=_entails, judge_name="test-overlap", judge_version="0")


def test_faithful_transform_scores_perfect_with_no_itemised_loss():
    sig = meaning_signal(SOURCE, FAITHFUL, _scorer())
    assert isinstance(sig, MeaningSignal)
    assert sig.loss == 0.0
    assert sig.score == 1.0
    assert sig.readout is not None
    assert sig.readout.dropped_claims == ()
    assert sig.readout.unsupported_claims == ()


def test_lossy_transform_scores_lower_and_names_drops_and_adds():
    faithful = meaning_signal(SOURCE, FAITHFUL, _scorer())
    lossy = meaning_signal(SOURCE, LOSSY, _scorer())
    assert lossy.score < faithful.score  # the gradient GEPA climbs
    assert "DROPPED" in lossy.feedback and "ADDED" in lossy.feedback
    # the actual dropped/added sentences are surfaced, not just counts
    assert any("Beta" in s for s in lossy.readout.dropped_claims)
    assert any("Delta" in t for t in lossy.readout.unsupported_claims)


def test_score_equals_one_minus_loss():
    sig = meaning_signal(SOURCE, LOSSY, _scorer())
    assert abs(sig.score - (1.0 - sig.loss)) < 1e-12


def test_dspy_metric_shape_supports_item_access():
    """Without dspy installed the metric returns a dict-and-attr shim exposing
    ['score']/['feedback'] — exactly the surface dspy.GEPA touches."""
    metric = make_gepa_metric(_scorer(), source_key="source", pred_key="summary")
    out = metric({"source": SOURCE}, {"summary": LOSSY})
    assert set(out.keys()) >= {"score", "feedback"}
    assert out["score"] == out.score  # attr access mirrors item access
    assert 0.0 <= out["score"] <= 1.0


def test_pred_field_autodetected_when_pred_key_omitted():
    metric = make_gepa_metric(_scorer(), source_key="source")  # no pred_key
    # 'summary' is one of the probed default output fields
    out = metric({"source": SOURCE}, {"summary": FAITHFUL})
    assert out["score"] == 1.0


def test_string_prediction_is_accepted():
    metric = make_gepa_metric(_scorer(), source_key="source")
    out = metric({"source": SOURCE}, FAITHFUL)  # pred is a bare string
    assert out["score"] == 1.0


def test_lexical_scorer_has_no_readout_and_feedback_flags_limitation():
    sig = meaning_signal(SOURCE, LOSSY, LexicalCoverageScorer())
    assert sig.readout is None  # lexical scorer has no .explain
    assert "cannot itemise" in sig.feedback
    assert 0.0 <= sig.score <= 1.0


# ---------------------------------------------------------------------------
# Regression pin: a prediction with none of the probed fields must RAISE, not
# have its Python repr scored.
#
# The old fallback was `return str(pred)`. A dspy Prediction whose output field
# is named something unexpected would have "Prediction(foo='...')" scored
# against the source; a faithfulness judge rates that as near-total loss, so
# the optimiser saw score ~0 on a perfectly faithful transform and steered
# away from it. Silently. Inside a THIRD PARTY's optimisation loop.
#
# This is the failure the project exists to prevent, aimed at the adoption
# channel chosen precisely because of the honesty. Fail loudly instead.
# ---------------------------------------------------------------------------

def test_unreadable_prediction_raises_instead_of_scoring_its_repr():
    from examples.gepa_meaning_metric import _extract_pred

    class Odd:
        def __init__(self):
            self.unexpected_field = "a perfectly good summary"

        def __str__(self):
            return "Odd(unexpected_field='a perfectly good summary')"

    with pytest.raises(KeyError) as ei:
        _extract_pred(Odd(), None)
    msg = str(ei.value)
    assert "pred_key" in msg, "the error must tell the caller how to fix it"
    assert "Odd" in msg, "the error must name the offending type"


def test_explicit_pred_key_still_reads_an_unusual_field():
    """The escape hatch the error message points at must actually work."""
    from examples.gepa_meaning_metric import _extract_pred

    class Odd:
        def __init__(self):
            self.unexpected_field = "a perfectly good summary"

    assert _extract_pred(Odd(), "unexpected_field") == "a perfectly good summary"


def test_default_fields_still_resolve_without_pred_key():
    """No regression: the happy path must not have become stricter."""
    from examples.gepa_meaning_metric import _extract_pred

    assert _extract_pred({"summary": "s"}, None) == "s"
    assert _extract_pred("already a string", None) == "already a string"
