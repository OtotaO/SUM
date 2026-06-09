"""Per-document meaning readout — `explain_meaning_loss` / `EntailmentScorer.explain`.

The readout is the #1 thing the 30-guest adoption sim asked for: "what changed
in MY text?" — what a transform kept, dropped, and added. It is a per-document
MEASUREMENT, not a certified bound; the load-bearing contract is that its
``loss`` equals exactly what ``EntailmentScorer.loss`` certifies (same number,
decomposed to claims), so a per-doc readout and a corpus certificate never
disagree about the number.

Torch-free: tested with a deterministic STUB entails judge (substring), so the
decomposition + loss-match are pinned in CI without a model.
"""
from __future__ import annotations

from sum_engine_internal.research.meaning.meaning_loss import (
    EntailmentScorer,
    explain_meaning_loss,
)


def _stub(premise: str, hypothesis: str) -> bool:
    # deterministic, model-free: the claim's text appears verbatim in the premise
    return hypothesis.strip().lower() in premise.strip().lower()


_SRC = "Alpha one. Bravo two. Charlie three."


def _explain(source, transform):
    return explain_meaning_loss(
        source, transform, entails=_stub, judge_name="stub", judge_version="1"
    )


def test_readout_loss_equals_scorer_loss():
    """The contract: the readout's loss is exactly the scorer's per-pair loss."""
    scorer = EntailmentScorer(entails=_stub, judge_name="stub", judge_version="1")
    for transform in (_SRC, "Alpha one. Bravo two.", "Delta four.", ""):
        r = explain_meaning_loss(_SRC, transform, entails=_stub,
                                 judge_name="stub", judge_version="1")
        assert abs(r.loss - scorer.loss(_SRC, transform)) < 1e-12
        # and the method delegates identically
        assert scorer.explain(_SRC, transform).loss == r.loss


def test_faithful_drops_nothing():
    r = _explain(_SRC, _SRC)
    assert r.loss == 0.0
    assert r.dropped_claims == () and r.unsupported_claims == ()
    assert r.preserved_claims == r.source_claims == 3


def test_dropped_claim_is_named():
    r = _explain(_SRC, "Alpha one. Bravo two.")  # drops "Charlie three."
    assert "Charlie three." in r.dropped_claims
    assert r.preserved_claims == 2 and r.source_claims == 3
    assert r.unsupported_claims == ()
    assert abs(r.recall - 2 / 3) < 1e-12


def test_added_unsupported_claim_is_flagged():
    r = _explain(_SRC, "Alpha one. Bravo two. Charlie three. Echo five.")
    assert "Echo five." in r.unsupported_claims   # not grounded in source
    assert r.dropped_claims == ()                  # nothing lost


def test_identity_empty_is_zero_loss():
    r = _explain("", "")
    assert r.loss == 0.0 and r.source_claims == 0


def test_source_present_transform_empty_drops_everything():
    # An empty transform omits all source claims (recall 0) but fabricates
    # nothing (fidelity 1.0), so the scorer's composite loss is w_recall=0.6,
    # NOT 1.0. The readout matches the scorer and names every dropped claim.
    r = _explain(_SRC, "")
    assert abs(r.loss - 0.6) < 1e-12
    assert set(r.dropped_claims) == {"Alpha one.", "Bravo two.", "Charlie three."}
    assert r.unsupported_claims == ()


def test_scope_is_honest_about_being_a_measurement():
    r = _explain(_SRC, _SRC)
    s = r.scope.lower()
    assert "measurement" in s and "not a certified bound" in s and "1-δ" in s


def test_weights_must_sum_to_one():
    import pytest
    with pytest.raises(ValueError):
        explain_meaning_loss(_SRC, _SRC, entails=_stub, judge_name="s",
                             w_recall=0.7, w_fidelity=0.4)
