"""Regression tests pinning the adversarial-review findings.

Each test corresponds to a confirmed finding from the multi-dimension
review of the meaning-loss module: a defect or overclaim the original
61-test suite did not catch. They are grouped here so the audit trail is
explicit — if any of these regresses, the fix it pins has been undone.
"""
from __future__ import annotations

import copy

import pytest

joserfc = pytest.importorskip(
    "joserfc",
    reason="[receipt-verify] extra not installed",
)

from joserfc.jwk import OKPKey

from sum_engine_internal.research.meaning import (
    EntailmentScorer,
    LexicalCoverageScorer,
    MeaningReceiptDisclosureError,
    MeaningReceiptReplayError,
    build_payload,
    certify_meaning_risk,
    sign_meaning_risk_receipt,
    verify_meaning_risk_receipt,
)


_SCORER = dict(
    scorer_name="lexical-coverage-bidirectional", scorer_version="1"
)


def _keypair(kid="harden-2026"):
    key = OKPKey.generate_key("Ed25519")
    priv = key.as_dict(private=True)
    priv.update(kid=kid, alg="EdDSA", use="sig")
    pub = key.as_dict(private=False)
    pub.update(kid=kid, alg="EdDSA", use="sig")
    return priv, {"keys": [pub]}, kid


def _receipt(losses, *, alpha_target=None, not_covered=None, disclosure=None):
    priv, jwks, kid = _keypair()
    g = certify_meaning_risk(losses, **_SCORER)
    kwargs = dict(
        guarantee=g, losses=losses, corpus_id="harden-v0",
        transform="t", loss_definition="d",
    )
    if alpha_target is not None:
        kwargs["alpha_target"] = alpha_target
    if not_covered is not None:
        kwargs["not_covered"] = not_covered
    if disclosure is not None:
        kwargs["disclosure"] = disclosure
    pl = build_payload(**kwargs)
    return sign_meaning_risk_receipt(pl, private_jwk=priv, kid=kid), jwks, pl


# ── Finding #1 (HIGH): controlled flag is forgeable ───────────────────


def test_flipped_controlled_flag_fails_replay():
    """An attacker who flips controlled False→True before signing must
    be caught by replay — the flag is recomputed from the bound."""
    losses = [0.9, 0.9, 0.9, 0.9]  # heavy loss → not controlled at 0.1
    priv, jwks, kid = _keypair()
    g = certify_meaning_risk(losses, **_SCORER)
    pl = build_payload(
        guarantee=g, losses=losses, corpus_id="harden-v0",
        transform="t", loss_definition="d", alpha_target=0.1,
    )
    assert pl["controlled"] is False
    pl["controlled"] = True  # forge the operational decision
    env = sign_meaning_risk_receipt(pl, private_jwk=priv, kid=kid)
    # signature is valid (they signed their lie); replay is not
    with pytest.raises(MeaningReceiptReplayError, match="controlled"):
        verify_meaning_risk_receipt(env, jwks, losses=losses)


def test_honest_controlled_flag_replays():
    losses = [0.0] * 80
    env, jwks, pl = _receipt(losses, alpha_target=0.5)
    assert pl["controlled"] is True
    out = verify_meaning_risk_receipt(env, jwks, losses=losses)
    assert out["controlled"] is True


# ── Finding #2 (MED): rounding is the single source of truth ──────────


def test_rounded_losses_replay_consistently():
    """A raw vector that shifts at the 7th dp must replay both when
    verified with the raw vector AND with the rounded loss file the hash
    actually commits — no false reject."""
    raw = [0.5 + 4e-7] * 1000  # rounds to 0.5
    env, jwks, pl = _receipt(raw)
    # verify with the raw losses
    verify_meaning_risk_receipt(env, jwks, losses=raw)
    # verify with the explicitly-rounded loss file (what losses_hash commits)
    rounded = [round(x, 6) for x in raw]
    out = verify_meaning_risk_receipt(env, jwks, losses=rounded)
    assert out["risk_upper_bound_micro"] == pl["risk_upper_bound_micro"]


# ── Finding #10 (LOW): n is re-validated against the committed losses ─


def test_inflated_n_fails_replay():
    losses = [0.1, 0.2, 0.3, 0.4]
    priv, jwks, kid = _keypair()
    g = certify_meaning_risk(losses, **_SCORER)
    pl = build_payload(
        guarantee=g, losses=losses, corpus_id="harden-v0",
        transform="t", loss_definition="d",
    )
    pl["n"] = 999999  # inflate calibration size
    env = sign_meaning_risk_receipt(pl, private_jwk=priv, kid=kid)
    with pytest.raises(MeaningReceiptReplayError, match="n does not replay"):
        verify_meaning_risk_receipt(env, jwks, losses=losses)


# ── Findings #4/#5/#9/#13: not_covered / disclosure enforced ──────────


def test_build_payload_rejects_empty_not_covered():
    g = certify_meaning_risk([0.1, 0.2], **_SCORER)
    with pytest.raises(ValueError, match="not_covered must be non-empty"):
        build_payload(
            guarantee=g, losses=[0.1, 0.2], corpus_id="x",
            transform="t", loss_definition="d", not_covered=[],
        )


def test_verify_rejects_missing_not_covered():
    losses = [0.1, 0.2]
    priv, jwks, kid = _keypair()
    g = certify_meaning_risk(losses, **_SCORER)
    pl = build_payload(
        guarantee=g, losses=losses, corpus_id="x",
        transform="t", loss_definition="d",
    )
    del pl["not_covered"]
    env = sign_meaning_risk_receipt(pl, private_jwk=priv, kid=kid)
    with pytest.raises(MeaningReceiptDisclosureError, match="not_covered"):
        verify_meaning_risk_receipt(env, jwks)  # even signature-only


def test_verify_rejects_empty_disclosure():
    losses = [0.1, 0.2]
    priv, jwks, kid = _keypair()
    g = certify_meaning_risk(losses, **_SCORER)
    pl = build_payload(
        guarantee=g, losses=losses, corpus_id="x",
        transform="t", loss_definition="d",
    )
    pl["disclosure"] = "   "  # blank
    env = sign_meaning_risk_receipt(pl, private_jwk=priv, kid=kid)
    with pytest.raises(MeaningReceiptDisclosureError, match="disclosure"):
        verify_meaning_risk_receipt(env, jwks)


# ── Finding #3 (MED): scorer identity is producer-asserted, not bound ─


def test_scorer_field_is_not_attested_documented_limitation():
    """Pins the documented limitation: the replay does NOT bind the
    scorer name to the losses. A receipt whose scorer label was changed
    still verifies — trust in the label is trust in the issuer, per the
    format doc §4 trust scope. If this ever starts FAILING, the trust
    scope changed and the doc must be updated."""
    losses = [0.1, 0.2, 0.3, 0.4]
    priv, jwks, kid = _keypair()
    g = certify_meaning_risk(losses, **_SCORER)
    pl = build_payload(
        guarantee=g, losses=losses, corpus_id="x",
        transform="t", loss_definition="d",
    )
    pl["scorer"] = "bidirectional-entailment[gpt-4o]"  # a different proxy
    pl["scorer_version"] = "999"
    env = sign_meaning_risk_receipt(pl, private_jwk=priv, kid=kid)
    out = verify_meaning_risk_receipt(env, jwks, losses=losses)
    # verifies (the limitation): the bound is a pure function of numbers
    assert out["scorer"] == "bidirectional-entailment[gpt-4o]"


# ── Finding #11 (LOW): build_payload re-validates loss range ──────────


def test_build_payload_rejects_out_of_range_losses():
    g = certify_meaning_risk([0.1, 0.2], **_SCORER)  # valid guarantee
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        build_payload(
            guarantee=g, losses=[5.0, -2.0], corpus_id="x",
            transform="t", loss_definition="d",
        )


# ── Findings #7/#8: EntailmentScorer weight validation + asymmetry ────


def _true_judge(premise, hypothesis):
    return True


def test_entailment_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        EntailmentScorer(entails=_true_judge, judge_name="j", w_recall=0.5, w_fidelity=0.4)


def test_lexical_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        LexicalCoverageScorer(w_drop=0.5, w_fab=0.4)


@pytest.mark.parametrize("w_recall,w_fidelity", [(0.6, 0.4), (0.5, 0.5), (0.8, 0.2)])
def test_entailment_identity_holds_for_any_valid_weights(w_recall, w_fidelity):
    s = EntailmentScorer(
        entails=_true_judge, judge_name="j",
        w_recall=w_recall, w_fidelity=w_fidelity,
    )
    assert s.loss("A sentence. Another.", "A sentence. Another.") == pytest.approx(0.0, abs=1e-9)


def test_entailment_weights_recall_above_fidelity():
    """Pure omission must cost more than equal-proportion fabrication
    under the default recall>fidelity weighting (the documented design)."""
    from sum_engine_internal.research.meaning.meaning_loss import _content_units

    def keyword_judge(premise, hypothesis):
        p, h = set(_content_units(premise)), set(_content_units(hypothesis))
        return bool(h) and h.issubset(p)

    s = EntailmentScorer(entails=keyword_judge, judge_name="kw")
    src = "Cats purr softly. Dogs bark loudly."
    omission = "Cats purr softly."                      # dropped a source unit
    fabrication = "Cats purr softly. Dogs bark loudly. Birds fly high."  # added one
    assert s.loss(src, omission) > s.loss(src, fabrication)


# ── Finding #12 (LOW): clopper_pearson is tighter on binary samples ───


def test_clopper_pearson_tighter_than_hoeffding_on_binary():
    binary = [0.0] * 8 + [1.0] * 2
    cp = certify_meaning_risk(binary, method="clopper_pearson", **_SCORER)
    h = certify_meaning_risk(binary, method="hoeffding", **_SCORER)
    assert cp.risk_upper_bound < h.risk_upper_bound


# ── Cross-runtime safety: the payload must be float-free ──────────────


def test_payload_is_float_free():
    """Every built-payload value is int | str | bool | list[str] — zero
    floats — so SUM's Node JCS (which rejects floats) can canonicalise it
    and the signature is verifiable cross-runtime. Rate/probability
    quantities ride as integer micro-units."""
    losses = [0.1, 0.2, 0.3, 0.4]
    g = certify_meaning_risk(losses, **_SCORER)
    pl = build_payload(
        guarantee=g, losses=losses, corpus_id="x", transform="t",
        loss_definition="d", alpha_target=0.5,
    )

    def _no_float(v):
        if isinstance(v, bool):
            return True  # bool is fine (and is an int subclass)
        if isinstance(v, float):
            return False
        if isinstance(v, list):
            return all(_no_float(x) for x in v)
        return True

    floats = {k: v for k, v in pl.items() if not _no_float(v)}
    assert not floats, f"payload must be float-free; found {floats}"
    # the four quantities are present as integer micro-units
    for f in ("delta_micro", "point_estimate_micro", "risk_upper_bound_micro",
              "alpha_target_micro"):
        assert isinstance(pl[f], int) and not isinstance(pl[f], bool)
