"""Property-based hardening for the Perspective Receipt + group-conditional.

Where the example suite pins specific tampers, these fuzz the invariants
for ANY corpus / cohort assignment / off-grid delta+alpha:
  1. round-trip: build → sign → verify(losses, groups) never false-rejects
     a genuine receipt (the regression for the #275-class off-grid bug,
     generalised — including the `simultaneous` Bonferroni path);
  2. float-free: every payload value is int|str|bool|list (so Node JCS,
     which rejects floats, can canonicalise it);
  3. tamper: any supra-rounding perturbation of a committed loss breaks
     the evidence anchor and fails replay.

Hypothesis: derandomize for reproducible CI; reduced max_examples on the
crypto path (real Ed25519 signing is not free).
"""
from __future__ import annotations

import pytest

hypothesis = pytest.importorskip("hypothesis")
joserfc = pytest.importorskip("joserfc", reason="[receipt-verify] not installed")

from hypothesis import given, settings, strategies as st
from joserfc.jwk import OKPKey

from sum_engine_internal.research.meaning import (
    MeaningReceiptReplayError,
    build_perspective_payload,
    certify_meaning_risk_by_group,
    sign_perspective_risk_receipt,
    verify_perspective_risk_receipt,
)

_S = dict(scorer_name="lexical-coverage-bidirectional", scorer_version="1")


@st.composite
def _corpus(draw):
    n = draw(st.integers(min_value=1, max_value=40))
    losses = draw(st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n,
    ))
    groups = draw(st.lists(st.sampled_from(["a", "b", "c", "d"]), min_size=n, max_size=n))
    return losses, groups


def _kp(kid="prop-persp"):
    k = OKPKey.generate_key("Ed25519")
    priv = k.as_dict(private=True); priv.update(kid=kid, alg="EdDSA", use="sig")
    pub = k.as_dict(private=False); pub.update(kid=kid, alg="EdDSA", use="sig")
    return priv, {"keys": [pub]}, kid


def _no_float(v):
    if isinstance(v, bool):
        return True
    if isinstance(v, float):
        return False
    if isinstance(v, (list, tuple)):
        return all(_no_float(x) for x in v)
    if isinstance(v, dict):
        return all(_no_float(x) for x in v.values())
    return True


@settings(derandomize=True, max_examples=200)
@given(corpus=_corpus(),
       delta=st.floats(min_value=0.001, max_value=0.49, allow_nan=False),
       alpha=st.floats(min_value=0.05, max_value=0.95, allow_nan=False),
       simultaneous=st.booleans())
def test_payload_is_float_free(corpus, delta, alpha, simultaneous):
    losses, groups = corpus
    g = certify_meaning_risk_by_group(losses, groups, delta=delta,
                                      simultaneous=simultaneous, **_S)
    pl = build_perspective_payload(
        grouped=g, losses=losses, group_ids=groups, corpus_id="x",
        transform="t", loss_definition="d", alpha_target=alpha,
    )
    bad = {k: v for k, v in pl.items() if not _no_float(v)}
    assert not bad, f"payload must be float-free; found {bad}"


@settings(derandomize=True, max_examples=30, deadline=None)
@given(corpus=_corpus(),
       delta=st.floats(min_value=0.001, max_value=0.49, allow_nan=False),
       alpha=st.floats(min_value=0.05, max_value=0.95, allow_nan=False),
       simultaneous=st.booleans())
def test_offgrid_round_trip(corpus, delta, alpha, simultaneous):
    """Any corpus + OFF-GRID delta/alpha must round-trip build→sign→verify
    — the generalised regression for the off-grid replay-symmetry bug."""
    losses, groups = corpus
    priv, jwks, kid = _kp()
    g = certify_meaning_risk_by_group(losses, groups, delta=delta,
                                      simultaneous=simultaneous, **_S)
    pl = build_perspective_payload(
        grouped=g, losses=losses, group_ids=groups, corpus_id="x",
        transform="t", loss_definition="d", alpha_target=alpha,
    )
    env = sign_perspective_risk_receipt(pl, private_jwk=priv, kid=kid)
    verify_perspective_risk_receipt(env, jwks, losses=losses, group_ids=groups)


@settings(derandomize=True, max_examples=30, deadline=None)
@given(corpus=_corpus(), data=st.data())
def test_perturbed_loss_fails_replay(corpus, data):
    losses, groups = corpus
    priv, jwks, kid = _kp()
    g = certify_meaning_risk_by_group(losses, groups, **_S)
    pl = build_perspective_payload(
        grouped=g, losses=losses, group_ids=groups, corpus_id="x",
        transform="t", loss_definition="d",
    )
    env = sign_perspective_risk_receipt(pl, private_jwk=priv, kid=kid)
    idx = data.draw(st.integers(min_value=0, max_value=len(losses) - 1))
    perturbed = list(losses)
    perturbed[idx] = perturbed[idx] + (0.01 if perturbed[idx] <= 0.5 else -0.01)
    with pytest.raises(MeaningReceiptReplayError, match="evidence_hash"):
        verify_perspective_risk_receipt(env, jwks, losses=perturbed, group_ids=groups)
