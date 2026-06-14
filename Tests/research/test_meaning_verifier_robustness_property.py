"""Property: the meaning-risk verifier is TOTAL on adversarial input.

`verify_meaning_risk_receipt` (research surface AND the shipped `sum_verify`
SDK) must, for ANY input, either return the verified payload or raise one of its
three DECLARED exception types — never an undeclared exception (TypeError,
ValueError, OverflowError, KeyError, …). An undeclared exception is a *crash*: a
denial-of-service surface and a contract violation for any caller that catches
only the declared set.

This is the invariant the 2026-06-13 mass-parallel fuzz campaign found violated
three ways (null micro field → TypeError; NaN/inf side-band loss → ValueError /
OverflowError; coercible-string micro field → silent accept). Those are fixed;
this property explores the space continuously so they cannot regress, and so a
*new* unguarded field added later trips here first.
"""
from __future__ import annotations

import copy

import pytest

joserfc = pytest.importorskip("joserfc", reason="[receipt-verify] extra not installed")
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from joserfc.jwk import OKPKey

from sum_engine_internal.research.meaning import (
    build_payload,
    certify_meaning_risk,
    sign_meaning_risk_receipt,
    verify_meaning_risk_receipt,
)
from sum_engine_internal.research.meaning.receipt import (
    MeaningReceiptDisclosureError,
    MeaningReceiptReplayError,
)
from sum_engine_internal.infrastructure.jose_envelope import JoseEnvelopeError

# The only exceptions the verifier is contractually allowed to raise.
DECLARED = (JoseEnvelopeError, MeaningReceiptReplayError, MeaningReceiptDisclosureError)
_SCORER = dict(scorer_name="lexical-coverage-bidirectional", scorer_version="1")


def _mint():
    key = OKPKey.generate_key("Ed25519")
    priv = key.as_dict(private=True); priv.update(kid="robust", alg="EdDSA", use="sig")
    pub = key.as_dict(private=False); pub.update(kid="robust", alg="EdDSA", use="sig")
    losses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    g = certify_meaning_risk(losses, **_SCORER)
    pl = build_payload(guarantee=g, losses=losses, corpus_id="robust-v0",
                       transform="t", loss_definition="d", alpha_target=0.5)
    return priv, {"keys": [pub]}, pl, losses


_PRIV, _JWKS, _BASE_PAYLOAD, _BASE_LOSSES = _mint()
_BASE_ENV = sign_meaning_risk_receipt(copy.deepcopy(_BASE_PAYLOAD), private_jwk=_PRIV, kid="robust")
_KEYS = list(_BASE_PAYLOAD.keys())

# Adversarial JSON values, including the leaves that previously crashed the
# verifier: None, non-finite floats, coercible strings, oversized ints.
_NASTY = (
    st.none() | st.booleans()
    | st.integers(min_value=-(10**19), max_value=10**19)
    | st.floats()                                   # includes nan / inf / -inf
    | st.floats(allow_nan=True, allow_infinity=True)
    | st.text(max_size=8)
    | st.sampled_from(["645438", "0.5", "", "​", "﻿", "64"])
)
_JSON = st.recursive(
    _NASTY,
    lambda c: st.lists(c, max_size=4) | st.dictionaries(st.text(max_size=4), c, max_size=4),
    max_leaves=8,
)


def _assert_total(fn):
    """Call fn(); pass iff it returns or raises a DECLARED exception. Any other
    exception is re-raised so Hypothesis reports it as a falsifying crash."""
    try:
        fn()
    except DECLARED:
        pass


@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(key=st.sampled_from(_KEYS), val=_JSON)
def test_resigned_field_mutation_never_crashes(key, val):
    """Mutate one signed field to an arbitrary value, re-sign with the real key
    (valid signature → exercises the disclosure + replay layers), verify WITH
    side-band losses. Must return or raise only a declared exception."""
    pl = copy.deepcopy(_BASE_PAYLOAD)
    pl[key] = val
    try:
        env = sign_meaning_risk_receipt(pl, private_jwk=_PRIV, kid="robust")
    except Exception:
        assume(False)   # payload not even serializable/signable → not a wire input
        return
    _assert_total(lambda: verify_meaning_risk_receipt(env, _JWKS, losses=_BASE_LOSSES))


@settings(max_examples=400, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(losses=st.lists(st.floats() | st.integers() | st.none() | st.text(max_size=3), max_size=24))
def test_arbitrary_side_band_losses_never_crash(losses):
    """Arbitrary side-band losses (incl nan/inf/out-of-range/wrong-type/wrong-
    length) must reject cleanly — never an unhandled numeric exception."""
    _assert_total(lambda: verify_meaning_risk_receipt(_BASE_ENV, _JWKS, losses=losses))


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(jws=st.text(max_size=120) | st.sampled_from(["", "a.b.c", "...", _BASE_ENV["jws"][:-3]]))
def test_mangled_jws_never_crashes(jws):
    """A structurally garbage `jws` string must raise a declared exception, not
    crash the splitter / base64 / JOSE layer."""
    env = copy.deepcopy(_BASE_ENV)
    env["jws"] = jws
    _assert_total(lambda: verify_meaning_risk_receipt(env, _JWKS))


@given(jwks=(
    st.none() | st.integers() | st.text(max_size=4) | st.lists(st.integers(), max_size=3)
    | st.dictionaries(st.text(max_size=3), st.integers(), max_size=3)            # no "keys"
    | st.fixed_dictionaries({"keys": st.one_of(st.integers(), st.none(), st.text(max_size=3))})
    | st.fixed_dictionaries({"keys": st.lists(st.integers() | st.none(), max_size=3)})  # non-dict entries
))
def test_malformed_jwks_never_crashes(jwks):
    """A malformed JWKS (the verifier's trusted input) — non-dict, missing/non-list
    `keys`, or non-dict key entries — must reject cleanly, not raise an unhandled
    AttributeError from dereferencing it. (Node parity for this is in
    Tests/test_differential_cross_runtime_fuzz.py.)"""
    _assert_total(lambda: verify_meaning_risk_receipt(_BASE_ENV, jwks))


def test_sum_verify_sdk_shares_totality():
    """The shipped SDK verifier shares the same fixes; spot-check the two
    previously-crashing inputs raise its declared types, not a bare crash."""
    sv = pytest.importorskip("sum_verify")
    pl = copy.deepcopy(_BASE_PAYLOAD); pl["n"] = None
    env = sign_meaning_risk_receipt(pl, private_jwk=_PRIV, kid="robust")
    with pytest.raises(sv.MeaningReceiptReplayError):
        sv.verify_meaning_risk_receipt(env, _JWKS, losses=_BASE_LOSSES)
    with pytest.raises(sv.MeaningReceiptReplayError):
        sv.verify_meaning_risk_receipt(_BASE_ENV, _JWKS, losses=[float("nan")] + _BASE_LOSSES[1:])
