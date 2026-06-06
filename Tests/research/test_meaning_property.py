"""Property-based tests for the meaning-loss frontier module.

Where the example suites pin "this input → this number", these pin the
invariants that must hold for *any* input — the contract the proxies and
the conformal bound claim, fuzzed:

Scorer invariants (LexicalCoverageScorer):
  1. boundedness   — loss(s, t) ∈ [0, 1] for any strings.
  2. identity      — loss(x, x) == 0 for any string.
  3. monotonicity  — deleting a source content-unit from the transform
                     never *decreases* the loss.

Conformal-bound invariants (certify_meaning_risk):
  4. validity      — risk_upper_bound ∈ [0, 1] for any losses in [0, 1].
  5. conservative  — risk_upper_bound ≥ observed mean loss.
  6. duality       — risk_upper_bound == 1 - hoeffding_lower_bound(1-loss).
  7. determinism   — same losses → identical guarantee.

Receipt invariants (sum.meaning_risk_receipt.v1):
  8. replay round-trip — for any losses, sign → verify(losses) reproduces
     the bound; any perturbation of the losses fails replay.

Hypothesis settings mirror Tests/test_property_receipt.py: derandomize
for reproducible CI, reduced max_examples on the crypto path (real
Ed25519 signing is not free).
"""
from __future__ import annotations

import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings, strategies as st

from sum_engine_internal.research.conformal.risk_control import (
    hoeffding_lower_bound,
)
from sum_engine_internal.research.meaning.meaning_loss import (
    LexicalCoverageScorer,
    _content_units,
)
from sum_engine_internal.research.meaning.conformal_meaning import (
    certify_meaning_risk,
)


# Text with a mix of letters, digits, punctuation, whitespace, and some
# non-ASCII — exercises the tokeniser's unicode handling.
_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), max_codepoint=0x2FF),
    min_size=0,
    max_size=120,
)
_losses = st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=60)
_SCORER = dict(scorer_name="lexical-coverage-bidirectional", scorer_version="1")


# ── Scorer invariants ─────────────────────────────────────────────────


@settings(derandomize=True, max_examples=300)
@given(source=_text, transform=_text)
def test_loss_is_bounded(source, transform):
    loss = LexicalCoverageScorer().loss(source, transform)
    assert 0.0 <= loss <= 1.0


@settings(derandomize=True, max_examples=300)
@given(x=_text)
def test_identity_is_zero(x):
    assert LexicalCoverageScorer().loss(x, x) == 0.0


@settings(derandomize=True, max_examples=300)
@given(source=_text, data=st.data())
def test_monotone_under_unit_deletion(source, data):
    """Build the transform from a subset of the source's content units,
    then delete one more unit; loss must not decrease."""
    units = list(dict.fromkeys(_content_units(source)))  # unique, ordered
    if len(units) < 2:
        return  # need at least two units to delete one and still vary
    keep = data.draw(
        st.lists(st.sampled_from(units), min_size=1, max_size=len(units), unique=True)
    )
    scorer = LexicalCoverageScorer()
    transform_full = " ".join(keep)
    transform_less = " ".join(keep[:-1])  # one fewer source unit
    loss_full = scorer.loss(source, transform_full)
    loss_less = scorer.loss(source, transform_less)
    assert loss_less >= loss_full - 1e-12


# ── Conformal-bound invariants ────────────────────────────────────────


@settings(derandomize=True, max_examples=400)
@given(losses=_losses)
def test_bound_is_valid_and_conservative(losses):
    g = certify_meaning_risk(losses, **_SCORER)
    assert 0.0 <= g.risk_upper_bound <= 1.0
    # The certified ceiling never sits below the observed mean.
    assert g.risk_upper_bound >= g.point_estimate - 1e-12


@settings(derandomize=True, max_examples=400)
@given(losses=_losses)
def test_bound_is_dual_of_rate_kernel(losses):
    g = certify_meaning_risk(losses, delta=0.05, method="hoeffding", **_SCORER)
    preservations = [1.0 - x for x in losses]
    lb = hoeffding_lower_bound(preservations, 0.05)
    assert g.risk_upper_bound == pytest.approx(max(0.0, min(1.0, 1.0 - lb)), abs=1e-12)


@settings(derandomize=True, max_examples=200)
@given(losses=_losses)
def test_certification_is_deterministic(losses):
    assert certify_meaning_risk(losses, **_SCORER) == certify_meaning_risk(losses, **_SCORER)


# ── Receipt replay round-trip (crypto path: fewer examples) ───────────


joserfc = pytest.importorskip(
    "joserfc",
    reason="install sum-engine[receipt-verify] to run receipt property tests",
)

from joserfc.jwk import OKPKey  # noqa: E402

from sum_engine_internal.research.meaning.receipt import (  # noqa: E402
    MeaningReceiptReplayError,
    build_payload,
    sign_meaning_risk_receipt,
    verify_meaning_risk_receipt,
)


def _keypair(kid="prop-meaning-2026"):
    key = OKPKey.generate_key("Ed25519")
    priv = key.as_dict(private=True)
    priv.update(kid=kid, alg="EdDSA", use="sig")
    pub = key.as_dict(private=False)
    pub.update(kid=kid, alg="EdDSA", use="sig")
    return priv, {"keys": [pub]}, kid


@settings(derandomize=True, max_examples=25, deadline=None)
@given(losses=_losses)
def test_replay_round_trip(losses):
    """For any losses: sign → verify(losses) reproduces the bound."""
    priv, jwks, kid = _keypair()
    g = certify_meaning_risk(losses, **_SCORER)
    pl = build_payload(
        guarantee=g, losses=losses, corpus_id="prop-v0",
        transform="t", loss_definition="d",
    )
    env = sign_meaning_risk_receipt(pl, private_jwk=priv, kid=kid)
    out = verify_meaning_risk_receipt(env, jwks, losses=losses)
    assert out["risk_upper_bound_micro"] == pl["risk_upper_bound_micro"]


@settings(derandomize=True, max_examples=25, deadline=None)
@given(losses=_losses, data=st.data())
def test_perturbed_losses_fail_replay(losses, data):
    """Any loss vector that differs from the committed one (above the
    hash rounding resolution) must fail the replay hash check."""
    priv, jwks, kid = _keypair()
    g = certify_meaning_risk(losses, **_SCORER)
    pl = build_payload(
        guarantee=g, losses=losses, corpus_id="prop-v0",
        transform="t", loss_definition="d",
    )
    env = sign_meaning_risk_receipt(pl, private_jwk=priv, kid=kid)
    # Perturb one entry by a clearly-supra-rounding amount, re-clamped.
    idx = data.draw(st.integers(min_value=0, max_value=len(losses) - 1))
    perturbed = list(losses)
    delta = 0.01 if perturbed[idx] <= 0.5 else -0.01
    perturbed[idx] = perturbed[idx] + delta
    with pytest.raises(MeaningReceiptReplayError, match="losses_hash"):
        verify_meaning_risk_receipt(env, jwks, losses=perturbed)
