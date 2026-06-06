"""sum.perspective_risk_receipt.v1 — sign + verify + per-cohort replay + tamper.

The signed, replayable form of group-conditional meaning-risk: it seals
the marginal bound AND every cohort's bound in one signature, and a
verifier handed the losses + cohort labels replays every one. These pin
the chain-of-custody for Perspective Receipts: per-cohort replay, the
evidence anchor, and that forging any cohort bound (or controls_all) is
caught even with a valid signature.

Skipped if joserfc isn't available (the [receipt-verify] extra).
"""
from __future__ import annotations

import copy

import pytest

joserfc = pytest.importorskip("joserfc", reason="[receipt-verify] not installed")

from joserfc.jwk import OKPKey

from sum_engine_internal.infrastructure.jose_envelope import (
    JoseEnvelopeError,
    JoseEnvelopeErrorClass,
)
from sum_engine_internal.research.meaning import (
    MeaningReceiptDisclosureError,
    MeaningReceiptReplayError,
    build_perspective_payload,
    certify_meaning_risk_by_group,
    sign_perspective_risk_receipt,
    verify_perspective_risk_receipt,
)

_S = dict(scorer_name="lexical-coverage-bidirectional", scorer_version="1")


@pytest.fixture(scope="module")
def keypair():
    kid = "test-perspective-2026"
    key = OKPKey.generate_key("Ed25519")
    priv = key.as_dict(private=True); priv.update(kid=kid, alg="EdDSA", use="sig")
    pub = key.as_dict(private=False); pub.update(kid=kid, alg="EdDSA", use="sig")
    return priv, {"keys": [pub]}, kid


# A corpus with a 'good' cohort (preserved) and a 'bad' cohort (heavy loss).
_LOSSES = [0.0] * 60 + [0.8] * 60
_GROUPS = ["en"] * 60 + ["legalese"] * 60


@pytest.fixture
def signed(keypair):
    priv, _jwks, kid = keypair
    grouped = certify_meaning_risk_by_group(_LOSSES, _GROUPS, **_S)
    payload = build_perspective_payload(
        grouped=grouped, losses=_LOSSES, group_ids=_GROUPS,
        corpus_id="perspective-demo", transform="slider:density=0.5",
        loss_definition="lexical overlap gap", alpha_target=0.3,
        signed_at="2026-06-06T12:00:00.000Z",
    )
    return sign_perspective_risk_receipt(payload, private_jwk=priv, kid=kid), payload


# ── happy path ────────────────────────────────────────────────────────


def test_payload_carries_marginal_and_cohorts(signed):
    _env, pl = signed
    assert pl["n"] == 120
    assert {g["group_id"] for g in pl["groups"]} == {"en", "legalese"}
    assert "marginal_risk_upper_bound_micro" in pl
    assert "evidence_hash" in pl and pl["evidence_hash"].startswith("sha256-")
    # the bad cohort is NOT controlled at 0.3; controls_all reflects it
    assert pl["controls_all"] is False


def test_verify_and_per_cohort_replay(signed, keypair):
    _priv, jwks, _kid = keypair
    out = verify_perspective_risk_receipt(
        signed[0], jwks, losses=_LOSSES, group_ids=_GROUPS
    )
    assert out["corpus_id"] == "perspective-demo"


def test_signature_only_passes_disclosure(signed, keypair):
    _priv, jwks, _kid = keypair
    out = verify_perspective_risk_receipt(signed[0], jwks)
    assert "arrangement" in out["not_covered"]


# ── tamper / replay-failure ───────────────────────────────────────────


def test_tampered_payload_fails_signature(signed, keypair):
    _priv, jwks, _kid = keypair
    bad = copy.deepcopy(signed[0])
    bad["payload"]["controls_all"] = True
    with pytest.raises(JoseEnvelopeError) as exc:
        verify_perspective_risk_receipt(bad, jwks)
    assert exc.value.error_class == JoseEnvelopeErrorClass.SIGNATURE_INVALID


def test_forged_cohort_bound_caught_by_replay(keypair):
    """Author lowers the bad cohort's bound before signing (valid sig over
    a lie); per-cohort replay recomputes and catches it."""
    priv, jwks, kid = keypair
    grouped = certify_meaning_risk_by_group(_LOSSES, _GROUPS, **_S)
    pl = build_perspective_payload(
        grouped=grouped, losses=_LOSSES, group_ids=_GROUPS,
        corpus_id="x", transform="t", loss_definition="d", alpha_target=0.3,
    )
    for g in pl["groups"]:
        if g["group_id"] == "legalese":
            g["risk_upper_bound_micro"] = 0  # forge it to "no loss"
    env = sign_perspective_risk_receipt(pl, private_jwk=priv, kid=kid)
    verify_perspective_risk_receipt(env, jwks)  # signature-only passes
    with pytest.raises(MeaningReceiptReplayError, match="legalese"):
        verify_perspective_risk_receipt(env, jwks, losses=_LOSSES, group_ids=_GROUPS)


def test_forged_controls_all_caught(keypair):
    priv, jwks, kid = keypair
    grouped = certify_meaning_risk_by_group(_LOSSES, _GROUPS, **_S)
    pl = build_perspective_payload(
        grouped=grouped, losses=_LOSSES, group_ids=_GROUPS,
        corpus_id="x", transform="t", loss_definition="d", alpha_target=0.3,
    )
    assert pl["controls_all"] is False
    pl["controls_all"] = True  # forge the headline decision
    env = sign_perspective_risk_receipt(pl, private_jwk=priv, kid=kid)
    with pytest.raises(MeaningReceiptReplayError, match="controls_all"):
        verify_perspective_risk_receipt(env, jwks, losses=_LOSSES, group_ids=_GROUPS)


def test_wrong_evidence_fails_hash(signed, keypair):
    _priv, jwks, _kid = keypair
    wrong = list(_LOSSES); wrong[0] = 0.5
    with pytest.raises(MeaningReceiptReplayError, match="evidence_hash"):
        verify_perspective_risk_receipt(signed[0], jwks, losses=wrong, group_ids=_GROUPS)


def test_relabelled_cohort_fails(signed, keypair):
    """Changing the cohort assignment (same losses) changes the evidence
    and must fail — the receipt binds losses TO cohorts."""
    _priv, jwks, _kid = keypair
    relabelled = ["legalese"] * 60 + ["en"] * 60  # swap labels
    with pytest.raises(MeaningReceiptReplayError):
        verify_perspective_risk_receipt(
            signed[0], jwks, losses=_LOSSES, group_ids=relabelled
        )


def test_missing_disclosure_rejected(keypair):
    priv, jwks, kid = keypair
    grouped = certify_meaning_risk_by_group(_LOSSES, _GROUPS, **_S)
    pl = build_perspective_payload(
        grouped=grouped, losses=_LOSSES, group_ids=_GROUPS,
        corpus_id="x", transform="t", loss_definition="d",
    )
    del pl["not_covered"]
    env = sign_perspective_risk_receipt(pl, private_jwk=priv, kid=kid)
    with pytest.raises(MeaningReceiptDisclosureError, match="not_covered"):
        verify_perspective_risk_receipt(env, jwks)


def test_build_rejects_empty_not_covered():
    grouped = certify_meaning_risk_by_group([0.1, 0.2], ["a", "b"], **_S)
    with pytest.raises(ValueError, match="not_covered"):
        build_perspective_payload(
            grouped=grouped, losses=[0.1, 0.2], group_ids=["a", "b"],
            corpus_id="x", transform="t", loss_definition="d", not_covered=[],
        )
