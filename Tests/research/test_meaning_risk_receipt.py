"""sum.meaning_risk_receipt.v1 — sign + verify + replay + tamper suite.

Mirrors the transform-receipt verifier test pattern: sign a known-good
certificate, assert the verifier accepts; supply the committed losses
and assert the bound REPLAYS; tamper one field / one loss at a time and
assert rejection with the right failure class.

The replay test is the load-bearing one — it is what makes this the
first *same-commit-reproducible* certificate over a meaning-space loss.

Skipped if joserfc isn't available (the [receipt-verify] extra).
"""
from __future__ import annotations

import copy

import pytest

joserfc = pytest.importorskip(
    "joserfc",
    reason="[receipt-verify] extra not installed",
)

from sum_engine_internal.infrastructure.jose_envelope import (
    JoseEnvelopeError,
    JoseEnvelopeErrorClass,
)
from sum_engine_internal.research.meaning import (
    LexicalCoverageScorer,
    MeaningReceiptReplayError,
    build_payload,
    certify_meaning_risk,
    score_pairs,
    sign_meaning_risk_receipt,
    verify_meaning_risk_receipt,
)
from sum_engine_internal.research.meaning.receipt import (
    DEFAULT_NOT_COVERED,
    SUPPORTED_SCHEMA,
    losses_hash,
)


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def keypair():
    from joserfc.jwk import OKPKey

    kid = "test-meaning-risk-2026"
    key = OKPKey.generate_key("Ed25519")
    private = key.as_dict(private=True)
    private["kid"] = kid
    private["alg"] = "EdDSA"
    private["use"] = "sig"
    public = key.as_dict(private=False)
    public["kid"] = kid
    public["alg"] = "EdDSA"
    public["use"] = "sig"
    return private, {"keys": [public]}, kid


# A tiny parallel-translation-style corpus: a dense source and a set of
# transforms (expansions / compressions). Stands in for the real
# Abrahamic parallel-translation bed described in the frontier doc.
_PAIRS = [
    (
        "The covenant was established between the parties and sealed with an oath.",
        "The covenant was established between the parties and sealed with an oath.",
    ),
    (
        "He who guards his tongue keeps his soul from troubles.",
        "Guarding the tongue keeps the soul from trouble.",
    ),
    (
        "The harvest is plentiful but the labourers are few.",
        "There is much harvest and few workers.",
    ),
    (
        "Justice rolls down like waters and righteousness like a mighty stream.",
        "Justice flows like water; righteousness like a strong river.",
    ),
]


@pytest.fixture(scope="module")
def losses():
    return score_pairs(_PAIRS, LexicalCoverageScorer())


@pytest.fixture(scope="module")
def guarantee(losses):
    return certify_meaning_risk(
        losses,
        scorer_name="lexical-coverage-bidirectional",
        scorer_version="1",
        delta=0.05,
        method="hoeffding",
    )


@pytest.fixture
def payload(guarantee, losses):
    return build_payload(
        guarantee=guarantee,
        losses=losses,
        corpus_id="abrahamic-parallel-translations-v0",
        transform="slider:density=0.5",
        alpha_target=0.5,
        loss_definition=(
            "bidirectional content-unit overlap gap in [0,1]; "
            "0 = full preservation, 1 = no preservation"
        ),
        signed_at="2026-06-05T12:00:00.000Z",
    )


@pytest.fixture
def signed(payload, keypair):
    private, _jwks, kid = keypair
    return sign_meaning_risk_receipt(payload, private_jwk=private, kid=kid)


# ── happy path ────────────────────────────────────────────────────────


def test_payload_shape(payload):
    for field in (
        "scorer", "scorer_version", "loss_definition", "n", "delta",
        "method", "point_estimate", "risk_upper_bound", "losses_hash",
        "corpus_id", "transform", "not_covered", "disclosure", "signed_at",
        "alpha_target", "controlled",
    ):
        assert field in payload, f"missing {field}"
    assert payload["not_covered"] == list(DEFAULT_NOT_COVERED)
    # the boundary the proxy cannot cross must be declared
    assert "arrangement" in payload["not_covered"]


def test_envelope_has_schema(signed):
    assert signed["schema"] == SUPPORTED_SCHEMA
    assert set(signed) == {"schema", "kid", "payload", "jws"}


def test_verify_signature_only(signed, keypair):
    _private, jwks, _kid = keypair
    out = verify_meaning_risk_receipt(signed, jwks)
    assert out["corpus_id"] == "abrahamic-parallel-translations-v0"


def test_verify_with_replay(signed, keypair, losses):
    """The crown jewel: hand the verifier the committed losses and the
    bound reproduces byte-for-byte."""
    _private, jwks, _kid = keypair
    out = verify_meaning_risk_receipt(signed, jwks, losses=losses)
    assert out["risk_upper_bound"] == signed["payload"]["risk_upper_bound"]


def test_replay_is_independent_recompute(signed, keypair):
    """Replay must not trust the receipt's numbers — recompute from the
    losses via a fresh certification and match."""
    _private, jwks, _kid = keypair
    recomputed = score_pairs(_PAIRS, LexicalCoverageScorer())
    out = verify_meaning_risk_receipt(signed, jwks, losses=recomputed)
    assert out is signed["payload"]


def test_controlled_flag_honest_at_small_n(payload):
    """Two honest behaviours at once, both recorded in the receipt:
      (1) the LEXICAL proxy over-reports loss on faithful paraphrase
          (it can't see through reworded meaning — the documented
          limitation that motivates EntailmentScorer), and
      (2) 4 samples cannot certify control at 0.5 distribution-free.
    The receipt records the honest verdict, never the optimistic one."""
    assert payload["point_estimate"] > 0.2   # lexical proxy penalises paraphrase
    assert payload["controlled"] is False    # and 4 samples can't certify anyway


def test_controlled_flag_true_with_enough_data(keypair):
    """Positive path: with enough faithful pairs the certified ceiling
    drops below the target and the receipt records controlled=True."""
    private, jwks, kid = keypair
    losses = [0.0] * 64 + [0.1] * 16   # 80 mostly-faithful pairs
    g = certify_meaning_risk(
        losses,
        scorer_name="lexical-coverage-bidirectional",
        scorer_version="1",
        delta=0.05,
    )
    pl = build_payload(
        guarantee=g,
        losses=losses,
        corpus_id="synthetic-faithful-v0",
        transform="slider:density=0.5",
        alpha_target=0.5,
        loss_definition="d",
    )
    assert pl["controlled"] is True
    env = sign_meaning_risk_receipt(pl, private_jwk=private, kid=kid)
    out = verify_meaning_risk_receipt(env, jwks, losses=losses)
    assert out["controlled"] is True


# ── tamper / replay-failure suite ─────────────────────────────────────


def test_tampered_payload_fails_signature(signed, keypair):
    _private, jwks, _kid = keypair
    bad = copy.deepcopy(signed)
    bad["payload"]["risk_upper_bound"] = 0.0  # forge a stronger claim
    with pytest.raises(JoseEnvelopeError) as exc:
        verify_meaning_risk_receipt(bad, jwks)
    assert exc.value.error_class == JoseEnvelopeErrorClass.SIGNATURE_INVALID


def test_wrong_schema_rejected(signed, keypair):
    _private, jwks, _kid = keypair
    bad = copy.deepcopy(signed)
    bad["schema"] = "sum.transform_receipt.v1"
    with pytest.raises(JoseEnvelopeError) as exc:
        verify_meaning_risk_receipt(bad, jwks)
    assert exc.value.error_class == JoseEnvelopeErrorClass.SCHEMA_UNKNOWN


def test_replay_rejects_wrong_losses(signed, keypair, losses):
    """A genuine receipt whose claimed losses don't match the side-band
    evidence fails the replay hash check — distinct from a forgery."""
    _private, jwks, _kid = keypair
    wrong = list(losses)
    wrong[0] = wrong[0] + 0.5  # different vector → different hash
    with pytest.raises(MeaningReceiptReplayError, match="losses_hash"):
        verify_meaning_risk_receipt(signed, jwks, losses=wrong)


def test_replay_rejects_bound_forgery_with_matching_hash(payload, keypair, losses):
    """If someone signs a receipt whose committed hash is honest but
    whose risk_upper_bound was hand-edited DOWN before signing, replay
    catches it: the hash matches but the re-certified bound doesn't."""
    private, jwks, kid = keypair
    forged = copy.deepcopy(payload)
    forged["risk_upper_bound"] = 0.0  # too-strong, but losses_hash honest
    signed_forged = sign_meaning_risk_receipt(forged, private_jwk=private, kid=kid)
    # signature is valid (they signed their own lie); replay is not
    verify_meaning_risk_receipt(signed_forged, jwks)  # sig-only passes
    with pytest.raises(MeaningReceiptReplayError, match="risk_upper_bound"):
        verify_meaning_risk_receipt(signed_forged, jwks, losses=losses)


def test_losses_hash_is_stable():
    a = losses_hash([0.0, 0.1, 0.2])
    b = losses_hash([0.0, 0.1, 0.2])
    assert a == b and a.startswith("sha256-")


def test_losses_hash_rounding_stable():
    """Last-bit float drift below the rounding resolution must not change
    the anchor."""
    assert losses_hash([0.1, 0.2]) == losses_hash([0.1 + 1e-12, 0.2 - 1e-12])


def test_build_payload_rejects_n_mismatch(guarantee, losses):
    with pytest.raises(ValueError, match="disagrees"):
        build_payload(
            guarantee=guarantee,
            losses=losses[:-1],  # one short
            corpus_id="x",
            transform="t",
            loss_definition="d",
        )
