"""sum.chain_receipt.v1 — build/sign/verify roundtrip + every tamper class.

The chain receipt binds ordered per-hop meaning-risk receipts (by canonical
hash), their integer-exact Bonferroni budget, and an optional directly
measured end-to-end leg. These tests generate real Ed25519 keys and real
per-hop receipts at runtime (no torch — losses are given, the scorer is
named), then attack each committed quantity.
"""
from __future__ import annotations

import copy

import pytest

joserfc = pytest.importorskip(
    "joserfc", reason="[receipt-verify] extra (joserfc) not installed"
)
from joserfc.jwk import OKPKey  # noqa: E402

from sum_engine_internal.research.meaning.chain_receipt import (  # noqa: E402
    BUDGET_SCOPE_STATEMENT,
    ChainReceiptDisclosureError,
    ChainReceiptReplayError,
    build_chain_payload,
    build_end_to_end_leg,
    canonical_receipt_hash,
    sign_chain_receipt,
    verify_chain_receipt,
)
from sum_engine_internal.research.meaning.conformal_meaning import (  # noqa: E402
    certify_meaning_risk,
)
from sum_engine_internal.research.meaning.receipt import (  # noqa: E402
    build_payload,
    sign_meaning_risk_receipt,
)

HOP1_LOSSES = [0.1, 0.2, 0.15, 0.05, 0.3, 0.25, 0.1, 0.2] * 4  # n=32
HOP2_LOSSES = [0.05, 0.1, 0.1, 0.2, 0.15, 0.05, 0.1, 0.1] * 4  # n=32
E2E_LOSSES = [0.2, 0.3, 0.25, 0.15, 0.4, 0.35, 0.2, 0.3] * 4   # n=32


@pytest.fixture(scope="module")
def keys():
    key = OKPKey.generate_key("Ed25519")
    private = key.as_dict(private=True)
    private.update(kid="chain-test-key-1", alg="EdDSA", use="sig")
    public = key.as_dict(private=False)
    public.update(kid="chain-test-key-1", alg="EdDSA", use="sig")
    return private, {"keys": [public]}


def _hop(losses, corpus_id, transform, private_jwk):
    g = certify_meaning_risk(
        losses,
        scorer_name="test-scorer",
        scorer_version="1",
        delta=0.05,
        method="hoeffding",
    )
    payload = build_payload(
        guarantee=g,
        losses=losses,
        corpus_id=corpus_id,
        transform=transform,
        loss_definition="1 - recall of source claims (test)",
    )
    return sign_meaning_risk_receipt(
        payload, private_jwk=private_jwk, kid="chain-test-key-1"
    )


@pytest.fixture(scope="module")
def chain(keys):
    private, jwks = keys
    hop1 = _hop(HOP1_LOSSES, "test-corpus", "compress", private)
    hop2 = _hop(HOP2_LOSSES, "test-corpus", "translate", private)
    leg = build_end_to_end_leg(
        E2E_LOSSES,
        scorer_name="test-scorer",
        scorer_version="1",
        loss_definition="direct source->final loss (test)",
    )
    payload = build_chain_payload([hop1, hop2], end_to_end=leg)
    envelope = sign_chain_receipt(
        payload, private_jwk=private, kid="chain-test-key-1"
    )
    return {"envelope": envelope, "hops": [hop1, hop2], "jwks": jwks}


def test_roundtrip_full_replay(chain):
    payload = verify_chain_receipt(
        chain["envelope"],
        chain["jwks"],
        hop_envelopes=chain["hops"],
        end_to_end_losses=E2E_LOSSES,
    )
    assert payload["n_hops"] == 2
    assert payload["budget_micro"] == sum(
        h["risk_upper_bound_micro"] for h in payload["hops"]
    )
    assert payload["joint_delta_micro"] == 100_000  # 2 x 0.05
    assert payload["budget_scope"] == BUDGET_SCOPE_STATEMENT


def test_verifies_without_side_band(chain):
    # Stage A + internal consistency only — no hops, no losses.
    payload = verify_chain_receipt(chain["envelope"], chain["jwks"])
    assert payload["chain_id"]


def test_reordered_hops_break_replay(chain):
    payload = verify_chain_receipt(chain["envelope"], chain["jwks"])
    assert payload  # sanity
    with pytest.raises(ChainReceiptReplayError, match="hashes to"):
        verify_chain_receipt(
            chain["envelope"],
            chain["jwks"],
            hop_envelopes=list(reversed(chain["hops"])),
        )


def test_substituted_hop_breaks_replay(chain, keys):
    private, _ = keys
    imposter = _hop([0.0] * 32, "test-corpus", "compress", private)
    with pytest.raises(ChainReceiptReplayError, match="hashes to"):
        verify_chain_receipt(
            chain["envelope"],
            chain["jwks"],
            hop_envelopes=[imposter, chain["hops"][1]],
        )


def test_tampered_budget_fails_closed(chain):
    doctored = copy.deepcopy(chain["envelope"])
    doctored["payload"]["budget_micro"] -= 1
    # Signature breaks first (the payload is signed); assert SOME failure.
    with pytest.raises(Exception):  # noqa: B017 - any SumVerifyError class
        verify_chain_receipt(doctored, chain["jwks"])


def test_internal_budget_sum_is_checked(chain, keys):
    """Even a RE-SIGNED payload with a wrong budget fails the integer-sum
    replay — the check does not rely on the signature alone."""
    private, jwks = keys
    payload = copy.deepcopy(
        verify_chain_receipt(chain["envelope"], chain["jwks"])
    )
    payload["budget_micro"] += 1
    resigned = sign_chain_receipt(
        payload, private_jwk=private, kid="chain-test-key-1"
    )
    with pytest.raises(ChainReceiptReplayError, match="budget_micro"):
        verify_chain_receipt(resigned, jwks)


def test_reordered_mirrors_break_chain_id(chain, keys):
    private, jwks = keys
    payload = copy.deepcopy(
        verify_chain_receipt(chain["envelope"], chain["jwks"])
    )
    h0, h1 = payload["hops"]
    h0["index"], h1["index"] = 1, 2  # keep indices ordered...
    payload["hops"] = [
        {**h1, "index": 1},
        {**h0, "index": 2},
    ]  # ...but swap the content (and the hash order)
    resigned = sign_chain_receipt(
        payload, private_jwk=private, kid="chain-test-key-1"
    )
    with pytest.raises(ChainReceiptReplayError, match="chain_id"):
        verify_chain_receipt(resigned, jwks)


def test_missing_budget_scope_fails_closed(chain, keys):
    private, jwks = keys
    payload = copy.deepcopy(
        verify_chain_receipt(chain["envelope"], chain["jwks"])
    )
    del payload["budget_scope"]
    resigned = sign_chain_receipt(
        payload, private_jwk=private, kid="chain-test-key-1"
    )
    with pytest.raises(ChainReceiptDisclosureError, match="budget_scope"):
        verify_chain_receipt(resigned, jwks)


def test_end_to_end_losses_replay_is_exact(chain):
    wrong = list(E2E_LOSSES)
    wrong[0] += 0.000001  # one micro off in one loss
    with pytest.raises(ChainReceiptReplayError, match="losses_hash"):
        verify_chain_receipt(
            chain["envelope"], chain["jwks"], end_to_end_losses=wrong
        )


def test_end_to_end_without_leg_is_refused(chain, keys):
    private, jwks = keys
    payload = copy.deepcopy(
        verify_chain_receipt(chain["envelope"], chain["jwks"])
    )
    del payload["end_to_end"]
    resigned = sign_chain_receipt(
        payload, private_jwk=private, kid="chain-test-key-1"
    )
    with pytest.raises(ChainReceiptReplayError, match="no end_to_end leg"):
        verify_chain_receipt(resigned, jwks, end_to_end_losses=E2E_LOSSES)


def test_single_hop_chain_is_refused_at_build(keys):
    private, _ = keys
    hop = _hop(HOP1_LOSSES, "c", "t", private)
    with pytest.raises(ValueError, match=">= 2 hops"):
        build_chain_payload([hop])


def test_sdk_dispatcher_handles_chain(chain):
    import sum_verify

    assert "sum.chain_receipt.v1" in sum_verify.SUPPORTED_SCHEMAS
    payload = sum_verify.verify(chain["envelope"], chain["jwks"])
    assert payload["n_hops"] == 2
    # Full side-band goes through the dedicated function.
    payload = sum_verify.verify_chain_receipt(
        chain["envelope"],
        chain["jwks"],
        hop_envelopes=chain["hops"],
        end_to_end_losses=E2E_LOSSES,
    )
    assert payload["end_to_end"]["n"] == 32


def test_budget_is_integer_exact_composition(chain):
    """The chain's budget equals drift_budget's payload composition —
    same math, byte-exact."""
    from sum_engine_internal.research.meaning.drift_budget import (
        compose_drift_budget_from_payloads,
    )

    payload = verify_chain_receipt(chain["envelope"], chain["jwks"])
    composed = compose_drift_budget_from_payloads(
        [env["payload"] for env in chain["hops"]]
    )
    assert round(composed.budget * 1_000_000) == payload["budget_micro"]
    assert round(composed.joint_delta * 1_000_000) == (
        payload["joint_delta_micro"]
    )


def test_hop_hash_matches_canonical_shape(chain):
    for env in chain["hops"]:
        h = canonical_receipt_hash(env)
        assert h.startswith("sha256-") and len(h) == 71


def test_internal_int_failures_use_chain_taxonomy(chain, keys):
    """A malformed integer field on the CHAIN payload raises
    ChainReceiptReplayError (not the meaning-receipt class) — the error
    taxonomy is part of the documented contract."""
    private, jwks = keys
    payload = copy.deepcopy(
        verify_chain_receipt(chain["envelope"], chain["jwks"])
    )
    payload["budget_micro"] = str(payload["budget_micro"])  # int -> str
    resigned = sign_chain_receipt(
        payload, private_jwk=private, kid="chain-test-key-1"
    )
    with pytest.raises(ChainReceiptReplayError, match="integer micro-unit"):
        verify_chain_receipt(resigned, jwks)


def test_max_age_windows_the_chain_not_the_hops(keys):
    """max_age_seconds applies to the CHAIN envelope only: a fresh chain
    over old hops verifies; an old chain fails its own window."""
    private, jwks = keys
    old_ts = "2020-01-01T00:00:00.000Z"
    g = certify_meaning_risk(
        HOP1_LOSSES, scorer_name="test-scorer", scorer_version="1",
        delta=0.05, method="hoeffding",
    )
    old_hops = []
    for transform in ("compress", "translate"):
        p = build_payload(
            guarantee=g, losses=HOP1_LOSSES, corpus_id="test-corpus",
            transform=transform,
            loss_definition="1 - recall of source claims (test)",
            signed_at=old_ts,
        )
        old_hops.append(
            sign_meaning_risk_receipt(
                p, private_jwk=private, kid="chain-test-key-1"
            )
        )
    fresh_chain = sign_chain_receipt(
        build_chain_payload(old_hops),
        private_jwk=private, kid="chain-test-key-1",
    )
    # Fresh chain + years-old hops + a 1-hour window: must PASS.
    payload = verify_chain_receipt(
        fresh_chain, jwks, hop_envelopes=old_hops, max_age_seconds=3600
    )
    assert payload["n_hops"] == 2
    # An old CHAIN envelope fails the same window on its own signed_at.
    stale_chain = sign_chain_receipt(
        build_chain_payload(old_hops, signed_at=old_ts),
        private_jwk=private, kid="chain-test-key-1",
    )
    from sum_engine_internal.infrastructure.jose_envelope import (
        JoseEnvelopeError,
    )

    with pytest.raises(JoseEnvelopeError):
        verify_chain_receipt(stale_chain, jwks, max_age_seconds=3600)


# ---- Totality property coverage for the chain verifier (2026-07-31 review #4) ----
# The chain verifier (added after the 06-14 hardening arc) had no hypothesis
# totality test, so malformed caller-supplied hop envelopes / end-to-end losses
# could raise an undeclared ValueError/TypeError out of canonicalize()/enumerate
# instead of failing closed in the chain taxonomy. These pin totality.
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from sum_engine_internal.infrastructure.jose_envelope import (  # noqa: E402
    JoseEnvelopeError as _JoseEnvelopeError,
)
# The research chain verifier re-exports sum_verify._chain.verify_chain_receipt,
# so the classes it actually raises are sum_verify's (the end-to-end losses path
# shares sum_verify's meaning validator).
from sum_verify._meaning import (  # noqa: E402
    MeaningReceiptReplayError as _MeaningReceiptReplayError,
)

_CK = OKPKey.generate_key("Ed25519")
_CPRIV = _CK.as_dict(private=True)
_CPRIV.update(kid="chain-prop", alg="EdDSA", use="sig")
_CPUB = _CK.as_dict(private=False)
_CPUB.update(kid="chain-prop", alg="EdDSA", use="sig")
_CJWKS = {"keys": [_CPUB]}


def _prop_hop(losses, transform):
    g = certify_meaning_risk(
        losses, scorer_name="s", scorer_version="1", delta=0.05, method="hoeffding"
    )
    pl = build_payload(
        guarantee=g, losses=losses, corpus_id="c",
        transform=transform, loss_definition="d",
    )
    return sign_meaning_risk_receipt(pl, private_jwk=_CPRIV, kid="chain-prop")


_PROP_HOP1 = _prop_hop(HOP1_LOSSES, "compress")
_PROP_HOP2 = _prop_hop(HOP2_LOSSES, "translate")
_PROP_LEG = build_end_to_end_leg(
    E2E_LOSSES, scorer_name="s", scorer_version="1", loss_definition="d"
)
_PROP_CHAIN = sign_chain_receipt(
    build_chain_payload([_PROP_HOP1, _PROP_HOP2], end_to_end=_PROP_LEG),
    private_jwk=_CPRIV, kid="chain-prop",
)

# Both replay classes count as declared: the end-to-end losses path shares the
# meaning validator, which raises MeaningReceiptReplayError (still a SumVerifyError).
_CHAIN_DECLARED = (
    _JoseEnvelopeError,
    ChainReceiptReplayError,
    ChainReceiptDisclosureError,
    _MeaningReceiptReplayError,
)

_NASTY_CHAIN = (
    st.none() | st.booleans()
    | st.integers(min_value=-(10 ** 9), max_value=10 ** 9)
    | st.floats() | st.text(max_size=6) | st.binary(max_size=6)
)
_CHAIN_JSON = st.recursive(
    _NASTY_CHAIN,
    lambda c: st.lists(c, max_size=3) | st.dictionaries(st.text(max_size=4), c, max_size=3),
    max_leaves=6,
)


def _assert_chain_total(fn):
    try:
        fn()
    except _CHAIN_DECLARED:
        pass


@settings(max_examples=250, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(hop_envelopes=st.one_of(_NASTY_CHAIN, _CHAIN_JSON, st.lists(_CHAIN_JSON, max_size=3)))
def test_chain_hop_envelopes_totality(hop_envelopes):
    """Malformed caller-supplied hop envelopes (non-sequence, bytes, a hop
    payload carrying NaN) must fail closed in the chain taxonomy, never an
    undeclared exception out of canonicalize()."""
    _assert_chain_total(
        lambda: verify_chain_receipt(_PROP_CHAIN, _CJWKS, hop_envelopes=hop_envelopes)
    )


@settings(max_examples=250, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(end_to_end_losses=st.one_of(
    _NASTY_CHAIN,
    _CHAIN_JSON,
    st.lists(st.floats() | st.none() | st.integers() | st.text(max_size=3), max_size=8),
))
def test_chain_end_to_end_losses_totality(end_to_end_losses):
    """With the REAL hops supplied, arbitrary end-to-end losses (incl. a bare
    non-sequence scalar and a missing/garbage vector) must reject cleanly."""
    _assert_chain_total(lambda: verify_chain_receipt(
        _PROP_CHAIN, _CJWKS,
        hop_envelopes=[_PROP_HOP1, _PROP_HOP2],
        end_to_end_losses=end_to_end_losses,
    ))
