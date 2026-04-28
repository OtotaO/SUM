"""Property-based tests for the v0.9.C receipt verifier + R0.2 trust-root verifier (Phase R0.4).

Both verifiers sit on the shared JOSE-envelope core in
``sum_engine_internal.infrastructure.jose_envelope``. The properties
asserted here exercise the cryptographic + protocol invariants that
the fixture-based tests pin by example. Where the fixture set
catches "this exact mutation must produce this exact error class,"
the property tests catch "for any generated valid envelope, ANY
single-field mutation MUST be rejected."

Properties asserted:

  1. **Sign + verify round-trip succeeds** — for any generated
     payload + keypair, the signed envelope verifies.

  2. **Mutation invariance** — for any generated valid envelope,
     mutating any signed payload field yields signature_invalid.

  3. **Schema mismatch invariance** — for any generated valid
     envelope, swapping the schema string to anything other than the
     supported value yields schema_unknown.

  4. **Empty-JWKS invariance** — for any generated valid envelope,
     verifying against ``{"keys": []}`` yields unknown_kid.

  5. **Wrong-key invariance** — for any generated valid envelope
     signed with key A, verifying against a JWKS containing key B
     under the same kid yields signature_invalid.

  6. **Cross-surface taxonomy invariance** — for any error raised by
     the trust-root verifier, the equivalent failure on the receipt
     verifier raises the same error_class string. Pins
     cross-runtime fixture-assertion compatibility.

Hypothesis settings: derandomize=True so failures are reproducible
across CI runs; reduced max_examples on the cryptographic tests
because each example signs/verifies with real Ed25519 (not free).
"""
from __future__ import annotations

import copy

import pytest

hypothesis = pytest.importorskip("hypothesis")
joserfc = pytest.importorskip(
    "joserfc",
    reason="install sum-engine[receipt-verify] to run receipt property tests",
)

from hypothesis import given, settings, strategies as st

from joserfc.jwk import OKPKey

from sum_engine_internal.infrastructure.jose_envelope import (
    sign_jose_envelope,
    verify_jose_envelope,
    JoseEnvelopeError,
    JoseEnvelopeErrorClass,
)
from sum_engine_internal.render_receipt import (
    SUPPORTED_SCHEMA as RECEIPT_SCHEMA,
    verify_receipt,
)
from sum_engine_internal.trust_root import (
    SUPPORTED_SCHEMA as TRUST_ROOT_SCHEMA,
    verify_trust_manifest,
)


# --------------------------------------------------------------------------
# Strategy helpers
# --------------------------------------------------------------------------


_ascii_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), max_codepoint=0x7E),
    min_size=1,
    max_size=20,
)


def _make_keypair(kid: str) -> tuple[dict, dict]:
    """Generate an Ed25519 OKP keypair as JWK dicts. Both carry the
    same kid; the public dict adds alg/use claims."""
    key = OKPKey.generate_key("Ed25519")
    private_jwk = key.as_dict(private=True)
    public_jwk = key.as_dict(private=False)
    private_jwk["kid"] = kid
    public_jwk["kid"] = kid
    public_jwk["alg"] = "EdDSA"
    public_jwk["use"] = "sig"
    return private_jwk, public_jwk


def _make_test_payload(extra: dict | None = None) -> dict:
    """A representative receipt-shaped payload. Hypothesis injects
    variation via the `extra` field; the rest stays stable so the
    test doesn't generate semantically nonsensical receipts."""
    payload = {
        "render_id": "0123456789abcdef",
        "sliders_quantized": {
            "audience": 0.5,
            "density": 1.0,
            "formality": 0.5,
            "length": 0.5,
            "perspective": 0.5,
        },
        "triples_hash": "sha256-" + "0" * 64,
        "tome_hash": "sha256-" + "1" * 64,
        "model": "claude-haiku-4-5-20251001",
        "provider": "anthropic",
        "signed_at": "2026-04-27T18:00:00.000Z",
        "digital_source_type": "trainedAlgorithmicMedia",
    }
    if extra:
        payload.update(extra)
    return payload


# --------------------------------------------------------------------------
# 1. Sign + verify round-trip
# --------------------------------------------------------------------------


@given(extra_str=_ascii_text)
@settings(deadline=None, derandomize=True, max_examples=20)
def test_round_trip_verifies(extra_str):
    """For any generated envelope, sign + verify succeeds. Fewer
    examples than the JCS property tests because real Ed25519 isn't
    free."""
    kid = "property-test-kid-1"
    priv, pub = _make_keypair(kid)

    payload = _make_test_payload({"extra": extra_str})
    envelope = sign_jose_envelope(payload, private_jwk=priv, kid=kid)
    envelope["schema"] = RECEIPT_SCHEMA

    result = verify_receipt(envelope, {"keys": [pub]})
    assert result.verified is True
    assert result.kid == kid
    assert result.payload == payload


# --------------------------------------------------------------------------
# 2. Mutation invariance — ANY signed-field mutation rejects
# --------------------------------------------------------------------------


# Identifies fields in the test payload by JSONPath-like dotted name
# so Hypothesis can pick which one to mutate.
_MUTABLE_FIELDS = [
    "tome_hash",
    "triples_hash",
    "model",
    "provider",
    "signed_at",
    "digital_source_type",
    "render_id",
]


@given(
    field_to_mutate=st.sampled_from(_MUTABLE_FIELDS),
    new_value=_ascii_text,
)
@settings(deadline=None, derandomize=True, max_examples=20)
def test_any_signed_field_mutation_rejects(field_to_mutate, new_value):
    """For any signed top-level payload field and any new value,
    mutating after signing yields signature_invalid. The fixture set
    pins this on specific values (`tampered_tome_hash`,
    `tampered_model`, etc.); this test runs the same property across
    arbitrary mutations to catch the case the fixture set didn't
    anticipate."""
    kid = "property-test-kid-2"
    priv, pub = _make_keypair(kid)
    payload = _make_test_payload()

    envelope = sign_jose_envelope(payload, private_jwk=priv, kid=kid)
    envelope["schema"] = RECEIPT_SCHEMA

    original_value = envelope["payload"][field_to_mutate]
    if new_value == original_value:
        # Hypothesis can land on a no-op mutation; skip without
        # losing the example budget.
        new_value = original_value + "x"
    envelope["payload"][field_to_mutate] = new_value

    with pytest.raises(JoseEnvelopeError) as excinfo:
        verify_receipt(envelope, {"keys": [pub]})
    assert excinfo.value.error_class == JoseEnvelopeErrorClass.SIGNATURE_INVALID


@given(
    slider_to_mutate=st.sampled_from(
        ["audience", "density", "formality", "length", "perspective"]
    ),
    new_slider_value=st.floats(
        min_value=-1.0, max_value=2.0, allow_nan=False, allow_infinity=False
    ).filter(lambda f: 0.0 <= f <= 1.0 and abs(f - 0.5) > 0.01),
)
@settings(deadline=None, derandomize=True, max_examples=20)
def test_any_slider_mutation_rejects(slider_to_mutate, new_slider_value):
    """Mutating any slider value in `sliders_quantized` rejects with
    signature_invalid. Tests the nested-field mutation path
    specifically — the flat-field test above doesn't exercise this."""
    kid = "property-test-kid-3"
    priv, pub = _make_keypair(kid)
    payload = _make_test_payload()

    envelope = sign_jose_envelope(payload, private_jwk=priv, kid=kid)
    envelope["schema"] = RECEIPT_SCHEMA
    original = envelope["payload"]["sliders_quantized"][slider_to_mutate]
    if abs(new_slider_value - original) < 1e-9:
        new_slider_value += 0.05
    envelope["payload"]["sliders_quantized"][slider_to_mutate] = new_slider_value

    with pytest.raises(JoseEnvelopeError) as excinfo:
        verify_receipt(envelope, {"keys": [pub]})
    assert excinfo.value.error_class == JoseEnvelopeErrorClass.SIGNATURE_INVALID


# --------------------------------------------------------------------------
# 3. Schema mismatch invariance
# --------------------------------------------------------------------------


@given(
    bogus_schema=st.text(
        alphabet=st.characters(blacklist_categories=("Cs",), max_codepoint=0x7E),
        min_size=1,
        max_size=40,
    ).filter(lambda s: s != RECEIPT_SCHEMA and s != TRUST_ROOT_SCHEMA),
)
@settings(deadline=None, derandomize=True, max_examples=20)
def test_unknown_schema_always_rejects(bogus_schema):
    """For any non-supported schema string, the verifier rejects
    closed via the schema gate (forward-compat lever from
    RENDER_RECEIPT_FORMAT.md §1.4 / TRUST_ROOT_FORMAT.md §1.2)."""
    kid = "property-test-kid-4"
    priv, pub = _make_keypair(kid)
    payload = _make_test_payload()

    envelope = sign_jose_envelope(payload, private_jwk=priv, kid=kid)
    envelope["schema"] = bogus_schema

    with pytest.raises(JoseEnvelopeError) as excinfo:
        verify_receipt(envelope, {"keys": [pub]})
    assert excinfo.value.error_class == JoseEnvelopeErrorClass.SCHEMA_UNKNOWN


# --------------------------------------------------------------------------
# 4. Empty / mismatched-kid JWKS
# --------------------------------------------------------------------------


@given(extra_str=_ascii_text)
@settings(deadline=None, derandomize=True, max_examples=10)
def test_empty_jwks_rejects_unknown_kid(extra_str):
    """For any generated envelope, an empty JWKS yields unknown_kid."""
    kid = "property-test-kid-5"
    priv, _pub = _make_keypair(kid)
    payload = _make_test_payload({"extra": extra_str})

    envelope = sign_jose_envelope(payload, private_jwk=priv, kid=kid)
    envelope["schema"] = RECEIPT_SCHEMA

    with pytest.raises(JoseEnvelopeError) as excinfo:
        verify_receipt(envelope, {"keys": []})
    assert excinfo.value.error_class == JoseEnvelopeErrorClass.UNKNOWN_KID


@given(
    other_kid=st.text(
        alphabet=st.characters(blacklist_categories=("Cs",), max_codepoint=0x7E),
        min_size=1,
        max_size=30,
    ).filter(lambda s: s != "property-test-kid-6"),
)
@settings(deadline=None, derandomize=True, max_examples=10)
def test_jwks_without_matching_kid_rejects(other_kid):
    """JWKS contains a key, but under a different kid → unknown_kid."""
    signing_kid = "property-test-kid-6"
    priv, _pub = _make_keypair(signing_kid)
    _, other_pub = _make_keypair(other_kid)

    payload = _make_test_payload()
    envelope = sign_jose_envelope(payload, private_jwk=priv, kid=signing_kid)
    envelope["schema"] = RECEIPT_SCHEMA

    with pytest.raises(JoseEnvelopeError) as excinfo:
        verify_receipt(envelope, {"keys": [other_pub]})
    assert excinfo.value.error_class == JoseEnvelopeErrorClass.UNKNOWN_KID


# --------------------------------------------------------------------------
# 5. Wrong-key invariance
# --------------------------------------------------------------------------


@given(extra_str=_ascii_text)
@settings(deadline=None, derandomize=True, max_examples=10)
def test_wrong_key_under_same_kid_rejects(extra_str):
    """Signed with key A, verified against key B (different keypair,
    same kid) → signature_invalid. Pins the kid-isn't-trust-on-its-own
    invariant."""
    kid = "property-test-kid-7"
    priv_a, _pub_a = _make_keypair(kid)
    _, pub_b = _make_keypair(kid)  # different keypair, same kid

    payload = _make_test_payload({"extra": extra_str})
    envelope = sign_jose_envelope(payload, private_jwk=priv_a, kid=kid)
    envelope["schema"] = RECEIPT_SCHEMA

    with pytest.raises(JoseEnvelopeError) as excinfo:
        verify_receipt(envelope, {"keys": [pub_b]})
    assert excinfo.value.error_class == JoseEnvelopeErrorClass.SIGNATURE_INVALID


# --------------------------------------------------------------------------
# 6. Cross-surface taxonomy: trust-root verifier === receipt verifier
#    on shared error classes for the same envelope shape
# --------------------------------------------------------------------------


@given(field_to_mutate=st.sampled_from(_MUTABLE_FIELDS))
@settings(deadline=None, derandomize=True, max_examples=10)
def test_trust_root_and_receipt_verifiers_agree_on_tampered(field_to_mutate):
    """For any single-field mutation that the receipt verifier
    rejects with signature_invalid, the trust-root verifier — given
    the same envelope but the trust-root schema — also rejects with
    signature_invalid. Cross-surface error-class compatibility."""
    kid = "property-test-kid-8"
    priv, pub = _make_keypair(kid)
    payload = _make_test_payload()

    envelope = sign_jose_envelope(payload, private_jwk=priv, kid=kid)
    envelope["payload"][field_to_mutate] = "TAMPERED"

    # Same envelope, both schemas, both verifiers
    receipt_envelope = copy.deepcopy(envelope)
    receipt_envelope["schema"] = RECEIPT_SCHEMA
    trust_envelope = copy.deepcopy(envelope)
    trust_envelope["schema"] = TRUST_ROOT_SCHEMA

    with pytest.raises(JoseEnvelopeError) as receipt_exc:
        verify_receipt(receipt_envelope, {"keys": [pub]})
    with pytest.raises(JoseEnvelopeError) as trust_exc:
        verify_trust_manifest(trust_envelope, {"keys": [pub]})

    assert receipt_exc.value.error_class == trust_exc.value.error_class == \
        JoseEnvelopeErrorClass.SIGNATURE_INVALID
