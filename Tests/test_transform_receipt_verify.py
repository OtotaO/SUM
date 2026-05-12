"""Transform-receipt sign + verify + adversarial-tamper suite.

Mirror of Tests/test_render_receipt_verifier.py for the new
sum.transform_receipt.v1 schema. Same fixture pattern: sign a
known-good receipt, assert verifier accepts; tamper one field at
a time, assert verifier rejects with the expected error class.

Skipped if joserfc isn't available (the optional dep that
sum-engine[receipt-verify] adds).
"""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

joserfc = pytest.importorskip(
    "joserfc",
    reason="[receipt-verify] extra not installed",
)

from sum_engine_internal.transform_receipt import (
    ErrorClass,
    SUPPORTED_SCHEMA,
    VerifyError,
    build_payload,
    canonical_hash,
    sign_transform_receipt,
    verify_transform_receipt,
)


# ─── Test fixtures: generate an Ed25519 keypair + JWKS ──────────────


@pytest.fixture(scope="module")
def keypair():
    """Generate a fresh Ed25519 keypair for the test module. Returns
    (private_jwk_dict, public_jwks_dict, kid)."""
    from joserfc.jwk import OKPKey

    kid = "test-transform-receipt-2026"
    key = OKPKey.generate_key("Ed25519")
    private = key.as_dict(private=True)
    private["kid"] = kid
    private["alg"] = "EdDSA"
    private["use"] = "sig"
    public = key.as_dict(private=False)
    public["kid"] = kid
    public["alg"] = "EdDSA"
    public["use"] = "sig"
    jwks = {"keys": [public]}
    return private, jwks, kid


@pytest.fixture
def signed_receipt(keypair):
    """A known-good, signed sum.transform_receipt.v1 envelope."""
    private, _jwks, kid = keypair

    # Use the slider transform's canonicalisation to produce
    # realistic hashes — this is what every transform implementation
    # will feed into build_payload at runtime.
    from sum_engine_internal.transforms.slider import SLIDER_TRANSFORM

    params = {
        "density": 1.0, "length": 0.5, "formality": 0.5,
        "audience": 0.5, "perspective": 0.5,
    }
    input_obj = {"triples": [("alice", "likes", "cats")]}
    output_tome = "The alice likes cats."

    parameters_hash = canonical_hash(SLIDER_TRANSFORM.canonicalize_parameters(params))
    input_hash = canonical_hash(SLIDER_TRANSFORM.canonicalize_input(input_obj))
    output_hash = canonical_hash(SLIDER_TRANSFORM.canonicalize_output(output_tome))

    payload = build_payload(
        transform="slider",
        parameters_hash=parameters_hash,
        input_hash=input_hash,
        output_hash=output_hash,
        model="canonical-deterministic-v0",
        provider="canonical-path",
        digital_source_type="algorithmicMedia",
    )

    return sign_transform_receipt(payload, private_jwk=private, kid=kid)


# ─── Positive path ──────────────────────────────────────────────────


def test_signed_receipt_has_correct_envelope_shape(signed_receipt):
    assert signed_receipt["schema"] == SUPPORTED_SCHEMA
    assert "kid" in signed_receipt
    assert "payload" in signed_receipt
    assert "jws" in signed_receipt
    payload = signed_receipt["payload"]
    assert payload["transform"] == "slider"
    assert payload["provider"] == "canonical-path"
    assert payload["model"] == "canonical-deterministic-v0"
    assert payload["digital_source_type"] == "algorithmicMedia"


def test_verifier_accepts_genuine_receipt(signed_receipt, keypair):
    _, jwks, _ = keypair
    result = verify_transform_receipt(signed_receipt, jwks)
    assert result.verified is True
    assert result.kid == keypair[2]


# ─── Tamper suite ───────────────────────────────────────────────────


def _tamper(receipt: dict, **kw) -> dict:
    """Helper: deep-copy + apply mutations via nested-key syntax.
    Example: _tamper(r, payload__model="attacker")."""
    out = copy.deepcopy(receipt)
    for k, v in kw.items():
        if "__" in k:
            outer, inner = k.split("__", 1)
            out[outer][inner] = v
        else:
            out[k] = v
    return out


def test_tamper_payload_model_rejects(signed_receipt, keypair):
    _, jwks, _ = keypair
    bad = _tamper(signed_receipt, payload__model="attacker-model")
    with pytest.raises(VerifyError) as exc:
        verify_transform_receipt(bad, jwks)
    assert exc.value.error_class == ErrorClass.SIGNATURE_INVALID


def test_tamper_payload_provider_rejects(signed_receipt, keypair):
    _, jwks, _ = keypair
    bad = _tamper(signed_receipt, payload__provider="openai")
    with pytest.raises(VerifyError) as exc:
        verify_transform_receipt(bad, jwks)
    assert exc.value.error_class == ErrorClass.SIGNATURE_INVALID


def test_tamper_payload_output_hash_rejects(signed_receipt, keypair):
    _, jwks, _ = keypair
    bad = _tamper(signed_receipt, payload__output_hash="sha256-" + "0" * 64)
    with pytest.raises(VerifyError) as exc:
        verify_transform_receipt(bad, jwks)
    assert exc.value.error_class == ErrorClass.SIGNATURE_INVALID


def test_tamper_payload_parameters_hash_rejects(signed_receipt, keypair):
    _, jwks, _ = keypair
    bad = _tamper(signed_receipt, payload__parameters_hash="sha256-" + "0" * 64)
    with pytest.raises(VerifyError) as exc:
        verify_transform_receipt(bad, jwks)
    assert exc.value.error_class == ErrorClass.SIGNATURE_INVALID


def test_tamper_payload_transform_name_rejects(signed_receipt, keypair):
    """Mutating the transform field invalidates the signature even
    though it's a string the verifier doesn't validate semantically.
    The JCS canonicalisation covers every payload byte."""
    _, jwks, _ = keypair
    bad = _tamper(signed_receipt, payload__transform="attacker-transform")
    with pytest.raises(VerifyError) as exc:
        verify_transform_receipt(bad, jwks)
    assert exc.value.error_class == ErrorClass.SIGNATURE_INVALID


def test_tamper_kid_to_unknown_rejects(signed_receipt, keypair):
    _, jwks, _ = keypair
    bad = _tamper(signed_receipt, kid="attacker-key-2026")
    with pytest.raises(VerifyError) as exc:
        verify_transform_receipt(bad, jwks)
    assert exc.value.error_class == ErrorClass.UNKNOWN_KID


def test_tamper_schema_to_render_receipt_rejects(signed_receipt, keypair):
    """A transform-receipt verifier MUST reject a renamed
    sum.render_receipt.v1 envelope (schema gate per §1.5 forward-
    compat). This is what keeps the two surfaces from masquerading
    as each other."""
    _, jwks, _ = keypair
    bad = _tamper(signed_receipt, schema="sum.render_receipt.v1")
    with pytest.raises(VerifyError) as exc:
        verify_transform_receipt(bad, jwks)
    assert exc.value.error_class == ErrorClass.SCHEMA_UNKNOWN


# ─── Application-layer integrity checks ─────────────────────────────


def test_recomputed_output_hash_matches_payload(signed_receipt):
    """Demonstrates the application-layer integrity check pattern:
    a caller who served the tome separately recomputes the hash and
    compares to payload.output_hash."""
    from sum_engine_internal.transforms.slider import SLIDER_TRANSFORM

    served_tome = "The alice likes cats."  # what the fixture signed
    expected = canonical_hash(SLIDER_TRANSFORM.canonicalize_output(served_tome))
    assert expected == signed_receipt["payload"]["output_hash"]


def test_recomputed_output_hash_mismatches_on_different_tome(signed_receipt):
    """The integrity check fires when the served bytes differ from
    what was signed."""
    from sum_engine_internal.transforms.slider import SLIDER_TRANSFORM

    different_tome = "Bob owns a dog."
    recomputed = canonical_hash(SLIDER_TRANSFORM.canonicalize_output(different_tome))
    assert recomputed != signed_receipt["payload"]["output_hash"]


def test_transform_id_is_deterministic(signed_receipt):
    """transform_id MUST be derivable from the other four hash inputs;
    re-deriving from the same tuple gives the same id."""
    payload = signed_receipt["payload"]

    from sum_engine_internal.transform_receipt.format import _derive_transform_id

    redrived = _derive_transform_id(
        payload["transform"],
        payload["parameters_hash"],
        payload["input_hash"],
        payload["output_hash"],
    )
    assert redrived == payload["transform_id"]
