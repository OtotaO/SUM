"""ShareableRender — round-trip + end-to-end verification tests.

T5 coverage:
  1. ShareableRender round-trips through JSON cleanly.
  2. share_id derives from the receipt's transform_id when present;
     falls back to a content-hash when no receipt.
  3. verify_share returns signature_verified=True on a genuine share.
  4. integrity_checks pin parameters_hash / input_hash / output_hash
     against the receipt's signed values.
  5. Tampering any of the share's embedded fields surfaces as
     False in the matching integrity_check (without invalidating
     the signature — the signature is over the receipt's payload,
     which is unchanged).
  6. Tampering the receipt itself surfaces as signature_verified=False.
  7. from_dict rejects unknown schemas / missing fields.
"""
from __future__ import annotations

import copy
import json

import pytest

joserfc = pytest.importorskip("joserfc", reason="[receipt-verify] required")

from sum_engine_internal.transform_receipt import (
    build_payload,
    canonical_hash,
    sign_transform_receipt,
)
from sum_engine_internal.transforms.share import (
    SUPPORTED_SCHEMA,
    ShareableRender,
    verify_share,
)
from sum_engine_internal.transforms.slider import SLIDER_TRANSFORM


# ─── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def keypair():
    from joserfc.jwk import OKPKey

    kid = "test-share-2026"
    key = OKPKey.generate_key("Ed25519")
    private = key.as_dict(private=True)
    private["kid"] = kid
    public = key.as_dict(private=False)
    public["kid"] = kid
    public["alg"] = "EdDSA"
    public["use"] = "sig"
    return private, {"keys": [public]}, kid


@pytest.fixture
def slider_share(keypair):
    """A genuine slider-canonical-path share, fully signed."""
    private, _jwks, kid = keypair
    params = {
        "density": 1.0, "length": 0.5, "formality": 0.5,
        "audience": 0.5, "perspective": 0.5,
    }
    input_obj = {"triples": [["alice", "likes", "cats"]]}
    output_tome = "The alice likes cats."

    payload = build_payload(
        transform="slider",
        parameters_hash=canonical_hash(
            SLIDER_TRANSFORM.canonicalize_parameters(params)
        ),
        input_hash=canonical_hash(
            SLIDER_TRANSFORM.canonicalize_input(input_obj)
        ),
        output_hash=canonical_hash(
            SLIDER_TRANSFORM.canonicalize_output(output_tome)
        ),
        model="canonical-deterministic-v0",
        provider="canonical-path",
        digital_source_type="algorithmicMedia",
    )
    receipt = sign_transform_receipt(payload, private_jwk=private, kid=kid)

    return ShareableRender(
        transform="slider",
        input=input_obj,
        parameters=params,
        output=output_tome,
        receipt=receipt,
    )


# ─── Round-trip ─────────────────────────────────────────────────────


def test_round_trip_through_json(slider_share):
    """A share serialised + deserialised gives back the same data."""
    s = slider_share.to_json()
    parsed = ShareableRender.from_json(s)
    assert parsed.transform == slider_share.transform
    assert parsed.input == slider_share.input
    assert parsed.parameters == slider_share.parameters
    assert parsed.output == slider_share.output
    assert parsed.receipt == slider_share.receipt


def test_to_dict_has_expected_keys(slider_share):
    d = slider_share.to_dict()
    assert d["schema"] == SUPPORTED_SCHEMA
    for k in ("transform", "input", "parameters", "output",
              "receipt", "created_at"):
        assert k in d


def test_share_id_uses_receipt_transform_id(slider_share):
    """When a receipt is present, share_id == receipt.payload.transform_id."""
    assert slider_share.share_id == slider_share.receipt["payload"]["transform_id"]


def test_share_id_fallback_when_no_receipt():
    """No receipt → share_id falls back to a content hash."""
    share = ShareableRender(
        transform="slider",
        input={"triples": [["a", "b", "c"]]},
        parameters={"density": 1.0, "length": 0.5, "formality": 0.5,
                    "audience": 0.5, "perspective": 0.5},
        output="The a b c.",
        receipt={},  # no signature embedded
    )
    sid = share.share_id
    assert isinstance(sid, str)
    assert len(sid) == 16  # 16 hex chars by design


# ─── from_dict validation ───────────────────────────────────────────


def test_from_dict_rejects_unknown_schema():
    with pytest.raises(ValueError, match="unsupported share schema"):
        ShareableRender.from_dict({
            "schema": "not.a.schema.v1",
            "transform": "slider",
            "input": {},
            "parameters": {},
            "output": "",
            "receipt": {},
        })


def test_from_dict_rejects_missing_required_field():
    with pytest.raises(ValueError, match="missing required field"):
        ShareableRender.from_dict({
            "schema": SUPPORTED_SCHEMA,
            "transform": "slider",
            "input": {},
            # parameters missing
            "output": "",
            "receipt": {},
        })


# ─── verify_share happy path ────────────────────────────────────────


def test_verify_share_accepts_genuine(slider_share, keypair):
    _, jwks, kid = keypair
    result = verify_share(slider_share, jwks)
    assert result.signature_verified is True
    assert result.receipt_kid == kid
    # All three integrity checks should pass on a genuine share.
    assert result.integrity_checks["parameters_hash"] is True
    assert result.integrity_checks["input_hash"] is True
    assert result.integrity_checks["output_hash"] is True


# ─── Tamper share fields → integrity_checks surface the drift ──────


def test_tamper_output_surfaces_in_integrity(slider_share, keypair):
    """If a third party mutates the share's `output` field
    post-signing, the receipt itself still verifies (signature is
    over the hash, not the bytes), but the application-layer
    integrity check on output_hash fails."""
    _, jwks, _ = keypair
    bad = ShareableRender(
        transform=slider_share.transform,
        input=slider_share.input,
        parameters=slider_share.parameters,
        output="THE BAD TOME — mutated.",  # different bytes
        receipt=slider_share.receipt,
    )
    result = verify_share(bad, jwks)
    assert result.signature_verified is True  # receipt unchanged
    assert result.integrity_checks["output_hash"] is False
    assert result.integrity_checks["parameters_hash"] is True
    assert result.integrity_checks["input_hash"] is True


def test_tamper_parameters_surfaces_in_integrity(slider_share, keypair):
    _, jwks, _ = keypair
    bad_params = dict(slider_share.parameters)
    bad_params["density"] = 0.5  # different quantize bin
    bad = ShareableRender(
        transform=slider_share.transform,
        input=slider_share.input,
        parameters=bad_params,
        output=slider_share.output,
        receipt=slider_share.receipt,
    )
    result = verify_share(bad, jwks)
    assert result.signature_verified is True
    assert result.integrity_checks["parameters_hash"] is False


def test_tamper_input_surfaces_in_integrity(slider_share, keypair):
    _, jwks, _ = keypair
    bad = ShareableRender(
        transform=slider_share.transform,
        input={"triples": [["different", "input", "triples"]]},
        parameters=slider_share.parameters,
        output=slider_share.output,
        receipt=slider_share.receipt,
    )
    result = verify_share(bad, jwks)
    assert result.signature_verified is True
    assert result.integrity_checks["input_hash"] is False


# ─── Tamper receipt → signature fails ───────────────────────────────


def test_tamper_receipt_fails_signature(slider_share, keypair):
    from sum_engine_internal.transform_receipt import VerifyError

    _, jwks, _ = keypair
    bad_receipt = copy.deepcopy(slider_share.receipt)
    bad_receipt["payload"]["model"] = "attacker-model"
    bad = ShareableRender(
        transform=slider_share.transform,
        input=slider_share.input,
        parameters=slider_share.parameters,
        output=slider_share.output,
        receipt=bad_receipt,
    )
    with pytest.raises(VerifyError):
        verify_share(bad, jwks)


# ─── Unknown transform-id passes signature, marks integrity ─────────


def test_unknown_transform_marked_in_integrity(slider_share, keypair):
    """A receipt whose `transform` value isn't in the registry passes
    the JOSE signature check (signature is over arbitrary bytes), but
    the application-layer can't recompute hashes — integrity_checks
    surfaces transform_known=False."""
    from joserfc.jwk import OKPKey
    from sum_engine_internal.transform_receipt import sign_transform_receipt

    # Build a receipt with a non-registered transform name. We sign
    # with the same key so the JOSE check passes; only the
    # transform-registry lookup fails.
    private, jwks, kid = keypair
    payload = build_payload(
        transform="not-a-real-transform-v0",
        parameters_hash="sha256-" + "a" * 64,
        input_hash="sha256-" + "b" * 64,
        output_hash="sha256-" + "c" * 64,
        model="canonical-deterministic-v0",
        provider="canonical-path",
        digital_source_type="algorithmicMedia",
    )
    receipt = sign_transform_receipt(payload, private_jwk=private, kid=kid)
    share = ShareableRender(
        transform="not-a-real-transform-v0",
        input={},
        parameters={},
        output="",
        receipt=receipt,
    )
    result = verify_share(share, jwks)
    assert result.signature_verified is True
    assert result.integrity_checks == {"transform_known": False}
