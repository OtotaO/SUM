"""Round-trip tests for the v0.9.C / R0.2 trust-root pipeline.

Exercises:
  - build → sign → verify round-trip with an in-memory test keypair.
  - tampered-field rejection across every signed payload field.
  - schema-mismatch rejection (forward-compat lever).
  - unknown-kid rejection (JWKS without the manifest's kid).
  - cross-runtime taxonomy: same error class strings the JS verifier
    uses (taxonomy is shared via JoseEnvelopeErrorClass).

Skipped if joserfc isn't installed (the optional dep ``sum-engine[
receipt-verify]`` introduced for v0.9.C; trust-root reuses it).
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest


joserfc = pytest.importorskip(
    "joserfc",
    reason="install sum-engine[receipt-verify] to run trust-root tests",
)


from joserfc.jwk import OKPKey  # noqa: E402

from sum_engine_internal.infrastructure.jose_envelope import (  # noqa: E402
    sign_jose_envelope,
)
from sum_engine_internal.trust_root import (  # noqa: E402
    ErrorClass,
    SUPPORTED_SCHEMA,
    VerifyError,
    verify_trust_manifest,
)


def _make_test_keypair(kid: str) -> tuple[dict, dict]:
    """Generate a fresh Ed25519 keypair as JWK dicts. In-memory; no
    disk artifacts. The returned tuple is (private_jwk, public_jwk).
    Both carry the same kid; the public dict adds alg/use claims."""
    key = OKPKey.generate_key("Ed25519")
    private_jwk = key.as_dict(private=True)
    public_jwk = key.as_dict(private=False)
    private_jwk["kid"] = kid
    public_jwk["kid"] = kid
    public_jwk["alg"] = "EdDSA"
    public_jwk["use"] = "sig"
    return private_jwk, public_jwk


def _make_sample_payload() -> dict:
    """A representative trust-root manifest payload. Values are
    realistic but arbitrary — these tests don't depend on the actual
    sum-engine release state."""
    return {
        "issued_at": "2026-04-27T18:00:00.000Z",
        "repo": "OtotaO/SUM",
        "commit": "a" * 40,
        "release": "v0.3.1",
        "artifacts": [
            {
                "name": "sum_engine-0.3.1-py3-none-any.whl",
                "kind": "pypi-wheel",
                "sha256": "1" * 64,
                "size_bytes": 184320,
                "pypi_provenance": "present",
                "github_attestation": "present",
                "cosign_bundle": "absent",
            },
            {
                "name": "sum_engine-0.3.1.tar.gz",
                "kind": "pypi-sdist",
                "sha256": "2" * 64,
                "size_bytes": 92160,
                "pypi_provenance": "present",
                "github_attestation": "present",
                "cosign_bundle": "absent",
            },
        ],
        "render_receipt_jwks": {
            "current_kids": ["sum-render-2026-04-27-1"],
            "revoked_kids": [],
            "jwks_sha256": "3" * 64,
        },
        "algorithm_registry": {
            "prime_scheme_current": "sha256_64_v1",
            "prime_scheme_next": "sha256_128_v2",
        },
    }


@pytest.fixture
def signed_manifest_and_jwks() -> tuple[dict, dict]:
    """Sign a sample payload with a fresh test keypair; return the
    signed manifest envelope + the matching JWKS for verification."""
    kid = "trust-root-test-key-1"
    private_jwk, public_jwk = _make_test_keypair(kid)
    payload = _make_sample_payload()
    envelope = sign_jose_envelope(payload, private_jwk=private_jwk, kid=kid)
    envelope["schema"] = SUPPORTED_SCHEMA
    jwks = {"keys": [public_jwk]}
    return envelope, jwks


# ---------------------------------------------------------------------------
# Round-trip happy path
# ---------------------------------------------------------------------------


def test_round_trip_verifies(signed_manifest_and_jwks):
    envelope, jwks = signed_manifest_and_jwks
    result = verify_trust_manifest(envelope, jwks)
    assert result.verified is True
    assert result.kid == envelope["kid"]
    assert result.payload["repo"] == "OtotaO/SUM"
    assert result.payload["release"] == "v0.3.1"
    assert result.protected_header["alg"] == "EdDSA"
    assert result.protected_header["b64"] is False
    assert "b64" in result.protected_header["crit"]


def test_payload_returned_matches_input():
    """Verifier must return the exact payload it verified — not a
    copy that's been mutated. The render-receipt verifier has the
    same contract; the trust-root verifier inherits it."""
    kid = "trust-root-test-key-2"
    priv, pub = _make_test_keypair(kid)
    payload = _make_sample_payload()
    envelope = sign_jose_envelope(payload, private_jwk=priv, kid=kid)
    envelope["schema"] = SUPPORTED_SCHEMA
    result = verify_trust_manifest(envelope, {"keys": [pub]})
    assert result.payload == payload


# ---------------------------------------------------------------------------
# Tampered-field rejection (signature_invalid for every signed field)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutator",
    [
        # commit hash tampering
        lambda p: p.__setitem__("commit", "b" * 40),
        # release tag tampering
        lambda p: p.__setitem__("release", "v9.9.9"),
        # repo tampering
        lambda p: p.__setitem__("repo", "evil/SUM"),
        # issued_at tampering
        lambda p: p.__setitem__("issued_at", "2099-01-01T00:00:00.000Z"),
        # artifact hash tampering
        lambda p: p["artifacts"][0].__setitem__("sha256", "f" * 64),
        # artifact name tampering
        lambda p: p["artifacts"][0].__setitem__("name", "evil-wheel.whl"),
        # JWKS state tampering
        lambda p: p["render_receipt_jwks"].__setitem__("jwks_sha256", "0" * 64),
        # JWKS revoked-kids tampering (drop entry)
        lambda p: p["render_receipt_jwks"].__setitem__("revoked_kids", ["evil-kid"]),
        # algorithm registry tampering
        lambda p: p["algorithm_registry"].__setitem__(
            "prime_scheme_current", "downgrade-scheme"
        ),
    ],
    ids=[
        "commit",
        "release",
        "repo",
        "issued_at",
        "artifact_sha256",
        "artifact_name",
        "jwks_sha256",
        "revoked_kids",
        "algorithm_registry",
    ],
)
def test_tampered_field_rejects_signature_invalid(mutator):
    kid = "trust-root-test-key-3"
    priv, pub = _make_test_keypair(kid)
    payload = _make_sample_payload()
    envelope = sign_jose_envelope(payload, private_jwk=priv, kid=kid)
    envelope["schema"] = SUPPORTED_SCHEMA

    # Mutate the signed-over payload AFTER signing; signature should
    # no longer verify against the mutated bytes.
    mutator(envelope["payload"])

    with pytest.raises(VerifyError) as excinfo:
        verify_trust_manifest(envelope, {"keys": [pub]})
    assert excinfo.value.error_class == ErrorClass.SIGNATURE_INVALID


# ---------------------------------------------------------------------------
# Forward-compat: schema gate
# ---------------------------------------------------------------------------


def test_schema_unknown_rejects(signed_manifest_and_jwks):
    envelope, jwks = signed_manifest_and_jwks
    envelope["schema"] = "sum.trust_root.v99"  # future version

    with pytest.raises(VerifyError) as excinfo:
        verify_trust_manifest(envelope, jwks)
    assert excinfo.value.error_class == ErrorClass.SCHEMA_UNKNOWN


def test_schema_mismatch_render_receipt_rejected(signed_manifest_and_jwks):
    """A trust-root verifier handed a render-receipt-shaped envelope
    must reject closed (different schema string). Cross-surface
    isolation."""
    envelope, jwks = signed_manifest_and_jwks
    envelope["schema"] = "sum.render_receipt.v1"

    with pytest.raises(VerifyError) as excinfo:
        verify_trust_manifest(envelope, jwks)
    assert excinfo.value.error_class == ErrorClass.SCHEMA_UNKNOWN


# ---------------------------------------------------------------------------
# JWKS-side failures
# ---------------------------------------------------------------------------


def test_unknown_kid_rejects(signed_manifest_and_jwks):
    envelope, _ = signed_manifest_and_jwks

    with pytest.raises(VerifyError) as excinfo:
        verify_trust_manifest(envelope, {"keys": []})
    assert excinfo.value.error_class == ErrorClass.UNKNOWN_KID


def test_wrong_key_rejects():
    """Manifest signed with key A but JWKS contains key B (same kid).
    Signature verification fails."""
    kid = "trust-root-test-key-4"
    priv_a, _ = _make_test_keypair(kid)
    _, pub_b = _make_test_keypair(kid)  # different key, same kid

    payload = _make_sample_payload()
    envelope = sign_jose_envelope(payload, private_jwk=priv_a, kid=kid)
    envelope["schema"] = SUPPORTED_SCHEMA

    with pytest.raises(VerifyError) as excinfo:
        verify_trust_manifest(envelope, {"keys": [pub_b]})
    assert excinfo.value.error_class == ErrorClass.SIGNATURE_INVALID


# ---------------------------------------------------------------------------
# Cross-surface taxonomy compatibility
# ---------------------------------------------------------------------------


def test_error_classes_match_render_receipt_taxonomy():
    """The trust-root and render-receipt verifiers share an error
    class taxonomy. Downstream consumers asserting by string match
    across surfaces should not need surface-specific branches."""
    from sum_engine_internal.render_receipt import ErrorClass as RRClass

    # Every shared name must have the same string value across both
    # surfaces. The two ErrorClass classes are aliases mapping into
    # JoseEnvelopeErrorClass.
    shared_names = [
        "MALFORMED_JWS",
        "MALFORMED_JWKS",
        "UNKNOWN_KID",
        "KID_MISMATCH",
        "SCHEMA_UNKNOWN",
        "CRIT_UNKNOWN_EXTENSION",
        "HEADER_INVARIANT_VIOLATED",
        "SIGNATURE_INVALID",
    ]
    for name in shared_names:
        assert getattr(ErrorClass, name) == getattr(RRClass, name), (
            f"taxonomy divergence on {name}: "
            f"trust_root={getattr(ErrorClass, name)!r} vs "
            f"render_receipt={getattr(RRClass, name)!r}"
        )

    # Surface-specific names map to the same shared envelope error.
    from sum_engine_internal.infrastructure.jose_envelope import (
        JoseEnvelopeErrorClass,
    )

    assert ErrorClass.MALFORMED_MANIFEST == JoseEnvelopeErrorClass.MALFORMED_ENVELOPE
    assert RRClass.MALFORMED_RECEIPT == JoseEnvelopeErrorClass.MALFORMED_ENVELOPE
    assert ErrorClass.MALFORMED_MANIFEST == RRClass.MALFORMED_RECEIPT


# ---------------------------------------------------------------------------
# Build script smoke (no network required)
# ---------------------------------------------------------------------------


def test_build_manifest_payload_shape(tmp_path):
    """The unsigned payload from build_trust_manifest matches the
    schema's payload contract. Doesn't run the script's CLI; calls
    build_manifest() directly with mocked inputs to keep the test
    network-free."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from build_trust_manifest import build_manifest

    payload = build_manifest(
        release="v0.3.1",
        commit="a" * 40,
        artifacts=[
            {
                "name": "test.whl",
                "kind": "pypi-wheel",
                "sha256": "1" * 64,
                "size_bytes": 100,
                "pypi_provenance": "present",
                "github_attestation": "present",
                "cosign_bundle": "absent",
            }
        ],
        jwks_raw=b'{"keys":[{"kid":"k1","kty":"OKP","crv":"Ed25519","x":"_"}]}',
        jwks_parsed={
            "keys": [{"kid": "k1", "kty": "OKP", "crv": "Ed25519", "x": "_"}]
        },
        algorithm_current="sha256_64_v1",
        algorithm_next="sha256_128_v2",
        revoked_kids=[],
    )

    # Required fields per spec §1.1
    assert payload["repo"] == "OtotaO/SUM"
    assert payload["commit"] == "a" * 40
    assert payload["release"] == "v0.3.1"
    assert "issued_at" in payload
    assert payload["artifacts"][0]["sha256"] == "1" * 64
    assert payload["render_receipt_jwks"]["current_kids"] == ["k1"]
    assert payload["render_receipt_jwks"]["jwks_sha256"]
    assert payload["algorithm_registry"]["prime_scheme_current"] == "sha256_64_v1"
    assert payload["algorithm_registry"]["prime_scheme_next"] == "sha256_128_v2"
