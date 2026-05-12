"""T4 — source-bytes binding tests.

Adds an optional `source_chain_hash` field to sum.transform_receipt.v1
payloads that binds the receipt to a list of (claim, source_uri,
byte_range) evidence links. Tests:

  1. compute_source_chain_hash is byte-stable across input ordering.
  2. Receipt with source_chain_hash signs + verifies cleanly.
  3. Receipt without source_chain_hash signs + verifies cleanly
     (the field is omitted, not present-with-null — receipts pre-T4
     remain byte-identical).
  4. Tampering source_chain_hash invalidates the signature.
  5. Application-layer integrity: recomputing the chain hash against
     the supplied chain matches the receipt field; tampering the
     chain mismatches.
"""
from __future__ import annotations

import copy

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
    compute_source_chain_hash,
    sign_transform_receipt,
    verify_transform_receipt,
)


# ─── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def keypair():
    from joserfc.jwk import OKPKey

    kid = "test-source-chain-2026"
    key = OKPKey.generate_key("Ed25519")
    private = key.as_dict(private=True)
    private["kid"] = kid
    public = key.as_dict(private=False)
    public["kid"] = kid
    public["alg"] = "EdDSA"
    public["use"] = "sig"
    return private, {"keys": [public]}, kid


@pytest.fixture
def chain():
    """A small representative evidence chain: two claims from the
    same source URI at different byte ranges."""
    return [
        {
            "claim": "alice||likes||cats",
            "provenance": {
                "source_uri": "https://example.com/article-42",
                "byte_start": 0,
                "byte_end": 18,
            },
        },
        {
            "claim": "bob||owns||dog",
            "provenance": {
                "source_uri": "https://example.com/article-42",
                "byte_start": 20,
                "byte_end": 35,
            },
        },
    ]


# ─── compute_source_chain_hash byte stability ───────────────────────


def test_empty_chain_returns_none():
    assert compute_source_chain_hash(None) is None
    assert compute_source_chain_hash([]) is None


def test_chain_hash_is_byte_stable_under_reordering(chain):
    """Producer-side ordering must not affect the chain hash —
    sorting key in compute_source_chain_hash takes care of it."""
    forward = compute_source_chain_hash(chain)
    reverse = compute_source_chain_hash(list(reversed(chain)))
    assert forward == reverse
    assert forward.startswith("sha256-")


def test_chain_hash_changes_on_byte_range_mutation(chain):
    """Mutating one link's byte_range changes the hash — the binding
    actually binds to bytes."""
    original = compute_source_chain_hash(chain)
    mutated = copy.deepcopy(chain)
    mutated[0]["provenance"]["byte_end"] = 999
    assert compute_source_chain_hash(mutated) != original


def test_chain_hash_changes_on_claim_mutation(chain):
    original = compute_source_chain_hash(chain)
    mutated = copy.deepcopy(chain)
    mutated[0]["claim"] = "alice||likes||dogs"
    assert compute_source_chain_hash(mutated) != original


def test_chain_hash_changes_on_source_uri_mutation(chain):
    original = compute_source_chain_hash(chain)
    mutated = copy.deepcopy(chain)
    mutated[0]["provenance"]["source_uri"] = "https://other.example.com/x"
    assert compute_source_chain_hash(mutated) != original


# ─── Receipt payload with source_chain_hash ─────────────────────────


def test_payload_includes_source_chain_hash_when_provided(chain):
    chain_hash = compute_source_chain_hash(chain)
    payload = build_payload(
        transform="slider",
        parameters_hash="sha256-" + "a" * 64,
        input_hash="sha256-" + "b" * 64,
        output_hash="sha256-" + "c" * 64,
        model="canonical-deterministic-v0",
        provider="canonical-path",
        digital_source_type="algorithmicMedia",
        source_chain_hash=chain_hash,
    )
    assert "source_chain_hash" in payload
    assert payload["source_chain_hash"] == chain_hash


def test_payload_omits_source_chain_hash_when_absent():
    """Pre-T4 receipts (no source_chain_hash) MUST remain byte-
    identical: the field is omitted entirely, not present-with-null."""
    payload = build_payload(
        transform="slider",
        parameters_hash="sha256-" + "a" * 64,
        input_hash="sha256-" + "b" * 64,
        output_hash="sha256-" + "c" * 64,
        model="canonical-deterministic-v0",
        provider="canonical-path",
        digital_source_type="algorithmicMedia",
    )
    assert "source_chain_hash" not in payload


# ─── Sign + verify with source_chain_hash ───────────────────────────


def test_receipt_with_chain_signs_and_verifies(chain, keypair):
    private, jwks, kid = keypair
    chain_hash = compute_source_chain_hash(chain)
    payload = build_payload(
        transform="slider",
        parameters_hash="sha256-" + "a" * 64,
        input_hash="sha256-" + "b" * 64,
        output_hash="sha256-" + "c" * 64,
        model="canonical-deterministic-v0",
        provider="canonical-path",
        digital_source_type="algorithmicMedia",
        source_chain_hash=chain_hash,
    )
    receipt = sign_transform_receipt(payload, private_jwk=private, kid=kid)
    assert receipt["schema"] == SUPPORTED_SCHEMA
    assert receipt["payload"]["source_chain_hash"] == chain_hash
    result = verify_transform_receipt(receipt, jwks)
    assert result.verified is True


def test_receipt_without_chain_still_signs_and_verifies(keypair):
    """T4 must not break receipts that don't carry source provenance."""
    private, jwks, kid = keypair
    payload = build_payload(
        transform="slider",
        parameters_hash="sha256-" + "a" * 64,
        input_hash="sha256-" + "b" * 64,
        output_hash="sha256-" + "c" * 64,
        model="canonical-deterministic-v0",
        provider="canonical-path",
        digital_source_type="algorithmicMedia",
    )
    receipt = sign_transform_receipt(payload, private_jwk=private, kid=kid)
    assert "source_chain_hash" not in receipt["payload"]
    result = verify_transform_receipt(receipt, jwks)
    assert result.verified is True


# ─── Tamper test on the new field ───────────────────────────────────


def test_tampering_source_chain_hash_rejects(chain, keypair):
    private, jwks, kid = keypair
    chain_hash = compute_source_chain_hash(chain)
    payload = build_payload(
        transform="slider",
        parameters_hash="sha256-" + "a" * 64,
        input_hash="sha256-" + "b" * 64,
        output_hash="sha256-" + "c" * 64,
        model="canonical-deterministic-v0",
        provider="canonical-path",
        digital_source_type="algorithmicMedia",
        source_chain_hash=chain_hash,
    )
    receipt = sign_transform_receipt(payload, private_jwk=private, kid=kid)
    bad = copy.deepcopy(receipt)
    bad["payload"]["source_chain_hash"] = "sha256-" + "0" * 64
    with pytest.raises(VerifyError) as exc:
        verify_transform_receipt(bad, jwks)
    assert exc.value.error_class == ErrorClass.SIGNATURE_INVALID


def test_adding_source_chain_hash_to_receipt_without_one_rejects(keypair):
    """A receipt signed WITHOUT source_chain_hash cannot be modified to
    include one without breaking the signature. JCS canonicalisation
    over the modified payload differs from the signed bytes."""
    private, jwks, kid = keypair
    payload = build_payload(
        transform="slider",
        parameters_hash="sha256-" + "a" * 64,
        input_hash="sha256-" + "b" * 64,
        output_hash="sha256-" + "c" * 64,
        model="canonical-deterministic-v0",
        provider="canonical-path",
        digital_source_type="algorithmicMedia",
    )
    receipt = sign_transform_receipt(payload, private_jwk=private, kid=kid)
    bad = copy.deepcopy(receipt)
    bad["payload"]["source_chain_hash"] = "sha256-" + "0" * 64
    with pytest.raises(VerifyError) as exc:
        verify_transform_receipt(bad, jwks)
    assert exc.value.error_class == ErrorClass.SIGNATURE_INVALID


# ─── Application-layer integrity ────────────────────────────────────


def test_recomputed_chain_hash_matches_genuine_receipt(chain, keypair):
    """A verifier that wants to bind a receipt to a SPECIFIC chain
    recomputes the hash and compares to payload.source_chain_hash."""
    private, jwks, kid = keypair
    chain_hash = compute_source_chain_hash(chain)
    payload = build_payload(
        transform="slider",
        parameters_hash="sha256-" + "a" * 64,
        input_hash="sha256-" + "b" * 64,
        output_hash="sha256-" + "c" * 64,
        model="canonical-deterministic-v0",
        provider="canonical-path",
        digital_source_type="algorithmicMedia",
        source_chain_hash=chain_hash,
    )
    receipt = sign_transform_receipt(payload, private_jwk=private, kid=kid)

    # Verify the signature first.
    verify_transform_receipt(receipt, jwks)

    # Then the application-layer integrity check: recompute the
    # chain hash against the supplied chain.
    recomputed = compute_source_chain_hash(chain)
    assert recomputed == receipt["payload"]["source_chain_hash"]


def test_recomputed_chain_hash_mismatches_tampered_chain(chain, keypair):
    """If the chain supplied to the verifier was modified after
    signing, recompute mismatches even though the receipt itself
    still verifies (signature was over the original hash)."""
    private, jwks, kid = keypair
    chain_hash = compute_source_chain_hash(chain)
    payload = build_payload(
        transform="slider",
        parameters_hash="sha256-" + "a" * 64,
        input_hash="sha256-" + "b" * 64,
        output_hash="sha256-" + "c" * 64,
        model="canonical-deterministic-v0",
        provider="canonical-path",
        digital_source_type="algorithmicMedia",
        source_chain_hash=chain_hash,
    )
    receipt = sign_transform_receipt(payload, private_jwk=private, kid=kid)

    # Signature still verifies — the receipt is genuine.
    verify_transform_receipt(receipt, jwks)

    # But a chain modified post-signing has a different hash.
    mutated_chain = copy.deepcopy(chain)
    mutated_chain[0]["provenance"]["byte_end"] = 999
    recomputed = compute_source_chain_hash(mutated_chain)
    assert recomputed != receipt["payload"]["source_chain_hash"]
