"""Tests for internal.infrastructure.verifiable_credential.

Covers:
- Unsigned credential construction (types, contexts, subject copy semantics)
- eddsa-jcs-2022 sign + verify round-trip
- Failure modes: tampered document, tampered proof, wrong key, missing proof,
  wrong cryptosuite
- Multibase base58btc encode/decode round-trip + leading-zero preservation
- Determinism: same input + same key → same signature (Ed25519 is deterministic)
- JSON-through-disk round-trip (the exercise any real VC ecosystem will do)
"""
from __future__ import annotations

import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from internal.infrastructure.verifiable_credential import (
    CRYPTOSUITE,
    DATA_INTEGRITY_PROOF_TYPE,
    VC_V2_CONTEXT,
    VerificationError,
    make_credential,
    multibase_base58btc_decode,
    multibase_base58btc_encode,
    sign_credential,
    verify_credential,
)


def _keypair() -> tuple[Ed25519PrivateKey, object]:
    sk = Ed25519PrivateKey.generate()
    return sk, sk.public_key()


# ─── Multibase base58btc ─────────────────────────────────────────────


class TestMultibase:
    def test_empty(self) -> None:
        assert multibase_base58btc_encode(b"") == "z"
        assert multibase_base58btc_decode("z") == b""

    def test_single_byte(self) -> None:
        assert multibase_base58btc_decode(multibase_base58btc_encode(b"\x01")) == b"\x01"

    def test_random_roundtrip(self) -> None:
        import os

        for _ in range(8):
            raw = os.urandom(64)
            encoded = multibase_base58btc_encode(raw)
            assert encoded.startswith("z")
            assert multibase_base58btc_decode(encoded) == raw

    def test_leading_zeros_preserved(self) -> None:
        raw = b"\x00\x00\x01\x02\x03"
        out = multibase_base58btc_encode(raw)
        decoded = multibase_base58btc_decode(out)
        assert decoded == raw

    def test_all_zeros(self) -> None:
        raw = b"\x00" * 5
        out = multibase_base58btc_encode(raw)
        assert multibase_base58btc_decode(out) == raw

    def test_non_z_prefix_rejected(self) -> None:
        with pytest.raises(ValueError, match="multibase prefix"):
            multibase_base58btc_decode("mAAAA")

    def test_invalid_char_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid base58btc character"):
            multibase_base58btc_decode("z0OIl")  # 0/O/I/l excluded from base58


# ─── make_credential ─────────────────────────────────────────────────


class TestMakeCredential:
    def test_minimal(self) -> None:
        cred = make_credential(
            subject={"id": "did:example:alice", "name": "Alice"},
            issuer="did:example:issuer",
            valid_from="2026-04-18T00:00:00Z",
        )
        assert cred["@context"] == [VC_V2_CONTEXT]
        assert cred["type"] == ["VerifiableCredential"]
        assert cred["issuer"] == "did:example:issuer"
        assert cred["credentialSubject"]["name"] == "Alice"
        assert cred["validFrom"] == "2026-04-18T00:00:00Z"
        assert "proof" not in cred

    def test_extra_type_prepends_base(self) -> None:
        cred = make_credential(
            subject={},
            issuer="did:example:x",
            credential_type="SumAttestation",
            valid_from="2026-04-18T00:00:00Z",
        )
        assert cred["type"] == ["VerifiableCredential", "SumAttestation"]

    def test_list_types_preserved_with_vc_first(self) -> None:
        cred = make_credential(
            subject={},
            issuer="did:example:x",
            credential_type=["SumAttestation", "MerkleCommitment"],
            valid_from="2026-04-18T00:00:00Z",
        )
        assert cred["type"] == [
            "VerifiableCredential",
            "SumAttestation",
            "MerkleCommitment",
        ]

    def test_extra_contexts(self) -> None:
        cred = make_credential(
            subject={},
            issuer="did:example:x",
            extra_contexts=["https://sum.ototao.dev/ns/v1"],
            valid_from="2026-04-18T00:00:00Z",
        )
        assert cred["@context"] == [
            VC_V2_CONTEXT,
            "https://sum.ototao.dev/ns/v1",
        ]

    def test_id_included_when_supplied(self) -> None:
        cred = make_credential(
            subject={},
            issuer="did:example:x",
            credential_id="urn:uuid:abc",
            valid_from="2026-04-18T00:00:00Z",
        )
        assert cred["id"] == "urn:uuid:abc"

    def test_empty_issuer_rejected(self) -> None:
        with pytest.raises(ValueError, match="issuer is required"):
            make_credential(subject={}, issuer="")

    def test_subject_is_copied(self) -> None:
        subject = {"a": 1}
        cred = make_credential(
            subject=subject,
            issuer="did:example:x",
            valid_from="2026-04-18T00:00:00Z",
        )
        subject["a"] = 999
        assert cred["credentialSubject"]["a"] == 1

    def test_reserved_extra_field_rejected(self) -> None:
        with pytest.raises(ValueError, match="reserved field"):
            make_credential(
                subject={},
                issuer="did:example:x",
                extra_fields={"@context": ["evil"]},
                valid_from="2026-04-18T00:00:00Z",
            )


# ─── sign + verify round-trip ────────────────────────────────────────


class TestSignVerify:
    def test_signed_credential_verifies(self) -> None:
        sk, pk = _keypair()
        cred = make_credential(
            subject={"name": "Alice"},
            issuer="did:example:issuer",
            valid_from="2026-04-18T00:00:00Z",
        )
        signed = sign_credential(
            cred,
            private_key=sk,
            verification_method="did:example:issuer#key-1",
            created="2026-04-18T00:00:00Z",
        )
        assert signed["proof"]["type"] == DATA_INTEGRITY_PROOF_TYPE
        assert signed["proof"]["cryptosuite"] == CRYPTOSUITE
        assert signed["proof"]["proofValue"].startswith("z")
        assert verify_credential(signed, pk) is True

    def test_sign_does_not_mutate_input(self) -> None:
        sk, _ = _keypair()
        cred = make_credential(
            subject={},
            issuer="did:example:x",
            valid_from="2026-04-18T00:00:00Z",
        )
        before = json.dumps(cred, sort_keys=True)
        _ = sign_credential(cred, sk, "did:example:x#k")
        after = json.dumps(cred, sort_keys=True)
        assert before == after

    def test_sign_is_deterministic_for_same_inputs(self) -> None:
        sk, _ = _keypair()
        cred = make_credential(
            subject={"x": 1},
            issuer="did:example:x",
            valid_from="2026-04-18T00:00:00Z",
        )
        a = sign_credential(
            cred, sk, "did:example:x#k", created="2026-04-18T00:00:00Z"
        )
        b = sign_credential(
            cred, sk, "did:example:x#k", created="2026-04-18T00:00:00Z"
        )
        assert a["proof"]["proofValue"] == b["proof"]["proofValue"]

    def test_double_sign_rejected(self) -> None:
        sk, _ = _keypair()
        cred = make_credential(
            subject={},
            issuer="did:example:x",
            valid_from="2026-04-18T00:00:00Z",
        )
        signed = sign_credential(cred, sk, "did:example:x#k")
        with pytest.raises(ValueError, match="already has a proof"):
            sign_credential(signed, sk, "did:example:x#k")

    def test_empty_verification_method_rejected(self) -> None:
        sk, _ = _keypair()
        cred = make_credential(
            subject={},
            issuer="did:example:x",
            valid_from="2026-04-18T00:00:00Z",
        )
        with pytest.raises(ValueError, match="verification_method"):
            sign_credential(cred, sk, "")


class TestVerifyFailureModes:
    def _signed(self) -> tuple[dict, object, object]:
        sk, pk = _keypair()
        cred = make_credential(
            subject={"x": 1},
            issuer="did:example:x",
            valid_from="2026-04-18T00:00:00Z",
        )
        signed = sign_credential(
            cred, sk, "did:example:x#k", created="2026-04-18T00:00:00Z"
        )
        return signed, sk, pk

    def test_tampered_subject_fails(self) -> None:
        signed, _sk, pk = self._signed()
        tampered = {**signed, "credentialSubject": {"x": 2}}
        with pytest.raises(VerificationError, match="does not verify"):
            verify_credential(tampered, pk)

    def test_tampered_proof_value_fails(self) -> None:
        signed, _sk, pk = self._signed()
        bad_proof = dict(signed["proof"])
        # Flip one character of the multibase-encoded signature.
        pv = bad_proof["proofValue"]
        flipped = pv[:2] + ("A" if pv[2] != "A" else "B") + pv[3:]
        bad_proof["proofValue"] = flipped
        tampered = {**signed, "proof": bad_proof}
        with pytest.raises(VerificationError):
            verify_credential(tampered, pk)

    def test_wrong_public_key_fails(self) -> None:
        signed, _sk, _pk = self._signed()
        other_sk, other_pk = _keypair()
        _ = other_sk
        with pytest.raises(VerificationError, match="does not verify"):
            verify_credential(signed, other_pk)

    def test_missing_proof_fails(self) -> None:
        _signed, _sk, pk = self._signed()
        cred = make_credential(
            subject={},
            issuer="did:example:x",
            valid_from="2026-04-18T00:00:00Z",
        )
        with pytest.raises(VerificationError, match="missing or malformed"):
            verify_credential(cred, pk)

    def test_wrong_cryptosuite_fails(self) -> None:
        signed, _sk, pk = self._signed()
        bad = {**signed, "proof": {**signed["proof"], "cryptosuite": "eddsa-rdfc-2022"}}
        with pytest.raises(VerificationError, match="cryptosuite"):
            verify_credential(bad, pk)

    def test_wrong_proof_type_fails(self) -> None:
        signed, _sk, pk = self._signed()
        bad = {**signed, "proof": {**signed["proof"], "type": "Other"}}
        with pytest.raises(VerificationError, match="proof type"):
            verify_credential(bad, pk)


# ─── JSON-on-disk round-trip (what a real consumer would do) ─────────


class TestOnDiskRoundtrip:
    def test_json_roundtrip_preserves_verification(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        sk, pk = _keypair()
        cred = make_credential(
            subject={
                "stateInteger": "12345",
                "axiomCount": 50,
                "branch": "main",
            },
            issuer="did:example:issuer",
            credential_type="SumStateAttestation",
            valid_from="2026-04-18T00:00:00Z",
        )
        signed = sign_credential(
            cred,
            sk,
            "did:example:issuer#key-1",
            created="2026-04-18T00:00:00Z",
        )

        path = tmp_path / "vc.jsonld"
        path.write_text(json.dumps(signed, ensure_ascii=False), encoding="utf-8")

        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert verify_credential(loaded, pk) is True

    def test_key_reordering_does_not_break_verification(self) -> None:
        # JCS canonicalizes key order before signing, so a consumer that
        # reorders keys (common in JSON parsers / network transfers) must
        # still verify. This is the whole point of using JCS.
        sk, pk = _keypair()
        cred = make_credential(
            subject={"a": 1, "b": 2, "c": 3},
            issuer="did:example:x",
            valid_from="2026-04-18T00:00:00Z",
        )
        signed = sign_credential(cred, sk, "did:example:x#k")

        reordered = {
            "credentialSubject": {"c": 3, "a": 1, "b": 2},
            "@context": signed["@context"],
            "validFrom": signed["validFrom"],
            "issuer": signed["issuer"],
            "type": signed["type"],
            "proof": {
                "proofValue": signed["proof"]["proofValue"],
                "type": signed["proof"]["type"],
                "verificationMethod": signed["proof"]["verificationMethod"],
                "proofPurpose": signed["proof"]["proofPurpose"],
                "cryptosuite": signed["proof"]["cryptosuite"],
                "created": signed["proof"]["created"],
            },
        }
        assert verify_credential(reordered, pk) is True
