"""Tests for sum_engine_internal.infrastructure.verifiable_credential.

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

from sum_engine_internal.infrastructure.verifiable_credential import (
    CRYPTOSUITE,
    DATA_INTEGRITY_PROOF_TYPE,
    DID_CONTEXT,
    ED25519_MULTICODEC_PREFIX,
    VC_V2_CONTEXT,
    VerificationError,
    build_did_web_document,
    did_web_verification_method,
    ed25519_public_key_multibase,
    ed25519_to_did_key,
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


# ─── DID helpers (did:key + did:web) ─────────────────────────────────


class TestEd25519DidKeyEncoding:
    def test_multibase_has_z_prefix(self) -> None:
        sk, pk = _keypair()
        mb = ed25519_public_key_multibase(pk)
        assert mb.startswith("z")
        # z-prefix + multicodec-encoded-in-base58 payload. Ed25519 keys
        # always produce a string that begins with z6Mk because the
        # 0xed 0x01 multicodec prefix + 32 bytes encodes to that form.
        assert mb.startswith("z6Mk")

    def test_did_key_format(self) -> None:
        sk, pk = _keypair()
        did = ed25519_to_did_key(pk)
        assert did.startswith("did:key:z6Mk")

    def test_multibase_decode_yields_prefix_plus_key(self) -> None:
        sk, pk = _keypair()
        mb = ed25519_public_key_multibase(pk)
        decoded = multibase_base58btc_decode(mb)
        assert decoded.startswith(ED25519_MULTICODEC_PREFIX)
        assert len(decoded) == len(ED25519_MULTICODEC_PREFIX) + 32

    def test_distinct_keys_distinct_dids(self) -> None:
        sk1, pk1 = _keypair()
        sk2, pk2 = _keypair()
        assert ed25519_to_did_key(pk1) != ed25519_to_did_key(pk2)


class TestDidWebVerificationMethod:
    def test_basic(self) -> None:
        assert did_web_verification_method("sum-demo.pages.dev") == "did:web:sum-demo.pages.dev#key-1"

    def test_custom_key_id(self) -> None:
        assert did_web_verification_method("example.org", "key-42") == "did:web:example.org#key-42"

    def test_rejects_scheme(self) -> None:
        with pytest.raises(ValueError, match="scheme"):
            did_web_verification_method("https://example.org")

    def test_rejects_trailing_slash(self) -> None:
        with pytest.raises(ValueError, match="scheme"):
            did_web_verification_method("example.org/")


class TestBuildDidWebDocument:
    def test_basic_shape(self) -> None:
        _, pk = _keypair()
        doc = build_did_web_document("sum-demo.pages.dev", pk)
        assert doc["@context"] == DID_CONTEXT
        assert doc["id"] == "did:web:sum-demo.pages.dev"
        assert len(doc["verificationMethod"]) == 1
        vm = doc["verificationMethod"][0]
        assert vm["id"] == "did:web:sum-demo.pages.dev#key-1"
        assert vm["type"] == "Multikey"
        assert vm["controller"] == "did:web:sum-demo.pages.dev"
        assert vm["publicKeyMultibase"].startswith("z6Mk")
        assert doc["assertionMethod"] == ["did:web:sum-demo.pages.dev#key-1"]
        assert doc["authentication"] == ["did:web:sum-demo.pages.dev#key-1"]

    def test_custom_key_id_propagates(self) -> None:
        _, pk = _keypair()
        doc = build_did_web_document("example.org", pk, key_id="issuer-2026")
        assert doc["verificationMethod"][0]["id"] == "did:web:example.org#issuer-2026"
        assert doc["assertionMethod"] == ["did:web:example.org#issuer-2026"]

    def test_also_known_as_included_when_given(self) -> None:
        _, pk = _keypair()
        did_key = ed25519_to_did_key(pk)
        doc = build_did_web_document("example.org", pk, also_known_as=[did_key])
        assert doc["alsoKnownAs"] == [did_key]

    def test_also_known_as_omitted_when_empty(self) -> None:
        _, pk = _keypair()
        doc = build_did_web_document("example.org", pk)
        assert "alsoKnownAs" not in doc


class TestDidBasedCredentialSigning:
    """Signing a credential with did:web / did:key verificationMethod
    still round-trips cleanly. The DID is an opaque string at the
    crypto layer — the verifier still needs the public key out-of-band
    today — but the `verificationMethod` IRI is now dereferenceable by
    any W3C-compliant resolver (DIF Universal Resolver, Veramo, etc.)."""

    def test_did_web_verification_method_signs_and_verifies(self) -> None:
        sk, pk = _keypair()
        cred = make_credential(
            subject={"axiom": "alice||like||cat"},
            issuer="did:web:sum-demo.pages.dev",
            valid_from="2026-04-20T00:00:00Z",
        )
        signed = sign_credential(
            cred,
            sk,
            did_web_verification_method("sum-demo.pages.dev"),
            created="2026-04-20T00:00:00Z",
        )
        assert signed["proof"]["verificationMethod"] == "did:web:sum-demo.pages.dev#key-1"
        assert verify_credential(signed, pk) is True

    def test_did_key_self_resolving_verification_method(self) -> None:
        sk, pk = _keypair()
        did_key = ed25519_to_did_key(pk)
        # did:key's verificationMethod convention is `did:key:z...#z...`
        # (the fragment equals the multibase-encoded key). For SUM's
        # purposes the DID itself is enough — verifiers derive the key
        # from it directly.
        vm = f"{did_key}#{did_key.split(':')[-1]}"
        cred = make_credential(
            subject={"axiom": "alice||like||cat"},
            issuer=did_key,
            valid_from="2026-04-20T00:00:00Z",
        )
        signed = sign_credential(cred, sk, vm, created="2026-04-20T00:00:00Z")
        assert verify_credential(signed, pk) is True
