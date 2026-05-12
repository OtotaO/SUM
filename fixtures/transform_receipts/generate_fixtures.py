#!/usr/bin/env python3
"""Generate the transform-receipt fixture set.

Mirrors ``fixtures/render_receipts/generate_fixtures.py`` for the
``sum.transform_receipt.v1`` schema. Consumed by:

  - Python verifier: ``Tests/test_transform_receipt_verifier_fixtures.py``
    via ``sum_engine_internal.transform_receipt.verify_transform_receipt``.
  - Browser/Node verifier: ``single_file_demo/transform_receipt_verifier.js``
    via a smoke harness that loops the same JSON fixtures.

Cross-runtime byte-identical outcomes on every fixture is the K-style
equivalence guarantee already proved for render receipts, extended to
transform receipts.

Deterministic by construction
-----------------------------
The signing key is derived from a fixed 32-byte seed (all zeros — the
classic RFC 8032 §7.1 test vector seed) so re-running this script
produces byte-identical fixtures. Ed25519 signatures are also
deterministic per RFC 8032, so the JWS segment is stable.

This is a TEST-ONLY key. It MUST NOT be used to sign anything outside
this fixture set. The private scalar is embedded in this script (not
committed as a file) so anyone regenerating fixtures gets the same
bytes; the key has no operator-side meaning.

Run
---
    python fixtures/transform_receipts/generate_fixtures.py

Idempotent: same inputs produce byte-identical outputs.
"""
from __future__ import annotations

import base64
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


HERE = Path(__file__).resolve().parent

# Fixed seed → fixed Ed25519 key. RFC 8032 §7.1 test vector 1 seed (all
# zero bytes). Test-only; the matching public key x parameter is
# `11qYAYKxCrfVS_7TyWQHOg7hcvPapiMlrwIaaPcHURo` per the RFC.
TEST_SEED = b"\x00" * 32
TEST_KID = "transform-fixture-key-2026"
# Fixed signed_at so re-running the script is fully byte-stable. The
# actual signed_at semantics (UTC ISO-8601 millis with trailing Z)
# match what build_payload would stamp; we just freeze the value.
FIXED_SIGNED_AT = "2026-05-12T12:00:00.000Z"


def b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _derive_test_jwks() -> tuple[dict, dict]:
    """Return (private_jwk, public_jwks) derived deterministically
    from TEST_SEED. private_jwk is for signing the source receipt;
    only the public_jwks is committed to disk."""
    sk = Ed25519PrivateKey.from_private_bytes(TEST_SEED)
    pk_bytes = sk.public_key().public_bytes_raw()
    private = {
        "kty": "OKP",
        "crv": "Ed25519",
        "d": b64url(TEST_SEED),
        "x": b64url(pk_bytes),
        "kid": TEST_KID,
        "alg": "EdDSA",
        "use": "sig",
    }
    public = {
        "kty": "OKP",
        "crv": "Ed25519",
        "x": b64url(pk_bytes),
        "kid": TEST_KID,
        "alg": "EdDSA",
        "use": "sig",
    }
    return private, {"keys": [public]}


def _sign_source_receipt(private_jwk: dict) -> dict:
    """Sign one canonical-path slider transform-receipt with the test
    key. Returns the four-key envelope dict — the basis every mutation
    fixture derives from."""
    from sum_engine_internal.transform_receipt import (
        build_payload,
        canonical_hash,
        sign_transform_receipt,
    )
    from sum_engine_internal.infrastructure.jcs import canonicalize

    # Frozen input + parameters so hashes are stable across re-runs.
    parameters = {
        "density": 1.0,
        "length": 0.5,
        "formality": 0.5,
        "audience": 0.5,
        "perspective": 0.5,
    }
    input_doc = {"triples": [["alice", "likes", "cats"], ["bob", "owns", "dog"]]}
    output_tome = "alice likes cats. bob owns dog."

    parameters_hash = canonical_hash(canonicalize(parameters))
    input_hash = canonical_hash(canonicalize(input_doc))
    output_hash = canonical_hash(output_tome.encode("utf-8"))

    payload = build_payload(
        transform="slider",
        parameters_hash=parameters_hash,
        input_hash=input_hash,
        output_hash=output_hash,
        model="canonical-deterministic-v0",
        provider="canonical-path",
        digital_source_type="algorithmicMedia",
        signed_at=FIXED_SIGNED_AT,
    )
    return sign_transform_receipt(payload, private_jwk=private_jwk, kid=TEST_KID)


def write_fixture(name: str, fixture: dict) -> None:
    out = HERE / f"{name}.json"
    text = json.dumps(fixture, indent=2, sort_keys=True) + "\n"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out.name}")


def base_fixture(receipt: dict, jwks: dict) -> dict:
    return {
        "name": "",
        "description": "",
        "expected_outcome": "verify",
        "expected_error_class": None,
        "receipt": copy.deepcopy(receipt),
        "jwks": copy.deepcopy(jwks),
    }


def main() -> int:
    private_jwk, public_jwks = _derive_test_jwks()
    receipt = _sign_source_receipt(private_jwk)

    # Commit the source receipt + public JWKS so consumers can
    # reconstruct the fixture set bit-for-bit without ever holding the
    # private key.
    (HERE / "source_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (HERE / "jwks_at_capture.json").write_text(
        json.dumps(public_jwks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # ---- positive control ----
    pos = base_fixture(receipt, public_jwks)
    pos["name"] = "positive_control"
    pos["description"] = (
        "Receipt signed by the deterministic fixture key (seed = 32 "
        "zero bytes per RFC 8032 §7.1 test vector 1). No mutations. A "
        "correct verifier MUST verify this fixture."
    )
    write_fixture("positive_control", pos)

    # ---- tampered signed-payload fields ----
    # Every payload field is signed, so mutating any one of them breaks
    # the JWS signature. Each fixture is its own case so a verifier
    # that handles one path but skips another fails specifically.
    signed_payload_fields = [
        (
            "tampered_parameters_hash",
            "payload.parameters_hash mutated; signed_over → invalid signature.",
            lambda r: r["payload"].__setitem__(
                "parameters_hash",
                "sha256-deadbeef" + r["payload"]["parameters_hash"][15:],
            ),
        ),
        (
            "tampered_input_hash",
            "payload.input_hash mutated; signed_over → invalid signature.",
            lambda r: r["payload"].__setitem__(
                "input_hash",
                "sha256-cafebabe" + r["payload"]["input_hash"][15:],
            ),
        ),
        (
            "tampered_output_hash",
            "payload.output_hash mutated; signed_over → invalid signature.",
            lambda r: r["payload"].__setitem__(
                "output_hash",
                "sha256-feedface" + r["payload"]["output_hash"][15:],
            ),
        ),
        (
            "tampered_transform",
            "payload.transform mutated from 'slider' to 'extract'.",
            lambda r: r["payload"].__setitem__("transform", "extract"),
        ),
        (
            "tampered_transform_id",
            "payload.transform_id mutated to a zero-string.",
            lambda r: r["payload"].__setitem__("transform_id", "0000000000000000"),
        ),
        (
            "tampered_model",
            "payload.model mutated to a different model string.",
            lambda r: r["payload"].__setitem__("model", "imposter-model-9000"),
        ),
        (
            "tampered_provider",
            "payload.provider mutated.",
            lambda r: r["payload"].__setitem__("provider", "evil-corp"),
        ),
        (
            "tampered_signed_at",
            "payload.signed_at shifted by an hour.",
            lambda r: r["payload"].__setitem__(
                "signed_at",
                r["payload"]["signed_at"].replace("Z", "+01:00"),
            ),
        ),
        (
            "tampered_digital_source_type",
            "payload.digital_source_type swapped algorithmicMedia → "
            "trainedAlgorithmicMedia. Catches the LLM-pretending-to-be-"
            "deterministic provenance flip.",
            lambda r: r["payload"].__setitem__(
                "digital_source_type", "trainedAlgorithmicMedia"
            ),
        ),
    ]
    for name, desc, mutator in signed_payload_fields:
        f = base_fixture(receipt, public_jwks)
        f["name"] = name
        f["description"] = desc
        f["expected_outcome"] = "reject"
        f["expected_error_class"] = "signature_invalid"
        mutator(f["receipt"])
        write_fixture(name, f)

    # ---- tampered JWS signature segment ----
    f = base_fixture(receipt, public_jwks)
    f["name"] = "tampered_signature"
    f["description"] = (
        "Last character of the JWS signature segment substituted; "
        "decoded bytes differ; signature verification fails."
    )
    f["expected_outcome"] = "reject"
    f["expected_error_class"] = "signature_invalid"
    proto, middle, sig = receipt["jws"].split(".")
    sig_mutated = sig[:-1] + ("A" if sig[-1] != "A" else "B")
    f["receipt"]["jws"] = f"{proto}.{middle}.{sig_mutated}"
    write_fixture("tampered_signature", f)

    # ---- tampered kid in protected header ----
    f = base_fixture(receipt, public_jwks)
    f["name"] = "tampered_kid_header"
    f["description"] = (
        "kid claim inside the JWS protected header mutated to "
        "'spoofed-kid'. Top-level receipt.kid stays the original. "
        "The signature was computed over the original header bytes; "
        "verification fails."
    )
    f["expected_outcome"] = "reject"
    f["expected_error_class"] = "signature_invalid"
    proto, middle, sig = receipt["jws"].split(".")
    proto_json = json.loads(b64url_decode(proto).decode("utf-8"))
    proto_json["kid"] = "spoofed-kid"
    proto_mutated = b64url(
        json.dumps(proto_json, separators=(",", ":")).encode("utf-8")
    )
    f["receipt"]["jws"] = f"{proto_mutated}.{middle}.{sig}"
    write_fixture("tampered_kid_header", f)

    # ---- malformed detached JWS (middle segment non-empty) ----
    f = base_fixture(receipt, public_jwks)
    f["name"] = "malformed_jws_middle_nonempty"
    f["description"] = (
        "Middle JWS segment populated with arbitrary bytes; per RFC "
        "7515 §A.5 the middle segment of a detached JWS MUST be "
        "empty. A correct verifier rejects with malformed_jws before "
        "attempting cryptographic verification."
    )
    f["expected_outcome"] = "reject"
    f["expected_error_class"] = "malformed_jws"
    proto, _, sig = receipt["jws"].split(".")
    f["receipt"]["jws"] = f"{proto}.YWJjZGVm.{sig}"
    write_fixture("malformed_jws_middle_nonempty", f)

    # ---- unknown kid (JWKS missing the receipt's kid) ----
    f = base_fixture(receipt, public_jwks)
    f["name"] = "unknown_kid"
    f["description"] = (
        "Receipt is the unmodified positive control, but the JWKS "
        "supplied to the verifier is empty. Verifier rejects with "
        "unknown_kid before any cryptographic operation."
    )
    f["expected_outcome"] = "reject"
    f["expected_error_class"] = "unknown_kid"
    f["jwks"] = {"keys": []}
    write_fixture("unknown_kid", f)

    # ---- forward-compat: schema_unknown ----
    f = base_fixture(receipt, public_jwks)
    f["name"] = "schema_unknown"
    f["description"] = (
        "receipt.schema mutated to 'sum.transform_receipt.v99' (a "
        "future-version this verifier doesn't know). Per "
        "TRANSFORM_RECEIPT_FORMAT.md §1.5, a v1-aware verifier MUST "
        "fail closed on a v99 receipt."
    )
    f["expected_outcome"] = "reject"
    f["expected_error_class"] = "schema_unknown"
    f["receipt"]["schema"] = "sum.transform_receipt.v99"
    write_fixture("schema_unknown", f)

    # ---- schema confusion: sum.render_receipt.v1 ----
    # Distinct forward-compat path: a render-receipt envelope MUST NOT
    # validate against the transform-receipt verifier even if it's a
    # cryptographically-valid render receipt. Locks the schema gate as
    # a cross-receipt-type firewall.
    f = base_fixture(receipt, public_jwks)
    f["name"] = "schema_confusion_render_receipt"
    f["description"] = (
        "receipt.schema mutated to 'sum.render_receipt.v1' (a "
        "different receipt format that uses the same JWS shape). The "
        "transform-receipt verifier MUST reject — receipt-type "
        "confusion is the same class of bug as JWT alg-confusion."
    )
    f["expected_outcome"] = "reject"
    f["expected_error_class"] = "schema_unknown"
    f["receipt"]["schema"] = "sum.render_receipt.v1"
    write_fixture("schema_confusion_render_receipt", f)

    # ---- forward-compat: crit_unknown_extension ----
    f = base_fixture(receipt, public_jwks)
    f["name"] = "crit_unknown_extension"
    f["description"] = (
        "JWS protected header `crit` array extended with "
        "'sum-future-feature' and a corresponding header claim. Per "
        "RFC 7515 §4.1.11, a verifier that doesn't understand a "
        "critical extension MUST reject closed."
    )
    f["expected_outcome"] = "reject"
    f["expected_error_class"] = "crit_unknown_extension"
    proto, middle, sig = receipt["jws"].split(".")
    proto_json = json.loads(b64url_decode(proto).decode("utf-8"))
    crit = list(proto_json.get("crit") or [])
    if "sum-future-feature" not in crit:
        crit.append("sum-future-feature")
    proto_json["crit"] = crit
    proto_json["sum-future-feature"] = "v2-only-claim"
    proto_mutated = b64url(
        json.dumps(proto_json, separators=(",", ":")).encode("utf-8")
    )
    f["receipt"]["jws"] = f"{proto_mutated}.{middle}.{sig}"
    write_fixture("crit_unknown_extension", f)

    # ---- unsupported_alg ----
    f = base_fixture(receipt, public_jwks)
    f["name"] = "unsupported_alg"
    f["description"] = (
        "Protected header `alg` claim mutated to 'HS256'. The in-tree "
        "algorithm registry only lists 'EdDSA'; verifier rejects with "
        "unsupported_alg before signature verification — the JWT "
        "history's classic 'alg downgrade' / 'alg confusion' pattern."
    )
    f["expected_outcome"] = "reject"
    f["expected_error_class"] = "unsupported_alg"
    proto, middle, sig = receipt["jws"].split(".")
    proto_json = json.loads(b64url_decode(proto).decode("utf-8"))
    proto_json["alg"] = "HS256"
    proto_mutated = b64url(
        json.dumps(proto_json, separators=(",", ":")).encode("utf-8")
    )
    f["receipt"]["jws"] = f"{proto_mutated}.{middle}.{sig}"
    write_fixture("unsupported_alg", f)

    # ---- T4 — source_chain_hash bound + tampered ----
    # T4 is the transform-receipt-specific signed field. Re-sign a
    # source receipt that includes source_chain_hash, then derive a
    # positive-with-chain control + a tampered-chain reject.
    from sum_engine_internal.transform_receipt import (
        build_payload,
        canonical_hash,
        compute_source_chain_hash,
        sign_transform_receipt,
    )
    from sum_engine_internal.infrastructure.jcs import canonicalize

    parameters = {
        "density": 1.0,
        "length": 0.5,
        "formality": 0.5,
        "audience": 0.5,
        "perspective": 0.5,
    }
    input_doc = {"triples": [["alice", "likes", "cats"]]}
    output_tome = "alice likes cats."
    evidence_chain = [
        {
            "claim": "alice||likes||cats",
            "provenance": {
                "source_uri": "fixture://source/alice.txt",
                "byte_start": 0,
                "byte_end": 17,
            },
        }
    ]
    source_chain_hash = compute_source_chain_hash(evidence_chain)
    payload_with_chain = build_payload(
        transform="slider",
        parameters_hash=canonical_hash(canonicalize(parameters)),
        input_hash=canonical_hash(canonicalize(input_doc)),
        output_hash=canonical_hash(output_tome.encode("utf-8")),
        model="canonical-deterministic-v0",
        provider="canonical-path",
        digital_source_type="algorithmicMedia",
        signed_at=FIXED_SIGNED_AT,
        source_chain_hash=source_chain_hash,
    )
    receipt_with_chain = sign_transform_receipt(
        payload_with_chain, private_jwk=private_jwk, kid=TEST_KID
    )

    pos_chain = base_fixture(receipt_with_chain, public_jwks)
    pos_chain["name"] = "positive_control_with_source_chain"
    pos_chain["description"] = (
        "T4 positive control. Source-chain-bound receipt: payload "
        "includes source_chain_hash, computed by "
        "compute_source_chain_hash over a single-link chain. "
        "Verifies cleanly across runtimes — locks that "
        "source_chain_hash is a covered field of the signature."
    )
    write_fixture("positive_control_with_source_chain", pos_chain)

    f = base_fixture(receipt_with_chain, public_jwks)
    f["name"] = "tampered_source_chain_hash"
    f["description"] = (
        "T4 negative case. source_chain_hash mutated post-signing; "
        "JWS signature was computed over the original chain hash. "
        "Verifier MUST reject with signature_invalid — the field is "
        "covered by the signature, not an unsigned annotation."
    )
    f["expected_outcome"] = "reject"
    f["expected_error_class"] = "signature_invalid"
    f["receipt"]["payload"]["source_chain_hash"] = (
        "sha256-" + "0" * 64
    )
    write_fixture("tampered_source_chain_hash", f)

    print()
    print(f"fixtures regenerated. Test kid: {TEST_KID}")
    print(f"Signed at: {FIXED_SIGNED_AT}")
    print(
        "Fixtures derive from a deterministic test key (RFC 8032 §7.1 "
        "test vector seed); rerunning yields byte-identical output."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
