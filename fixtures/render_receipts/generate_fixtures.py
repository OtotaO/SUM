#!/usr/bin/env python3
"""Generate the receipt-fixture set from a captured live render.

This fixture set is shared between v0.9.B (browser, JS) and v0.9.C
(Python) receipt verifiers. Each output JSON names the mutation,
the expected outcome, and (when the outcome is reject) the
expected error class. Authoring fixtures once and consuming from
both verifiers means the negative-path matrix is byte-identical
across runtimes — the K-style cross-runtime equivalence we already
have for CanonicalBundle, applied to render receipts.

Inputs (committed alongside this script):
  source_render.json    — full /api/render response captured live.
  jwks_at_capture.json  — /.well-known/jwks.json from the same
                          moment, so the captured receipt's kid is
                          present and the positive control verifies.

Outputs (also committed):
  positive_control.json
  tampered_tome_hash.json
  tampered_triples_hash.json
  tampered_sliders_quantized.json
  tampered_model.json
  tampered_provider.json
  tampered_signed_at.json
  tampered_digital_source_type.json
  tampered_render_id.json
  tampered_signature.json
  tampered_kid_header.json
  malformed_jws_middle_nonempty.json
  unknown_kid.json
  schema_unknown.json
  crit_unknown_extension.json

Each fixture file has shape:
  {
    "name": "<fixture-name>",
    "description": "<what was mutated, in plain English>",
    "expected_outcome": "verify" | "reject",
    "expected_error_class": "<err-class>" | null,
    "receipt": { schema, kid, payload, jws },
    "jwks": { keys: [...] }
  }

Error classes (verifier-runtime-neutral):
  "signature_invalid"     — JWS signature does not verify against the
                            JCS-canonical payload bytes. Catches every
                            tampered-signed-field case.
  "malformed_jws"         — detached JWS contract violated (middle
                            segment non-empty, etc.).
  "unknown_kid"           — receipt.kid not present in the JWKS.
  "kid_mismatch"          — protected-header kid != top-level
                            receipt.kid (header tampering).
  "schema_unknown"        — receipt.schema is not a value this
                            verifier accepts.
  "crit_unknown_extension" — protected header `crit` contains an
                             extension this verifier doesn't
                             understand; per RFC 7515 §4.1.11, fail
                             closed.

Run:
  python fixtures/render_receipts/generate_fixtures.py

Idempotent: same inputs produce byte-identical outputs.
"""
from __future__ import annotations

import base64
import copy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def write_fixture(name: str, fixture: dict) -> None:
    out = HERE / f"{name}.json"
    text = json.dumps(fixture, indent=2, sort_keys=True) + "\n"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out.name}")


def base_fixture(receipt: dict, jwks: dict) -> dict:
    """Common shape — child fixtures override fields as needed."""
    return {
        "name": "",
        "description": "",
        "expected_outcome": "verify",
        "expected_error_class": None,
        "receipt": copy.deepcopy(receipt),
        "jwks": copy.deepcopy(jwks),
    }


def main() -> int:
    source = json.loads((HERE / "source_render.json").read_text())
    jwks = json.loads((HERE / "jwks_at_capture.json").read_text())
    receipt = source["render_receipt"]

    # Sanity: the captured receipt's kid IS present in the captured
    # JWKS. Without this, the positive control would fail and the
    # whole fixture set would be moot.
    captured_kid = receipt["kid"]
    jwks_kids = [k["kid"] for k in jwks["keys"]]
    if captured_kid not in jwks_kids:
        raise SystemExit(
            f"captured receipt kid {captured_kid!r} not in captured jwks "
            f"{jwks_kids!r} — re-capture both at the same moment"
        )

    # ---- positive control ----
    pos = base_fixture(receipt, jwks)
    pos["name"] = "positive_control"
    pos["description"] = (
        "Receipt and JWKS captured live from the production worker at "
        "https://sum-demo.ototao.workers.dev. No mutations. A correct "
        "verifier MUST verify this fixture."
    )
    write_fixture("positive_control", pos)

    # ---- tampered signed-payload fields ----
    # Each of these mutates one signed field; the JWS signature
    # therefore fails because what was signed was the original
    # value. All resolve to the same error class (signature_invalid)
    # but each fixture is its own case so a verifier that handles
    # one path correctly but skips another fails specifically.
    signed_payload_fields = [
        ("tampered_tome_hash",
         "payload.tome_hash mutated; signed_over → invalid signature.",
         lambda r: r["payload"].__setitem__(
             "tome_hash",
             "sha256-deadbeef" + r["payload"]["tome_hash"][15:],
         )),
        ("tampered_triples_hash",
         "payload.triples_hash mutated; signed_over → invalid signature.",
         lambda r: r["payload"].__setitem__(
             "triples_hash",
             "sha256-cafebabe" + r["payload"]["triples_hash"][15:],
         )),
        ("tampered_sliders_quantized",
         "payload.sliders_quantized.formality mutated 0.7→0.6.",
         lambda r: r["payload"]["sliders_quantized"].__setitem__(
             "formality", 0.6
         )),
        ("tampered_model",
         "payload.model mutated to a different model string.",
         lambda r: r["payload"].__setitem__(
             "model", "imposter-model-9000"
         )),
        ("tampered_provider",
         "payload.provider mutated.",
         lambda r: r["payload"].__setitem__(
             "provider", "evil-corp"
         )),
        ("tampered_signed_at",
         "payload.signed_at shifted by an hour.",
         lambda r: r["payload"].__setitem__(
             "signed_at",
             # Shift the captured ISO timestamp by adding a suffix
             # that's still valid ISO-8601 but a different instant.
             r["payload"]["signed_at"].replace("Z", "+01:00"),
         )),
        ("tampered_digital_source_type",
         "payload.digital_source_type swapped trainedAlgorithmicMedia "
         "↔ algorithmicMedia.",
         lambda r: r["payload"].__setitem__(
             "digital_source_type",
             "algorithmicMedia"
             if r["payload"]["digital_source_type"] == "trainedAlgorithmicMedia"
             else "trainedAlgorithmicMedia",
         )),
        ("tampered_render_id",
         "payload.render_id mutated.",
         lambda r: r["payload"].__setitem__(
             "render_id", "0000000000000000"
         )),
    ]
    for name, desc, mutator in signed_payload_fields:
        f = base_fixture(receipt, jwks)
        f["name"] = name
        f["description"] = desc
        f["expected_outcome"] = "reject"
        f["expected_error_class"] = "signature_invalid"
        mutator(f["receipt"])
        write_fixture(name, f)

    # ---- tampered JWS signature segment ----
    # Flip the last base64url char of the signature segment. The
    # decoded signature bytes therefore differ; verification fails.
    # We don't decode + re-encode — just substitute one character so
    # the fixture is obviously a one-character mutation.
    f = base_fixture(receipt, jwks)
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
    # The protected header is base64url-encoded JSON; mutating kid
    # there means the header bytes the signature was computed over
    # no longer match → signature_invalid.
    f = base_fixture(receipt, jwks)
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
    proto_mutated = b64url_encode(
        json.dumps(proto_json, separators=(",", ":")).encode("utf-8")
    )
    f["receipt"]["jws"] = f"{proto_mutated}.{middle}.{sig}"
    write_fixture("tampered_kid_header", f)

    # ---- malformed detached JWS (middle segment non-empty) ----
    # Detached JWS per RFC 7515 §A.5 has an empty middle segment;
    # the canonical-payload bytes are the detached payload. A
    # non-empty middle segment is malformed.
    f = base_fixture(receipt, jwks)
    f["name"] = "malformed_jws_middle_nonempty"
    f["description"] = (
        "Middle JWS segment populated with arbitrary bytes; per RFC "
        "7515 §A.5 the middle segment of a detached JWS MUST be "
        "empty. A correct verifier rejects with malformed_jws "
        "before attempting cryptographic verification."
    )
    f["expected_outcome"] = "reject"
    f["expected_error_class"] = "malformed_jws"
    proto, _, sig = receipt["jws"].split(".")
    f["receipt"]["jws"] = f"{proto}.YWJjZGVm.{sig}"  # 'abcdef' base64url
    write_fixture("malformed_jws_middle_nonempty", f)

    # ---- unknown kid (JWKS doesn't contain the receipt's kid) ----
    # Ship a JWKS with no entries; the verifier rejects at the kid
    # lookup step before attempting cryptographic verification.
    f = base_fixture(receipt, jwks)
    f["name"] = "unknown_kid"
    f["description"] = (
        "Receipt is the unmodified positive control, but the JWKS "
        "supplied to the verifier is empty (no key for the receipt's "
        "kid). Verifier rejects with unknown_kid before any "
        "cryptographic operation."
    )
    f["expected_outcome"] = "reject"
    f["expected_error_class"] = "unknown_kid"
    f["jwks"] = {"keys": []}
    write_fixture("unknown_kid", f)

    # ---- forward-compat: schema_unknown ----
    # The receipt's wrapping schema is bumped to v99; a v1 verifier
    # MUST reject closed per RENDER_RECEIPT_FORMAT.md §1.4 ("MUST
    # bump the schema identifier for breaking changes").
    f = base_fixture(receipt, jwks)
    f["name"] = "schema_unknown"
    f["description"] = (
        "receipt.schema mutated to 'sum.render_receipt.v99' (a "
        "future-version this verifier doesn't know). Per "
        "RENDER_RECEIPT_FORMAT.md §1.4, a v1-aware verifier MUST "
        "fail closed on a v99 receipt."
    )
    f["expected_outcome"] = "reject"
    f["expected_error_class"] = "schema_unknown"
    f["receipt"]["schema"] = "sum.render_receipt.v99"
    write_fixture("schema_unknown", f)

    # ---- forward-compat: crit_unknown_extension ----
    # Add a new field to the JWS protected header AND list it in
    # `crit`. Per RFC 7515 §4.1.11, a verifier that doesn't
    # understand a critical extension MUST reject closed.
    f = base_fixture(receipt, jwks)
    f["name"] = "crit_unknown_extension"
    f["description"] = (
        "JWS protected header `crit` array extended with "
        "'sum-future-feature' and a corresponding header claim "
        "added. Per RFC 7515 §4.1.11, a verifier that doesn't "
        "understand a critical extension MUST reject closed. This "
        "is the future-compat lever we'll use when v2 lands."
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
    proto_mutated = b64url_encode(
        json.dumps(proto_json, separators=(",", ":")).encode("utf-8")
    )
    # Note: this fixture's signature is the original, which doesn't
    # match the mutated header — a correctly-implemented verifier
    # rejects on `crit` before attempting signature verification, so
    # the test asserts crit handling specifically. If a verifier
    # checks signatures first, it would reject as signature_invalid
    # (still a closed rejection, just classified differently). The
    # spec ordering puts crit-check first; this fixture pins that.
    f["receipt"]["jws"] = f"{proto_mutated}.{middle}.{sig}"
    write_fixture("crit_unknown_extension", f)

    print(f"\nfixtures regenerated. Captured kid: {captured_kid}")
    print(f"Captured at: {receipt['payload']['signed_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
