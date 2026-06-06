"""Regenerate the golden ``sum.meaning_risk_receipt.v1`` fixture.

Mirrors ``fixtures/transform_receipts/generate_fixtures.py``: a fully
DETERMINISTIC generator (fixed Ed25519 seed → fixed key → fixed
signature via RFC 8032; fixed ``signed_at``) so re-running it writes
byte-identical output. The private key is derived from the seed in this
script and is **never written to the fixtures directory** — only the
signed receipt and the public JWKS are committed.

What it produces:
  - ``meaning_risk_receipt.golden.json`` — the signed envelope.
  - ``jwks.json``                        — the public key to verify it.

The calibration evidence (``corpus_2026-06-06.json``) is committed
separately; a verifier recomputes the per-pair losses from it with the
named scorer and re-runs the certifier to replay the bound. See
``Tests/research/test_meaning_golden_fixture.py``.

Run:  python fixtures/meaning_receipts/generate_fixtures.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sum_engine_internal.research.meaning import (
    LexicalCoverageScorer,
    build_payload,
    certify_meaning_risk,
    score_pairs,
    sign_meaning_risk_receipt,
)

HERE = Path(__file__).parent
# RFC 8032 §7.1 test-vector-1 seed (32 zero bytes) — the same deterministic
# convention the transform-receipt fixtures use. Throwaway demo key; its
# only job is a byte-stable golden signature.
SEED = b"\x00" * 32
KID = "meaning-fixture-key-2026"
SIGNED_AT = "2026-06-06T12:00:00.000Z"
LOSS_DEFINITION = (
    "bidirectional content-unit overlap gap in [0,1]; "
    "0 = full preservation, 1 = no preservation"
)


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _keypair() -> tuple[dict, dict]:
    sk = Ed25519PrivateKey.from_private_bytes(SEED)
    pk_raw = sk.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    x = _b64u(pk_raw)
    private = {
        "kty": "OKP", "crv": "Ed25519", "d": _b64u(SEED), "x": x,
        "kid": KID, "alg": "EdDSA", "use": "sig",
    }
    public = {
        "kty": "OKP", "crv": "Ed25519", "x": x,
        "kid": KID, "alg": "EdDSA", "use": "sig",
    }
    return private, public


def build() -> tuple[dict, dict]:
    """Build the signed golden receipt + public JWKS from the committed
    corpus. Deterministic: same corpus + seed → byte-identical output."""
    corpus = json.loads((HERE / "corpus_2026-06-06.json").read_text("utf-8"))
    pairs = [(p["source"], p["rendering"]) for p in corpus["pairs"]]

    scorer = LexicalCoverageScorer()
    losses = score_pairs(pairs, scorer)
    guarantee = certify_meaning_risk(
        losses,
        scorer_name=scorer.name,
        scorer_version=scorer.version,
        delta=0.05,
        method="hoeffding",
    )
    payload = build_payload(
        guarantee=guarantee,
        losses=losses,
        corpus_id=corpus["corpus_id"],
        transform=corpus["transform"],
        alpha_target=0.5,
        loss_definition=LOSS_DEFINITION,
        signed_at=SIGNED_AT,
    )
    private, public = _keypair()
    receipt = sign_meaning_risk_receipt(payload, private_jwk=private, kid=KID)
    return receipt, {"keys": [public]}


def main() -> None:
    receipt, jwks = build()
    (HERE / "meaning_risk_receipt.golden.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    (HERE / "jwks.json").write_text(
        json.dumps(jwks, indent=2) + "\n", encoding="utf-8"
    )
    pl = receipt["payload"]
    print(
        f"wrote golden receipt (n={pl['n']}, "
        f"point_estimate_micro={pl['point_estimate_micro']}, "
        f"risk_upper_bound_micro={pl['risk_upper_bound_micro']}, "
        f"controlled={pl.get('controlled')}) + jwks.json"
    )


if __name__ == "__main__":
    main()
