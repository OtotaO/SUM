"""Regenerate the golden ``sum.perspective_risk_receipt.v1`` fixture.

Mirrors fixtures/meaning_receipts/generate_fixtures.py: a fully
DETERMINISTIC generator (fixed Ed25519 seed → fixed key → fixed signature
via RFC 8032; fixed signed_at) so re-running writes byte-identical output.
The private key is derived from the seed here and is NEVER written to the
fixtures directory — only the signed receipt + public JWKS are committed.

The corpus is a small set of (source, rendering, cohort) triples; the
receipt certifies the marginal meaning-loss bound PLUS a per-cohort bound
(the Perspective Receipt). The golden exists so the JS verifier
(single_file_demo/meaning_receipt_verifier.js) can prove the
Python-signed perspective receipt verifies in Node — the cross-runtime
claim for the new receipt family.

Run:  python fixtures/perspective_receipts/generate_fixtures.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sum_engine_internal.research.meaning import (
    LexicalCoverageScorer,
    build_perspective_payload,
    certify_meaning_risk_by_group,
    score_pairs,
    sign_perspective_risk_receipt,
)

HERE = Path(__file__).parent
SEED = b"\x00" * 32  # RFC 8032 §7.1 test-vector-1 seed (throwaway demo key)
KID = "perspective-fixture-key-2026"
SIGNED_AT = "2026-06-07T12:00:00.000Z"
LOSS_DEFINITION = (
    "bidirectional content-unit overlap gap in [0,1]; "
    "0 = full preservation, 1 = no preservation"
)


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _keypair() -> tuple[dict, dict]:
    sk = Ed25519PrivateKey.from_private_bytes(SEED)
    x = _b64u(sk.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw))
    private = {"kty": "OKP", "crv": "Ed25519", "d": _b64u(SEED), "x": x,
               "kid": KID, "alg": "EdDSA", "use": "sig"}
    public = {"kty": "OKP", "crv": "Ed25519", "x": x,
              "kid": KID, "alg": "EdDSA", "use": "sig"}
    return private, public


def build() -> tuple[dict, dict]:
    corpus = json.loads((HERE / "corpus_2026-06-07.json").read_text("utf-8"))
    triples = corpus["triples"]
    scorer = LexicalCoverageScorer()
    losses = score_pairs([(t["source"], t["rendering"]) for t in triples], scorer)
    cohorts = [t["cohort"] for t in triples]
    grouped = certify_meaning_risk_by_group(
        losses, cohorts, scorer_name=scorer.name,
        scorer_version=scorer.version, delta=0.05, method="hoeffding",
    )
    payload = build_perspective_payload(
        grouped=grouped, losses=losses, group_ids=cohorts,
        corpus_id=corpus["corpus_id"], transform=corpus["transform"],
        loss_definition=LOSS_DEFINITION, alpha_target=0.5, signed_at=SIGNED_AT,
    )
    private, public = _keypair()
    receipt = sign_perspective_risk_receipt(payload, private_jwk=private, kid=KID)
    return receipt, {"keys": [public]}


def main() -> None:
    receipt, jwks = build()
    (HERE / "perspective_risk_receipt.golden.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (HERE / "jwks.json").write_text(json.dumps(jwks, indent=2) + "\n", encoding="utf-8")
    pl = receipt["payload"]
    print(f"wrote perspective golden (n={pl['n']}, cohorts="
          f"{[g['group_id'] for g in pl['groups']]}, controls_all={pl.get('controls_all')})")


if __name__ == "__main__":
    main()
