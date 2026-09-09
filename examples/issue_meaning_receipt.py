#!/usr/bin/env python3
"""Issue a signed `sum.meaning_risk_receipt.v1` over YOUR OWN text — the
issuer side of the trust loop, as a runnable recipe.

The `sum mint-meaning` CLI is the normal issuer entry point. This script is a
small library orchestration example for developers integrating the same primitives.

INPUT — a JSON file of (source, transform) pairs, your own data:

    [
      ["The full original sentence.", "The transformed/rewritten version."],
      ["Another source ...",          "Its transform ..."],
      ...
    ]

Each pair is one calibration example: `source` is the original, `transform`
is what an AI (or anyone) produced from it. This example records descriptive
batch measurements and conditional bound arithmetic under a named proxy. A
population confidence interpretation needs independent calibration draws from
the target population, a fixed policy and no unaccounted adaptive selection.
Exchangeability alone is insufficient; source meaning and factual truth are not guaranteed.

USAGE

    # instant, no extra deps — fair only for EXTRACTIVE compression:
    python examples/issue_meaning_receipt.py pairs.json --out out/ --scorer lexical \
        --corpus-id my-corpus-v0 --transform "summarize:my-pipeline"

    # honest, paraphrase-aware (needs `pip install "sum-engine[judge]"`):
    python examples/issue_meaning_receipt.py pairs.json --out out/ --scorer nli ...

OUTPUT (in --out): `receipt.json` (the signed envelope), `jwks.json` (your
PUBLIC key — share this), `losses.json` (the per-pair losses, side-band
replay evidence), and `private_jwk.json` (your SECRET key — never share;
needed only to re-sign).

VERIFY what you issued (the consumer side, shipped surface):

    python -m sum_verify out/receipt.json --jwks out/jwks.json --losses out/losses.json
    # → {"verified": true, "replayed": true, ...}

Needs the `[research]` extra (the issuance primitives) plus `[judge]` for the
nli/embedding scorers. Apache-2.0.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _fresh_or_load_keypair(kid: str, private_jwk_path: Path | None):
    """Load a private JWK if given, else generate a fresh Ed25519 keypair.
    Returns (private_jwk, jwks)."""
    if private_jwk_path and private_jwk_path.exists():
        private = json.loads(private_jwk_path.read_text())
        kid = private.get("kid", kid)
    else:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )

        sk = Ed25519PrivateKey.generate()
        seed = sk.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        x = _b64u(
            sk.public_key().public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw
            )
        )
        private = {
            "kty": "OKP", "crv": "Ed25519", "d": _b64u(seed), "x": x,
            "kid": kid, "alg": "EdDSA", "use": "sig",
        }
    public = {k: v for k, v in private.items() if k != "d"}
    return private, {"keys": [public]}


def _load_scorer(name: str):
    from sum_engine_internal.research.meaning.meaning_loss import (
        LexicalCoverageScorer,
    )
    if name == "lexical":
        return LexicalCoverageScorer()
    from sum_engine_internal.research.meaning.local_judge import (
        embedding_entailment_scorer,
        nli_entailment_scorer,
    )
    return embedding_entailment_scorer() if name == "embedding" else nli_entailment_scorer()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Issue a signed meaning-risk receipt over your own (source, transform) pairs.")
    p.add_argument("pairs", help='JSON file: [["source","transform"], ...]')
    p.add_argument("--out", default="out", help="output directory (default: out/)")
    p.add_argument("--scorer", default="nli", choices=["lexical", "nli", "embedding"],
                   help="meaning-loss judge (default nli; lexical is instant but fair only for extractive compression)")
    p.add_argument("--corpus-id", required=True, help="names YOUR measured corpus; does not establish independent sampling, e.g. 'support-emails-v0'")
    p.add_argument("--transform", required=True, help="names what produced the pairs, e.g. 'summarize:gpt-4o' or 'translate:en-fr'")
    p.add_argument("--loss-definition", default="bidirectional-entailment meaning-loss in [0,1]; 0 = judge detects no loss",
                   help="one-line human description of what the [0,1] number means")
    p.add_argument("--delta", type=float, default=0.05, help="miscoverage (confidence = 1 - delta); default 0.05 → 95%%")
    p.add_argument("--alpha", type=float, default=0.5, help="risk level you want controlled; receipt records whether the bound met it")
    p.add_argument("--method", default="auto", choices=["auto", "hoeffding", "clopper_pearson", "empirical_bernstein"])
    p.add_argument("--kid", default="my-issuer-key-1", help="key id stamped in the receipt + JWKS")
    args = p.parse_args(argv)

    try:
        from sum_engine_internal.research.meaning.conformal_meaning import certify_meaning_risk
        from sum_engine_internal.research.meaning.meaning_loss import score_pairs
        from sum_engine_internal.research.meaning.receipt import (
            build_payload,
            sign_meaning_risk_receipt,
        )
    except ImportError as e:
        print(f"issue: needs the [research] extra (pip install 'sum-engine[research]'): {e}", file=sys.stderr)
        return 2

    try:
        pairs = json.loads(Path(args.pairs).read_text(encoding="utf-8"))
        pairs = [(str(a), str(b)) for a, b in pairs]
    except Exception as e:  # noqa: BLE001
        print(f"issue: could not read pairs file (expected [[source,transform],...]): {e}", file=sys.stderr)
        return 2
    if not pairs:
        print("issue: need at least one (source, transform) pair", file=sys.stderr)
        return 2

    try:
        scorer = _load_scorer(args.scorer)
    except ImportError as e:
        print(f"issue: --scorer {args.scorer} needs the [judge] extra (pip install 'sum-engine[judge]'): {e}", file=sys.stderr)
        return 2

    # 1. score YOUR pairs → per-pair meaning-loss in [0,1]
    losses = score_pairs(pairs, scorer)
    # 2. certify a distribution-free upper bound on the expected loss
    guarantee = certify_meaning_risk(
        losses, scorer_name=scorer.name, scorer_version=scorer.version,
        delta=args.delta, method=args.method,
    )
    # 3. assemble the wire payload (re-certifies over the committed quantised losses)
    from sum_engine_internal.research.meaning.evidence import build_evaluation_manifest
    evidence = build_evaluation_manifest(pairs, scorer, transform_configuration={"description": args.transform, "generation_provenance": "caller_supplied_outputs"})
    payload = build_payload(
        guarantee=guarantee, losses=losses, corpus_id=args.corpus_id,
        transform=args.transform, alpha_target=args.alpha,
        loss_definition=args.loss_definition, evaluation_manifest=evidence,
    )
    # 4. sign it with YOUR key → the receipt
    private_jwk, jwks = _fresh_or_load_keypair(args.kid, None)
    envelope = sign_meaning_risk_receipt(payload, private_jwk=private_jwk, kid=private_jwk["kid"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(out / "private_jwk.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        print("Output already contains a private key; choose a fresh --out directory.", file=sys.stderr)
        return 2
    with os.fdopen(fd, "w") as key_file:
        key_file.write(json.dumps(private_jwk, indent=2))
    (out / "receipt.json").write_text(json.dumps(envelope, indent=2))
    (out / "jwks.json").write_text(json.dumps(jwks, indent=2))
    (out / "losses.json").write_text(json.dumps(
        {"judge": scorer.name, "judge_version": scorer.version, "note": "per-pair meaning-loss; side-band replay evidence", "losses": losses},
        indent=2,
    ))

    ub = guarantee.risk_upper_bound
    print(f"Issued sum.meaning_risk_receipt.v1 over {len(losses)} pairs under {scorer.name}")
    print("  scope: descriptive batch; sampling assumptions are not established")
    print(f"  conditional bound arithmetic: expected proxy loss ≤ {ub:.4f} at {100*(1-args.delta):.0f}% (mean {guarantee.point_estimate:.4f}, n={guarantee.n})")
    print(f"  controlled at alpha={args.alpha}: {payload.get('controlled')}")
    if ub >= 0.95 or payload.get("controlled") is False:
        ctrl = "" if payload.get("controlled") is not False else ", and NOT controlled at your alpha"
        print()
        print(
            f"  ⚠️  WARNING: this bound is near-vacuous (≤ {ub:.4f}{ctrl}, n={guarantee.n}). "
            "Small samples can produce a vacuous bound near 1.0. More pairs alone do not "
            "establish independent sampling, distribution match or a fixed policy. "
            "This example defaults to descriptive batch scope.",
            file=sys.stderr,
        )
    print(f"  wrote: {out}/receipt.json  {out}/jwks.json  {out}/losses.json  {out}/private_jwk.json (SECRET)")
    print()
    print("Verify it (the consumer side):")
    print(f"  python -m sum_verify {out}/receipt.json --jwks {out}/jwks.json --losses {out}/losses.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
