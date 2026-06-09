"""Generate a REAL ``sum.meaning_risk_receipt.v1`` over a real public-domain
corpus — the arXiv Paper-1 binding-gate artifact (meaning-preserving
COMPRESSION).

Corpus: the first ``N`` examples (dataset order) of the **BillSum** test
split (``FiscalNote/billsum``) — US Congressional bills + reference
summaries. BillSum is **CC0-1.0** (US-government works are public domain),
so the pairs are committed here in full for offline auditability.

Transform = abstractive summarization (the bill → its reference summary).
Judge = the local, offline ``EmbeddingJudge`` (mean-pooled
``all-MiniLM-L6-v2`` cosine), the paraphrase-aware no-$ scorer. Bound =
the shipped default (``method="auto"`` → Hoeffding for these fractional
losses); at n=64 / var≈0.0037 empirical-Bernstein does not yet beat it
(eB's win regime is larger n / lower variance — PR #288), so we commit the
default rather than cherry-pick a method.

HONEST PROOF BOUNDARY (the load-bearing point):
  * The **certificate is replayable offline**: the receipt's ``losses_hash``
    anchors the committed integer-micro loss vector (``losses_billsum.json``);
    a verifier re-runs the *pure-Python* certifier over those losses and
    reproduces the bound byte-for-byte — no model, no GPU, deterministic
    everywhere (this is what the CI golden test checks, Stage A + Stage B).
  * The **loss computation is machine-pinned** (F23/F26): re-deriving the
    losses from raw bill text needs the MiniLM forward pass, whose float
    output can drift across hardware/torch versions. So the receipt is
    *conditional on the named judge*, and the judge step is reproduced only
    on a matching stack. The receipt discloses this; it does not hide it.

Determinism: fixed Ed25519 seed (RFC 8032 zero-seed throwaway demo key,
private key NEVER written) + fixed ``signed_at`` → byte-stable signature on
a matching judge stack. Re-fetching is avoided when the corpus file is
already committed.

Run:  python fixtures/meaning_receipts_billsum/generate_billsum_fixture.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sum_engine_internal.research.meaning import (
    build_payload,
    certify_meaning_risk,
    score_pairs,
    sign_meaning_risk_receipt,
)
from sum_engine_internal.research.meaning.local_judge import (
    embedding_entailment_scorer,
)

HERE = Path(__file__).parent
N = 64
SEED = b"\x00" * 32
KID = "billsum-fixture-key-2026"
SIGNED_AT = "2026-06-08T12:00:00.000Z"
CORPUS_ID = "billsum-test-first64-cc0"
TRANSFORM = "summarize:billsum-reference"
LOSS_DEFINITION = (
    "1 - bidirectional sentence-entailment preservation (recall 0.6 / "
    "fidelity 0.4) under the named MiniLM-cosine judge; 0 = full "
    "preservation, 1 = none"
)
DISCLOSURE = (
    "Bounds the EXPECTED value of a NAMED meaning-loss proxy "
    "(bidirectional-entailment over a local all-MiniLM-L6-v2 cosine judge), "
    "MARGINALLY over the first 64 BillSum test bills (CC0-1.0), under "
    "exchangeability. NOT a per-document claim and NOT meaning itself. The "
    "CERTIFICATE replays offline over the committed integer-micro loss "
    "vector; the LOSS COMPUTATION is machine-pinned (model-judge float drift, "
    "F23/F26) and reproduced only on a matching torch/MiniLM stack."
)
CORPUS_FILE = HERE / "corpus_billsum_test_first64.json"
LOSSES_FILE = HERE / "losses_billsum.json"
RECEIPT_FILE = HERE / "meaning_risk_receipt.billsum.golden.json"
JWKS_FILE = HERE / "jwks.json"


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _keypair() -> tuple[dict, dict]:
    sk = Ed25519PrivateKey.from_private_bytes(SEED)
    pk_raw = sk.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    x = _b64u(pk_raw)
    private = {"kty": "OKP", "crv": "Ed25519", "d": _b64u(SEED), "x": x,
               "kid": KID, "alg": "EdDSA", "use": "sig"}
    public = {"kty": "OKP", "crv": "Ed25519", "x": x,
              "kid": KID, "alg": "EdDSA", "use": "sig"}
    return private, public


def _load_or_fetch_corpus() -> dict:
    if CORPUS_FILE.exists():
        return json.loads(CORPUS_FILE.read_text("utf-8"))
    from datasets import load_dataset  # lazy: only needed on first fetch
    ds = load_dataset("FiscalNote/billsum", split="test", streaming=True)
    pairs = []
    for i, ex in enumerate(ds):
        if i >= N:
            break
        pairs.append({"id": f"billsum-test-{i}",
                      "source": ex["text"], "rendering": ex["summary"]})
    corpus = {
        "corpus_id": CORPUS_ID,
        "source_dataset": "FiscalNote/billsum",
        "license": "CC0-1.0 (US-government works, public domain)",
        "selection": "first 64 examples of the test split in dataset order",
        "transform": TRANSFORM,
        "pairs": pairs,
    }
    CORPUS_FILE.write_text(json.dumps(corpus, indent=2) + "\n", encoding="utf-8")
    return corpus


def build() -> tuple[dict, dict, list[float]]:
    corpus = _load_or_fetch_corpus()
    pairs = [(p["source"], p["rendering"]) for p in corpus["pairs"]]
    # Scorer identity is fixed and named even when we replay committed losses,
    # so the certificate's provenance is unchanged.
    scorer = embedding_entailment_scorer()
    if LOSSES_FILE.exists():
        # Re-sign without re-running the (machine-pinned, slow) judge: the
        # committed integer-micro loss vector is the evidence the receipt
        # anchors. Regeneration is then judge-free + deterministic everywhere.
        losses = json.loads(LOSSES_FILE.read_text("utf-8"))["losses"]
    else:
        losses = score_pairs(pairs, scorer)
    # Method = the shipped default ("auto" → Hoeffding for these fractional
    # [0,1] losses). The regime check below is printed (not snooped): at
    # n=64 / var≈0.0037 the empirical-Bernstein additive term still slightly
    # dominates, so eB does NOT beat Hoeffding here (eB's win regime is larger
    # n / lower variance — see PR #288). Using the default avoids any
    # method cherry-picking; both bounds are individually valid at 1-δ.
    import numpy as np
    _eb = certify_meaning_risk(losses, scorer_name=scorer.name,
                               scorer_version=scorer.version, delta=0.05,
                               method="empirical_bernstein")
    guarantee = certify_meaning_risk(
        losses, scorer_name=scorer.name, scorer_version=scorer.version,
        delta=0.05, method="auto",
    )
    print(f"  regime check (n={guarantee.n}, var={float(np.var(losses, ddof=1)):.5f}): "
          f"auto/{guarantee.method} risk_ub={guarantee.risk_upper_bound:.4f}  "
          f"eB risk_ub={_eb.risk_upper_bound:.4f}  "
          f"-> committing auto ({'tighter' if guarantee.risk_upper_bound <= _eb.risk_upper_bound else 'looser'})")
    payload = build_payload(
        guarantee=guarantee, losses=losses, corpus_id=corpus["corpus_id"],
        transform=corpus["transform"], alpha_target=0.7,
        loss_definition=LOSS_DEFINITION, disclosure=DISCLOSURE,
        signed_at=SIGNED_AT,
    )
    private, public = _keypair()
    receipt = sign_meaning_risk_receipt(payload, private_jwk=private, kid=KID)
    return receipt, {"keys": [public]}, losses


def main() -> None:
    receipt, jwks, losses = build()
    RECEIPT_FILE.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    JWKS_FILE.write_text(json.dumps(jwks, indent=2) + "\n", encoding="utf-8")
    LOSSES_FILE.write_text(json.dumps({
        "judge": "bidirectional-entailment[minilm-cosine-0.5]",
        "judge_version": "1",
        "note": ("machine-pinned: computed via transformers all-MiniLM-L6-v2 "
                 "on CPU; cross-hardware float drift may alter the last "
                 "micro-unit (F23/F26). The receipt replays over these "
                 "committed losses; only re-deriving them from raw text needs "
                 "the judge."),
        "losses": [round(x, 6) for x in losses],
    }, indent=2) + "\n", encoding="utf-8")
    pl = receipt["payload"]
    print(f"wrote BillSum golden: n={pl['n']} "
          f"point_estimate={pl['point_estimate_micro']/1e6:.4f} "
          f"risk_upper_bound={pl['risk_upper_bound_micro']/1e6:.4f} "
          f"controlled={pl.get('controlled')} (alpha=0.7)")


if __name__ == "__main__":
    main()
