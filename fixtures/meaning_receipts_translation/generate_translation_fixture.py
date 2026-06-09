"""Generate a REAL cross-lingual ``sum.meaning_risk_receipt.v1`` — the
paraphrase-robustness half of the binding-gate pair (the "Both" the operator
chose). Demonstrates the moat directly: faithful EN→FR translations preserve
meaning (~0 loss) despite **zero lexical overlap** — the dial scores by
meaning, not surface form, robust to the most extreme rewriting (a different
language).

Corpus: the first ``N`` substantive, length-aligned (EN, FR) pairs of the
**opus-100** test split (``Helsinki-NLP/opus-100``, en-fr), a standard MT
benchmark (Zhang et al. 2020). Transform = translation (EN → FR).
Judge = the local, offline multilingual NLI judge
(``MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`` — the F24
model, ``model_id``-parameterised, no new code).

LICENSE-CLEAN HANDLING: opus-100 aggregates mixed-licence OPUS web sources,
so — unlike the CC0 BillSum fixture — this generator does **not redistribute
the raw text**. It commits the per-pair **loss vector** plus a **hash-pinned
corpus pointer** (dataset + deterministic selection + sha256 of the pairs).
The CERTIFICATE replays offline over the committed losses; re-deriving the
losses re-fetches opus-100 under its own terms (and the hash verifies the
same pairs were used). Same machine-pinned-judge boundary as BillSum
(F23/F26), disclosed.

Run:  python fixtures/meaning_receipts_translation/generate_translation_fixture.py
"""
from __future__ import annotations

import base64
import hashlib
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
from sum_engine_internal.research.meaning.local_judge import nli_entailment_scorer

HERE = Path(__file__).parent
N = 64
MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
SEED = b"\x00" * 32
KID = "translation-fixture-key-2026"
SIGNED_AT = "2026-06-08T12:00:00.000Z"
CORPUS_ID = "opus100-en-fr-test-first64-filtered"
TRANSFORM = "translate:en->fr"
LOSS_DEFINITION = (
    "1 - bidirectional sentence-entailment preservation (recall 0.6 / "
    "fidelity 0.4) under the named multilingual-NLI judge; 0 = full "
    "cross-lingual preservation, 1 = none"
)
DISCLOSURE = (
    "Bounds the EXPECTED value of a NAMED cross-lingual meaning-loss proxy "
    "(bidirectional-entailment over a local mDeBERTa-v3 multilingual NLI "
    "judge), MARGINALLY over the first 64 length-aligned opus-100 en-fr test "
    "pairs (a min-word + length-ratio FILTERED subset, not all en-fr), under "
    "exchangeability. Demonstrates paraphrase/rewriting robustness: faithful "
    "translations score ~0 despite NEAR-zero lexical overlap (EN and FR share "
    "only names, numbers, and cognates). The judge emits a small discrete set "
    "of values. NOT a per-document claim and NOT meaning itself. The "
    "CERTIFICATE replays "
    "offline over the committed integer-micro loss vector; re-deriving the "
    "losses re-fetches opus-100 and runs the judge (machine-pinned, F23/F26). "
    "The raw text is NOT redistributed (mixed-licence corpus); the committed "
    "corpus pointer is sha256-pinned."
)
# Deterministic, documented selection filter (NOT result-tuned): the first N
# pairs in dataset order that are substantive on both sides and length-aligned
# (similar word counts ⇒ sentence-aligned, screening out the merge/split
# misalignments that loose corpora carry). Fixed before any bound was seen.
MIN_EN_WORDS, MIN_FR_WORDS = 6, 4
LEN_RATIO_LO, LEN_RATIO_HI = 0.5, 2.0

CORPUS_POINTER_FILE = HERE / "corpus_pointer.json"
LOSSES_FILE = HERE / "losses_translation.json"
RECEIPT_FILE = HERE / "meaning_risk_receipt.translation.golden.json"
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


def _fetch_pairs() -> list[tuple[str, str]]:
    from datasets import load_dataset
    # Pinned dataset revision → the from-scratch selection is reproducible
    # against the exact bytes the corpus_sha256 pointer was computed over.
    ds = load_dataset("Helsinki-NLP/opus-100", "en-fr", split="test",
                      streaming=True,
                      revision="805090dc28bf78897da9641cdf08b61287580df9")
    pairs: list[tuple[str, str]] = []
    for r in ds:
        t = r["translation"]
        en = (t.get("en") or "").strip()
        fr = (t.get("fr") or "").strip()
        we, wf = len(en.split()), len(fr.split())
        if (we >= MIN_EN_WORDS and wf >= MIN_FR_WORDS and en != fr
                and LEN_RATIO_LO <= wf / max(we, 1) <= LEN_RATIO_HI):
            pairs.append((en, fr))
        if len(pairs) >= N:
            break
    return pairs


def _corpus_hash(pairs: list[tuple[str, str]]) -> str:
    blob = json.dumps([[en, fr] for en, fr in pairs],
                      ensure_ascii=False, separators=(",", ":"))
    return "sha256-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build() -> tuple[dict, dict, list[float]]:
    scorer = nli_entailment_scorer(model_id=MODEL)
    if LOSSES_FILE.exists():
        # Judge-free, fetch-free re-sign: the committed loss vector is the
        # evidence the receipt anchors. Regeneration is deterministic
        # everywhere (this is the CI path).
        losses = json.loads(LOSSES_FILE.read_text("utf-8"))["losses"]
    else:
        pairs = _fetch_pairs()
        corpus_hash = _corpus_hash(pairs)
        losses = score_pairs(pairs, scorer)
        CORPUS_POINTER_FILE.write_text(json.dumps({
            "corpus_id": CORPUS_ID,
            "source_dataset": "Helsinki-NLP/opus-100",
            "config": "en-fr", "split": "test",
            "dataset_revision": "805090dc28bf78897da9641cdf08b61287580df9",
            "license": "mixed-OPUS (benchmark; raw text NOT redistributed here)",
            "selection": (f"first {N} test pairs in dataset order with "
                          f"en>={MIN_EN_WORDS} words, fr>={MIN_FR_WORDS} words, "
                          f"en!=fr, word-count ratio in "
                          f"[{LEN_RATIO_LO},{LEN_RATIO_HI}]"),
            "n": len(pairs),
            "corpus_sha256": corpus_hash,
        }, indent=2) + "\n", encoding="utf-8")

    guarantee = certify_meaning_risk(
        losses, scorer_name=scorer.name, scorer_version=scorer.version,
        delta=0.05, method="auto",
    )
    payload = build_payload(
        guarantee=guarantee, losses=losses, corpus_id=CORPUS_ID,
        transform=TRANSFORM, alpha_target=0.5,
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
    if not LOSSES_FILE.exists():
        LOSSES_FILE.write_text(json.dumps({
            "judge": f"bidirectional-entailment[nli:{MODEL}]",
            "judge_version": "1",
            "note": ("machine-pinned: multilingual NLI judge on CPU; "
                     "cross-hardware float drift may alter the last micro-unit "
                     "(F23/F26). The receipt replays over these committed "
                     "losses; re-deriving them re-fetches opus-100 + the judge."),
            "losses": [round(x, 6) for x in losses],
        }, indent=2) + "\n", encoding="utf-8")
    pl = receipt["payload"]
    n_zero = sum(1 for x in losses if x == 0.0)
    print(f"wrote translation golden: n={pl['n']} "
          f"point_estimate={pl['point_estimate_micro']/1e6:.4f} "
          f"risk_upper_bound={pl['risk_upper_bound_micro']/1e6:.4f} "
          f"controlled={pl.get('controlled')} (alpha=0.5) "
          f"| {n_zero}/{len(losses)} pairs at exactly 0 loss")


if __name__ == "__main__":
    main()
