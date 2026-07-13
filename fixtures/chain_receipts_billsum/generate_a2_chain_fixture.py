"""Generate the world's first REAL certified multi-hop meaning chain
(``sum.chain_receipt.v1``) over a real public-domain corpus.

The chain composes the meaning-loss of TWO real transforms of the same
document, hop by hop, into a Bonferroni-additive budget with a joint
confidence, plus a DIRECT end-to-end leg. It is the "drift budget" that the
meaning-loss frontier converged on, now materialised as a signed, offline
replayable certificate.

CORPUS
    The first N=32 examples (dataset order) of the **BillSum** test split
    (``FiscalNote/billsum``) -- US Congressional bills + reference summaries,
    **CC0-1.0** (US-government works, public domain), reused from the committed
    binding-gate corpus ``../meaning_receipts_billsum/corpus_billsum_test_first64.json``.

THE TWO HOPS (honest labels -- read these before quoting any number)
    hop 1  document -> reference summary.
           transform = "summarize:billsum-reference". This is the DATASET's OWN
           reference summarization; SUM did NOT perform it. Identical framing to
           the committed binding-gate golden -- we certify the meaning-loss of a
           transform someone else performed.
    hop 2  reference summary -> lead-N extractive compression.
           transform = "compress:lead-extractive-keep0.5". A real, DETERMINISTIC,
           OFFLINE transform: keep the first ceil(0.5 * n_sentences) sentences of
           the summary (lead-N, a standard extractive-summarization baseline).
           llm_calls_made = 0; no model, no key, no network. Single-sentence
           summaries pass through unchanged (identity, loss ~ 0) -- that is
           honest behaviour, not a bug.

JUDGE
    The local, offline NLI judge ``nli_entailment_scorer()``
    (bidirectional entailment over DeBERTa-v3-mnli-fever-anli, pinned revision),
    the load-bearing scorer. It is STRICT and recall-weighted: a short
    abstractive summary of a ~6.7k-char bill does not entail the whole bill, so
    hop 1's loss is high and its Hoeffding upper bound clamps at 1.0. THAT IS THE
    SYSTEM WORKING: the certificate reports, honestly, that abstractive
    summarization blows the budget while the deterministic extractive hop is
    gentler. We do NOT swap to a lenient judge to get prettier numbers; that
    would be the exact cherry-pick this project refuses.

BOUNDS
    method="hoeffding" everywhere. At n=32 the bounds are WIDE; that is fine and
    said plainly. The budget is the SUM of per-hop expected-loss bounds
    (Bonferroni union bound), so it can exceed 1.0 -- it bounds a sum, not a
    metric distance (see ``budget_scope`` in the chain receipt; no triangle
    inequality is claimed).

HONEST PROOF BOUNDARY (identical discipline to the binding-gate golden)
    * The CERTIFICATE (each hop + the chain) replays offline over the committed
      integer-micro loss vectors: a verifier re-runs the pure-Python certifier
      and reproduces every bound byte-for-byte -- no model, no GPU.
    * The LOSS COMPUTATION is machine-pinned (F23/F26): re-deriving the losses
      from raw text needs the NLI forward pass, whose float output can drift
      across hardware/torch versions, and the long bills are truncated to the
      judge's 512-token window before scoring. The receipts disclose this; they
      do not hide it. Regeneration below replays the committed losses so it is
      judge-free and deterministic everywhere.

Determinism: fixed Ed25519 seed (RFC 8032 zero-seed throwaway demo key, private
key NEVER written) + pinned ``signed_at`` -> byte-stable receipts on any stack
(because regeneration replays the committed losses). The public JWKS is the only
key material committed.

Run:  python fixtures/chain_receipts_billsum/generate_a2_chain_fixture.py
"""
from __future__ import annotations

import base64
import json
import math
import re
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sum_engine_internal.research.meaning import (
    build_chain_payload,
    build_end_to_end_leg,
    build_payload,
    certify_meaning_risk,
    score_pairs,
    sign_chain_receipt,
    sign_meaning_risk_receipt,
)
from sum_engine_internal.research.meaning.local_judge import nli_entailment_scorer

HERE = Path(__file__).parent
CORPUS_FILE = (
    HERE.parent / "meaning_receipts_billsum" / "corpus_billsum_test_first64.json"
)

N = 32
LEAD_KEEP = 0.5
SEED = b"\x00" * 32
KID = "billsum-chain-fixture-key-2026"
SIGNED_AT = "2026-07-12T12:00:00.000Z"
CORPUS_ID = "billsum-test-first32-cc0"
DELTA = 0.05
METHOD = "hoeffding"

HOP1_TRANSFORM = "summarize:billsum-reference"
HOP2_TRANSFORM = "compress:lead-extractive-keep0.5"

_NLI_BASE = (
    "1 - bidirectional sentence-entailment preservation (recall 0.6 / "
    "fidelity 0.4) under the named NLI judge (DeBERTa-v3-mnli-fever-anli, "
    "pinned revision) at a 0.5 entailment cut; 0 = full preservation, 1 = none"
)
HOP1_LOSS_DEF = (
    _NLI_BASE + ". Hop 1 = the DATASET's own reference summarization (bill -> "
    "reference summary); SUM did not perform it. Long bills are truncated to the "
    "judge's ~512-token window before scoring (F23/F26 machine-pinning), so a "
    "short abstractive summary of a full bill scores high loss under this strict "
    "recall-weighted judge -- the bound is expected to clamp near 1.0."
)
HOP2_LOSS_DEF = (
    _NLI_BASE + ". Hop 2 = deterministic lead-N extractive compression (keep the "
    "first ceil(0.5*n) sentences of the summary; lead-N is a standard extractive "
    "baseline), offline, llm_calls_made=0. It is NOT a SUM slider render. "
    "Single-sentence summaries pass through unchanged."
)
E2E_LOSS_DEF = (
    _NLI_BASE + ". Direct end-to-end leg: bill -> lead-N extractive of its "
    "reference summary (hop 1 then hop 2 composed), scored directly. Long bills "
    "are truncated to the judge's ~512-token window (F23/F26)."
)

_DISCLOSURE_TAIL = (
    " Bounds the EXPECTED value of a NAMED meaning-loss proxy MARGINALLY over the "
    "first 32 BillSum test bills (CC0-1.0), under exchangeability. NOT a "
    "per-document claim and NOT meaning itself. The CERTIFICATE replays offline "
    "over the committed integer-micro loss vector; the LOSS COMPUTATION is "
    "machine-pinned (NLI float drift + 512-token truncation, F23/F26) and "
    "reproduced only on a matching torch/DeBERTa stack."
)
HOP1_DISCLOSURE = "Meaning-loss of the dataset's reference summarization." + _DISCLOSURE_TAIL
HOP2_DISCLOSURE = "Meaning-loss of deterministic lead-N extractive compression." + _DISCLOSURE_TAIL

LOSSES_HOP1_FILE = HERE / "losses_hop1.json"
LOSSES_HOP2_FILE = HERE / "losses_hop2.json"
LOSSES_E2E_FILE = HERE / "losses_e2e.json"
FINALS_FILE = HERE / "finals_lead_extractive.json"
HOP1_FILE = HERE / "hop1_summarize.golden.json"
HOP2_FILE = HERE / "hop2_extractive.golden.json"
CHAIN_FILE = HERE / "chain_receipt.billsum.golden.json"
JWKS_FILE = HERE / "jwks.json"

# --- deterministic lead-N extractive (offline, no model) ---
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def sentences(text: str) -> "list[str]":
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


def lead_extractive(text: str, keep: float = LEAD_KEEP) -> str:
    ss = sentences(text)
    k = max(1, math.ceil(keep * len(ss)))
    return " ".join(ss[:k])


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _keypair() -> "tuple[dict, dict]":
    sk = Ed25519PrivateKey.from_private_bytes(SEED)
    x = _b64u(
        sk.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
    )
    private = {"kty": "OKP", "crv": "Ed25519", "d": _b64u(SEED), "x": x,
               "kid": KID, "alg": "EdDSA", "use": "sig"}
    public = {"kty": "OKP", "crv": "Ed25519", "x": x,
              "kid": KID, "alg": "EdDSA", "use": "sig"}
    return private, public


def _load_or_score(losses_file: Path, pairs, scorer) -> "list[float]":
    """Replay committed losses when present (judge-free, deterministic
    everywhere); otherwise run the machine-pinned judge once."""
    if losses_file.exists():
        return json.loads(losses_file.read_text("utf-8"))["losses"]
    return score_pairs(pairs, scorer)


def _write_losses(losses_file: Path, losses, note: str) -> None:
    losses_file.write_text(
        json.dumps(
            {
                "judge": "bidirectional-entailment[nli:DeBERTa-v3-mnli-fever-anli]",
                "judge_version": "1",
                "note": note,
                "losses": [round(x, 6) for x in losses],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _mint_hop(losses, *, transform, loss_def, disclosure, scorer, private):
    guarantee = certify_meaning_risk(
        losses, scorer_name=scorer.name, scorer_version=scorer.version,
        delta=DELTA, method=METHOD,
    )
    payload = build_payload(
        guarantee=guarantee, losses=losses, corpus_id=CORPUS_ID,
        transform=transform, alpha_target=None, loss_definition=loss_def,
        disclosure=disclosure, signed_at=SIGNED_AT,
    )
    return sign_meaning_risk_receipt(payload, private_jwk=private, kid=KID), guarantee


def build():
    corpus = json.loads(CORPUS_FILE.read_text("utf-8"))
    pairs = corpus["pairs"][:N]
    sources = [p["source"] for p in pairs]
    summaries = [p["rendering"] for p in pairs]
    finals = [lead_extractive(s, LEAD_KEEP) for s in summaries]

    scorer = nli_entailment_scorer()
    hop1_pairs = list(zip(sources, summaries))
    hop2_pairs = list(zip(summaries, finals))
    e2e_pairs = list(zip(sources, finals))

    losses_hop1 = _load_or_score(LOSSES_HOP1_FILE, hop1_pairs, scorer)
    losses_hop2 = _load_or_score(LOSSES_HOP2_FILE, hop2_pairs, scorer)
    losses_e2e = _load_or_score(LOSSES_E2E_FILE, e2e_pairs, scorer)

    private, public = _keypair()
    hop1, g1 = _mint_hop(
        losses_hop1, transform=HOP1_TRANSFORM, loss_def=HOP1_LOSS_DEF,
        disclosure=HOP1_DISCLOSURE, scorer=scorer, private=private,
    )
    hop2, g2 = _mint_hop(
        losses_hop2, transform=HOP2_TRANSFORM, loss_def=HOP2_LOSS_DEF,
        disclosure=HOP2_DISCLOSURE, scorer=scorer, private=private,
    )
    e2e_leg = build_end_to_end_leg(
        losses_e2e, scorer_name=scorer.name, scorer_version=scorer.version,
        loss_definition=E2E_LOSS_DEF, delta=DELTA, method=METHOD,
    )
    chain_payload = build_chain_payload(
        [hop1, hop2], end_to_end=e2e_leg, signed_at=SIGNED_AT,
    )
    chain = sign_chain_receipt(chain_payload, private_jwk=private, kid=KID)
    return {
        "corpus": corpus, "pairs": pairs, "finals": finals, "scorer": scorer,
        "losses_hop1": losses_hop1, "losses_hop2": losses_hop2,
        "losses_e2e": losses_e2e, "hop1": hop1, "hop2": hop2, "chain": chain,
        "jwks": {"keys": [public]}, "g1": g1, "g2": g2,
    }


def main() -> None:
    r = build()
    for f, obj in [
        (HOP1_FILE, r["hop1"]), (HOP2_FILE, r["hop2"]),
        (CHAIN_FILE, r["chain"]), (JWKS_FILE, r["jwks"]),
    ]:
        f.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    _write_losses(
        LOSSES_HOP1_FILE, r["losses_hop1"],
        "hop 1 (bill -> reference summary), NLI judge; machine-pinned, "
        "512-token source truncation. Receipt replays over these committed losses.",
    )
    _write_losses(
        LOSSES_HOP2_FILE, r["losses_hop2"],
        "hop 2 (reference summary -> lead-N extractive, keep 0.5), NLI judge. "
        "Receipt replays over these committed losses.",
    )
    _write_losses(
        LOSSES_E2E_FILE, r["losses_e2e"],
        "end-to-end (bill -> lead-N extractive of its reference summary), NLI "
        "judge; 512-token source truncation. Chain end_to_end leg replays over these.",
    )
    FINALS_FILE.write_text(
        json.dumps(
            {
                "note": ("hop-2 outputs: deterministic lead-N extractive "
                         "(keep first ceil(0.5*n_sentences)) of each reference "
                         "summary. Offline, llm_calls_made=0. Committed for full "
                         "auditability of the (summary -> final) pairs."),
                "keep": LEAD_KEEP,
                "finals": [
                    {"id": p["id"], "final": f}
                    for p, f in zip(r["pairs"], r["finals"])
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    cp = r["chain"]["payload"]
    print(
        f"wrote certified chain: n_hops={cp['n_hops']} "
        f"chain_id={cp['chain_id']} budget={cp['budget_micro']/1e6:.4f} "
        f"joint_delta={cp['joint_delta_micro']/1e6:.4f}"
    )
    print(
        f"  hop1 ub={r['g1'].risk_upper_bound:.4f} (pe={r['g1'].point_estimate:.4f})  "
        f"hop2 ub={r['g2'].risk_upper_bound:.4f} (pe={r['g2'].point_estimate:.4f})  "
        f"e2e ub={cp['end_to_end']['risk_upper_bound_micro']/1e6:.4f}"
    )


if __name__ == "__main__":
    main()
