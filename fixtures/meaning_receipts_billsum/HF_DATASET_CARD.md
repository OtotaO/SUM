---
license: cc0-1.0
pretty_name: "SUM BillSum binding-gate meaning-risk receipt"
tags:
  - provenance
  - faithfulness
  - distribution-free-bounds
  - ai-transparency
  - chain-of-custody
language:
  - en
size_categories:
  - n<1K
---

# SUM BillSum binding-gate meaning-risk receipt

A **signed, independently re-verifiable** receipt recording a named
meaning-loss proxy over 64 public-domain BillSum (bill, reference summary)
pairs: a mean loss and a Hoeffding bound value that a verifier reproduces
exactly. The summaries are the dataset's human-written references, not AI
output, and the values describe these 64 pairs only. This is a worked, citable example of
`sum.meaning_risk_receipt.v1` from [SUM](https://github.com/OtotaO/SUM); the
files here are copies of `fixtures/meaning_receipts_billsum/` in that repo.

## What's in it

| File | What it is |
|---|---|
| `meaning_risk_receipt.billsum.golden.json` | the signed receipt (Ed25519 detached JWS over JCS-canonical bytes) |
| `jwks.json` | the issuer public key set (verify against this) |
| `losses_billsum.json` | the committed per-pair meaning-loss vector (integer-micro), to replay the bound arithmetic |
| `corpus_billsum_test_first64.json` | the 64 BillSum test bills (CC0-1.0) the values are computed over |

Corpus: the first 64 bills of the [BillSum](https://huggingface.co/datasets/billsum)
test split (US Congressional/California legislation, public domain / CC0).

## Verify it yourself, offline, in 5 lines

The verifier is dependency-light (no numpy/scipy/torch, no GPU, no network):

```bash
pip install "sum-engine[verify]"
python -m sum_verify meaning_risk_receipt.billsum.golden.json \
  --jwks jwks.json --losses losses_billsum.json
# → {"verified": true, "replayed": true, "risk_upper_bound": 0.645438, ...}
# (or, with no files at all: `python -m sum_verify --demo` replays this same golden)
```

`verified: true` + `replayed: true` means the committed losses hash to the
receipt's anchor and reproduce its stated bound arithmetic by exact integer
equality, on your machine. **Demo key:** the signing key is derived from a publicly known all-zero Ed25519 seed, so anyone can sign under this JWKS: the signature authenticates no issuer, and only the arithmetic replay is meaningful. Schema +
verification algorithm: [`docs/RECEIPT_FAMILY_SPEC.md`](https://github.com/OtotaO/SUM/blob/main/docs/RECEIPT_FAMILY_SPEC.md).

## What it records — and what it does NOT (read this)

The receipt's own disclosure, verbatim:

> Bounds the EXPECTED value of a NAMED meaning-loss proxy (bidirectional-entailment
> over a local all-MiniLM-L6-v2 cosine judge), MARGINALLY over the first 64 BillSum
> test bills (CC0-1.0), under exchangeability. NOT a per-document claim and NOT
> meaning itself. The CERTIFICATE replays offline over the committed integer-micro
> loss vector; the LOSS COMPUTATION is machine-pinned (model-judge float drift,
> F23/F26) and reproduced only on a matching torch/MiniLM stack.

- **Recorded values:** mean loss 0.4925 and a Hoeffding bound value of **0.6455** at δ = 0.05 (rounded up from the signed 0.645438; n = 64), with `controlled = true` against the issuer's illustrative 0.70 target. The first 64 bills are a fixed prefix, not an independent random draw, so these values describe these 64 pairs; they are not a 95% bound on expected meaning-loss. (The signed wording above says "under exchangeability"; it predates the correction that exchangeability alone would not support such a bound either.)
- **`not_covered`:** `arrangement`, `sound`, `connotation`, `implicature` — the layers the proxy explicitly does not measure.
- The meaning proxy tracks *human* faithfulness only **modestly** (Spearman ρ = 0.267–0.291, pooled summary-level, on SummEval). The values describe a *named proxy* and are **not** a substitute for human judgment. The signature shows only that the holder of the demo key signed these bytes (see **Demo key** above); the arithmetic replay is the meaningful check, and re-deriving the losses from text is machine-pinned.

## Cite / reuse

CC0-1.0 — public domain. If you reuse it as prior art or a baseline, a link to
the [SUM repo](https://github.com/OtotaO/SUM) is appreciated but not required.
