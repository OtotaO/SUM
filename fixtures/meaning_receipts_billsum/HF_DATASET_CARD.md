---
license: cc0-1.0
pretty_name: "SUM BillSum binding-gate meaning-risk receipt"
tags:
  - provenance
  - faithfulness
  - conformal-prediction
  - ai-transparency
  - chain-of-custody
language:
  - en
size_categories:
  - n<1K
---

# SUM BillSum binding-gate meaning-risk receipt

A **signed, independently re-verifiable** certificate of how much meaning an AI
transformation could have lost — bounded with a distribution-free conformal
guarantee over a public-domain corpus. This is a worked, citable example of
`sum.meaning_risk_receipt.v1` from [SUM](https://github.com/OtotaO/SUM); the
files here are copies of `fixtures/meaning_receipts_billsum/` in that repo.

## What's in it

| File | What it is |
|---|---|
| `meaning_risk_receipt.billsum.golden.json` | the signed receipt (Ed25519 detached JWS over JCS-canonical bytes) |
| `jwks.json` | the issuer public key set (verify against this) |
| `losses_billsum.json` | the committed per-pair meaning-loss vector (integer-micro), to replay the bound |
| `corpus_billsum_test_first64.json` | the 64 BillSum test bills (CC0-1.0) the bound is over |

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
receipt's anchor and re-certify to its stated bound by exact integer equality —
on your machine, against the issuer's key, trusting nobody. Schema +
verification algorithm: [`docs/RECEIPT_FAMILY_SPEC.md`](https://github.com/OtotaO/SUM/blob/main/docs/RECEIPT_FAMILY_SPEC.md).

## What it bounds — and what it does NOT (read this)

The receipt's own disclosure, verbatim:

> Bounds the EXPECTED value of a NAMED meaning-loss proxy (bidirectional-entailment
> over a local all-MiniLM-L6-v2 cosine judge), MARGINALLY over the first 64 BillSum
> test bills (CC0-1.0), under exchangeability. NOT a per-document claim and NOT
> meaning itself. The CERTIFICATE replays offline over the committed integer-micro
> loss vector; the LOSS COMPUTATION is machine-pinned (model-judge float drift,
> F23/F26) and reproduced only on a matching torch/MiniLM stack.

- **Bound:** expected meaning-loss ≤ **0.6454** at **95%** (n = 64), `controlled = true`.
- **`not_covered`:** `arrangement`, `sound`, `connotation`, `implicature` — the layers the proxy explicitly does not measure.
- The meaning proxy tracks *human* faithfulness only **modestly** (Spearman ρ ≈ 0.27–0.33 on SummEval). This is a directionally-valid, distribution-free bound on a *named proxy*, **not** a substitute for human judgment. Provenance is cryptographically solid; the meaning recomputation is advisory.

## Cite / reuse

CC0-1.0 — public domain. If you reuse it as prior art or a baseline, a link to
the [SUM repo](https://github.com/OtotaO/SUM) is appreciated but not required.
