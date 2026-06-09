# BillSum golden meaning-risk receipt — the binding-gate artifact

A **real** `sum.meaning_risk_receipt.v1` over a **real public-domain
corpus** — the "one real signed receipt over a real corpus" gate the arXiv
Paper-1 plan ([`docs/arxiv/SUBMISSION_OUTLINE_2026-06-07.md`](../../docs/arxiv/SUBMISSION_OUTLINE_2026-06-07.md))
and the product-vision roadmap both name. Demonstrates **compression with a
certified meaning-loss bound** (bill → summary) — *"bounded", not
"preserving"*: aggressive summarization loses ~49% of the named proxy on
average; the receipt's job is to **certify how much**, not to claim little
was lost.

## What it certifies

> With 95% confidence, the **expected meaning-loss** of the BillSum
> bill→summary transform is **≤ 0.6454** — measured by the named
> `bidirectional-entailment[minilm-cosine-0.5]` judge, marginally over the
> first 64 BillSum test bills, under exchangeability. Controlled at the
> 0.70 target. (n=64, mean loss 0.4925.)

*(The 0.70 `alpha_target` is an **illustrative** control threshold to
exercise the `controlled` flag — not a tuned or claimed quality bar. The
bound 0.6454 and mean 0.4925 are the load-bearing measured numbers.)*

- **Corpus:** [BillSum](https://huggingface.co/datasets/FiscalNote/billsum)
  test split, first 64 examples in dataset order. **CC0-1.0** (US-government
  works are public domain), so the full (bill, summary) pairs are committed
  here for offline auditability — `corpus_billsum_test_first64.json`.
- **Judge:** local, offline `EmbeddingJudge` (mean-pooled `all-MiniLM-L6-v2`
  cosine), the paraphrase-aware no-`$` scorer. Named + versioned in the
  receipt, so the bound is explicitly conditional on it.
- **Bound:** the shipped default (`method="auto"` → Hoeffding for these
  fractional losses). At n=64 / var≈0.0037 empirical-Bernstein does not yet
  beat Hoeffding (eB's win regime is larger n / lower variance), so we
  commit the default rather than cherry-pick a method.

## The honest proof boundary (the load-bearing point)

| | replayable where? |
|---|---|
| **The certificate** (bound matches the committed losses) | **offline, everywhere** — pure-Python certifier over the committed integer-micro loss vector (`losses_billsum.json`); no model, no GPU. The receipt's `losses_hash` anchors it. **This is what CI checks.** |
| **The loss computation** (raw text → losses) | **machine-pinned** — needs the MiniLM forward pass, whose float output can drift across hardware/torch versions (F23/F26). Reproduced only on a matching stack. |

So the receipt is a *replayable certificate over a named judge's losses*,
honest that re-deriving those losses is judge- and machine-dependent. It is
a **batch** primitive — a single bill uses the per-document measurement, not
this certificate.

## Files

| File | What |
|---|---|
| `corpus_billsum_test_first64.json` | the 64 CC0 (bill, summary) pairs |
| `losses_billsum.json` | the per-pair meaning-loss vector (the committed evidence the receipt anchors) + the machine-pinning note |
| `meaning_risk_receipt.billsum.golden.json` | the signed receipt envelope |
| `jwks.json` | public key to verify the signature |
| `generate_billsum_fixture.py` | deterministic generator (private key never written; reads the committed losses, so regeneration is judge-free) |

## Reproduce / verify

```bash
# Verify + replay (offline, no judge) — Stage A + Stage B:
python -m pytest Tests/research/test_meaning_golden_billsum.py

# Regenerate byte-identically (judge-free; reads committed losses):
python fixtures/meaning_receipts_billsum/generate_billsum_fixture.py

# Re-derive the losses from raw text (needs the [judge] extra + a matching
# stack; this is the machine-pinned step): delete losses_billsum.json first,
# then run the generator with transformers + torch installed.
```
