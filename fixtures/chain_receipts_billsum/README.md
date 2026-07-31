# BillSum certified chain — the first real `sum.chain_receipt.v1`

The world's first **real certified multi-hop meaning chain** over a real
public-domain corpus: two real transforms of the same 32 US Congressional
bills, with their meaning-loss composed hop by hop into a Bonferroni budget,
plus a directly measured end-to-end leg. This is the "drift budget" the
meaning-loss frontier converged on, materialised as a signed, offline-replayable
certificate.

## The two hops (read the labels before quoting any number)

| Hop | Transform | What performed it |
|---|---|---|
| **1** | `summarize:billsum-reference` (bill → reference summary) | the **dataset's own** reference summarization. **SUM did not perform it.** Same framing as the binding-gate golden: we certify the meaning-loss of a transform someone else performed. |
| **2** | `compress:lead-extractive-keep0.5` (reference summary → lead-N extractive) | a real, **deterministic, offline** transform: keep the first `ceil(0.5 * n_sentences)` sentences of the summary (lead-N, a standard extractive-summarization baseline). `llm_calls_made = 0`; no model, no key, no network. Single-sentence summaries pass through unchanged (identity, loss ~ 0) — honest behaviour, not a bug. |

## What it certifies

> With 95% confidence per hop (joint confidence **0.90** under Bonferroni),
> over the first 32 BillSum test bills (CC0-1.0), under exchangeability, by the
> named strict NLI judge:
>
> | leg | expected meaning-loss upper bound | point estimate |
> |---|---|---|
> | hop 1 — abstractive summarization | **≤ 0.865768** | 0.649416 |
> | hop 2 — deterministic extractive compression | **≤ 0.488860** | 0.272507 |
> | **budget** (sum of hops; Bonferroni) | **≤ 1.354628** | — |
> | direct end-to-end (bill → final) | **≤ 0.874216** | — |
>
> `chain_id = 9a8ab39f08522c50`.
>
> *Micro-unit rounding: each `≤` value is the true bound rounded to nearest at
> 1e-6 resolution (the signed `*_micro` wire convention — see
> `docs/RECEIPT_FAMILY_SPEC.md` §2). A strictly-conservative reading adds 1e-6.*

**Read this honestly, it is the point.** Strict recall-weighted NLI reports
that abstractive summarization of a full bill loses a lot of the named proxy
(hop 1), while deterministic extractive compression is gentler (hop 2). The
chain surfaces **where** meaning is lost. The bounds are **wide** at n=32
(Hoeffding); that is fine and stated plainly. We do **not** swap to a lenient
judge to get prettier numbers — that would be the exact cherry-pick this
project refuses.

**The budget is not the end-to-end loss.** The budget (1.354628) bounds the
*sum* of per-hop expected losses (a Bonferroni union bound). The *direct*
end-to-end measurement (0.874216) is **lower** than the budget, because the
proxy is a **directed loss, not a metric** — no triangle inequality holds in
either direction. The receipt's mandatory `budget_scope` field says this; the
verifier fails closed without it.

## The honest proof boundary (identical discipline to the binding-gate golden)

| | replayable where? |
|---|---|
| **The certificate** (each hop + the chain bound matches the committed losses) | **offline, everywhere** — the pure-Python certifier re-runs over the committed integer-micro loss vectors (`losses_hop1.json` / `losses_hop2.json` / `losses_e2e.json`); no model, no GPU. **This is what CI checks.** |
| **The loss computation** (raw text → losses) | **machine-pinned** — needs the NLI forward pass, whose float output can drift across hardware/torch versions, and long bills are truncated to the judge's ~512-token window (F23/F26). Reproduced only on a matching stack. |

## Files

| File | What |
|---|---|
| `hop1_summarize.golden.json` | signed hop-1 meaning-risk receipt (bill → summary) |
| `hop2_extractive.golden.json` | signed hop-2 meaning-risk receipt (summary → lead-N) |
| `chain_receipt.billsum.golden.json` | the signed `sum.chain_receipt.v1` envelope |
| `jwks.json` | public key to verify every signature |
| `losses_hop1.json` / `losses_hop2.json` / `losses_e2e.json` | committed integer-micro loss vectors the receipts anchor + machine-pinning notes |
| `finals_lead_extractive.json` | hop-2 outputs (the deterministic lead-N of each summary), committed for full auditability of the pairs |
| `generate_a2_chain_fixture.py` | deterministic generator (private key never written; reads the committed losses, so regeneration is judge-free) |

The corpus itself is the CC0 slice already committed at
`../meaning_receipts_billsum/corpus_billsum_test_first64.json` (first 32 used
here).

## Reproduce / verify

```bash
# Verify + replay the whole chain offline (no judge) via the SDK CLI:
python -m sum_verify fixtures/chain_receipts_billsum/chain_receipt.billsum.golden.json \
  --jwks fixtures/chain_receipts_billsum/jwks.json \
  --hops fixtures/chain_receipts_billsum/hop1_summarize.golden.json \
         fixtures/chain_receipts_billsum/hop2_extractive.golden.json \
  --losses fixtures/chain_receipts_billsum/losses_e2e.json
# -> {"verified": true, "replayed": true, "hops_replayed": true, "end_to_end_replayed": true, ...}

# Full replay + regression + byte-stable regeneration test (numpy + joserfc only):
python -m pytest Tests/research/test_chain_golden_billsum.py

# Regenerate byte-identically (judge-free; reads the committed losses):
python fixtures/chain_receipts_billsum/generate_a2_chain_fixture.py

# Re-derive the losses from raw text (needs the [judge] extra + a matching
# stack; the machine-pinned step): delete the losses_*.json first, then run the
# generator with transformers + torch installed.
```

All three receipts are witnessed in the public transparency log
(`transparency/log.jsonl`); `python scripts/witness_receipt.py verify` recomputes
the hash chain.
