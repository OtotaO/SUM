# BillSum signed chain — the first real `sum.chain_receipt.v1`

The first **real signed multi-hop meaning chain** in this project over a real
public-domain corpus: two real transforms of the same 32 US Congressional
bills, with their per-hop Hoeffding bound values summed into a Bonferroni
budget, plus a directly measured end-to-end leg. This is the "drift budget" the
meaning-loss frontier converged on, recorded as a signed, offline-replayable
receipt of descriptive measurements.

## The two hops (read the labels before quoting any number)

| Hop | Transform | What performed it |
|---|---|---|
| **1** | `summarize:billsum-reference` (bill → reference summary) | the **dataset's own** reference summarization. **SUM did not perform it.** Same framing as the binding-gate golden: we compute the meaning-loss of a transform someone else performed. |
| **2** | `compress:lead-extractive-keep0.5` (reference summary → lead-N extractive) | a real, **deterministic, offline** transform: keep the first `ceil(0.5 * n_sentences)` sentences of the summary (lead-N, a standard extractive-summarization baseline). `llm_calls_made = 0`; no model, no key, no network. Single-sentence summaries pass through unchanged (identity, loss ~ 0) — honest behaviour, not a bug. |

## What it measures

> Hoeffding bound values at δ = 0.05 per hop (joint δ = **0.10** under
> Bonferroni), over the first 32 BillSum test bills (CC0-1.0), by the named
> strict NLI judge. The 32 bills are a fixed prefix of the split, not an
> independent random draw, so every value below describes these 32 bills
> only; none is a bound on expected meaning-loss for other bills. (The
> signed hop disclosures say "under exchangeability"; that wording predates
> the correction and the receipts are kept byte-for-byte.)
>
> | leg | Hoeffding bound value (δ 0.05, descriptive) | mean loss |
> |---|---|---|
> | hop 1 — abstractive summarization | **≤ 0.865768** | 0.649416 |
> | hop 2 — deterministic extractive compression | **≤ 0.488860** | 0.272507 |
> | **budget** (sum of the two hop values; Bonferroni) | **≤ 1.354628** | — |
> | direct end-to-end (bill → final) | **≤ 0.874216** | 0.657864 |
>
> `chain_id = 9a8ab39f08522c50`.
>
> *Micro-unit rounding: each `≤` value is the computed bound value rounded to
> nearest at 1e-6 resolution (the signed `*_micro` wire convention — see
> `docs/RECEIPT_FAMILY_SPEC.md` §2). A strictly-conservative reading adds 1e-6.*

**Read this honestly, it is the point.** Strict recall-weighted NLI reports
that abstractive summarization of a full bill loses a lot of the named proxy
(hop 1), while deterministic extractive compression is gentler (hop 2). The
chain surfaces **where** the proxy registers loss. The bound values are
**wide** at n=32 (Hoeffding); that is fine and stated plainly. We do **not** swap to a lenient
judge to get prettier numbers — that would be the exact cherry-pick this
project refuses.

**The budget is not the end-to-end loss.** The budget (1.354628) is the
*sum* of the two per-hop bound values (the Bonferroni union composition), a
descriptive figure for these 32 bills. The *direct* end-to-end value
(0.874216) is **lower** than the budget, because the proxy is a **directed
loss, not a metric** — no triangle inequality holds in either direction.
The receipt's mandatory `budget_scope` field (the verifier fails closed
without it) says the budget does not bound the end-to-end loss. Like every
chain minted before PR #531, this golden's same field also
states the Bonferroni reading unconditionally ("budget_micro bounds the SUM
of per-hop expected proxy losses ... joint confidence >= 1 - joint_delta").
The signed bytes stay as they are; every verdict adds an unsigned
`budget_scope_note` saying that reading holds only under each hop's
sampling assumptions, which these fixed-prefix samples do not meet. Chains
minted from PR #531 on sign the conditional wording instead.

## The honest proof boundary (identical discipline to the binding-gate golden)

| | replayable where? |
|---|---|
| **The bound arithmetic** (each hop value + the chain budget match the committed losses) | **offline, everywhere** — the pure-Python bound code re-runs over the committed integer-micro loss vectors (`losses_hop1.json` / `losses_hop2.json` / `losses_e2e.json`); no model, no GPU. **This is what CI checks.** |
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

**Demo key:** the signing key is derived from a publicly known all-zero Ed25519 seed, so anyone can sign under this JWKS: the signature authenticates no issuer, and only the arithmetic replay is meaningful.

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
# -> {"verified": true, "replayed": true, "hop_envelopes_checked": true, "end_to_end_replayed": true,
#     "issuer_asserted_budget": 1.354628, "joint_delta": 0.1, "budget_scope_note": "...", ...}

# Full replay + regression + byte-stable regeneration test (numpy + joserfc + cryptography):
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
