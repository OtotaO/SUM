# Cross-lingual translation golden meaning-risk receipt — the moat artifact

A **real** `sum.meaning_risk_receipt.v1` over a real EN→FR translation
corpus — the paraphrase-robustness half of the binding-gate pair (the
"Both" alongside the BillSum *compression* receipt in
[`../meaning_receipts_billsum/`](../meaning_receipts_billsum/)).

## What it demonstrates — the moat, directly

> With 95% confidence, the **expected meaning-loss** of EN→FR translation is
> **≤ 0.4124** — measured by the named multilingual NLI judge
> (`mDeBERTa-v3-base-xnli`), marginally over the first 64 length-aligned
> opus-100 en-fr test pairs, under exchangeability. **Controlled** at the
> 0.50 target. (n=64, mean loss 0.2594; **39/64 pairs at exactly 0 loss**.)

The headline is the **39/64 at zero loss**: faithful translations preserve
meaning despite **zero lexical overlap** between English and French. The
dial scores by *meaning*, not surface form — robust to the most extreme
rewriting, a different language. (A watermark or lexical scheme cannot do
this; this is the chain-of-custody-for-meaning moat.)

Pair it with the BillSum receipt and the dial **grades** transforms:
translation preserves more (mean 0.26) than aggressive summarization
(mean 0.49).

- **Corpus:** [opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
  en-fr test, first 64 pairs that are substantive (en ≥6 / fr ≥4 words) and
  length-aligned (word-count ratio in [0.5, 2.0] — a fixed, documented
  alignment-quality filter set before any bound was seen, not result-tuned).
- **Judge:** local, offline `mDeBERTa-v3-base-xnli` multilingual NLI (the
  F24 model; the scorer is `model_id`-parameterised, so no new code).
- **Method:** the shipped default (`auto` → Hoeffding for these fractional
  losses).

## License-clean handling (why no raw text here)

opus-100 aggregates mixed-licence OPUS web sources, so — unlike the **CC0**
BillSum fixture, which commits its text in full — this fixture does **not
redistribute the raw text**. It commits:

- `losses_translation.json` — the per-pair meaning-loss vector (the evidence
  the receipt's `losses_hash` anchors) + the machine-pinning note;
- `corpus_pointer.json` — a **sha256-pinned** pointer (dataset + exact
  deterministic selection + hash of the pairs), so the corpus is reproducible
  and verifiable without redistribution.

The **certificate replays offline** over the committed losses (CI-checked,
no model/GPU/fetch); re-deriving the losses re-fetches opus-100 under its own
terms and runs the (machine-pinned, F23/F26) judge — the hash verifies the
same pairs were used.

## Reproduce / verify

```bash
python -m pytest Tests/research/test_meaning_golden_translation.py   # offline, no judge
python fixtures/meaning_receipts_translation/generate_translation_fixture.py  # byte-stable regen
```
