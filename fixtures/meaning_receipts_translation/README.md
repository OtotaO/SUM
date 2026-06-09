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
meaning despite **near-zero lexical overlap** between English and French
(the few shared tokens are names, numbers, and cognates). The dial scores by
*meaning*, not surface form — robust to the most extreme rewriting, a
different language. (A watermark or lexical scheme cannot do this; this is
the chain-of-custody-for-meaning moat.)

**Honest scope (read before quoting the number).**
- The bound is over the **filtered subset** named in `corpus_id`
  (`…-filtered`), *not* all opus-100 en-fr. The dominant filter is the
  min-word (short-segment) exclusion; short segments are where the NLI judge
  is noisiest and loss tends to be highest, so the filtered sample plausibly
  biases the proxy **low** vs the unfiltered split. Quote the number *with*
  the "filtered subset" qualifier.
- The NLI judge emits a **small discrete set** of per-pair loss values
  (recall/fidelity are booleans per sentence), so the loss vector is coarse,
  not continuous — the mean/bound rest on that grid.
- The 0.50 `alpha_target` is an **illustrative** control threshold, not a
  tuned or claimed quality bar.
- **No clean cross-receipt grading:** the BillSum receipt uses a *different*
  judge (MiniLM-cosine embedding) than this one (mDeBERTa NLI), so the means
  (0.26 vs 0.49) are **not** directly comparable as a single "the dial grades
  transforms" scale — each transform is certified under its own appropriate,
  named judge. A clean cross-transform comparison would require the same
  judge on both.

- **Corpus:** [opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
  en-fr test (revision `805090d`), first 64 pairs that are substantive
  (en ≥6 / fr ≥4 words) and length-aligned (word-count ratio in [0.5, 2.0] —
  a fixed, documented alignment-quality filter set before any bound was seen,
  not result-tuned).
- **Judge:** local, offline `mDeBERTa-v3-base-xnli` multilingual NLI (the
  F24 model; the scorer is `model_id`-parameterised, so no new code).
- **Method:** the shipped default (`auto` → Hoeffding for these fractional
  losses; at n=64 / this variance, the variance-adaptive empirical-Bernstein
  is *not* tighter, so we commit the default — not a cherry-pick).

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
