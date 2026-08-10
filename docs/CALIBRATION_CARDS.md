# Calibration cards — the judge's validity, as a product surface

**Status: shipped with the `[research]` surface (unsigned, machine-readable).**
Cards live in [`fixtures/calibration_cards/`](../fixtures/calibration_cards/);
each is a `sum.calibration_card.v1` JSON. A CI-run test
(`Tests/research/test_calibration_cards.py`) hard-locks every number in every
card to the committed measurement artifact it cites, so a card can never
drift from its measurement.

## Why cards, not caveats

Every SUM receipt already carries the honest caveat: the bound is over a
*named proxy*, and vs human judgments the proxy correlates only modestly.
A caveat tells a reader that validity is limited; a **card** tells them *how
much, measured where, at what aggregation level, with what confidence
interval* — and gives them the committed script + artifact to recompute it.

That inversion is deliberate. The provenance market signs bytes and disclaims
meaning; SUM signs a meaning proxy and **ships the evidence for what that
proxy is worth**. The card family is the second half of the honesty
discipline: not just "every number wears its scope" but *the scope itself is
a first-class, versioned, machine-readable artifact*.

Cards are **unsigned by design**: they are measurements *about the
instrument*, made on calibration corpora — not certified properties of any
document a receipt covers. Baking a card's ρ into a signed receipt field
would be a cross-corpus overclaim (the card was measured on SummEval/FRANK,
not on whatever corpus the receipt certifies). The receipt cites the judge;
the card family documents the judge; the two must stay separate surfaces.

## The current cards (measured 2026-07-02, carded 2026-07-10)

| Card | Corpus | Aggregation | Headline | The point |
|------|--------|-------------|----------|-----------|
| [`summeval_pooled_2026-07-02.json`](../fixtures/calibration_cards/summeval_pooled_2026-07-02.json) | SummEval | pooled summary-level | ρ ≈ 0.27–0.29 (all three judges) | The strict read. Modest. This is the number that keeps headlines honest. |
| [`summeval_system_level_2026-07-02.json`](../fixtures/calibration_cards/summeval_system_level_2026-07-02.json) | SummEval (same data) | system-level, n=16 | ρ/r ≈ 0.57–0.75 (NLI composite 0.70 [0.49, 0.83]) | The aggregation most published "0.6-class" numbers use. Publishing both cards makes the aggregation effect explicit. |
| [`frank_xsum_2026-07-02.json`](../fixtures/calibration_cards/frank_xsum_2026-07-02.json) | FRANK-XSum (abstractive) | pooled | NLI **replicates** (0.290); embedding **collapses** (0.032, CI spans 0) | The failure-mode card. Why `--scorer nli` is the load-bearing default. |
| [`frank_cnndm_2026-07-02.json`](../fixtures/calibration_cards/frank_cnndm_2026-07-02.json) | FRANK-CNN/DM (extractive-leaning) | pooled | NLI stable (0.290); lexical strong here (0.47) but collapses on XSum | The bracket: judges whose validity depends on text style must be re-checked per corpus. |

## Correction, 2026-08-10: the headline range was mis-scoped

Every surface in this repo used to quote **"ρ ≈ 0.27–0.33 (pooled
summary-level, SummEval)"**. That range was wrong, and it was wrong in the
flattering direction. Corrected here and on all five other surfaces that
carried it.

**What the pooled card actually measures** (its own `results` block, against
the `meaning_composite` target):

| scorer | ρ | n |
|---|---|---|
| lexical-coverage-bidirectional | 0.2907 | 800 |
| embedding-minilm@0.5 | 0.2672 | 800 |
| nli-deberta-mnli-fever-anli@0.5 | 0.2741 | 192 |

So the honest pooled range is **0.267–0.291**. The maximum is 0.2907, not 0.33.

**Where 0.33 actually came from.** It is real, but it is a *different target*:
`scorers['nli-deberta-mnli-fever-anli@0.5'].consistency.spearman = 0.3273`
(n=192) in `Tests/benchmarks/meaning_proxy_human_calibration.result.json` — the
NLI judge against the **consistency axis alone**, not the meaning composite.
Splicing it onto the composite range produced a single number spanning two
targets at two sample sizes, with the upper endpoint borrowed from whichever
measurement happened to look best.

**Why this correction matters more than its size.** `NORTH_STAR.md` invariant 2
is "every number ships with its scope", and it used *this very range* as its
worked example of a correctly-scoped number. The rule was being violated by its
own illustration, on the four surfaces a stranger reads first. Nobody outside
the project found this; an adversarial audit of our own materials did.

If you are citing this work, cite 0.267–0.291 pooled summary-level against the
meaning composite, and cite 0.3273 separately if you want the consistency-axis
figure. They are not the same claim.

## Reading rules (the ones that keep this honest)

1. **Never quote a card number without its aggregation level and corpus.**
   "ρ = 0.267–0.291 (pooled summary-level, SummEval, meaning-composite
   target)" is a claim;
   "ρ ≈ 0.3" is an overclaim by omission.
2. **A card is evidence about the judge, not about your document.** Per-document
   trust still comes from reading the receipt's own losses/diff, not the card.
3. **Cards never enter signed fields.** See above.
4. **The failure modes are part of the card.** A card family that only carried
   the replications would be marketing; the embedding-collapse card is the one
   a skeptic should read first.

## Adding a card

A new card requires, in the same PR: (a) the measurement script committed
under `Tests/benchmarks/`, (b) the result artifact committed, (c) the card
JSON whose every number is asserted equal to the artifact by
`Tests/research/test_calibration_cards.py`, (d) scope limits written by
someone trying to *refute* the number, not sell it. Cards for corpora we
cannot redistribute must still commit the deterministic sampling recipe
(seeds, splits) so a holder of the corpus can recompute.

The natural next cards, in priority order (each $0-gated on nothing, just
time — or a grant deliverable): a per-transform card (translation vs
compression vs perspective on the same corpus); a card for the deterministic
INT8 judge mode (does quantization change validity?); and the eventual
**human-anchor card** — a small fresh human-judgment round, itself committed,
re-calibrating the proxy on a cadence.
