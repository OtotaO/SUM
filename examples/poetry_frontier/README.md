# Poetry on the frontier — the measurement the poetry-LLM papers skip

Poetry is an *extreme-compression rendering* of meaning. The interesting
question isn't "can an LLM write a sonnet of this contract" (it can) — it's
**how much meaning survived, and exactly what dropped.** Every adversarial-poetry
/ style-transfer paper transforms the form and never measures preservation.
SUM's named judge *is* that missing measurement — and it needs **zero new
substrate**: `sum meaning-diff` and `sum frontier` already host it. (No
meter/rhyme-aware scorer, no "poetic codec" — that would be decoration;
compression and error-correction are opposite operations.)

## The artifact

`source.txt` — a four-clause lease notice (rent, late fee, entry notice,
termination). Two renderings of it on the faithful↔compressed frontier:

- `plain_summary.txt` — a faithful paraphrase that keeps the specifics.
- `sonnet.txt` — a verse rendering that keeps the *gesture* and drops the numbers.

## Run it

```bash
sum meaning-diff source.txt plain_summary.txt --scorer embedding
sum meaning-diff source.txt sonnet.txt       --scorer nli   # nli: the figurative-language judge
```

## What the judges measured — and where they disagree

| rendering | embedding (MiniLM-cosine) | NLI (DeBERTa cross-encoder) |
| --- | --- | --- |
| plain summary | loss **0.000**, 4/4 preserved | — |
| sonnet | loss **1.000**, 0/4 preserved, 4 "unsupported" | loss **0.45**, **1/4 preserved**, 3 dropped, **0 fabricated** |

The plain summary preserves everything. The sonnet is where it gets honest: the
two judges *disagree sharply*, and the disagreement is the finding.

- The **embedding** judge scores the sonnet as **total loss** — every claim
  dropped, every verse line flagged as unsupported. Figurative phrasing ("twelve
  hundred dollars greet each month's first light") has low cosine similarity to
  the literal claim, so the judge can't see the meaning that *is* there.
- The **NLI** judge scores **0.45** — it **credits** the figurative rent clause
  (1/4 preserved), correctly drops the three specifics it really did lose (the
  *50 dollar fee*, *24 hours notice*, *30 days written notice*), and — crucially
  — flags **zero** lines as fabricated. It reads the verse the way a human does:
  the gist of the rent survived, the numbers didn't, nothing was invented.

## The honest finding (two things, both real)

1. **The judge works as advertised on aggressive rewriting.** It doesn't just
   produce a number — it *localizes* what an extreme compression dropped. That is
   the visceral, non-statistician-legible demonstration of "we measure
   meaning-loss under rewriting," and a direct rebuttal to form-transfer work
   that never measures preservation.

2. **It surfaces the judge choice, honestly and concretely.** The embedding-cosine
   judge under-credits *figurative* preservation (it scored the sonnet 1.000 —
   total loss); the NLI cross-encoder credited the figurative rent clause and
   scored it 0.45. Same text, loss 1.000 vs 0.45 — the largest judge disagreement
   we have, and it lands exactly on figurative language. This is the "embedding is
   brittle, prefer NLI" guidance the frontier already carries, made vivid: for
   anything but extractive trimming, use `--scorer nli`. Poetry is the sharpest
   possible stress-test of that choice, which is what makes it a good demo.

## Honest scope

Per-document **measurements** under a named judge, not certified bounds (a
`meaning_risk` receipt over a corpus is the (1-δ) guarantee). Poetry is mostly a
**high-loss** case — this demo advances the destination weakly (it stresses the
*number*, it doesn't encode knowledge). Its value is adoption-funnel clarity at
near-zero cost: drop a contract, get a sonnet, and *see, claim by claim, what the
beauty cost.*
