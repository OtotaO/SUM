# Tags in bits

`scripts/tags_bits.py` (method `sum.tags_bits.v1`) asks how much a list of notes
tells a fixed language model about a text. The model is SmolLM2-360M, pinned by
revision, loaded from safetensors and run in float32 on CPU. For each source
sentence, it counts the bits the model needs for that sentence after reading
the notes, and compares that with the bits after reading no notes. The gain is
the difference summed over sentences; the share is the gain over the no-notes
total.

Nothing is generated. Every note list comes from a file already committed:

- the recorded equal-budget notes (`../tags_roundtrip/equal_budget/`);
- the same kinds of notes for the other EFSO text, which has the same author
  and series, so it is not an unrelated control;
- the hand-written controls in `controls.json`:
  - the fact@60 tags, negated with every content word kept;
  - the fact@60 tags with one detail swapped;
  - an unrelated list;
- a filler list of notes that say nothing.

The two runs of the round-trip judges disagreed on about half the sentences
they credited, because the rewrites differ between runs. This measure has no
rewrite.

## Why it was built

The round-trip judges cannot read tag lists fairly. Reading
`../tags_roundtrip/equal_budget/` again shows two faults:

- **Gist judge:** it credits a bare tag list with more of the source than the
  rewrite written from it. Net of what the list gets alone, a rewrite gained
  0.00 to 0.08 in all 29 cells. That figure was computed after the results were
  seen, not predicted.
- **Exact judge:** it does not see tag lists at all. True fact tags alone score
  0.02 to 0.06, the same as rewrites written with no tags (0.05 and 0.07).

A search of the literature found no paper that tests tag or keyword lists as
the premise of these judges.

This measure scores the notes themselves. The question it had to answer first
is whether it measures claims or wording, so the deciding control is the
negated fact tags.

## Tuning text: EFSO Vaults #1

The context template, the model, the unit and the no-notes context were set
before this run and not changed after it. The run only informed the thresholds
below. `results_efso_vaults_1.json`:

| check | EFSO #1 |
|---|---|
| largest difference between two runs | 0.0 bits |
| negated fact tags' gain over true fact tags' gain | 0.93 |
| swapped-detail fact tags over true | 0.67 |
| unrelated list share | -0.020 |
| filler share | -0.011 |
| a sentence as its own note, mean share of its bits | 0.85 |

Negated tags kept 0.93 of the gain, so on this text the measure is about
wording, not claims: "author never used World_sim simulator" tells the model
nearly as much as "author used World_sim simulator". The negated lists are
also about 20 words longer, because of the added "never" and "did not".

## Prediction for EFSO Vaults #2 (committed before its run)

Thresholds for the run on `efso_vaults_2`, which has not been scored with this
measure:

1. Two CPU runs differ by 0.0 bits.
2. Negated fact tags keep at least 0.7 of the true fact tags' gain. If they
   keep 0.5 or less, the measure tells claims apart; the prediction is that it
   does not.
3. Swapped-detail fact tags keep between 0.4 and 0.85 of the gain.
4. The unrelated list's share is between -0.04 and 0.01; the filler's share is
   between -0.03 and 0.01.
5. A sentence as its own note recovers on average at least 0.75 of its bits.
6. Each of the other text's note lists (topic, fact and mixed at 60 words) has
   a share between 0 and 0.06, below this text's own list of the same kind.
7. At 60 and at 80 words, the spread extract's share is above every tag
   list's share at the same budget.
8. At 60 words, topic, fact and mixed tags have shares within 0.03 of each
   other.

What a result would mean:

- If 2 holds, gains in bits stay a measure of wording, and no claim-recall
  wording may rest on them.
- If 4, 5 or 6 fails, the measure is not separating notes about the text from
  other notes, and its gains are not usable at all.
