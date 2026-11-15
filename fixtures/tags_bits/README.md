# Tags in bits

`scripts/tags_bits.py` (method `sum.tags_bits.v1`) asks how much a list of notes
tells a fixed language model about a text. The model is SmolLM2-360M, pinned by
revision, loaded from safetensors and run in float32 on CPU. For each source
sentence, it counts the bits the model needs for that sentence after reading
the notes, and compares that with the bits after reading no notes. The gain is
the difference summed over sentences; the share is the gain over the no-notes
total. The units are the sentences as the round-trip judges split them, so
they include bare lines such as code fences (4 of the 49 units of EFSO #2).

Nothing is generated. The tag and sentence lists come from files already
committed; the filler list and the self condition (each sentence as its only
note) are built in code:

- the recorded equal-budget notes (`../tags_roundtrip/equal_budget/`);
- the same kinds of notes for the other EFSO text, which has the same author
  and series, so it is not an unrelated control;
- the hand-written controls in `controls.json`:
  - the fact@60 tags negated, each content word kept as its stem, with
    "never" or "did not" added. In 13 of the 27, "did not" turns the verb to
    its base form ("used" becomes "did not use"), so the wording changes as
    well, and the lists grow by about 20 words. The description inside
    `controls.json` says "every content word kept"; the file is left
    byte-identical because both results files bind its sha256;
  - the fact@60 tags with one detail swapped;
  - an unrelated list;
- a filler list of notes that say nothing.

On the exact judge, the run and the rerun of EFSO #1 with the same tags and
seeds agree on 45 of the 96 sentence credits that either one gives, pooled over
six rewrite pairs (topic and mixed at 32 tags, three seeds), because the
rewrites differ between runs. This measure has no rewrite.

## Why it was built

The round-trip judges cannot read tag lists fairly. Reading
`../tags_roundtrip/equal_budget/` again shows two faults:

- **Gist judge:** topic and mixed tag lists alone get more gist credit than
  the rewrites written from them, in all 11 cells. Take the source sentences
  the gist judge credits to a rewrite but not to its notes read alone, as a
  share of all source sentences, averaged over the three rewrites: for the 17
  topic, fact and mixed cells this is 0.00 to 0.06, and for the 12 extract and
  lead cells 0.01 to 0.08. These figures were computed after the results were
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

## Result for EFSO Vaults #2 (added after the run)

`results_efso_vaults_2.json`. Seven of the eight predictions held. The one
that failed, 6, is one the prediction named as disqualifying: "its gains are
not usable at all". By the prediction's own terms, no gain here measures what
notes carry about a text. The observations after the table are descriptive and
sit outside what the prediction allows.

| prediction | threshold | EFSO #2 | held |
|---|---|---|---|
| 1. two CPU runs | 0.0 bits | 0.0 bits | yes |
| 2. negated over true fact tags | at least 0.7 | 0.91 | yes |
| 3. swapped over true | 0.4 to 0.85 | 0.46 | yes |
| 4. unrelated share; filler share | -0.04 to 0.01; -0.03 to 0.01 | -0.034; -0.003 | yes |
| 5. a sentence as its own note | at least 0.75 | 0.82 | yes |
| 6. the other text's notes at 60 words | 0 to 0.06, below this text's | topic 0.087, fact 0.090, mixed 0.094 (this text's: 0.108, 0.123, 0.107) | no |
| 7. spread extract above every tag list | at 60 and 80 words | 0.218 against at most 0.123 at 60; 0.236 against at most 0.144 at 80 | yes |
| 8. topic, fact, mixed at 60 within 0.03 | 0.03 | 0.108, 0.123, 0.107 | yes |

What this means:

- **Wording, not claims.** On both texts, negated fact tags kept 0.91 to 0.93
  of the true tags' gain. A gain in bits says how much of the text's wording
  the notes give the model, not which of its claims they state.
- **On EFSO #2, the tag lists mostly carry what the two essays share.** The
  first essay's tags (same author, same series, both about the World_sim
  simulator) give 0.09 of its bits. EFSO #2's own tags give 0.11 to 0.12. So
  the difference between a text's own tags and the other text's is 0.01 to
  0.03 of the text's bits, and the measure cannot separate what a tag list
  carries about this text from what it carries about the other essay.
  On EFSO #1, the second essay's tags gave 0.03 to 0.05 against 0.12 to 0.13
  for its own, so the two directions differ.
- **The extract's lead is mostly its copied sentences scoring themselves.**
  Prediction 7 held: at 60 and 80 words the spread extract's share (0.18 to
  0.24) is above every tag list's (at most 0.14), on both texts. On the
  sentences not copied into the extract, it is 0.12 to 0.14 against 0.10 to
  0.14 for the tag lists, and above every tag list in 2 of the 4 text and
  budget cells: EFSO #2 at 60 words, and EFSO #1 at 80 words by 4 bits
  (0.142 against 0.141 for fact tags).
- **Deterministic and blind to unrelated notes.** Two CPU runs gave identical
  bits, and the unrelated and filler lists gave slightly negative shares.

Neither this measure nor the round-trip judges can yet say how much of a
specific text a tag list carries back. A measure of claims needs units a
person has checked: the text's atomic claims, written once per essay and
frozen, then scored by the exact judge with a weight that discounts claims
any on-topic text entails.
