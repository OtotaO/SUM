# Tags against claims

`scripts/tags_claims.py` (method `sum.tags_claims.v1`) asks which of a text's
own claims a list of notes states. The units are atomic claims written once
per text and frozen in `claims.json`; the judge is the exact judge of the
round-trip scripts (DeBERTa-v3-large NLI, windows of at most 80 tokens,
probability at least 0.9), reading the notes one per line. Nothing is
generated, and two runs gave the same probabilities.

## The claims

A model wrote the claims from each essay alone, with no tag list in its
context: one fact per claim, 4 to 16 words, self-contained, each with a
verbatim quote that the tests find in the source. The judge then read each
essay against its own claims. Claims it did not find were reworded from the
source, never from a tag list: first by removing an "In the simulation,"
frame the essay never states, which fixed most, then by a fresh model for the
20 still not found. One claim per essay still failed and is listed under
`dropped`. That leaves 94 claims for EFSO #1 and 56 for EFSO #2.

**None is checked yet.** Each claim has `"checked": false`, which only the
essay's author changes, after reading the claim against the essay (keep, fix
or drop). Results record how many were checked.

## The floor

The judge credits text about something else with a few claims. A recorded
rewrite written with no tags, a story set in a bookstore, is credited with
"The author thought of simulating a business super intelligence." at 0.98.
So the six recorded no-tags rewrites (three per essay) are the floor: a claim
any of them is credited with is loose, and the measure counts only the firm
claims left. The floor texts score zero on firm claims by construction; the
other controls do not, so they still test the measure.

## Tuning text: EFSO Vaults #1

The judge and premise were fixed before the first run on this text. That
run, on the first draft of the claims, showed the judge crediting the no-tags
rewrites, and the floor rule was added after it; the thresholds below come
from these runs. `results_efso_vaults_1.json`, claims not yet checked (a
budget is a cap: a list at 60 words has at most 60):

| | firm claims credited (of 79) |
|---|---|
| the source | 79 |
| topic, fact and mixed tags, budgets of 40 to 80 words | 2 to 5 |
| fact tags at 60 words | 3 |
| spread extract, lead at 80 words | 8, 8 |
| negated, swapped fact tags at 60 words | 0, 0 |
| unrelated list, filler | 0, 0 |
| EFSO #2's topic, fact and mixed tags at 60 words | 0, 0, 0 |

15 of the 94 claims are loose. Every control states none of the firm claims,
and no tag list states more than 5 of 79. The counts are small: true fact
tags beat their negations by 3 claims to 0.

## Prediction for EFSO Vaults #2 (committed before its run)

The judge has read EFSO #2's claims against its own source only, in the check
above, and against no note list. (The tests run a word-overlap stub over
them; its numbers are not recorded.) The run waits until its author has
checked its claims.

1. Two runs give the same probabilities.
2. The source states at least 0.95 of the firm claims. This holds by
   construction unless the author's edits add a claim the judge cannot find.
3. At most 0.25 of the claims are loose.
4. On firm claims, the negated fact tags, the swapped fact tags, the
   unrelated list, the filler and each of EFSO #1's topic, fact and mixed
   lists at 60 words are each credited with at most 1.
5. Fact tags at 60 words are credited with at least 2 firm claims, and the
   negated fact tags with fewer than half as many.
6. Every topic, fact and mixed list, at every budget, has a firm recall of at
   most 0.15.
7. At 80 words, the spread extract is credited with at least as many firm
   claims as every tag list.

What a result would mean:

- If 1, 2, 3 or 4 fails, the measure does not separate notes about the text
  from other notes, and its counts are not usable.
- If 5 fails, the measure cannot tell a stated claim from its negation on
  this text, or tags state too few claims to tell.
- 6 and 7 are about the tags, not the measure: whether, at these budgets,
  tag lists state little of a text and copied sentences state more.
