# Tags or sentences, at the same number of words

Every earlier run here compared tags with no tags or with other tags. None asked whether a
few dozen words of tags bring back more of a text than the same number of words copied from
it. This run asks that, on the two worked examples in `../efso_vaults_1/` and
`../efso_vaults_2/`, reusing the tags stored in their `results.json` (`--tags-from`).

## Stated before the run

Written and committed before any result in this folder was produced.

At budgets of 40, 60 and 80 words, five kinds of notes are written again by the same model
(gpt-4o-mini-2024-07-18) through one shared prompt, so only the notes differ. The notes are
numbered and unlabelled:

- topic, fact and mixed tags: the largest tag count whose words fit in the budget;
- extract: whole sentences of the text chosen by TF-IDF centrality with a redundancy penalty
  (no model involved);
- lead: sentences in text order, passing over any that does not fit.

Plus the no-tags control. Three rewrites each (seeds 7 to 9), scored by the frozen gist and
exact judges. Each rung also records the notes read as if they were the rewrite: for an
extract, the credit a rewrite gets just by repeating its notes.

```
PYTHONPATH=. python scripts/tags_roundtrip.py fixtures/tags_roundtrip/efso_vaults_N/source.md \
  --tags-from fixtures/tags_roundtrip/efso_vaults_N/results.json --counts 0 --mixed-counts "" \
  --budgets 40,60,80 --out fixtures/tags_roundtrip/equal_budget/results_efso_vaults_N.json
```

Two texts times three budgets make six cells. Expected, comparing means of three rewrites:

1. Gist recall: mixed tags above the spread extract in at least 5 of the 6 cells.
2. Exact fidelity: the spread extract above mixed tags in all 6 cells.
3. Gist recall: the spread extract above lead in most of the cells where their notes differ.
4. Exact recall: fact tags above topic tags in most of the cells where both run.
5. Control: gist recall at most 0.05 and exact recall at most 0.10 on both texts.

No direction is predicted for exact recall between mixed tags and the spread extract; it is
reported.

Known before the run, and to be read with any result:

- **Not the same input at the same word count.** Tags spend few words on function words, so at
  equal words they give the model more characters and tokens. Each rung records both. Topic
  tags stop at 32, which is 62 and 60 words, so topic at 80 is not at 80. On text 2, topic at
  80 repeats topic at 60 and is skipped.
- **Different pickers.** The tags were picked by gpt-4o-mini reading the whole text; the
  extracts by bag-of-words centrality. A win for tags is partly a win for the picker.
- **Extracts are scored in their own units.** Extract notes are source sentences, split exactly
  as recall is scored, so a rewrite that repeats them is credited for them automatically. The
  notes-only scores show that floor.
- **Tags not in text order.** The prompt says the notes are in text order; tags whose quote was
  not found in the text come last (0 to 3 per rung, recorded).
- **Text 2 has four code-fence lines,** each a recall unit (4 of 49). The exact judge has
  credited them in earlier rewrites that held no fence.
- **Different prompt from the earlier runs,** so these numbers do not line up with the tables in
  `../efso_vaults_1/` and `../efso_vaults_2/`.
- **Two texts, both by one author.** Text 1 is the text the judge settings were chosen on. A
  third text, a bill from `fixtures/meaning_receipts_billsum/`, was considered and not used:
  its hard line breaks split sentences into line fragments under the scorer's unitizer. The
  three rewrites of a condition often open alike, so they are not independent draws.

## Result (added after the run)

Both runs completed as stated, with the command above, and the result files are committed
unedited. Means of three rewrites, with the lowest and highest in brackets. "Words" is the
words the notes actually used. "Notes alone" scores the numbered notes, one per line, as if they
were the rewrite.

Text 1, EFSO Vaults #1:

| Notes | Words | Chars | Tokens | Gist recall | Exact recall | Gist fidelity | Exact fidelity | Notes alone: gist recall | Notes alone: exact recall |
|---|---|---|---|---|---|---|---|---|---|
| control | 0 |  |  | 0.00 [0.00-0.00] | 0.05 [0.02-0.10] | 0.00 | 0.02 |  | |
| topic@40 | 40 | 353 | 57 | 0.32 [0.24-0.37] | 0.11 [0.08-0.15] | 0.52 | 0.20 | 0.40 | 0.05 |
| fact@40 | 36 | 331 | 46 | 0.17 [0.16-0.18] | 0.06 [0.05-0.08] | 0.81 | 0.15 | 0.16 | 0.02 |
| mixed@40 | 39 | 337 | 51 | 0.26 [0.26-0.26] | 0.09 [0.03-0.13] | 0.62 | 0.36 | 0.29 | 0.05 |
| extract@40 | 40 | 248 | 49 | 0.16 [0.11-0.19] | 0.18 [0.16-0.21] | 0.60 | 0.18 | 0.11 | 0.06 |
| lead@40 | 40 | 237 | 52 | 0.09 [0.08-0.10] | 0.08 [0.06-0.10] | 0.56 | 0.11 | 0.06 | 0.03 |
| topic@60 | 60 | 539 | 85 | 0.26 [0.24-0.31] | 0.12 [0.08-0.16] | 0.57 | 0.33 | 0.42 | 0.05 |
| fact@60 | 58 | 518 | 77 | 0.25 [0.23-0.26] | 0.10 [0.08-0.11] | 0.82 | 0.07 | 0.27 | 0.03 |
| mixed@60 | 59 | 519 | 80 | 0.27 [0.21-0.32] | 0.13 [0.10-0.15] | 0.68 | 0.25 | 0.35 | 0.06 |
| extract@60 | 59 | 361 | 72 | 0.15 [0.15-0.15] | 0.20 [0.18-0.23] | 0.41 | 0.18 | 0.11 | 0.11 |
| lead@60 | 60 | 346 | 77 | 0.12 [0.11-0.15] | 0.10 [0.10-0.10] | 0.31 | 0.17 | 0.11 | 0.08 |
| topic@80 | 62 | 558 | 88 | 0.27 [0.26-0.31] | 0.15 [0.15-0.16] | 0.55 | 0.25 | 0.42 | 0.03 |
| fact@80 | 77 | 676 | 109 | 0.31 [0.29-0.32] | 0.16 [0.15-0.18] | 0.87 | 0.24 | 0.31 | 0.06 |
| mixed@80 | 79 | 703 | 106 | 0.26 [0.23-0.29] | 0.22 [0.21-0.24] | 0.64 | 0.31 | 0.44 | 0.06 |
| extract@80 | 80 | 458 | 99 | 0.25 [0.23-0.27] | 0.27 [0.26-0.31] | 0.55 | 0.25 | 0.19 | 0.13 |
| lead@80 | 79 | 472 | 98 | 0.18 [0.16-0.19] | 0.16 [0.15-0.18] | 0.54 | 0.12 | 0.18 | 0.11 |

Text 2, EFSO Vaults #2 (topic at 80 skipped, as stated):

| Notes | Words | Chars | Tokens | Gist recall | Exact recall | Gist fidelity | Exact fidelity | Notes alone: gist recall | Notes alone: exact recall |
|---|---|---|---|---|---|---|---|---|---|
| control | 0 |  |  | 0.00 [0.00-0.00] | 0.07 [0.04-0.10] | 0.00 | 0.01 |  | |
| topic@40 | 40 | 334 | 49 | 0.25 [0.18-0.33] | 0.16 [0.10-0.20] | 0.72 | 0.28 | 0.39 | 0.02 |
| fact@40 | 38 | 245 | 45 | 0.22 [0.20-0.24] | 0.16 [0.10-0.20] | 0.51 | 0.37 | 0.22 | 0.04 |
| mixed@40 | 37 | 276 | 46 | 0.26 [0.22-0.31] | 0.10 [0.10-0.10] | 0.41 | 0.25 | 0.37 | 0.02 |
| extract@40 | 39 | 194 | 49 | 0.06 [0.04-0.08] | 0.05 [0.04-0.08] | 0.20 | 0.05 | 0.02 | 0.02 |
| lead@40 | 39 | 194 | 49 | 0.05 [0.04-0.08] | 0.03 [0.02-0.04] | 0.22 | 0.06 | 0.02 | 0.02 |
| topic@60 | 60 | 466 | 72 | 0.29 [0.29-0.31] | 0.14 [0.14-0.14] | 0.68 | 0.17 | 0.41 | 0.04 |
| fact@60 | 59 | 404 | 71 | 0.24 [0.22-0.27] | 0.07 [0.06-0.08] | 0.58 | 0.40 | 0.31 | 0.04 |
| mixed@60 | 60 | 434 | 74 | 0.28 [0.18-0.35] | 0.12 [0.08-0.16] | 0.58 | 0.24 | 0.43 | 0.02 |
| extract@60 | 60 | 354 | 79 | 0.15 [0.12-0.16] | 0.17 [0.12-0.22] | 0.32 | 0.25 | 0.24 | 0.10 |
| lead@60 | 59 | 329 | 78 | 0.16 [0.16-0.16] | 0.14 [0.12-0.16] | 0.37 | 0.30 | 0.18 | 0.10 |
| fact@80 | 78 | 543 | 91 | 0.32 [0.31-0.33] | 0.12 [0.08-0.14] | 0.66 | 0.44 | 0.37 | 0.06 |
| mixed@80 | 76 | 582 | 95 | 0.27 [0.24-0.29] | 0.12 [0.10-0.12] | 0.62 | 0.23 | 0.45 | 0.04 |
| extract@80 | 80 | 447 | 101 | 0.30 [0.29-0.31] | 0.16 [0.10-0.20] | 0.78 | 0.45 | 0.24 | 0.10 |
| lead@80 | 79 | 430 | 98 | 0.20 [0.12-0.29] | 0.25 [0.16-0.35] | 0.32 | 0.37 | 0.20 | 0.10 |

On text 2 at 40 words, the extract and lead notes came out identical (the same one
sentence), so that pair is left out of expectation 3. The same notes, prompt and seeds gave
gist recall 0.06 against 0.05 and exact recall 0.05 against 0.03. That is a lower bound on
rerun noise, measured near the floor where scores move least. A rerun of EFSO Vaults #1 with
the same tags and seeds (`../efso_vaults_1/`) moved exact recall from 0.23 to 0.16, so
differences below about 0.07 are not read as results here. The expectations below are still
scored by their stated rule, and most of their margins are smaller than that.

Each expectation, from the unrounded means:

1. **Held by the stated rule (5 of 6), but weaker than it reads.** Mixed tags recalled more
   gist than the spread extract in four cells clearly (text 1 at 40 and 60, text 2 at 40 and
   60). On text 1 at 80 it is a tie (0.258 against 0.253, ranges overlapping). On text 2 at 80
   the extract is above (0.30 against 0.27). See also the gist finding below.
2. **Failed (2 of 6).** The spread extract's rewrites were more often supported by the source
   only on text 2 at 60 (0.25 against 0.24, a tie) and at 80 (0.45 against 0.23). In the other
   four cells the mixed-tag rewrites scored higher exact fidelity. Copied sentences did not
   keep the model from writing claims the exact judge cannot find in the source.
3. **Held (4 of 5).** The spread extract recalled more gist than lead on text 1 at all three
   budgets and on text 2 at 80; lead was above on text 2 at 60 (0.16 against 0.15).
4. **Failed (2 of 5).** Fact tags recalled more exact sentences than topic tags only on text 1
   at 80, where topic tags used 62 words against fact's 77, and on text 2 at 40 (0.163 against
   0.156, a tie). Topic tags were above in the other three cells.
5. **Held.** The no-tags control scored gist recall 0.00 on both texts and exact recall 0.05 on
   text 1 and 0.07 on text 2.

Not predicted, and the main finding: **tag notes alone out-score their own rewrites on gist
recall.** For topic and mixed tags, the notes read as the rewrite recalled more gist than the
rewrites written from them in all 11 cells (text 1 mixed at 80: 0.44 alone against 0.26
rewritten). Mixed rewrites stay near 0.26 to 0.28 at every budget, while their notes alone
climb from 0.29 to 0.45. Each tag is one unit on the rewrite side, and the gist judge counts an
original sentence as carried when any unit reaches cosine 0.5 with it, so a short phrase can
carry a sentence on the same topic. The gist judge therefore credits a bare tag
list, and expectation 1 holding may show the judge crediting tag words as much as the rewrite
bringing anything back. This also applies to the gist recall of tag
rewrites in `../efso_vaults_1/` and `../efso_vaults_2/`, and to the recorded card on the review
page. On the exact judge the order reverses: every tag list alone scored at most 0.06.

The open comparison (exact recall, mixed tags against the spread extract) depends on the floor.
Raw, the extract's rewrites recalled more exact sentences in 5 of 6 cells (all but text 2 at
40). But a rewrite that repeats extracted sentences is credited for them, and the extract
notes alone already score up to 0.13 exact recall (0.02 to 0.13), against at most 0.06 for mixed tags.
Subtracting each kind's notes-alone score, the extract is ahead in 2 of 6 cells (text 1 at 40:
0.11 against 0.04; text 1 at 60: 0.09 against 0.06) and mixed tags are ahead in the other four.
Only text 1 at 40 (0.18 against 0.09 raw) clearly exceeds the rerun noise above.

Read together: at 40 to 80 words, neither tags nor copied sentences bring back much of either
text: at most about a third of its sentences by gist (0.32) and a quarter exactly (0.27). Tags name more of
the text's topics than copied sentences of the same length. That shows up on the gist judge,
which credits naming. Copied sentences give the rewrite a few exact sentences for free. Once
that free credit is taken off, neither kind of note leads on exact recall consistently. Two texts
by one author, three rewrites per cell that are not independent, and differences mostly within
rerun noise: descriptive only.

One pre-run caveat did not hold as written: tags gave the model more characters than the
extract at every budget, but on text 2 the mixed and fact notes used fewer tokens than the
extract (mixed 46, 74 and 95 against 49, 79 and 101).
