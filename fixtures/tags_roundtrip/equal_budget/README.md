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
