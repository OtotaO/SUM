# Depth reader gate

One question, asked once on texts nobody has tuned on: does the depth reader's
**brief** view (about a tenth of a text) keep the sentences a reader marked as
must-read more often than simply reading the opening of the text at the same
length?

If it does not, the depth reader's ranking is not earning its place, and the
page should fall back to a plain outline view (headings and the first sentence
of each block).

`depth_gate.mjs` is the harness. It was written and committed before any
marked text existed. The texts and marks stay private. The JSON record holds
hashes and counts, never their wording; error messages name a file and a mark's
line number only; `check` prints file names.

## The rule

Proposed on 2026-11-22. **The operator's answer is pending.** The held-out run
does not happen until the operator says yes to this rule or changes it.

- 8 texts, split by the SHA-256 of each text file's bytes: the 4 lowest values
  are dev, the 4 highest are held out.
- On the 4 held-out texts, pool the marks. The depth reader **passes** when its
  brief keeps at least half of the pooled marks (`--min-recall 0.5`) **and**
  strictly more marks than the lead arm. A tie fails.
- Before the dev run, `check --write-split` freezes the SHA-256 of every text
  and marks file, and that file is committed. `dev` and `heldout` refuse a
  folder whose files differ from it by a byte, so the split cannot be moved
  after a dev run.
- The held-out set is scored once. The command refuses to run when its output
  file already exists; that guards against an accident, and the commit history
  is the record that it ran once.
- Dev runs may lead to changes in the depth reader before the held-out run.
  The held-out record carries the SHA-256 of `distiller.js` and its method ID,
  so the result applies to that version only.

## How to mark the texts

1. Pick 8 long texts that have never been opened in the depth reader or the
   review page. Each must be plain text or Markdown, UTF-8, under 100,000
   characters.
2. Put them in one folder **outside this repository**, each as `NAME.txt` or
   `NAME.md`.
3. Before looking at any view of a text, read it and copy 3 to 5 sentences
   someone must read to get it into `NAME.marks.txt` next to it, one per line.
   Copy them character for character from the text (line breaks inside a
   sentence may become single spaces; quotes, dashes and case must match).
   Each mark must occur exactly once in its text, and marks must not overlap.
4. Check the folder. This validates every text and mark and prints the split;
   it scores nothing:

   ```
   node fixtures/depth_gate/depth_gate.mjs check /path/to/folder
   ```

   A mark that is missing, repeated or overlapping stops the check with the
   file's name and the mark's number (counting non-blank lines of the marks
   file). Fix the marks file and check again; once it passes, do not edit the
   texts or marks again.

## What is compared

Each arm is a set of the depth reader's own passages (its sentences), chosen
by the reader's own budget rule: the longest prefix of the arm's ordering whose
passages fit a share of the text's passage characters, and always at least one
passage. Only the ordering differs, so the test isolates the reader's ranking.
The gate reads the 0.1 share. Shares 0.25 and 0.5 are reported, never gated.

- **distill**: the reader's ranking, so the arm is the reader's own view at
  that depth. When the reader dropped a depth because it repeated the one
  before, the one before is used.
- **lead**: the passages in text order.
- **outline**: block openers in text order (the first passage, every heading
  line, the first passage after a heading, the first passage after a blank
  line), then the other passages in text order.

The characters each arm uses differ by text and are recorded. An arm goes over
the budget only when its first passage alone is longer than it; such rows are
flagged, and the verdict counts them beside the result rather than dropping
them.

A mark counts as kept only when every non-whitespace character of it lies
inside the passages the arm kept.

## Commands

```
node fixtures/depth_gate/depth_gate.mjs check   DIR --write-split fixtures/depth_gate/split.json
node fixtures/depth_gate/depth_gate.mjs dev     DIR --split fixtures/depth_gate/split.json
node fixtures/depth_gate/depth_gate.mjs heldout DIR --split fixtures/depth_gate/split.json --out fixtures/depth_gate/heldout_result.json --min-recall 0.5
```

The record holds, for each text, its SHA-256, the SHA-256 of its marks file,
its length, and for each share and arm the characters used, the number of
passages, whether it went over the budget and the number of marks kept; then
the pooled counts, the hashes of `distiller.js`, the harness and the split
file, and, for the held-out run, the verdict.

## What this does not show

- Four held-out texts and one marker. The marker is also the project's author.
  A pass is a reason to keep going, not evidence that the reader helps people
  in general.
- Kept marks are not usefulness. A view can keep every marked sentence and
  still read badly.
- The numbers carry no confidence interval; with about 16 pooled marks, one
  mark moves recall by about 0.06.

Tests: `fixtures/depth_gate/test_depth_gate.mjs`, run by `npm test` in
`single_file_demo/`.
