# Depth reader gate

One question, asked once on texts nobody has tuned on: does the depth reader's
**brief** view (about a tenth of a text) keep the sentences a reader marked as
must-read more often than simply reading the opening of the text at the same
length?

If it does not, the depth reader's ranking is not earning its place, and the
page should fall back to a plain outline view (headings and the first sentence
of each block).

`depth_gate.mjs` is the harness. It was written and committed before any
marked text existed. The texts and marks stay private; nothing it writes
contains their wording.

## The rule

Proposed on 2026-11-22. **The operator's answer is pending.** The held-out run
does not happen until the operator says yes to this rule or changes it.

- 8 texts, split by the SHA-256 of each text file's bytes: the 4 lowest values
  are dev, the 4 highest are held out.
- On the 4 held-out texts, pool the marks. The depth reader **passes** when its
  brief keeps at least half of the pooled marks (`--min-recall 0.5`) **and**
  strictly more marks than the lead arm. A tie fails.
- The held-out set is scored once. The command refuses to run when its output
  file already exists.
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
   file's name.

## What is compared

Each arm is a set of character ranges of the text, at a budget of a share of
the text's passage characters (passages are the depth reader's own sentences).
The gate reads the 0.1 share. Shares 0.25 and 0.5 are reported, never gated.

- **distill**: the passages in the depth reader's own view at that depth. When
  the reader dropped a depth because it repeated the one before, the one before
  is used.
- **lead**: the passages in text order, cut at exactly the budget, so the last
  one may end mid-sentence. It always uses the whole budget.
- **outline**: block openers in text order (the first passage, every heading
  line, the first passage after a heading, the first passage after a blank
  line), then the other passages in text order. It takes each whole passage
  that still fits and passes over one that does not.

The reader's own rule stops at its first ranked passage that does not fit, so
it can use less of the budget than lead, and more only when its first
passage alone is longer than the budget. The comparison leans against the
reader.

A mark counts as kept only when every non-whitespace character of it lies
inside the arm's ranges.

## Commands

```
node fixtures/depth_gate/depth_gate.mjs check   DIR
node fixtures/depth_gate/depth_gate.mjs dev     DIR
node fixtures/depth_gate/depth_gate.mjs heldout DIR --out fixtures/depth_gate/heldout_result.json --min-recall 0.5
```

The record holds, for each text, its SHA-256, the SHA-256 of its marks file,
its length, and for each share and arm the characters used, the number of
ranges and the number of marks kept; then the pooled counts and, for the
held-out run, the verdict.

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
