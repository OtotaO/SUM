# Worked example: EFSO Vaults #1

`source.md` is the text of "EFSO Vaults #1: Wholly Mammon Discovered", published 2024-04-30 at
https://ototao.substack.com/p/efso-vaults-1 by the project's operator, who agreed on 2026-11-06 to
its use here as the public worked example for `scripts/tags_roundtrip.py`. The text was taken from
the published page on 2026-11-06 (paragraph text only; the closing image is not included). It
contains a quoted transcript of output from Nous Research's World_sim simulator, as published in the
post. 667 words, SHA-256 `fc7824200e4293fe98ae8b171546ec9d089740493cc95c8a7f179cf2241dc147`.

This text is also the development text on which the judge settings in `scripts/tags_roundtrip.py`
were chosen (the window size and the two thresholds), so results on it are not an evaluation of
those settings.

## Stated before the run

Written and committed before `results.json` was produced. The question: at the same tag count, do
tags that also say how the text is written (form) and what it states (fact) bring back more of it
than topic tags alone?

Conditions, three rewrites each: the no-tags control; topic tags alone at 4, 8, 16 and 32; mixed at
8, 16 and 32 (half topic, a quarter form, a quarter fact, merged in the order their evidence appears).

Expected at 32 tags, comparing mixed with topic alone:

1. Mean gist recall is higher for mixed.
2. Mean exact recall is at least as high for mixed.
3. The mean first-person share of the rewrites is closer to the source's for mixed.

Known confound, stated now: fact tags are up to 8 words and topic and form tags up to 4, so mixed
carries more tag words at the same tag count. `results.json` records the tag words of every
condition. With one text and three rewrites per condition, any difference is descriptive, not a
finding about other texts.

## Result (added after the run)

From `results.json`, produced by the commit after the prediction. Means of three rewrites, with the
range in brackets. The source's first-person share is 0.049 and it averages 10.8 words per sentence.
The two form-signal columns were recomputed from the stored rewrites after review found that the
first version of the counter treated curly and straight apostrophes differently and did not count
first-person contractions (I'm, we've, let's); tags, rewrites and judge scores are the run's
original output.

| Condition | Tags | Tag words | Gist recall | Exact recall | Gist fidelity | Exact fidelity | First-person share | Words per sentence |
|---|---|---|---|---|---|---|---|---|
| control | 0 | 0 | 0.00 [0.00-0.00] | 0.05 [0.03-0.10] | 0.00 | 0.02 | 0.010 | 14.1 |
| topic | 4 | 9 | 0.12 [0.10-0.13] | 0.05 [0.03-0.08] | 0.56 | 0.04 | 0.012 | 20.2 |
| topic | 8 | 17 | 0.24 [0.23-0.26] | 0.06 [0.05-0.08] | 0.56 | 0.10 | 0.012 | 21.5 |
| topic | 16 | 32 | 0.33 [0.27-0.37] | 0.16 [0.13-0.18] | 0.57 | 0.26 | 0.008 | 20.4 |
| topic | 32 | 62 | 0.25 [0.23-0.27] | 0.18 [0.16-0.21] | 0.42 | 0.23 | 0.030 | 20.5 |
| mixed | 8 | 22 | 0.16 [0.13-0.18] | 0.10 [0.08-0.11] | 0.49 | 0.02 | 0.079 | 19.4 |
| mixed | 16 | 43 | 0.22 [0.19-0.24] | 0.11 [0.10-0.13] | 0.30 | 0.06 | 0.089 | 14.2 |
| mixed | 32 | 83 | 0.27 [0.26-0.29] | 0.23 [0.19-0.26] | 0.42 | 0.13 | 0.076 | 13.4 |

Against the three expectations at 32 tags:

1. Gist recall: mixed 0.27, topic 0.25. Held, but the ranges overlap (0.26-0.29 and 0.23-0.27), so
   three rewrites cannot separate them.
2. Exact recall: mixed 0.23, topic 0.18. Held.
3. First-person share: mixed 0.076 and topic 0.030, against the source's 0.049. Did not hold: from
   the unrounded means, mixed overshoots by 0.027 and topic undershoots by 0.019.

Not predicted, seen here: mixed rewrites come much closer to the source's sentence length (13.4 words
against 10.8; topic 20.5), and every mixed rewrite opens as a first-person story ("I remember the
moment I decided..." at 32 tags) where every topic rewrite at 32 tags opens as an essay ("In a world
increasingly shaped by..."). Their exact fidelity is lower at every count (0.02 to 0.13 against 0.10
to 0.26), meaning more of what they write is not entailed by the source. At 8 and 16 tags they
recover less gist than topic tags alone.

Two confounds limit what this shows. Mixed carries more tag words at every count, as stated before
the run. And the mixed rewrites used a different prompt, which labels each tag by school and tells
the model to follow the form tags; the prediction did not name this. So the change in voice comes
from the form tags and that instruction together, and this run cannot separate them.

The three rewrites of each condition open alike: in topic 4 and all three mixed conditions their
first sentences are identical, and in the others they differ in wording (temperature 0, seeds 7 to
9). They are not independent draws, and the ranges above understate how much a rewrite can vary.

## Second run: stated before the run

Written and committed before `results_run2.json` was produced. It reuses the tags in `results.json`
(`--tags-from`), so the tags are identical across conditions and only the rewrite prompt changes.
All conditions use 32 tags (the mixed ones share them as 16 topic, 8 form, 8 fact) and three
rewrites each, in one run:

- control: no tags.
- topic: topic tags, plain prompt.
- mixed: mixed tags, the prompt from the first run (labels, and "Follow the form tags for voice and
  structure").
- mixed_plain: mixed tags, the same prompt without that instruction.
- mixed_grounded: mixed tags, the first-run prompt plus "Write only what the tags state or directly
  imply; where they leave a gap, leave it out instead of filling it", with "at most about" the
  source's length.

Expected, comparing means:

1. mixed_grounded has higher exact fidelity than mixed.
2. mixed_grounded has gist recall no higher than mixed.
3. mixed_plain has a first-person share strictly between topic's and mixed's.

The first two ask whether an instruction can cut invention, and at what cost in recall. The third
asks whether the voice change in the first run came partly from the instruction rather than from the
form tags alone. As before: one text, the judge's development text, rewrites that may open alike.
