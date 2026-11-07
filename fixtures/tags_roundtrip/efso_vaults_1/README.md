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
range in brackets. The source's first-person share is 0.046 and it averages 10.9 words per sentence.

| Condition | Tags | Tag words | Gist recall | Exact recall | Gist fidelity | Exact fidelity | First-person share | Words per sentence |
|---|---|---|---|---|---|---|---|---|
| control | 0 | 0 | 0.00 [0.00-0.00] | 0.05 [0.03-0.10] | 0.00 | 0.02 | 0.010 | 14.2 |
| topic | 4 | 9 | 0.12 [0.10-0.13] | 0.05 [0.03-0.08] | 0.56 | 0.04 | 0.012 | 20.2 |
| topic | 8 | 17 | 0.24 [0.23-0.26] | 0.06 [0.05-0.08] | 0.56 | 0.10 | 0.012 | 21.5 |
| topic | 16 | 32 | 0.33 [0.27-0.37] | 0.16 [0.13-0.18] | 0.57 | 0.26 | 0.008 | 20.4 |
| topic | 32 | 62 | 0.25 [0.23-0.27] | 0.18 [0.16-0.21] | 0.42 | 0.23 | 0.030 | 20.5 |
| mixed | 8 | 22 | 0.16 [0.13-0.18] | 0.10 [0.08-0.11] | 0.49 | 0.02 | 0.078 | 19.5 |
| mixed | 16 | 43 | 0.22 [0.19-0.24] | 0.11 [0.10-0.13] | 0.30 | 0.06 | 0.086 | 14.4 |
| mixed | 32 | 83 | 0.27 [0.26-0.29] | 0.23 [0.19-0.26] | 0.42 | 0.13 | 0.074 | 13.6 |

Against the three expectations at 32 tags:

1. Gist recall: mixed 0.27, topic 0.25. Held, but the ranges overlap (0.26-0.29 and 0.23-0.27), so
   three rewrites cannot separate them.
2. Exact recall: mixed 0.23, topic 0.18. Held.
3. First-person share: mixed 0.074 and topic 0.030, against the source's 0.046. Did not hold: mixed
   overshoots by 0.028, topic undershoots by 0.016.

Not predicted, seen here: mixed rewrites come much closer to the source's sentence length (13.6 words
against 10.9; topic 20.5), and they read as a first-person story where topic rewrites read as an
essay about the topics. Their exact fidelity is lower at every count (0.02 to 0.13 against 0.10 to
0.26), meaning more of what they write is not entailed by the source. At 8 and 16 tags they recover
less gist than topic tags alone. Mixed also carries more tag words at every count, the confound
stated before the run.

The three rewrites of each condition open with the same sentence (temperature 0, seeds 7 to 9), so
they are not independent draws, and the ranges above understate how much a rewrite can vary.
