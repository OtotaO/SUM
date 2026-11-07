# Worked example: EFSO Vaults #2

`source.md` is the text of "EFSO Vaults #2: Oh the Humanitas", published 2024-05-10 at
https://ototao.substack.com/p/efso-vaults-2 by the project's operator, who sent it on 2026-11-07
as a further worked example for `scripts/tags_roundtrip.py`. The text was taken from the published
page on 2026-11-07: paragraph text and the two code blocks, which hold a quoted transcript of
output from Nous Research's World_sim simulator as published in the post; the closing image and the
horizontal rule after the last paragraph are not included. 464 words, SHA-256
`665921ff21f62d93f3fa07c340e16d0dda707c679d5dcd4dc183472002475ae2`.

Unlike EFSO Vaults #1, this text played no part in choosing the judge settings (window size and the
two thresholds) or the cut rule below, all of which were fixed on EFSO Vaults #1 and are unchanged.

## The cut, fixed before this run

`--cut` scores a second version of every rewrite: only the sentences that the tags themselves entail.
The tags given to that rewrite are written as sentences ("The text is about X." for a topic tag,
"The text uses X." for a form tag, the phrase itself for a fact tag), and a rewrite sentence is kept
if some 80-token window of them entails it at probability 0.5 or more under the exact judge's model.
The source is not consulted. A cut that keeps nothing counts as recall 0 and is left out of fidelity.

On the stored second-run rewrites of EFSO Vaults #1 (development, two rewrites per condition), this
cut kept 2 to 8 of 25 to 61 sentences, raised exact fidelity in 7 of 8 rewrites (one fell to 0.00)
and lowered exact recall in all 8. A looser rule (gist similarity to some tag of at least 0.5 or 0.6) kept more but
moved fidelity both ways and was not chosen.

## Stated before the run

Written and committed before `results.json` was produced. One run with `--cut`: the no-tags control,
topic tags at 4, 8, 16 and 32, mixed tags at 8, 16 and 32, three rewrites each.

1. The judges do not credit unrelated prose on this text either: control mean gist recall is at most
   0.05 and control mean exact recall at most 0.10.
2. At mixed 32, the cut's mean exact fidelity (over non-empty cuts) is higher than the uncut mean
   exact fidelity.
3. At mixed 32, the cut's mean exact recall is lower than the uncut mean exact recall.
4. Topic 32 has higher mean gist recall than topic 4.
5. Mixed 32 has a higher mean first-person share than topic 32.

One text, three rewrites per condition that may open alike: any difference is descriptive.

## Result (added after the run)

From `results.json`. Means of three rewrites, with the range in brackets. The source's first-person
share is 0.056. The last three columns are the cut: sentences kept per rewrite, its exact recall, and
its exact fidelity over the cuts that kept anything (n of 3).

| Condition | Tags | Tag words | Gist recall | Exact recall | Gist fidelity | Exact fidelity | First-person share | Cut: kept | Cut: exact recall | Cut: exact fidelity |
|---|---|---|---|---|---|---|---|---|---|---|
| control | 0 | 0 | 0.00 [0.00-0.00] | 0.03 [0.02-0.04] | 0.00 | 0.01 | 0.008 | - | - | - |
| topic | 4 | 6 | 0.14 [0.12-0.14] | 0.12 [0.08-0.18] | 0.72 | 0.15 | 0.027 | 0.3 | 0.00 | 1.00 (n=1) |
| topic | 8 | 15 | 0.24 [0.22-0.27] | 0.10 [0.08-0.12] | 0.73 | 0.18 | 0.024 | 3.0 | 0.02 | 0.40 (n=3) |
| topic | 16 | 27 | 0.25 [0.24-0.27] | 0.11 [0.10-0.12] | 0.63 | 0.21 | 0.028 | 3.0 | 0.03 | 0.81 (n=3) |
| topic | 32 | 60 | 0.24 [0.22-0.27] | 0.09 [0.08-0.10] | 0.69 | 0.25 | 0.036 | 5.3 | 0.13 | 0.53 (n=3) |
| mixed | 8 | 18 | 0.12 [0.10-0.14] | 0.05 [0.02-0.08] | 0.26 | 0.09 | 0.099 | 0.7 | 0.00 | 1.00 (n=2) |
| mixed | 16 | 41 | 0.23 [0.22-0.24] | 0.18 [0.10-0.33] | 0.43 | 0.03 | 0.096 | 0.3 | 0.00 | 1.00 (n=1) |
| mixed | 32 | 76 | 0.40 [0.35-0.45] | 0.27 [0.14-0.49] | 0.52 | 0.24 | 0.092 | 7.3 | 0.02 | 0.81 (n=3) |

Against the five expectations:

1. Control gist recall 0.00 and exact recall 0.03, within the stated limits (0.05 and 0.10). Held: on
   a text they were not tuned on, the judges still gave unrelated prose almost no credit.
2. Mixed 32: the cut's exact fidelity 0.81 against 0.24 uncut. Held.
3. Mixed 32: the cut's exact recall 0.02 against 0.27 uncut. Held. Of about 40 sentences per rewrite,
   the cut kept 7.3 on average.
4. Topic 32 gist recall 0.24 against topic 4's 0.14. Held, though topic 8, 16 and 32 are all about
   0.24 to 0.25: past 8 tags, more topic tags added no gist here.
5. Mixed 32 first-person share 0.092 against topic 32's 0.036. Held. As on EFSO Vaults #1, mixed
   overshoots the source (0.056) and topic undershoots it.

Not predicted, seen here: at 32 tags mixed recalled clearly more gist than topic tags alone (0.40
[0.35-0.45] against 0.24 [0.22-0.27]; the ranges do not overlap), unlike EFSO Vaults #1, where the
ranges overlapped. The best mixed rewrite (seed 7) recalled 49% of the source's sentences on the
exact judge; its first paragraph follows the essay's own sequence (staying up late, "a bit of binary
banter", sleeping until 9, inbox zero, starting a simulation). The same confounds as before apply to
any topic-against-mixed comparison: different prompts and more tag words (76 against 60).

On the cut: at topic 32 its exact recall (0.13) is higher than the full rewrite's (0.09), which a
subset of the rewrite's sentences cannot really achieve; the exact judge reads the shorter text in
different windows, so differences of that size on this judge are noise. Where the cut kept one or
two sentences, its fidelity of 1.00 rests on one or two sentences.

In every condition the three rewrites open with the same sentence, so they are not independent draws
and the ranges understate how much a rewrite can vary. One text: descriptive only.
