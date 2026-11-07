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
