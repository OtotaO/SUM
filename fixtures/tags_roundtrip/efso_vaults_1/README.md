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
