# Critical-detail flags: second registration (checker v1.1)

Registered 2026-11-06, before any run on `corpus_v2.json`. `PREREG.md` registered the first evaluation; its
measures, label rule and decision rule apply here unchanged, with `corpus_v2.json` in place of `corpus_v1.json`
and the checker named below. Of its procedure, the single run and the rule that nothing changes after it apply;
the clause that the detector is written against hand-made examples only does not, because v1.1 was developed on
`corpus_v1.json` (see below).

- Corpus: `corpus_v2.json`, SHA-256 `2a694486ac1529a152127ca8b5fdf618cd7c6d8e1aec34d407ed905a34b0035e`. 144
  synthetic pairs (84 detail-altered, 4 per category and change type; 60 detail-preserving) written and
  validated by a fresh run of the same agent procedure, from the same rules. Pair ids start with `v2-`.
- Checker: `single_file_demo/detail_flags.js` with method `sum.detail_flags.lexical_v1.1`, SHA-256
  `5b22b253fcac96741b5e9863b4886ef4c346538735c310a0c213fef46a9efd3e`, frozen before this registration. It was
  developed on `corpus_v1.json` after the v1 result was committed, so v1 is its development set and its v1
  numbers are not an evaluation. Its author has not read `corpus_v2.json`; the validators' short summaries,
  which mention a few names and details, were seen. This is enforced by discipline, not mechanism.
- Changes from v1, all fixed before this registration: no failure on "half" without a following word; bounded
  running time (crafted inputs that took from 14 to over 100 seconds under v1 took under 0.6 seconds at the
  100,000-character limit in the author's and reviewers' measurements, within the 300-span limit); hours, days and weeks compared as one unit; "$1,000" equals "1000" and
  "$2 million" equals "$2,000,000"; rates ("$10 a month", "twice a day") no longer count "a month" or "a day"
  as a duration; "once" counts only as a frequency; relative dates ("the end of the month"); "should not" is a
  recommendation, not a prohibition; "when" is no longer a condition; a kept condition or exception whose
  clause loses at least two content words, and at least half of them, is flagged as changed.
- The run happens once, with the checker above, receiving only each pair's original and rewrite; the flags
  and score are committed as produced, and the page then states the v2 result for v1.1.
