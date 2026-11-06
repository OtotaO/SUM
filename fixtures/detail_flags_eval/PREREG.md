# Critical-detail flags: pre-registered evaluation

Registered 2026-11-06, before any detector run on this corpus. This implements the
playbook's Horizon 1 bullet "Workbench first-touch value": deterministic, in-browser flags for changed or
missing numbers, dates, negations, modals, conditions, exceptions and names, whose false-alarm and recall
evaluation is registered before results.

## Corpus (`corpus_v1.json`, frozen by this commit)

- SHA-256 of `corpus_v1.json`: `02028e35c3f7d931b1a194a6fec0c4b276b106e0fe423872b729de7f9724127b`.
- 144 synthetic pairs of an original notice and a plain-language rewrite, in ten everyday settings (consumer
  terms, leases, patient instructions, workplace policy, insurance, school notices, travel refunds, bank fees,
  benefits letters, warranties). All names are invented.
- 84 detail-altered pairs: exactly one change each, 12 per category (number, date, negation, modal, condition,
  exception, name), 4 each of changed, dropped and added. 60 detail-preserving pairs: no change.
- Authorship: written by Claude subagents from the category and label rules only, each shard checked by a
  separate validator agent; they were told not to read any repository file. The detector was drafted at the
  same time, against hand-made examples only. Its author has not read the pairs and will not before the single
  run below; the validators' short summaries, which mention a few details, were seen. This is enforced by
  discipline, not mechanism.
- Label rule for hard cases, fixed now: a value written differently but equal (30 and thirty, 14 days and two
  weeks, 50% and half), the same date in another format, a modal replaced within its class (obligation,
  permission, prohibition, recommendation, commitment), the same polarity in other words, a condition or
  exception with the same logic, and a shorter or longer form of the same name are preserved. Approximation or
  rounding (30 days and about a month, $24.99 and $25), a modal moved across classes (may and must, must and
  should), weakened polarity (never and rarely), and a condition, exception or name that differs, is removed or
  is added are changes.
- Known limits: synthetic text, short notices (3 to 6 sentences), one change per altered pair, English only.
  Results describe this corpus; they are not an estimate for any population of real rewrites.

## Measures (computed by `score.mjs`, frozen by this commit)

- A flag is an object with exactly the fields `category` (one of the seven), `kind`, `source` and `output`; `source`
  is a span of the original and `output` a span of the rewrite, each a non-empty `{start, end}` inside its text (an
  optional `text` field is ignored) or null, in UTF-16 code units, and at least one is a span. `score.mjs` refuses malformed input. A flag credits an altered
  pair if a span of at most 80 code units overlaps the changed detail: the source detail for changed and
  dropped details, the rewrite detail for changed and added details.
- Recall (primary): altered pairs with at least one credited flag, out of 84.
- False-alarm rate (primary): preserving pairs with at least one flag of any kind, out of 60.
- Secondary, reported but not decisive: recall with the category also correct; recall by category and by
  change type; mean flags per altered and per preserving pair.

## Decision rule (from the playbook, fixed now)

If recall is below 50% or more than half of the detail-preserving pairs are flagged, the workbench ships a
neutral detail table (details found in each text, side by side, with no "changed" claim) instead of flags.
Otherwise the flags ship, labelled as lexical checks that can miss changes and can flag preserved details.

## Procedure

The detector is written against hand-made examples only. When it is complete, it runs once over
`corpus_v1.json`, receiving only each pair's `original` and `rewrite` text; the flags and the `score.mjs` output are committed with the detector, whatever they show.
No threshold, rule or corpus pair is changed after that run. Any later detector version is evaluated on a new
corpus registered the same way.
