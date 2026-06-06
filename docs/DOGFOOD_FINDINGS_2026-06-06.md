# Dogfood findings — 2026-06-06 (`sum frontier` on real prose)

A deliberate dogfood session: use `sum frontier` as a first-pass writer
would, on a real ~120-word paragraph, with three versions a real writer
would actually produce (a faithful extractive trim, a faithful
paraphrase, a one-line tag). Goal per the fresh-eyes audit: stop building
substrate, *use* the thing, produce a falsifiable finding.

It produced one — and it changes the priority.

## F18 (load-bearing): the lexical scorer *misranks* paraphrase, not just over-reports it

Measured meaning-loss, source paragraph about the printing press:

| Version | Human judgement | `meaning_loss` (lexical) |
|---|---|---|
| source (identity) | — | 0.000 |
| extractive trim (keeps source words) | ok | **0.394** |
| faithful paraphrase (best gist, reworded) | **best compression** | **0.762** |
| one-line tag (discards almost all) | worst | 0.742 |

A human ranks preservation: paraphrase > extractive trim > tag. The tool
ranks loss: extractive (0.394) < tag (0.742) < paraphrase (0.762) — i.e.
it says the **faithful paraphrase is the *most* lossy of the three**,
worse even than the near-empty tag. The ranking is **inverted** for the
exact use-case the product vision names ("how much meaning did I lose
compressing this?", PRODUCT_VISION §7).

Root cause is known and named in the code (the `LexicalCoverageScorer`
is bidirectional content-word overlap; a reword shares few words with the
source, so it scores as both high *drop* and high *fabrication*). The
existing caveat said it "over-reports loss on a faithful reword." The
dogfood shows that is **understated**: it doesn't just over-report, it
**reorders** — it would actively steer a writer toward the worse summary.

**Implication (the priority change):** the `EntailmentScorer` (an NLI /
LLM judge, currently `$`/operator-gated and deferred) is **not an
optional enhancement — it is the load-bearing requirement** for
`sum frontier` to be useful on real (paraphrased) writing at all. With
only the lexical scorer, the tool is trustworthy *only* for extractive
(sentence-dropping) compression, and is misleading for paraphrase.

**Do NOT** promote `sum frontier` to users — nor ship an MCP/Worker
`frontier` tool (rung B) — with only the lexical scorer as the default,
without a loud, specific warning. A misleading instrument in a user's
hands is worse than no instrument. This is the proof-boundary discipline
applied to product: the number must not mislead.

## F19 (ergonomics): `--version` takes files; a writer has text

To compare versions, each must be written to a file and passed as
`--version PATH`. A writer has *text* (drafts in an editor), not a
directory of files. Round-tripping through temp files is real friction.
Lower priority than F18 — but fixing ergonomics on an instrument that
misranks (F18) would be polishing the wrong thing. Defer until F18 is
addressed.

## F20 (vision gap): the offline CLI scores, it does not render

The product vision is "drop text → slide → watch it summarize." The
shipped `sum frontier` is **bring-your-own-versions**: it scores
renderings the user already has; it cannot *generate* the
faithful↔compressed path offline (that needs the slider's LLM render
path, `$`/operator-gated). So the "drop text and slide" magic is gated on
the same LLM the entailment scorer is gated on. The honest offline story
today is "score versions you already made," not "summarize my text."

## What this session changes

The fresh-eyes audit said: get a real human using it, get a falsifiable
finding, let signal drive the next move. The signal is now in:

- The **binding constraint is the judge**, not more substrate. Both the
  useful scorer (F18) and the "drop text and slide" render path (F20) are
  the *same* `$`/operator-gated pull. Building more format/UI/MCP surface
  on top of the lexical scorer is motion; it does not move that
  constraint.
- The **immediate, no-`$` response** (this PR): sharpen the user-facing
  honesty so the lexical scorer cannot mislead — the CLI now warns, when
  `scorer=lexical`, that it misranks paraphrase and is trustworthy only
  for extractive compression. The number stays; the framing stops
  steering a writer wrong.
- The next **real** move is operator/`$`-gated: provide a deterministic
  NLI judge (a local model, or a temperature-0 LLM) so `EntailmentScorer`
  can be exercised on real prose — that is when the tool becomes useful,
  and the meaning-risk receipt becomes meaningful on paraphrase. Until
  that pull, the honest position is: the tool is for extractive
  compression, clearly labelled.
