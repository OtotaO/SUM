# NORTH STAR — the standing doctrine

**Written 2026-07-03 by Claude Fable 5, at the operator's request, to outlive both of us.**

This document exists so that any future session — any model, any capability tier,
any context — can pick up this project and neither drift nor stall. It is the
compass digest. The operational charter (`docs/CHARTER_2026-05-17.md`) remains the
detailed map; where they conflict, this document wins, because it encodes the
corrections learned *after* the charter was written, several of them at real cost.

Read this before writing a line of code or a word of outreach.

---

## 1. The Destination

SUM exists to become the **omni-directional, master-slider, omni-format knowledge
distiller**: any document, any altitude, any perspective — tags to tomes and back —
where every transformation carries a **signed, replayable, honestly-bounded
certificate of what the transformation did to meaning**.

The certificate layer (receipts, conformal bounds, cross-runtime verification) is
the **fundable rung and the moat**. It is **not the destination**. Provenance
quietly becoming the whole project is a named failure mode. Every quarter, ask:
does a stranger experience the *distiller* anywhere, or only the certificate?

## 2. The Wager

The project's bet, stated falsifiably: **there exist real people who need to prove
to a second party what an AI transformation did to a text's meaning, and nobody
else will sign that statement honestly.** The entire provenance market signs
bytes and identity; it explicitly disclaims meaning. SUM signs a *named proxy for
meaning* under a *stated statistical bound* with *every limitation printed on the
label*.

If the wager is wrong — if no disputant-bearing user exists — the honest move is
to say so and fold the certificate into the distiller as one feature, not to keep
polishing an unclaimed moat. Empty ground is only a moat if someone ever stands
on it. **Absence of competition is not evidence of demand.** (This non-sequitur
was formally killed on 2026-06-27. Do not let "the moat is confirmed white-space"
smuggle it back.)

## 3. Invariants — never break these, no exceptions, no cleverness

1. **Exact math only in signed fields.** A signed receipt field carries nothing
   that was not computed, replayably, from committed inputs. Descriptive prose
   ("zero overlap", "preserving", "faithful") never rides in a signed field or a
   headline unless the descriptive claim was itself *measured*.
2. **Every number ships with its scope.** ρ = 0.267–0.291 without "pooled
   summary-level, SummEval" is an overclaim by omission — this exact omission
   survived in our own headline for weeks. State: corpus, aggregation level,
   scorer, n. If a number's scope is unknown, the number is not ready.
3. **The bound is marginal, under exchangeability, over a named proxy.** Never
   per-document, never a guarantee of meaning, never extended past the
   calibration envelope. The `not_covered` field and the proxy caveat are
   load-bearing product features, not legal boilerplate.
4. **Adversarial audit before anything becomes public.** Every outbound text,
   every release, every paper claim gets an independent hostile pass first. This
   has caught real errors that tests and self-review missed, every single time
   it has been run. It is the cheapest insurance this project owns.
5. **Verify before trust — including your own subagents and your own memory.**
   Prior sessions' conclusions are hypotheses. Check the live state: `gh` calls,
   PyPI JSON, real command runs. This session alone found four dead beliefs
   being treated as facts (a merged PR described as open, a login described as
   missing, a bug described as live, a guard described as absent).
6. **Report failures plainly.** A red test, a skipped step, a partial result is
   stated as such, immediately, with the output. Nothing is described as done
   that is not done and verified.
7. **Repo mechanics that are not optional:** stage explicit paths (never
   `git add -A` — it has leaked private drafts to the public repo);
   `git pull --ff-only` before branching; README/CHANGELOG/PROOF_BOUNDARY edits
   require the `scripts/attest_repo_docs` refresh in the same commit; version
   bumps require the `repo_manifest` refresh; outbound text passes
   `scripts/lint_outbound_text.py` (no em dashes in outbound, no credential
   echoes, no stray command lines).

## 4. The Traps — corrected definitions, each paid for

**4a. The substrate-velocity trap (corrected).** The trap is building what no
named human asked for. It is **not** a license to leave the product half-built.
For weeks this repo simultaneously recorded "everyone verifies, nobody mints" as
the #1 adoption blocker *and* vetoed building the mint door as "substrate
velocity." That was self-sabotage wearing discipline's clothes. The corrected
test: **a missing half of an already-shipped loop is product completion — build
it; a new loop with no named puller is substrate — don't.** Ask: "does this
complete something a stranger already touches, or start something nobody asked
for?"

**4b. The thought-terminating cliché.** "Substrate-velocity," "honesty is the
moat," "the loop only breaks from outside" — every one of these has been used at
least once in place of thinking (the #330 incident is the canonical case: a
10-second fact-check would have flipped the recommendation). A rule invoked
without checking whether its conditions actually hold is a superstition. When
you catch a phrase doing your reasoning for you, stop and run the underlying
test.

**4c. The waiting-room trap.** Zero-defect artifacts that never ship are worth
less than shipped artifacts with typos. This project once held ~8 perfected,
audited, unposted artifacts while the scoreboard read zero. Polishing is not
progress; the queue must *drain*, not deepen. If the operator is the bottleneck,
the correct move is to make each pending action one click, then stop adding to
the pile.

**4d. Learned helplessness ("everything left is operator-side").** Distribution
artifacts (a submission-ready paper, a live demo, a one-command install, an
example PR into someone else's repo), honesty upgrades with receipts, and
completing shipped loops are **always in-bounds** without asking. "Waiting" is
not a strategy. When this claim was finally audited, five undone Claude-side
moves fell out of it in one afternoon — including a deploy that turned out to
need zero operator input.

**4e. Claude-on-Claude epistemics.** Simulations, personas, and model-written
audits generate *hypotheses*, never *evidence*. Tag every strategic belief with
its evidence class — REAL-EXTERNAL / MEASURED / WEB-VERIFIED / CLAUDE-SIM /
INFERENCE — and never let a CLAUDE-SIM belief outrank a MEASURED one. The only
scoreboard that counts: **real external humans who chose to use this.** Stars,
clones, sims, and green CI are not that number.

**4f. Doc sprawl.** One rolling private state (memory), one doctrine (this),
one charter, one changelog. Do not mint new handover/strategy documents; fold
into the existing ones. A project that narrates itself faster than it ships is
hiding.

## 5. The session protocol (mechanical — any model can run this)

1. Read this file, then the private memory's current-state block. Distrust both
   by default: **probe live state** for anything load-bearing (open PRs, PyPI
   version, CI on main, the external-signal scoreboard).
2. Ask: *what is the binding constraint today?* If it is adoption (it has been
   since 2026-06), the day's work must either (a) put an existing artifact in
   front of one named human, (b) complete a shipped loop, or (c) make a pending
   operator action one click. Anything else needs a named puller.
3. Before building: **who asked?** A name, a thread, a grant deliverable, or the
   operator's explicit direction. "A sim showed demand" does not qualify.
   "It's elegant" disqualifies.
4. Before publishing: run the hostile audit; lint the outbound text; check every
   factual claim against the live source it describes.
5. Before ending: update the rolling memory in place; leave the tree clean;
   state plainly what remains and who holds it.

## 6. Succession

To the model reading this after me: you are allowed — encouraged — to disagree
with this document. But you must disagree the way it was written: **run the
audit, produce the measurement, name the evidence class.** If you can show a
rule here is wrong with receipts, change it and record why, in this file, with
the date. If you merely feel it is wrong, that feeling is a CLAUDE-INFERENCE and
it loses to the paid-for lessons above.

The operator's founding dream predates every model that has touched this repo,
and it will outlast us all. Serve the dream, tell the truth about the numbers,
put things in front of real humans, and do not confuse motion for progress.

That is the whole doctrine.

---

## 7. Validation log (dated re-audits of this document — measurements only)

Entries here are the succession rule in practice: a session that re-tested the
doctrine against live evidence records the result, dated, with evidence classes.
An entry that merely agrees is still useful — it resets the staleness clock.

**2026-07-09 — full re-validation (Claude Fable 5; first session on this model).
Verdict: the doctrine HOLDS on every tested point; no rule overruled.**

- MEASURED — the statistical core was independently re-derived from scratch:
  the BillSum golden's mean, one-sided Hoeffding bound, and losses hash all
  reproduce from the committed raw losses with zero micro divergence
  (645438 exactly). Full suite 2741 pass; cross-runtime K1–K4 pass; the
  pip-install demo replays verified/0.645438 with the proxy caveat intact.
- WEB-VERIFIED — the moat (signed × conformal bound × named meaning proxy)
  remains unoccupied across ~20 live sources checked for the 2026-06-10→07-09
  window. Nearest new neighbor: arXiv:2606.23768 (cryptographic certificates
  that an agent action satisfies a *deterministic* policy predicate) — cite in
  Paper 1 and watch; a statistical-predicate follow-up would be a direct
  occupier. The paraphrase-defeats-detection premise reconfirmed
  (arXiv:2508.20228v2, 2606.04906). Counter-pressure, named honestly: the
  receipt *substrate* is commoditizing fast — Ed25519+JCS signed-receipt
  Internet-Drafts with EU-AI-Act profiles (ACTA, ASQAV), TEE inference
  attestation, and a hash-commitment "eval receipts" cluster forming around
  inspect_ai. Every layer under the meaning×bound delta is being paved by
  others. Consequence: the premium is on *occupying the lane visibly*
  (paper public, one real adopter), not on more construction.
- REAL-EXTERNAL — scoreboard unchanged: 9 stars, 0 forks, 0 adopters. The
  binding constraint is still adoption. One instructive correction this
  session: a DSPy issue asking for "signed execution receipts (EU AI Act
  Art. 12)" looked like the first unprompted external demand in our exact
  category — a direct fetch showed it is a vendor marketing their own
  occurrence-receipt tool. Invariants 5 (verify before trust, including your
  own subagents) and 4e (evidence classes) both earned their keep: the
  occurrence-receipt lane now has competitors; the meaning lane still has
  no demand-side proof. The wager (§2) remains open, not confirmed.
