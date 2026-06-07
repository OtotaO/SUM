# Dogfood findings — 2026-06-07 (simulating users through complete cycles)

Per the operator's "simulate users yourself through a complete cycle or
two," I played two real personas end-to-end on the *actual* surface
(issue in Python → verify in Node, as a party that did not issue the
receipt). Both cycles ran. The spine works; the on-ramp does not.

## The cycles (both ran, no `$`)

**Cycle A — agent handoff.** Agent A summarises a document, scores
candidates with the **NLI judge** (faithful → loss 0.00; a fluent on-topic
*contradiction* → loss 1.00 — the judge caught the hallucination), issues
a signed `meaning_risk_receipt`, and hands it to **Agent B**, who
**verifies it in Node** (the JS verifier) without trusting Agent A:
signature + disclosure valid.

**Cycle B — Art-50 compliance.** A publisher transforms content for a
`plain` and a `technical` audience, issues a signed **Perspective
Receipt**, and a **Regulator verifies it in Node** and reads the
per-cohort verdict.

The cross-runtime, external-party trust loop **works**. That is the moat,
and it functions outside Python now.

## F21 (the big one): there is no user-facing path for the cycle

To run *either* cycle I had to **hand-write ~40 lines of Python to issue
and shell out to `node` to verify.** A real agent developer or publisher
cannot do this — there is no `sum`-level workflow for the meaning/
perspective receipts: no `sum verify-meaning`, no issue command, no SDK
one-liner. **The substrate exists; the workflow does not.** This is the
#1 adoption gap — larger than any feature. A standard nobody can *run*
without reading the source is a spec, not a standard (the bird's-eye
review's exact point, now confirmed from the user's seat).
*Fix:* a `sum verify-meaning <receipt> --jwks <jwks> [--losses …]` command
(the external-party on-ramp) — shipped alongside these findings — and,
next, an issue path.

## F22: small corpora produce vacuous bounds — receipts are a BATCH primitive

Agent A's 3-document receipt certified meaning-loss **≤ 0.947** — useless.
You **cannot certify a single handoff**: the distribution-free radius
needs a meaningful `n`. The honest framing the product must adopt: the
*signed receipt* certifies a **body of work / a batch / a day's output**;
for a *single* item, use the per-document **measurement** (the frontier
number), clearly labelled "measured, not certified." Conflating the two
is the trap; the receipt is for batches.

## F23: the replay-stable scorer is paraphrase-blind; the accurate one isn't replay-stable

The central tension, made concrete: Cycle B showed *both* cohorts
"not controlled" — but that was largely the **lexical** scorer
over-reporting paraphrase (+ small `n`), not real meaning loss. The
**NLI** judge would score it correctly (Cycle A proved it) — but a model
judge is **machine-pinned** for *signed-receipt replay* (cross-hardware
float drift can flip a boundary decision), while the lexical scorer is
deterministic-everywhere but paraphrase-blind. So today a signed,
cross-machine-replayable receipt is forced onto the *worse* scorer.
Honest resolutions, for the spec to choose: (a) NLI for the *frontier*
(measurement) + lexical for the *signed bound*, disclosed; (b) coarse-
quantise the NLI decision so cross-machine drift can't flip it; (c)
declare model-judge receipts **machine-pinned** for replay (already the
documented stance) and pin model+runtime in the receipt. This is a
real product/spec decision, not a bug.

## What this reshapes

The bird's-eye sequence was: JS verifier → doc-currency → spec → external
verifier. The simulation **confirms** the JS verifier was the right
prerequisite (the external party really can verify now) and **sharpens
the rest**: before a standards pitch, the receipts need a **runnable
on-ramp** (F21) and an honest **batch-vs-single / scorer-determinism
story** (F22, F23) — because a standard is judged on what a stranger can
*do* with it, and right now a stranger can verify (good) but cannot issue
or invoke it without the SDK. The binding constraint remains adoption;
the simulation shows the precise shape of the on-ramp that's missing.
