# Exchangeability advisory — is a meaning-risk bound applicable to *your* text?

> Every conformal bound is valid only **under exchangeability** between the
> calibration corpus and deployment. This turns that caveat from prose into a
> **measured advisory** — without ever gating or re-signing the bound.

A `sum.meaning_risk_receipt.v1` certifies an upper bound on expected
meaning-loss over a *named calibration corpus*, marginally, **under
exchangeability**. The receipt names `corpus_id`, but until now it offered no
signal that *your* deployment text is actually exchangeable with that corpus —
the caveat lived only in the proof boundary. This is the missing guardrail: the
"is the conformal scale still valid out here?" check the skeptic-statistician
named as the #1 gap.

## The mechanism (proven, not asserted)

Embed the calibration corpus and a deployment batch with the same named
embedder, then run the in-repo **MMD two-sample permutation test**
(`research/mmd/`) between them — a kernel test for "do these two samples come
from the same distribution?". A **significant** result is evidence the
deployment distribution differs from calibration → the bound may not apply.

Validated on real CC0 text (BillSum, MiniLM embedder):

| compared against the calibration bills | MMD² | p | verdict |
| --- | --- | --- | --- |
| held-out bills (same distribution) | 0.027 | 0.270 | no shift — consistent with exchangeability |
| their summaries (same topic, different register) | 0.080 | 0.0005 | **shift detected** |
| generic short text (different topic) | 0.215 | 0.0005 | **shift detected** |

The control is non-significant; both out-of-distribution batches are
significant — and MMD² even *grades* the shift (0.027 < 0.080 < 0.215).

## Use it

```bash
sum exchangeability calibration.json deployment.json --corpus-id billsum-test-cc0
# calibration.json / deployment.json = JSON list of texts (or one doc per line)
```
```
Exchangeability advisory — measured (ADVISORY, never gating)
  calibration corpus: billsum-test-cc0  (n=32)
  MMD² = 0.08035   p = 0.0005   (α=0.05, 2000 permutations)
  → SHIFT DETECTED — the bound for billsum-test-cc0 may be OUT-OF-SCOPE here; do not quote it.
```

Library: `from sum_engine_internal.research.meaning import assess_exchangeability,
embed_texts, advisory_report`. Needs `[research]` + `[judge]`.

## The honest boundary — the whole point

- **ADVISORY, NEVER GATING.** It does not change, invalidate, or re-sign any
  bound. A meaning-risk receipt's cryptographic verification and bound-replay
  are unaffected. This is a *separate* measurement a consumer runs to decide
  whether the bound is *applicable* to their data.
- **Asymmetric evidence.** A *significant* p is evidence **against**
  exchangeability (don't quote the bound). A *non-significant* p is
  **consistent with** exchangeability but does **not prove** it — a two-sample
  test cannot accept the null. Absence of a detected shift is not a certificate.
- **Judge/hardware-pinned, so it's a measurement — not a signed field.** The
  embeddings come from a model forward pass (F23/F26), reproducible only on a
  matching stack. It is therefore emitted as an **unsigned report**
  (`sum.exchangeability_advisory.v1`), deliberately **not** folded into a signed
  receipt payload — putting an unreplayable number in a signed field is exactly
  the overclaim discipline this project refuses
  ([`feedback_overclaim_in_signed_fields`](PROOF_BOUNDARY.md)).

## Where it sits in the destination

Stripped of mysticism, SUM's knowledge structure is a **weighted causal poset of
receipts**: the receipt DAG is the *order* (who-derived-from-whom, the
provenance partial order — already built in the drift-budget), and the judge's
per-hop loss is the *number* (the scalar you provably cannot recover from graph
topology alone). The exchangeability advisory is the one piece that makes each
edge's **number refuse to mislead out-of-distribution** — the guardrail before a
multi-hop drift-budget can be trusted end-to-end. It is also the recognized
on-ramp to **weighted-conformal** validity-restoration under shift (re-weighting
the calibration sample by the density ratio the MMD test is detecting) — a
named, no-wire-change future tightening.
