# Machine Studying → SUM: the verifiable cheatsheet

*Status: research (`[research]` extra). Not a cataloged production feature
(same convention as v3 / sheaf / the meaning-loss frontier). 2026-06-18.*

## Why this doc exists

Jacob Xiaochen Li, *Machine Studying*
(<https://jacobxli.com/blog/2026/machine-studying>), names a gap: how does
an agent autonomously build **expertise** over a corpus *before* it knows
the downstream task? Three of his load-bearing ideas map almost one-to-one
onto primitives SUM already ships. This doc records the mapping, the one
new piece it motivated (`sum study` + the `expertise` scalar), and — per
the charter — the honest boundary on how far the analogy reaches and why
the work lives behind `[research]` rather than in the product surface.

## The blog in three claims

1. **Expertise is a curve, not a ceiling.** Define expertise as efficiency
   at converting inference-compute into accuracy — a weighted area under
   the accuracy-vs-compute curve, with a log-scale decay
   `w(x) = ln(10)·10^(-x)` that *favours cheap budgets*. Two agents with
   the same peak accuracy can differ wildly in expertise.
2. **Studying = what the agent does to itself over the corpus before
   evaluation.** The corpus stays available at test time; the point is
   efficiency in using it.
3. **The cheatsheet won.** Of three studying paradigms tested
   (continual pre-training, synthetic SFT, amortized context management),
   only *note/cheatsheet writing* gave consistent expertise gains — and the
   gains concentrated at the cheap end. Memorisation (SFT) actually *lowered*
   expertise (verbose answers, higher cost per query). Pure retrieval is not
   expertise ("you wouldn't hire us as a lawyer just because we can Google
   the legal literature").

## The mapping onto SUM

| Blog concept | SUM primitive | Path |
|---|---|---|
| Expertise = weighted AUC of accuracy vs. compute (curves, not ceilings) | **`RenderFrontier`** — an ordered faithful→compressed path, measured meaning-loss per point | `sum_engine_internal/research/frontier.py` |
| Studying = condense the corpus into reusable notes (the winning cheatsheet) | `extract` each doc → **`compose`** (LCM-union of triples → one SUM-of-SUMs) → `slider`-render at a density | `transforms/extract.py`, `transforms/compose.py`, `ensemble/slider_renderer.py` |
| The blog's open gap: a cheatsheet is **unverified** ("did my notes silently drop something load-bearing?") | **`sum.meaning_risk_receipt.v1`** — signed, conformal, replayable bound on meaning-loss | `research/meaning/{conformal_meaning,receipt}.py` |

The convergent move these three suggest is **the verifiable cheatsheet**:
produce the winning study artifact (compressed corpus notes), but close the
silent-drop gap with a signed meaning-loss bound. That is exactly what
`sum study` emits.

## What shipped: `sum study` and the `expertise` scalar

`sum study --corpus DIR [--doc FILE ...]` composes the substrate above:

1. **extract** each document → a per-doc triple bundle (offline sieve);
2. **compose** the bundles → one SUM-of-SUMs (the corpus knowledge base —
   `state_integer` + the unioned axioms; the persistent artifact the engine
   previously lacked);
3. **render** the merged bundle down a faithful→compressed density path
   (the study notes at several compression levels) and **score** each point
   under a named meaning-loss proxy (`--scorer nli|embedding|lexical`);
4. report SUM's native **`expertise`** scalar and emit the cheatsheet at
   `--study-density` (default = the floor: the smallest, cheapest note);
5. with `--certify --signing-jwk … --kid …`, seal a
   `sum.meaning_risk_receipt.v1` over the **per-document loss of consulting
   the cheatsheet** (each document is one exchangeable unit).

Output is a `sum.study_artifact.v1` JSON. Code: `sum_engine_internal/research/study.py`
(the `StudyArtifact` container + the `expertise` function) and `cmd_study`
in `sum_cli/main.py`.

### The expertise scalar (the one new measure)

```
expertise = Σ wᵢ · fidelityᵢ  /  Σ wᵢ ,   wᵢ = exp(-decay · (1 - positionᵢ))
```

where `fidelity = 1 - meaning_loss`, `position ∈ [0,1]` is the point's place
on the faithful→compressed path (`1.0` = most compressed = cheapest to
consult), and the *consultation budget* of a point is `1 - position`. The
weight `exp(-decay · budget)` favours the cheap (compressed) end exactly as
the blog's `10^(-x)` favours cheap compute; the default `decay = ln(10)`
carries the blog's constant over. So an artifact that stays faithful **when
small** scores near `1.0`; one that collapses at the cheap end scores well
below its naïve mean (a 2-point `[1.0, 0.0]` faithful→compressed curve
scores `≈0.09`, not `0.5`).

## The honest boundary (proof discipline)

This is where the analogy must not be oversold — the same discipline the
proof-boundary and meaning-loss frontier docs hold one layer up:

- **`expertise` is a MEASUREMENT, never a guarantee.** It is a per-run number
  under a named scorer. No "fast / efficient / better-studied" prose may ride
  on it without the same-commit number in hand.
- **It is an *analogy*, not the blog's metric.** Li measures downstream
  **task accuracy**; SUM measures **meaning-fidelity** under a proxy whose
  declared blind spots (arrangement, sound, connotation, implicature) it
  inherits. A high expertise number means "this study artifact stays faithful
  even when small," nothing about downstream task performance.
- **Only the embedded receipt is certified.** The `sum.meaning_risk_receipt.v1`
  carries the one distribution-free, marginal, replayable claim; the expertise
  scalar and the frontier losses are measurements. The `measurement_note`
  travels on every emitted artifact saying so.
- **The lexical scorer misranks paraphrase** (F18): for a real corpus prefer
  `--scorer nli`.

## Why it lives in `[research]`, not the product surface (charter gate)

Per `docs/CHARTER_2026-05-17.md`: the standing direction is *wait + dogfood*,
and every change must pass the **buyer-or-dream filter** (a named buyer, a
funder commitment, or a dream element — math-natural work is rejected) while
respecting **Constraint 2** (no auto-pivot to substrate shipping to fill
time). Machine-studying is not something the writers' wedge ICP (journalists,
academics, newsletter writers) is asking for today, and an agent-builder
wedge does not yet exist. It *does* serve the origin **dream** (bi-directional
tags↔tomes, knowledge reshaped across forms without loss) and it composes
existing substrate rather than adding new algebra (process intensification —
the only new code is the `expertise` scalar). So it ships **behind
`[research]`, is NOT added to `FEATURE_CATALOG.md`**, and waits for real pull
before any product-surface (Worker / API / MCP) promotion — the same path the
meaning-loss frontier took.

## Reproduce

```bash
pip install 'sum-engine[research,sieve]'        # + [judge] for --scorer nli
sum study --corpus path/to/corpus --scorer nli --pretty
# verifiable cheatsheet:
sum study --corpus path/to/corpus --certify \
          --signing-jwk key.jwk --kid mykey --corpus-id my-corpus-v0
```

Tests: `Tests/test_study.py` (the `expertise` contract + the end-to-end
extract→compose→render→certify pipeline).

## See also

- `docs/MEANING_LOSS_FRONTIER.md` — the sub-factual layer the scorer measures.
- `docs/PRODUCT_VISION.md` — the `RenderFrontier` the curve is built on.
- `docs/RECEIPT_FAMILY_SPEC.md` §3.5 — the `sum.study_artifact.v1` container.
- `docs/CHARTER_2026-05-17.md` — the gate this work is held behind.
