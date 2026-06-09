# The multi-hop drift budget — composing meaning-loss across a chain

> A single receipt certifies one hop. Text in the wild takes many. This is
> how the per-hop pieces compose into a chain-level statement — **honestly**.

A `sum.meaning_risk_receipt.v1` certifies the expected meaning-loss of
**one** transform, over one corpus, under one named proxy. But an
autonomous agent loop (propose → transform → measure → commit, the "loopy
era") and a human editing pipeline both run a document through **many**
transforms. The question that matters is cumulative: *how much meaning has
drifted after N hops, and is the chain still within budget?*

This is the convergent frontier — it's simultaneously the deepest dream
(compose the custody chain), the loudest external tailwind (autonomous
transform loops make drift urgent), and a named adopter ask (demand #4
from the 30-guest adoption simulation). Module:
`sum_engine_internal/research/meaning/drift_budget.py` (research-grade,
behind `[research]`).

## Two quantities, kept rigorously apart

The one real trap here is conflating two different numbers. The module
refuses to.

### Leg A — the per-document **measurement** (`measure_chain_drift`)

Run one document `x0 → x1 → … → xN` through a named scorer. You get:

| quantity | meaning |
| --- | --- |
| per-hop loss `Lᵢ` | `scorer.loss(x_{i-1}, xᵢ)` — drift at hop *i* |
| **additive budget** `Σ Lᵢ` | drift consumed hop-by-hop |
| **end-to-end loss** `L_e2e` | `scorer.loss(x0, xN)` — measured *directly* between endpoints |
| **slack** `Σ Lᵢ − L_e2e` | the gap (sign reported, **not assumed**) |

```bash
sum drift-budget x0.txt x1.txt x2.txt --scorer nli      # or --scorer lexical (no model)
```
```
Drift budget — measured for THIS chain (not a certified bound)
  judge: lexical-coverage-bidirectional v1
  hops: 2
     hop 1: loss 0.350   (x0 → x1)  ← most expensive
     hop 2: loss 0.233   (x1 → x2)
  additive budget (Σ Lᵢ): 0.583  — drift consumed hop-by-hop
  end-to-end loss (x0 → x2): 0.467  — measured directly
  slack: +0.117 — additive is CONSERVATIVE here (it did not miss end-to-end drift)
```

This is the multi-hop analogue of [`sum meaning-diff`](MEANING_LOSS_FRONTIER.md):
a measurement for one chain, honestly labelled — not a certified bound.

### Leg B — the corpus-level **certified** composition (`compose_drift_budget`)

Given a per-hop `MeaningRiskGuarantee` for each transform (each certifying
`E[Lᵢ] ≤ Uᵢ` at confidence `1 − δᵢ`), the **union bound** gives:

> With confidence ≥ `1 − Σ δᵢ`, *every* per-hop bound holds simultaneously,
> hence the cumulative expected per-hop loss `Σ E[Lᵢ] ≤ Σ Uᵢ`.

That `Σ Uᵢ` is the **certified drift budget**. It is provable (Bonferroni +
monotonicity of summation), and it **cannot disagree with the single-hop
receipts because it is literally their sum**, carried at the joint
confidence. `compose_drift_budget_from_payloads` does the same over verified
receipt payloads in integer micro-units, so the chain budget is byte-exact
against the receipts it composes — feed it the dicts returned by
[`sum_verify.verify_meaning_risk_receipt`](VERIFY_SDK.md) and the chain
number matches the receipts to the last micro.

```python
from sum_verify import verify_meaning_risk_receipt
from sum_engine_internal.research.meaning.drift_budget import (
    compose_drift_budget_from_payloads,
)
payloads = [verify_meaning_risk_receipt(r, jwks, losses=L) for r, L in hops]
budget = compose_drift_budget_from_payloads(payloads)
assert budget.within(0.50)        # is the whole chain under a 0.50 ceiling?
print(budget.joint_confidence)    # 1 − Σ δᵢ
```

## Why additive `Σ Lᵢ` is NOT claimed to bound end-to-end loss

It is tempting to assert a triangle inequality — that hop-by-hop loss can
only *over-count* end-to-end loss — and ship `Σ Lᵢ` as a guaranteed ceiling
on `L_e2e`. **We do not.** The entailment proxy is not a metric and
claim-survival along a chain is not monotone, so both regimes occur:

- **Recovery → additive over-counts (conservative).** Hop 1 drops claim A;
  hop 2 re-derives A. `L₁ > 0` but `L_e2e ≈ 0`, so `Σ Lᵢ > L_e2e`.
- **Compounding brittleness → additive UNDER-counts.** Each hop is a
  faithful paraphrase the judge scores at `≈ 0`, but the compounded rewrite
  drifts far enough that the judge scores `L_e2e > 0`. Then `Σ Lᵢ < L_e2e` —
  the additive budget *misses* drift the end-to-end measurement catches.

So the additive↔end-to-end relationship is **measured, not asserted**
(`audit_additive_vs_end_to_end`, the discipline the slider's
[T4 drift-composition audit](BENCH_HARDENING_FROM_QCVV.md) applied to
`drift_pct`). The test suite deliberately exhibits *both* regimes with
deterministic judges. The certified leg (B) sidesteps the question
entirely: it bounds `Σ E[Lᵢ]`, a quantity that is additive by definition,
never `E[L_e2e]`.

## Honest boundary

Inherited from the receipt family ([`PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md)):
every loss is a **named proxy**, **marginal**, valid only under
**exchangeability** with each hop's calibration corpus. Nothing here covers
arrangement (*naẓm*), sound, connotation, or implicature. The certified
budget bounds the *sum of per-hop expectations*, each within its own
corpus's scope — not the end-to-end expected loss, and not any single
document's realised path.

## Status and the named next rung

Shipped: the composition primitive (both legs), the empirical audit, and
the `sum drift-budget` CLI. **Not yet shipped:** a signed
`sum.drift_budget_receipt.v1` envelope that wraps a composed chain budget
into its own replayable certificate (binding the ordered list of per-hop
receipt hashes + the Bonferroni `joint_delta`). That is the natural
capstone — the chain-level analogue of a single meaning-risk receipt — and
the next increment on this frontier.
