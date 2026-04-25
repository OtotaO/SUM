# Slider v0.2 — Research Findings & Roadmap

Curated extraction from a research survey on verifiable bidirectional
knowledge distillation engines (April 2026), filtered to **work that
materially improves SUM's slider in the next 1–3 PRs**. Items are
ordered by payoff, not by where they appeared in the source survey.
Frontier items SUM cannot meaningfully advance today are listed at
the end as awareness, not commitments.

The full research survey covered AIT (Solomonoff–Kolmogorov–Levin–
Rissanen), category theory (sheaves, monoidal functors, DisCoCat,
Categorical Deep Learning), rate–distortion / Information Bottleneck,
proof-carrying code / Lean 4 / zkML, GEPA / DSPy / GRPO/DAPO,
hierarchical PRMs, weak-to-strong generalization, AI-safety-via-
debate, W3C PROV / RO-Crate / Merkle DAGs, and metamorphic testing.

## Validation: what SUM is already doing right

These observations from the survey confirm the architectural
direction of Phase E rather than calling for change. Documented so
future contributors know which choices are load-bearing.

- **Verifiable rewards, not learned reward models.** SUM's per-axis
  drift formulas are deterministic functions of the renderer's
  output (set-membership, word counts, pronoun ratios, register
  markers). The 2025–2026 RL-post-training literature converged on
  *verifiable* rewards as the path past reward-hacking. SUM is
  already in this regime.
- **Cycle-consistency as the load-bearing claim.** SUM's
  `fact_preservation = |source ∩ reextracted| / |source|` is
  CycleGAN-style round-trip identity at the triple level. The
  survey identifies this pattern as core to verifiable
  transformation systems.
- **Content-addressed everything.** SUM's bundle / AkashicLedger /
  cross-runtime trust triangle is exactly the Merkle DAG +
  W3C PROV pattern the survey recommends as the provenance backbone.
- **The 5-axis slider is the IB Pareto frontier.** Information
  Bottleneck exposes a tunable rate-vs-fidelity trade-off; the
  survey calls out `pareto_weights` as the right user-facing knob.
  SUM's slider IS that knob. Density is the rate axis; the four
  LLM axes are the perception/distortion axes.
- **Per-axis drift = process supervision (PRM-shaped).** The survey
  calls out hierarchical Process Reward Models as the dense per-
  step signal that makes RL sample-efficient. SLIDER_CONTRACT.md's
  per-axis drift formulas are the deterministic counterpart —
  process supervision without RL.

## Status as of v0.2 (2026-04-25)

| Item | Status | Where it landed |
|---|---|---|
| 1. MontageLie-resistant fact preservation | ✅ shipped | `order_preservation` + regression test; bench reports per cell |
| 2. Constrained decoding for the renderer | deferred → v0.3 | requires Pydantic schema for render path; out of scope this round |
| 3. Audience classifier 2000 → 5000 words | ✅ shipped | `data/common_english_5000.txt` (Brown corpus top-5K) |
| 4. Metamorphic relation tests | deferred → v0.3 | Hypothesis-based property tests not yet written |
| Bonus — three-layer fact preservation | ✅ shipped | strict / normalized / semantic; v0.2 substrate |

The headline insight from the v0.2 bench run: the original "fact
preservation = 1.000" claim was an artifact of measuring against
`triples_used` (trivially equal to source at density=1.0) rather
than against re-extracted triples. The corrected three-layer metric
shows median semantic preservation = 1.000 / p10 = 0.455 across 160
LLM-axis cells. Both the original claim was overconfident AND the
strict-key correction was over-pessimistic; the semantic layer is
the honest middle.

## Actionable in the next 1–3 PRs

### 1. MontageLie-resistant fact preservation *(✅ shipped in v0.2)*

**Problem.** Set-based fact preservation is exploitable. The
MontageLie benchmark (Zheng et al., May 2025) shows fact-decomposition
metrics drop to AUC 0.51 — barely above random — when true atomic
facts are reordered into deceptive narratives. SUM's
`fact_preservation = 1.000` headline is currently measured the same
way. A render that preserves all triples but rearranges
causal/temporal order would still score 1.000 here.

**Fix.** Extend the renderer's measurement from set membership to
ordered/causal relations between triples. Two options:

- **Option A — pairwise order-preservation.** For each pair of
  source triples `(t_i, t_j)` where order matters in the input,
  check that the same order is present in the rendered tome. Score
  is the fraction of preserved orderings.
- **Option B — DoveScore-style event-order-aware extraction.**
  Extract not just `(s, p, o)` but `(s, p, o, position)` where
  position is the source-text index. Re-extract from the tome with
  the same field. Score against ordered relations.

Option A is simpler and ships first. Option B is the v0.3 evolution.

**Acceptance.** A test that constructs a permutation of preserved
triples and shows the new metric drops below 1.000 while the old
set-based metric stays at 1.000. This proves we can measure the
attack the literature flagged.

### 2. Constrained decoding for the renderer

**Problem.** The renderer currently asks the LLM for free-form prose,
then re-extracts triples. Two LLM calls (render + extract) per
non-canonical render. The re-extraction is also where the
`LengthFinishReasonError` failures happen (LLM emits more triples
than fit in the completion budget).

**Fix.** Use XGrammar / Outlines / llguidance to constrain the
render LLM to emit a JSON object `{tome: str, claimed_triples:
list[Triple]}`. The DOMINO algorithm (ICML 2024) reports zero or
negative overhead for grammar-constrained decoding via subword-
aligned masking.

**Wins.**
- One LLM call per render instead of two (~50% latency cut).
- 100% format-validity guarantee — no parse errors.
- `claimed_triples` is the LLM's self-attestation; we still
  cross-check against an independent re-extraction for the headline
  fact-preservation claim, but adversarial signal becomes:
  `claimed_triples ⊕ reextracted_triples` (the symmetric difference
  is where the LLM hallucinated or omitted).

**Acceptance.** Renderer ships a `structured=True` mode; bench
runs at parity or better on fact-preservation; latency drops
measurably; the two `LengthFinishReasonError` cells from STATE 5b
no longer error.

### 3. Audience classifier upgrade *(2000 → 5000+ word table)*

**Problem.** STATE 5b's 2000-word Brown-corpus frequency table
covers ~85% of typical English but still under-covers the
vocabulary tail of technical prose. p90 audience drift at neutral
sits at 0.39 — within the relaxed threshold but with no headroom.

**Fix.** Swap to a 5000-word frequency-derived list (SCOWL /
COCA / OpenSubtitles-frequency). Same loader path; just a bigger
data file. Expected to cut p90 drift to ≤0.25.

**Alternative if (3) plateaus.** Rescale the audience target
formula: anchor against measured LLM baseline jargon density at
neutral (≈0.40 with the 2000-word table) rather than the linear
`target = audience * 0.30` assumption. Pure formula change, no
additional data.

### 4. Metamorphic relation tests for the renderer

**Background.** The survey identifies metamorphic testing
(T.Y. Chen, 1998) as the canonical approach to ML-system testing
under the oracle problem. Three workhorse relations: invariance,
monotonic-increase, monotonic-decrease.

**For SUM specifically:**
- *Compression invariance:* `len(render(triples, sliders))` should be
  approximately equal under paraphrasing of the source triples
  (different surface forms of the same facts).
- *Density monotonicity:* `len(triples_used(render(t, density=0.3)))`
  ≤ `len(triples_used(render(t, density=0.7)))`.
- *Length monotonicity:* `len(tome(render(t, length=0.1)))` ≤
  `len(tome(render(t, length=0.9)))` — already implicit in our
  bench data; should be a property test.
- *Round-trip identity (up to ε):* `d(triples,
  reextract(render(triples, neutral_sliders))) ≤ ε`.

**Acceptance.** `Tests/test_slider_metamorphic.py` lands with at
least four MR tests. Hypothesis library generates the source
triple sets.

## Awareness, defer

Items the survey calls out as state-of-the-art that SUM should NOT
build now. Listed so future PR authors don't re-evaluate from scratch.

- **zkML proof of LLM inference (NanoZK / Jolt Atlas / EZKL).** Real
  technology in 2025–2026 — full GPT-2 inference proven in seconds —
  but only relevant when SUM has paying users who care about
  delegated-compute trust. Premature today.
- **Lean 4 formal verification of paragraph-level meaning preservation.**
  Possible for restricted controlled languages; an open frontier
  problem (survey §8) for unrestricted natural language. Section-8
  research, not engineering.
- **GEPA outer loop / DSPy module graph / GRPO inner training.**
  SUM doesn't train models — it uses pretrained LLMs through API.
  Adding the optimizer stack is a different project (a model-
  training infrastructure for SUM-specific PRMs). The survey is
  right that this is the path past hand-tuning, but we have no
  trained-component scope today.
- **Sheaf neural networks / Categorical Deep Learning / DisCoCat.**
  Beautiful framing; no immediate code that improves SUM. Note for
  multimodal extension (Phase F+).
- **Compression-rate-on-held-out-data as North-Star metric.** The
  survey recommends this as the universal Solomonoff metric.
  Wrong metric for SUM specifically — our claim is *fact
  preservation across axis changes*, not raw compression. We're
  shipping a slider, not a codec.

## Frontier problems (Section 8 of the survey)

These are open in the literature; SUM cannot resolve them but should
track them. If any closes during Phase F or later, revisit.

1. **Closed-form RDP for natural language.** Gaussian RDP has
   closed-form solutions; for token sequences it does not. A
   non-trivial bound on minimum bits/token for given semantic
   fidelity is open.
2. **Composable zk-SNARK proofs for transformer pipelines.** Single-
   stage proofs ship today; multi-step pipeline composition where
   each stage's witness becomes the next stage's input is not yet
   practical.
3. **Lean-formalisable semantic preservation at paragraph scale.**
   Sentence-level sometimes works; paragraph-level requires modal-
   temporal-narrative logics that don't yet exist in usable form.
4. **Fact-decomposition-resistant verifiers.** MontageLie is
   item 1 above; full v0.2 mitigation is non-trivial.
5. **Provably calibrated PRMs.** Conformal-prediction-style
   guarantees on PRM scores remain a research thread.
6. **Round-trip fidelity bounds for very long contexts.** Theoretical
   bounds on `K(book | tag(book))` analogous to Solomonoff prediction-
   error theorem are absent.
7. **Cross-modal certificates.** Asymmetric cycle frameworks for
   Text↔Image cycles (where image space is too rich for cycle-loss
   alone) are partial; principled certificates remain open. Relevant
   if SUM goes multimodal.

## Reading list (for future contributors)

Curated to material directly relevant to SUM, not the full survey
bibliography.

- Delétang et al., "Language Modeling is Compression" (ICLR 2024).
  The empirical instantiation of the AIT prediction-compression
  equivalence. Validates the core slider thesis.
- Dhuliawala et al., "Chain of Verification" (ACL 2024). The
  draft → plan-questions → answer-independently → finalise pattern
  reduces hallucinations 50–70%. Good template for adding a
  verification step at the renderer's LLM call.
- Setlur et al., "Rewarding Progress / PAVs" (ICLR 2025). PRMs as
  step-level dense supervision; relevant when SUM ever builds
  trained components.
- T.Y. Chen, "Metamorphic Testing" (1998 + modern follow-ups).
  Direct guide for action item 4 above.
- MontageLie / DoveScore work (Zheng et al., May 2025). The threat
  model for action item 1 above.
