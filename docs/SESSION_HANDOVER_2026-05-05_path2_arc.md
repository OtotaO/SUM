# Session handover — Path 2 / multi-LLM arc (2026-05-05)

This handover covers the three-PR arc that closes the §7
load-bearing asterisk in the preprint and corroborates the finding
across LLM families. Read this if the §4.7.x narrative is in
question; otherwise the prior
[`SESSION_HANDOVER_2026-05-05_sprint_7_5_arc.md`](SESSION_HANDOVER_2026-05-05_sprint_7_5_arc.md)
remains current for everything else.

## What landed

- **PR #156** — Path 2 real-LLM-rendered adversarial bench. New
  `scripts/research/sheaf_path2_v3_bench.py` with
  capture-once-replay-forever architecture: Phase 1 calls
  `gpt-4o-mini-2024-07-18` to render each `seed_long_paragraphs`
  doc at four prompt classes (neutral + a1/a2/a4 adversarial);
  snapshot committed; Phase 2 re-extracts via `DeterministicSieve`
  and scores deterministically. Receipt:
  `fixtures/bench_receipts/path2_v3_bench_2026-05-05.json`,
  `bench_digest 7b364fc6…cc4b75e`. Pinned in
  `Tests/research/test_sheaf_path2_v3.py`.

- **PR #157** — Multi-LLM Path 2 comparison harness. New
  `scripts/research/sheaf_path2_multi_llm_compare.py` extends
  Path 2 across LLM families via the vendor-agnostic dispatcher
  (`llm_dispatch.get_adapter`). Joint findings classify into one
  of `STRUCTURAL_GAP_*`, `HYBRID_BEATS_*`,
  `MIXED_VERDICTS_MODEL_DEPENDENT`, or
  `SINGLE_MODEL_<verdict>` (n=1 honest label). Wrapper verified
  no-op on the scoring path.

- **PR #158** — Phase 1 capture for `claude-haiku-4-5-20251001`
  (operator-gated, 64 LLM calls); cross-family run produces
  **`STRUCTURAL_GAP_ALL_MODELS_LOSE`** (gpt-4o-mini Δ = −0.021;
  claude-haiku-4.5 Δ = −0.032; spread 0.011). The
  synthetic-vs-real gap from §4.7.2 is **not**
  gpt-4o-mini-specific. New §4.7.3 in
  `docs/arxiv/sheaf-detector-note-v0.md`.

## Substantive verdicts at HEAD

| Bench | Verdict | Δ vs B2 | Per-model digest |
|---|---|---:|---|
| §4.7.1 synthetic | `HYBRID_BEATS_BASELINE` | +0.043 | `dc6e0260…` |
| §4.7.2 Path 2 (gpt-4o-mini) | `HYBRID_LOSES_TO_BASELINE_ON_REAL_LLM` | −0.021 | `7b364fc6…` |
| §4.7.3 Path 2 (claude-haiku-4.5) | `HYBRID_LOSES_TO_BASELINE_ON_REAL_LLM` | −0.032 | `d0f9f175…` |
| §4.7.3 joint | **`STRUCTURAL_GAP_ALL_MODELS_LOSE`** | spread 0.011 | — |

The synthetic-vs-real gap is the load-bearing finding. Synthetic
A1/A4 cleanly change the entity set (B2's strongest case); real
LLM perturbations don't share that property. The hybrid's apparent
synthetic-bench advantage doesn't survive contact with real LLM
hallucinations from either family.

## Open follow-ups

- **Phase 1 / Phase 2 same-process digest contamination** (task
  #22). When Phase 1 (LLM API capture) and Phase 2 (deterministic
  scoring) run in the same Python process, the Phase 2 digest
  differs slightly from running Phase 2 in a fresh process against
  the same cached snapshot. Concretely: capture-time gave
  Δ = −0.027 / digest `782ea1a1…` for Claude; fresh-process
  reproduces Δ = −0.032 / digest `d0f9f175…`. **Substantive
  verdict is unchanged** (both LOSES; structural gap holds), but
  determinism is the substrate's primitive. Likely cause: HTTP
  client / asyncio / Anthropic SDK import warming up some BLAS or
  torch state. Cleanest fix: have `_ensure_snapshot_for_model`
  spawn a subprocess for Phase 1 so the parent process never has
  the API client warmed up before scoring. The canonical pin uses
  the fresh-process digest, which is what tests reproduce.

- **v0.4+ candidate: open-weights extension**. Two LLM families is
  a thin sample; an open-weights run (Llama-3.x or Mistral) would
  broaden the cross-family sample beyond proprietary frontier APIs.

- **v0.4+ candidate: real-LLM-aware per-triple V training**. The
  per-triple V channel was calibrated for synthetic
  entity-set-preserving perturbations. Training on a corpus of
  LLM-rendered perturbations would let the trained restriction
  maps reflect the real-LLM perturbation distribution.

- **v0.4+ candidate: naturalistic perturbation synthesis**. Have
  an LLM generate A1/A2/A4-class perturbations on the source
  TRIPLE set directly (not the rendered prose), so the
  perturbation structure matches synthetic but the perturbation
  choice is LLM-natural. Decouples "what gets perturbed" from
  "how it propagates through rendering."

## What's now closed

- The §7 bounded-claims asterisk ("not generalising to real-LLM-
  rendered hallucinations") is no longer load-bearing — both §4.7.2
  (single model) and §4.7.3 (cross-family) have measured the gap
  and reported it honestly. The hybrid's synthetic-bench WIN is
  *real on its corpus* but *doesn't generalise*; this is named in
  preprint prose and pinned by digest in CI.

## Operator-only items remaining

Same as the prior arc: optional v0.6.0 release, pre-circulation
packet, arXiv submit. The §4.7.3 cross-family corroboration
strengthens the §4.7.x narrative in the same direction §4.7.2
established — no further substantive work is needed before submit.
