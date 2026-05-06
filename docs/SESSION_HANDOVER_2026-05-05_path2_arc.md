# Session handover — Path 2 / multi-LLM arc (2026-05-05 → 2026-05-06)

This handover covers the six-PR arc that closes the §7 load-bearing
asterisk in the preprint, corroborates the finding across LLM
families, extends to open-weights, and root-causes a determinism bug.
Read this if the §4.7.x narrative is in question; otherwise the prior
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
  `MIXED_VERDICTS_MODEL_DEPENDENT`, or `SINGLE_MODEL_<verdict>`
  (n=1 honest label). Wrapper verified no-op on the scoring path.

- **PR #158** — Phase 1 capture for `claude-haiku-4-5-20251001`
  (operator-gated, 64 LLM calls); n=2 cross-family run produced
  `STRUCTURAL_GAP_ALL_MODELS_LOSE` (gpt-4o-mini Δ = −0.021;
  claude-haiku-4.5 Δ = −0.032; spread 0.011). New §4.7.3 in
  `docs/arxiv/sheaf-detector-note-v0.md`.

- **PR #159** — In-repo doc sync. New `path2_arc` session
  handover (this file's first revision); CLAUDE.md onboarding
  pointer reordered; `docs/PROOF_BOUNDARY.md` §2.9 cross-family
  bullet; `docs/SHEAF_HALLUCINATION_DETECTOR.md` future-work
  item closed.

- **PR #160** — Root-causes the misdiagnosed "Phase 1 same-process
  contamination" (task #22). The actual bug was dict iteration
  order: in-memory snapshots from Phase 1 are in **corpus order**
  (insertion order); cached snapshots from JSON round-trip are in
  **alphabetical order** (`json.dumps(..., sort_keys=True)`).
  `run_path2_v3_bench` iterated `snapshot["renders"]` directly to
  build `all_triples`, which fed `train_restriction_maps`; even
  with `seed=0`, training-data order changed the trained sheaf and
  hence the digest. One-line fix:
  `docs_with_src.sort(key=lambda x: x[0])` before training. New
  regression test
  `test_path2_v3_bench_invariant_to_snapshot_dict_order` runs the
  bench in both orders and asserts identical digests. Existing pins
  unchanged — they were always against alphabetical/cached order.
  Investigation also probed and ruled out asyncio / httpx /
  anthropic SDK import / `AsyncAnthropic` construction / sequential
  in-process Phase 2 invocations as contamination triggers.

- **PR #161** — Open-weights extension via Hugging Face Inference
  Providers; bumps n=2 → n=6. `llm_dispatch.get_adapter` recognises
  HF-namespaced model ids (containing `/`) and constructs
  `OpenAIAdapter` pointed at `https://router.huggingface.co/v1`
  with `HF_TOKEN`; `OpenAIAdapter` gains optional `base_url`. Four
  open-weights captures landed (Meta Llama-3.3-70B, Alibaba
  Qwen3.6-35B-A3B, DeepSeek V3-0324, Google Gemma-3-27B). Joint
  finding upgraded to **`STRUCTURAL_GAP_NO_MODEL_BEATS`**: four LOSE
  (gpt-4o-mini, claude, Llama, Gemma); two TIE (Qwen +0.003,
  DeepSeek +0.018); zero BEAT. `CAPTURE_TIMEOUT_S` raised
  60s → 180s + 2 retries on timeout. §4.7.3 prose rewritten with
  full per-model table + texture analysis.

## Substantive verdicts at HEAD

| Bench | Verdict | Δ vs B2 | bench_digest |
|---|---|---:|---|
| §4.7.1 synthetic (Borda hybrid) | `HYBRID_BEATS_BASELINE` | +0.043 | `dc6e0260…` |
| §4.7.2 Path 2 (gpt-4o-mini) | `HYBRID_LOSES_TO_BASELINE_ON_REAL_LLM` | −0.021 | `7b364fc6…` |
| §4.7.3 Path 2 (claude-haiku-4.5) | `HYBRID_LOSES_TO_BASELINE_ON_REAL_LLM` | −0.032 | `d0f9f175…` |
| §4.7.3 Path 2 (Llama-3.3-70B) | `HYBRID_LOSES_TO_BASELINE_ON_REAL_LLM` | −0.047 | `f1c17c3e…` |
| §4.7.3 Path 2 (Qwen3.6-35B-A3B) | `HYBRID_TIES_BASELINE_ON_REAL_LLM` | +0.003 | `23da3ecb…` |
| §4.7.3 Path 2 (DeepSeek V3-0324) | `HYBRID_TIES_BASELINE_ON_REAL_LLM` | +0.018 | `619a413f…` |
| §4.7.3 Path 2 (Gemma-3-27B) | `HYBRID_LOSES_TO_BASELINE_ON_REAL_LLM` | −0.028 | `fe76913e…` |
| §4.7.3 joint (n=6) | **`STRUCTURAL_GAP_NO_MODEL_BEATS`** | spread 0.065 | — |

The synthetic-vs-real gap is the load-bearing finding. Synthetic
A1/A4 cleanly change the entity set (B2's strongest case); real
LLM perturbations don't share that property. The hybrid's apparent
synthetic-bench advantage doesn't survive contact with real LLM
hallucinations from any of the six tested LLM families. Qwen and
DeepSeek are the only two with weakly positive Δ but neither
crosses the +0.030 BEATS threshold.

## What's now closed

- The §7 bounded-claims asterisk ("not generalising to real-LLM-
  rendered hallucinations") is no longer load-bearing — §4.7.2
  (single model), §4.7.3 (n=2 closed pair), and §4.7.3 extended
  (n=6 closed + open-weights) have measured the gap and reported
  it honestly. The hybrid's synthetic-bench WIN is *real on its
  corpus* but *doesn't generalise to any LLM family in the
  sample*; this is named in preprint prose and pinned by digest
  in CI.

- **Task #22** (Phase 1 / Phase 2 same-process digest contamination)
  — closed by PR #160. Root cause was dict iteration order, not
  BLAS or asyncio state.

- **v0.4+ candidate: open-weights extension** — closed by PR #161.
  Four open-weights families (Meta, Qwen, DeepSeek, Gemma) added;
  none flip the verdict to BEATS.

## Open follow-ups

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

- **v0.4+ candidate: cross-corpus extension**. The §4.7.3 sample
  is one corpus × six LLMs. A complementary scaling axis is one
  LLM × N corpora. Combined with the cross-family scaling
  already landed, this would give a 2-D sample sufficient to
  argue structural gap independent of both corpus and LLM choice.

## Operator-only items remaining

Same as the prior arc: optional v0.6.0 release, pre-circulation
packet, arXiv submit. The §4.7.3 cross-family corroboration —
now spanning six LLM lineages from six organisations, closed and
open-weights — strengthens the §4.7.x narrative in the same
direction §4.7.2 established. No further substantive work is
needed before submit.
