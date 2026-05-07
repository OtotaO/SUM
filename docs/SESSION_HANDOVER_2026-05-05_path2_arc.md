# Session handover — Path 2 / multi-LLM arc (2026-05-05 → 2026-05-07)

This handover covers the eight-PR arc that closes the §7 load-bearing
asterisk in the preprint, corroborates the finding across LLM
families, extends to open-weights, root-causes a determinism bug,
extends to cross-corpus where the §4.7.3 finding initially appeared
corpus-specific, and then resolves the lone BEATS cell as
extremal-Goodhart at small n — at controlled sample sizes
(n ≥ 16), the §4.7.3 STRUCTURAL_GAP_NO_MODEL_BEATS finding holds.
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

- **PR #163** — Cross-corpus extension (§4.7.4); §4.7.3 finding
  initially appears corpus-specific. Multi-LLM compare gains
  `--corpus` flag (backward-compatible). New aggregator
  `scripts/research/sheaf_path2_cross_corpus_aggregate.py` loads
  N per-corpus receipts and produces a 2-D joint finding. Two new
  corpora authored: `seed_paragraphs` (8 docs, encyclopedic
  shorter, already in repo) and `seed_news_briefs` (16 docs, new,
  news-wire prose, deliberately out-of-distribution). Across 3
  corpora — `seed_long_paragraphs` carries the n=6 set with claude,
  `seed_paragraphs` and `seed_news_briefs` are n=5 (Anthropic key
  unavailable during the §4.7.4 capture); jagged matrix totals
  16 cells: 1 BEATS, 8 TIES, 7 LOSES. Joint finding
  **`CROSS_CORPUS_VERDICTS_DIVERGE`** — `seed_paragraphs` produces
  one BEATS cell (gpt-4o-mini Δ=+0.032 right at the +0.030
  threshold) which drives that corpus's joint finding to
  `MIXED_VERDICTS_MODEL_DEPENDENT`; the other two corpora
  reproduce `STRUCTURAL_GAP_NO_MODEL_BEATS`. Bug fix in
  `_receipt_paths.py` glob to require date suffix (prevents
  prefix-of-prefix false positives on receipt paths).

- **PR #164** — §4.7.4.1 resolves the lone BEATS cell as
  extremal-Goodhart at small n. New 16-doc corpus
  `seed_paragraphs_16.json` (same encyclopedic voice as
  `seed_paragraphs`, eight originals retained verbatim plus eight
  new docs: Mount Everest, Marie Curie, Great Wall of China,
  Titanic, Renaissance, atomic structure, jet stream, blockchain).
  At doubled n the lone BEATS cell flips to TIES: gpt-4o-mini
  Δ=+0.032 BEATS (n=8) → Δ=−0.013 TIES (n=16) for the same model
  on the same style. Joint finding on `seed_paragraphs_16`:
  `STRUCTURAL_GAP_NO_MODEL_BEATS`, matching the other two corpora
  at n≥16. Updated 4-corpus aggregate: 21 cells (1 BEATS, 10 TIES,
  10 LOSES) — the lone BEATS cell is now *explained*, not
  unresolved. Substantive consequence: at controlled sample sizes
  (n ≥ 16) across 3 corpora × 4-6 LLM lineages, the hybrid does
  NOT BEAT B2 on real-LLM perturbations. The synthetic-bench WIN
  (+0.043) is now read as a Goodhart artifact: hybrid selected to
  compose well on a measure, measure stops being a good measure
  once it is the target. §4.7.2 gains the *deception register*
  frame from biological signal-reward contracts (Schiestl et al.
  1999; Cook & Rasplus 2003). §7 restructured into four-tier
  audit (holds up / corrected / real but narrow / limits).
  PROOF_BOUNDARY §2.10 reframed as *continuous-enforcement*
  against mutualism breakdown (Sachs et al. 2004), with PR #160
  dict-order fix as worked example.

## Substantive verdicts at HEAD

| Bench | Verdict | Δ vs B2 |
|---|---|---:|
| §4.7.1 synthetic (Borda hybrid) | `HYBRID_BEATS_BASELINE` | +0.043 |
| §4.7.3 joint (n=6, `seed_long_paragraphs`) | `STRUCTURAL_GAP_NO_MODEL_BEATS` | spread 0.065 |
| §4.7.4 joint (3 corpora, 16 cells: 6+5+5) | `CROSS_CORPUS_VERDICTS_DIVERGE` | 1 BEATS, 8 TIES, 7 LOSES |
| §4.7.4.1 joint (4 corpora, 21 cells: 6+5+5+5) | **`CROSS_CORPUS_VERDICTS_DIVERGE`** (small-n artifact) | 1 BEATS, 10 TIES, 10 LOSES |
| §4.7.4.1 at n≥16 (3 corpora, 16 cells) | **all 3 → `STRUCTURAL_GAP_NO_MODEL_BEATS`** | 0 BEATS, 7 TIES, 9 LOSES |

§4.7.4 cross-corpus matrix (`seed_long_paragraphs` is n=6 with
claude; the two new corpora are n=5 because Anthropic was
unavailable during the §4.7.4 capture):

| Model | `seed_long_paragraphs` | `seed_paragraphs` | `seed_news_briefs` |
|---|---:|---:|---:|
| gpt-4o-mini-2024-07-18 | −0.021 LOSES | **+0.032 BEATS** | −0.023 LOSES |
| claude-haiku-4-5-20251001 | −0.032 LOSES | — | — |
| meta-llama/Llama-3.3-70B | −0.047 LOSES | +0.005 TIES | +0.025 TIES |
| Qwen/Qwen3.6-35B-A3B | +0.003 TIES | +0.027 TIES | −0.016 TIES |
| deepseek-ai/DeepSeek-V3-0324 | +0.018 TIES | −0.042 LOSES | −0.007 TIES |
| google/gemma-3-27b-it | −0.028 LOSES | −0.014 TIES | −0.038 LOSES |

The synthetic-vs-real gap is the load-bearing finding. The lone
real-LLM BEATS cell (`seed_paragraphs` × gpt-4o-mini, +0.032) sits
right at the threshold and on a small sample (n=6 effective docs
post-partition); the synthetic-bench WIN at +0.043 is
substantially larger. Honest reading: the hybrid does not
consistently BEAT baseline on real-LLM perturbations across
LLM families × corpora, but isolated cells can produce positive
Δ at the threshold.

## What's now closed

- The §7 bounded-claims asterisk ("not generalising to real-LLM-
  rendered hallucinations") is no longer load-bearing — §4.7.2
  (single model), §4.7.3 (n=2 → n=6 cross-family), and §4.7.4
  (n=5 × 3-corpus cross-corpus) have measured the gap honestly.
  The hybrid's synthetic-bench WIN is *real on its corpus* but
  *does not consistently generalise across LLM families ×
  corpora*; the synthetic-vs-real magnitude gap (+0.043 vs +0.032
  BEATS-cell ceiling) is real even where the verdict-class gap
  narrows. All claims pinned by digest in CI.

- **Task #22** (Phase 1 / Phase 2 same-process digest contamination)
  — closed by PR #160. Root cause was dict iteration order, not
  BLAS or asyncio state.

- **v0.4+ candidate: open-weights extension** — closed by PR #161.
  Four open-weights families (Meta, Qwen, DeepSeek, Gemma) added;
  none flip the verdict to BEATS on `seed_long_paragraphs`.

- **v0.4+ candidate: cross-corpus extension** — closed by PR #163.
  Two new corpora added (`seed_paragraphs`, `seed_news_briefs`).
  §4.7.3 finding turned out to be corpus-specific:
  `seed_paragraphs` produces one BEATS cell. New §4.7.4 captures
  the corrected, weaker claim.

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

- **v0.4+ candidate: deeper corpus sampling**. The §4.7.4 result
  rests on three corpora. Expanding to 5-10 stylistically distinct
  corpora (scientific abstracts, fiction, legal/policy, code
  commentary, spoken transcripts) would distinguish the lone
  `seed_paragraphs` BEATS cell from threshold-noise.

- **Re-run §4.7.4 with claude-haiku-4.5 once a current Anthropic
  key is available**. The cross-corpus matrix is currently n=5;
  the §4.7.3 n=6 set includes claude. Adding claude on the two
  new corpora would tighten the cross-corpus claim from n=5×3 to
  n=6×3 (18 cells).

## Operator-only items remaining

Same as the prior arc: optional v0.6.0 release, pre-circulation
packet, arXiv submit. The §4.7.3 cross-family corroboration —
now spanning six LLM lineages from six organisations, closed and
open-weights — strengthens the §4.7.x narrative in the same
direction §4.7.2 established. No further substantive work is
needed before submit.
