# Text-First Moonshot Plan

## Mission
Build a **bi-directional distillation/extrapolation engine** that can move between:

- **Tag-scale output** (atomic concepts)
- **Sentence + paragraph digests**
- **Full structured documents/books**

...through a single, controllable **knowledge density slider**.

## Why Text-First
Before expanding deeply into audio/video/image, the system should become best-in-class on text because:

1. Text is the shared representation for most knowledge workflows.
2. Text allows precise quality evaluation (faithfulness, coverage, compression ratio).
3. Text provides the best proving ground for scaling to arbitrary size.

## Product Contract (v1)

### 1) Universal Density Slider
Define slider levels as deterministic contracts:

- **0 = Tags**: keyword/entity set
- **1 = Essence**: 1-2 sentence ultra-summary
- **2 = Digest**: short multi-paragraph brief
- **3 = Executive**: ~1 page summary with key sections
- **4 = Article**: detailed long-form synthesis
- **5 = Expansion**: full multi-section generation from sparse prompts
- **6 = Book**: chapterized extrapolation with table of contents + chapter drafts

Each level should specify target ranges for:
- output token budget
- section structure
- citations/traceability requirements

### 2) Arbitrary-Size Input Support
Implement a hierarchical pipeline designed for very large corpora:

1. **Ingestion**: stream + normalize files into text blocks.
2. **Segmentation**: semantic chunking with overlap and adaptive sizing.
3. **Local Pass**: per-chunk distill/extract representations.
4. **Regional Merge**: combine chunk outputs into section-level syntheses.
5. **Global Synthesis**: produce requested density level.
6. **Refinement Pass**: enforce style, factual consistency, and formatting.

This map-reduce style architecture avoids context-window hard limits and supports 1KB → TB-scale inputs.

### 3) Running-Total Progress Window
Expose progress as a first-class API object (and SSE stream), not only as UI percentages.

Core fields:
- `job_id`
- `phase` (ingest/chunk/local/merge/synthesis/refine)
- `items_total`
- `items_done`
- `tokens_in_total`
- `tokens_in_processed`
- `tokens_out_generated`
- `elapsed_seconds`
- `eta_seconds`
- `throughput_items_per_sec`
- `throughput_tokens_per_sec`

This provides the user an honest running-total view on massive jobs and makes scheduling + autoscaling measurable.

## Engineering Milestones

### Milestone A — Deterministic Density Contracts
- Lock per-level output specs.
- Add regression fixtures: same input, predictable output envelope by level.
- Add quality gates: faithfulness + non-hallucination checks at compression levels.

### Milestone B — Scalable Text Processing Core
- Upgrade chunker to support adaptive chunk sizes and resumable processing.
- Add checkpointing every N chunks for long-running jobs.
- Add cache keys by `(input_hash, slider_level, model_profile, prompt_version)`.

### Milestone C — Unified Progress Telemetry
- Standardize progress schema across all endpoints.
- Stream updates over SSE/WebSocket with phase transitions.
- Persist job timeline for observability and postmortems.

### Milestone D — Multi-Model Orchestration (Text)
- Route by task:
  - extraction/classification model for tags
  - summarization model for contraction levels
  - generation/planning model for extrapolation levels
- Add policy engine for cost/latency/quality targets.
- Add fallback model chain for reliability.

### Milestone E — Evaluation Harness
- Build benchmark suites:
  - short docs
  - long technical docs
  - multi-document corpora
  - noisy OCR text
- Track metrics: faithfulness, coverage, latency, cost, stability.

## Initial API Shape

```json
{
  "input": "...",
  "slider_level": 2,
  "mode": "auto",
  "model_profile": "balanced",
  "stream": true,
  "progress": {
    "running_total": true,
    "emit_interval_ms": 500
  }
}
```

## Definition of Done (Text v1)
- Handles large text jobs via hierarchical streaming pipeline.
- Maintains stable quality from tags to book-scale generation.
- Exposes transparent, accurate running totals during processing.
- Supports multi-model routing with robust fallbacks.
