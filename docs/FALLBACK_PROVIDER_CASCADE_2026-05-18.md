# FALLBACK_PROVIDER_CASCADE_2026-05-18.md

**Operational design + roadmap for the four-tier LLM-provider cascade SUM uses (and will use) to serve the slider's LLM-axis without operator-wallet exposure.**

The hosted Worker is public-facing; the operator is broke. Naive single-provider dispatch means either (a) the operator pays for every render and runs out of credit, or (b) callers without OpenAI keys can't use the slider at all. The cascade closes both gaps by routing each request to the cheapest tier that can serve it.

This document is the design + research-snapshot. The substrate's lowest tiers (HF, Ollama, llama.cpp, local, NVIDIA NIM, Groq, Cerebras) are wired today via `LiveLLMAdapter.from_model`; the orchestrator that chooses between them automatically is planned-not-shipped.

## The four tiers

### Tier 0 — In-browser (zero cost, zero network)

**2026 reality, summarised from the research pass on 2026-05-17:**

- WebGPU shipped by default across Chrome, Firefox, Edge, Safari in November 2025. Desktop coverage ~82.7%; mobile ~70–75%.
- WebGPU delivers 3–5× speedups over WebGL for transformer models; 10–15× over WebAssembly.
- Transformers.js v4 (Feb 2026) rewrote runtime in C++ with a WebGPU backend — 3–10× faster than v3. 200+ model architectures, 1,200+ converted models. Llama 3.2 3B at ~60 tok/s in browser.
- WebLLM (mlc-ai) is the fastest in-browser engine; its API mirrors OpenAI's. **Endpoint swap, not rewrite.**

**Right fit for SUM:** the *small* operations — re-extraction during slider rendering (NLI audit per cell, axis-token classification, hedge/conditional detection in the negative-control corpus), embeddings for similarity, short-prose generation. NOT large-context (16-paragraph slider corpus) or high-stakes generation.

**Implementation status:** not yet shipped. The single-file browser demo currently uses a sieve-style fallback for in-browser extraction (no LLM). The path to add WebLLM:

1. Vendor a `webllm.js` ESM bundle into `single_file_demo/vendor/` (already pattern-matched by the vendored jose).
2. Wire `single_file_demo/index.html` to expose a "use in-browser LLM" toggle that initialises WebLLM with a small model (Llama 3.2 3B Q4 quantised, ~2GB download).
3. Slider client logic: if all axes are at 0.5, run canonical-path (no LLM); if displacement is small and a single-document context, prefer in-browser; else hit the Worker.
4. The receipt path is unchanged — in-browser-generated tomes still sign through the Worker's signing key (no per-browser keypair issuance in v1).

**Why it's not in this PR:** ~3–5 days of focused single-file-demo work, plus a substantive review of the receipt-issuance story for client-generated tomes. Worth it; just not now.

### Tier 1 — Free hosted-API tiers (zero $ until quota exhausted)

Three providers wired in this PR via `LiveLLMAdapter.from_model`:

| Provider | Free quota | Prefix | Env var | Models |
|---|---|---|---|---|
| **NVIDIA NIM** | 1000 credits on signup (up to 5000 with request); 40 req/min/model | `nim:<model>` | `NVIDIA_API_KEY` | 80+ models incl. Llama, Mistral, Microsoft open weights, Zhipu GLM-4 (free of credit) |
| **Groq** | Daily token quota | `groq:<model>` | `GROQ_API_KEY` | Llama 3.3 70B, Mixtral, Gemma2. Fastest TTFT (<300ms) |
| **Cerebras** | Daily token quota | `cerebras:<model>` | `CEREBRAS_API_KEY` | gpt-oss-120B at 3000 tok/sec (fastest end-to-end) |

Worked examples:

```bash
# NVIDIA NIM (1000 free credits on signup — get key at build.nvidia.com)
export NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxx
export SUM_TRANSFORM_MODEL=nim:meta/llama-3.3-70b-instruct
sum transform apply slider --input doc.json --parameters '{"density":1.0,"length":0.9,...}'

# Groq (free daily quota — get key at console.groq.com)
export GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
export SUM_TRANSFORM_MODEL=groq:llama-3.3-70b-versatile
sum transform apply slider --input doc.json --parameters '{...}'

# Cerebras (free daily quota — get key at cloud.cerebras.ai)
export CEREBRAS_API_KEY=csk-xxxxxxxxxxxxxxxxxxxx
export SUM_TRANSFORM_MODEL=cerebras:llama-4-scout-17b-16e-instruct
sum transform apply slider --input doc.json --parameters '{...}'
```

**Implementation status:** wired in this PR. Tests lock the routing.

### Tier 2 — User's existing credit pile (already paid for)

- **Hugging Face Inference Providers**: `org/model` route, `HF_TOKEN`. Wired in PR #238.
- **Modal**: `local:<model>` + `$SUM_LOCAL_LLM_BASE` pointed at a Modal-hosted OpenAI-compatible endpoint. Wired in PR #238.
- **Fireworks.ai**: same shape as Modal (`local:` + base URL). Wired in PR #238.

### Tier 3 — Paid pay-as-you-go (cheapest fallback)

- **DeepInfra**: lowest cost-per-token (76% cheaper than Together on Llama 4 Maverick). Widest open-model catalog. OpenAI-compatible — route via `local:` with `SUM_LOCAL_LLM_BASE=https://api.deepinfra.com/v1/openai`.
- **Together AI**: pricier but full fine-tuning support.
- **Fireworks**: best all-rounder, faster TTFT than Together.
- **OpenAI / Anthropic**: default routes. Anthropic via Worker BYO-header only (Python slider is OpenAI-SDK-shaped).
- **AWS Bedrock**: NOT WIRED. Bedrock uses SigV4 auth, not OpenAI-compatible. The $200 new-account credit is one-shot and expires in 6 months. **Not worth a new adapter for the ROI**. Recommendation: route to DeepInfra or Together instead — they host most of the same open-weights models on OpenAI-compatible APIs.

## The orchestrator (planned, not shipped)

The cascade today requires manual routing — caller picks the prefix. The orchestrator at `sum_engine_internal/ensemble/cascade_dispatch.py` (not yet shipped) would automate:

1. **Classify request:** small (≤500 tokens, single doc) vs. medium (≤5K tokens) vs. large (multi-doc).
2. **Pick tier:** in-browser if applicable AND browser supports WebGPU; else cheapest free tier with quota remaining; else BYO credits; else paid pay-as-you-go.
3. **Fall through on errors:** 401 (auth fail) → next tier; 429 (quota) → next tier; 500 (provider down) → next tier.
4. **Record provenance:** `extra.llm_endpoint.tier` reports which tier served. Operator audit + dogfood findings can both see the cascade in motion.

This is the "beautiful fallback system" the user asked for on 2026-05-18. Single PR's worth of work once the small-PR cadence permits.

## What this PR ships

- Three free-tier prefixes (`nim:`, `groq:`, `cerebras:`) in `LiveLLMAdapter.from_model`.
- Six new tests locking the routing + error contracts.
- BYOK doc extension with the three new recipes.
- This planning artifact.

## What this PR does NOT ship (deferred)

- The orchestrator (Tier-picker + fall-through logic).
- In-browser LLM via WebLLM (Tier 0).
- AWS Bedrock adapter.
- A new Provider literal value for the receipt schema (so HF / Ollama / NIM all still report `provider: "openai"` because the API SHAPE is OpenAI's — the actual routing target is in `extra.llm_endpoint`).

## Re-entry conditions

Update this document when:

- A new free-tier provider lands worth wiring (currently watching: any provider that exposes an OpenAI-compatible router AND has a free daily quota AND ships open-weights models).
- WebGPU coverage crosses ≥90% mobile (would justify shipping Tier 0).
- The first paid-provider quota actually exhausts in practice (would inform the orchestrator's fall-through tuning).
- A research subpackage in `sum_engine_internal/research/` matures into something that should ride the cascade (e.g., the bench-hardening T2 capability-region work would want cheap mass inference on a free tier).

## Pointers

- `sum_engine_internal/ensemble/live_llm_adapter.py::LiveLLMAdapter.from_model` — wired prefixes
- `sum_engine_internal/ensemble/llm_dispatch.py` — base URLs + env-var names
- [`docs/BYOK_AND_FREE_PROVIDERS.md`](BYOK_AND_FREE_PROVIDERS.md) — user-facing recipes
- [`docs/PUBLIC_API_RATE_LIMITS.md`](PUBLIC_API_RATE_LIMITS.md) — Worker-side BYO-key gate
- [`docs/CHARTER_2026-05-17.md`](CHARTER_2026-05-17.md) §3 — strategy: "substrate first → adoption through writers"; cascade is substrate, not product
- [`docs/ZENITH_FRAMING_2026-05-16.md`](ZENITH_FRAMING_2026-05-16.md) — destination framing; the orchestrator implementing the cascade is the user-facing surface that makes "Tier 0 / 1 / 2 / 3" legible to the Epistemic Nutrition Label

## Sources

- NVIDIA NIM developer program — [build.nvidia.com](https://build.nvidia.com), [NIM for Developers](https://developer.nvidia.com/nim)
- Groq Cloud — [console.groq.com](https://console.groq.com)
- Cerebras Cloud — [cloud.cerebras.ai](https://cloud.cerebras.ai)
- WebLLM — [mlc-ai/web-llm](https://github.com/mlc-ai/web-llm)
- Transformers.js v4 — [WebGPU backend release notes](https://huggingface.co/docs/transformers.js)
- DeepInfra cost comparison — research pass 2026-05-17 (76% cheaper than Together on Llama 4 Maverick)
- AWS Bedrock $200 new-account credit policy — [aws.amazon.com/bedrock/pricing](https://aws.amazon.com/bedrock/pricing/)
