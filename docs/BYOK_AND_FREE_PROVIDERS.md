# BYOK_AND_FREE_PROVIDERS.md

**Run SUM's slider LLM-axis on credits you already have. No OpenAI bill required.**

Operator (Umar) is broke; users following the DOGFOOD_QUICKSTART without OpenAI credits are too. This doc enumerates every free or cheap path that's wired in the substrate today, with copy-pasteable env-var blocks.

## Six routes, one switch

The Python slider's LLM-axis dispatch routes through `LiveLLMAdapter.from_model(model_id, api_key=...)`. The routing decision is made from the model id shape:

| Model id shape | Routes to | Auth | Cost |
|---|---|---|---|
| `gpt-...` / `o1-*` / `o3-*` / `o4-*` | OpenAI (`api.openai.com`) | `OPENAI_API_KEY` | pay-as-you-go |
| `claude-...` | **not yet supported on Python path** — use Worker `/api/render` with `X-Render-LLM-Key-Anthropic` header | `X-Render-LLM-Key-Anthropic` | pay-as-you-go |
| `org/model` (any `/`-namespaced) | **Hugging Face Inference Providers** (`router.huggingface.co/v1`) | `HF_TOKEN` | **HF credits** or pay-as-you-go |
| `ollama:<model>` | Ollama daemon at `localhost:11434/v1` | none | **free, local** |
| `llamacpp:<model>` | llama.cpp server at `localhost:8080/v1` | none | **free, local** |
| `local:<model>` | Custom OpenAI-compatible base (`$SUM_LOCAL_LLM_BASE`) | optional bearer | varies (Modal, Fireworks, vLLM-on-anything) |

The model id is selected via `SUM_TRANSFORM_MODEL` env var (CLI) or `TransformEnv.model` (library callers).

## Recipe: Hugging Face Inference Providers ($800 in HF credits)

The simplest path if you have HF credits.

```bash
# 1. Get a token at https://huggingface.co/settings/tokens (read scope)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 2. Pick a model. The model id MUST contain a `/` for the HF route
#    to fire. Open models that work well at low cost include:
#      meta-llama/Llama-3.3-70B-Instruct
#      Qwen/Qwen2.5-72B-Instruct
#      mistralai/Mistral-Large-Instruct-2411
export SUM_TRANSFORM_MODEL=meta-llama/Llama-3.3-70B-Instruct

# 3. Run a slider render with off-centre LLM axes
echo '{"triples":[["alice","graduated","2012"]]}' \
  | sum transform apply slider \
    --input - \
    --parameters '{"density":1.0,"length":0.9,"formality":0.5,"audience":0.5,"perspective":0.5}'
```

The resulting transform-receipt's `extra.llm_endpoint` will show:
```json
{"model":"meta-llama/Llama-3.3-70B-Instruct","base_url":"https://router.huggingface.co/v1"}
```

— honest provenance about which endpoint actually served. The receipt's `provider` field stays as `"openai"` (the API shape; HF exposes an OpenAI-compatible router) until a v2 schema bumps in dedicated provider literals.

## Recipe: Ollama (free, local)

If you have a Mac / Linux box with enough RAM, Ollama is the cheapest path.

```bash
# 1. Install Ollama (macOS: brew install ollama; Linux: https://ollama.com/download)
brew install ollama
ollama serve  # in another terminal; daemon listens on localhost:11434

# 2. Pull a model
ollama pull llama3.1   # ~4.7GB; 8B params; good for slider testing

# 3. Set SUM to route through Ollama
export SUM_TRANSFORM_MODEL=ollama:llama3.1

# 4. Run a slider render
echo '{"triples":[["alice","graduated","2012"]]}' \
  | sum transform apply slider \
    --input - \
    --parameters '{"density":1.0,"length":0.9,"formality":0.5,"audience":0.5,"perspective":0.5}'
```

No API key required. No bill. Everything local. Receipt's `extra.llm_endpoint`: `{"model":"llama3.1","base_url":"http://localhost:11434/v1"}`.

## Recipe: llama.cpp server (free, local)

Same shape as Ollama but uses llama.cpp's HTTP server. Useful if you've already quantized a model and don't want Ollama's overhead.

```bash
# 1. Start llama.cpp's server with your model
./llama-server -m models/llama-3.3-70b-instruct.Q4_K_M.gguf --port 8080

# 2. Point SUM at it
export SUM_TRANSFORM_MODEL=llamacpp:llama-3.3-70b-instruct

# 3. Run as above
```

## Recipe: Modal ($600 in Modal credits)

Modal can host any model with an OpenAI-compatible API. The pattern:

```python
# modal_serve.py — minimal vLLM-on-Modal recipe (pseudocode)
import modal

app = modal.App("sum-llm-endpoint")

@app.function(image=modal.Image.debian_slim().pip_install("vllm"), gpu="A10G")
@modal.asgi_app()
def serve():
    from vllm.entrypoints.openai.api_server import app as vllm_app
    return vllm_app  # exposes /v1/chat/completions

# `modal deploy modal_serve.py` → https://<your-org>--sum-llm-endpoint-serve.modal.run
```

Then point SUM at it:

```bash
export SUM_LOCAL_LLM_BASE=https://your-org--sum-llm-endpoint-serve.modal.run/v1
export SUM_TRANSFORM_MODEL=local:meta-llama/Llama-3.3-70B-Instruct
# optional: export OPENAI_API_KEY=any-bearer-token-modal-accepts

sum transform apply slider --input doc.json --parameters '...'
```

Modal's pricing rewards short-burst inference; the slider's per-render LLM call fits well within free-tier-friendly cold-start windows for small models.

## Recipe: Fireworks.ai (credits)

Fireworks exposes an OpenAI-compatible API.

```bash
export SUM_LOCAL_LLM_BASE=https://api.fireworks.ai/inference/v1
export OPENAI_API_KEY=fw_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # Fireworks key, OpenAI SDK reuses the env var
export SUM_TRANSFORM_MODEL=local:accounts/fireworks/models/llama-v3p3-70b-instruct
```

## Recipe: OpenAI directly (pay-as-you-go)

The default. No SUM_TRANSFORM_MODEL needed.

```bash
export OPENAI_API_KEY=sk-...
sum transform apply slider --input doc.json --parameters '{"density":1.0,"length":0.9,...}'
```

Defaults to `gpt-4o-mini`. To pick a different OpenAI model:

```bash
export SUM_TRANSFORM_MODEL=gpt-4o
```

## Worker BYO-key route (public demo, no install needed)

For ad-hoc use via the hosted Worker, BYO key by HTTP header. Rate-limited per [`PUBLIC_API_RATE_LIMITS.md`](PUBLIC_API_RATE_LIMITS.md):

```bash
curl -sX POST https://sum-demo.ototao.workers.dev/api/render \
  -H 'content-type: application/json' \
  -H "x-render-llm-key-anthropic: $YOUR_ANTHROPIC_KEY" \
  -d '{"triples":[["alice","graduated","2012"]],"slider_position":{"density":1.0,"length":0.9,"formality":0.5,"audience":0.5,"perspective":0.5}}'
```

Or with OpenAI:

```bash
curl -sX POST https://sum-demo.ototao.workers.dev/api/render \
  -H 'content-type: application/json' \
  -H "x-render-llm-key-openai: $YOUR_OPENAI_KEY" \
  -d '{...}'
```

BYO-keyed Worker calls get **100/hr per IP**. Operator-keyed (no header) is **5/24h per IP**. Drop into the CLI for unlimited.

## What does NOT work yet

- **Anthropic via Python CLI.** The Worker handles it (header above); the Python slider's LLM-axis path is OpenAI-SDK-shaped. Anthropic dispatch through the Python registry is a separate PR.
- **Cohere / Mistral-direct / Together / DeepInfra.** Not wired. Most are OpenAI-compatible — adding them is a matter of either using the `local:` prefix pointed at their base URL OR adding a new prefix to `llm_dispatch._LOCAL_PREFIXES`.
- **Free Hugging Face usage without credits.** HF Inference Providers requires credits for most modern models. Some smaller models may have free quota.

## Sanity check after setup

Quick verification any route works:

```bash
# Should print the slider transform's spec
sum transform list --pretty | jq '.transforms[] | select(.id=="slider")'

# Should succeed with the routed-endpoint info in .extra.llm_endpoint
echo '{"triples":[["a","b","c"]]}' \
  | sum transform apply slider \
    --input - \
    --parameters '{"density":1.0,"length":0.9,"formality":0.5,"audience":0.5,"perspective":0.5}' \
    --pretty \
  | jq '.extra.llm_endpoint'
```

If you see `{"model":"...","base_url":"..."}` matching the route you chose, dispatch is wired correctly.

## Pointers

- `sum_engine_internal/ensemble/live_llm_adapter.py::LiveLLMAdapter.from_model` — routing implementation.
- `sum_engine_internal/ensemble/llm_dispatch.py::get_adapter` — the original adapter-routing helper this builds on.
- `sum_engine_internal/transforms/_base.py::TransformEnv.model` — the env field.
- `sum_engine_internal/transforms/slider.py::_apply_llm_axis` — call site.
- `docs/PUBLIC_API_RATE_LIMITS.md` — Worker-side rate limits and the BYO header surface.
- `docs/DOGFOOD_QUICKSTART.md` — five-minute end-to-end.
