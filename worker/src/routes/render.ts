// /api/render — Slider-conditioned tome rendering.
//
// Phase E.3 endpoint. POST {triples, slider_position, force_render?}
// → RenderResult per worker/src/cache/bin_cache.ts. Cache-first; LLM
// call only on cache miss. Drift measured server-side; UI just
// displays.
//
// SCAFFOLD STATE: handler returns 501 with the activation plan
// inline. Logic ships in EXECUTE state.

import type { Env } from "../index";
import {
  deriveCacheKey,
  getCached,
  putCached,
  type RenderResult,
} from "../cache/bin_cache";

interface RenderRequest {
  triples: Array<[string, string, string]>;
  slider_position: {
    density: number;
    length: number;
    formality: number;
    audience: number;
    perspective: number;
  };
  force_render?: boolean;       // bypass cache; always re-LLM
  cache_ttl_seconds?: number;   // override default 24h
}

const DEFAULT_TTL_SECONDS = 24 * 60 * 60;
const SLIDER_BINS_PER_AXIS = 5;

/**
 * Snap a continuous [0, 1] value to the centre of one of N bins.
 * Mirror of sum_engine_internal.ensemble.tome_sliders.snap_to_bin.
 * Pure function. Same input → same output; identical to Python side.
 */
function snapToBin(value: number, bins: number = SLIDER_BINS_PER_AXIS): number {
  if (value < 0 || value > 1) throw new Error(`slider value out of [0, 1]: ${value}`);
  const idx = Math.min(Math.floor(value * bins), bins - 1);
  return (idx + 0.5) / bins;
}

function quantizeSliders(sliders: RenderRequest["slider_position"]): RenderResult["quantized_sliders"] {
  return {
    density: snapToBin(sliders.density),
    length: snapToBin(sliders.length),
    formality: snapToBin(sliders.formality),
    audience: snapToBin(sliders.audience),
    perspective: snapToBin(sliders.perspective),
  };
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

export async function handleRender(
  request: Request,
  env: Env,
  _ctx: ExecutionContext,
): Promise<Response> {
  if (request.method !== "POST") {
    return json({ error: "method not allowed; use POST" }, 405);
  }

  let body: RenderRequest;
  try {
    body = (await request.json()) as RenderRequest;
  } catch {
    return json({ error: "invalid JSON body" }, 400);
  }

  if (!Array.isArray(body.triples) || body.triples.length === 0) {
    return json({ error: "missing or empty 'triples' array" }, 400);
  }
  if (!body.slider_position) {
    return json({ error: "missing 'slider_position'" }, 400);
  }

  // Quantize → cache key.
  let quantized: RenderResult["quantized_sliders"];
  try {
    quantized = quantizeSliders(body.slider_position);
  } catch (e) {
    return json({ error: (e as Error).message }, 400);
  }

  const key = await deriveCacheKey(body.triples, quantized);

  // Cache-first path (unless force_render).
  if (!body.force_render) {
    const cached = await getCached(env, key);
    if (cached) {
      return json({ ...cached, cache_status: "hit" });
    }
  }

  // SCAFFOLD STATE: no LLM call yet. Return 501 with the activation plan.
  // STATE 4 will:
  //   1. Build system_prompt via build_system_prompt(quantized) shipped
  //      from a Pythonic /api/internal/render-prompt route OR replicate
  //      the prompt-building logic in TS to keep the Worker self-
  //      contained. Decision deferred to EXECUTE.
  //   2. Call /api/complete (existing Anthropic/OpenAI proxy) with the
  //      conditioned prompts.
  //   3. Re-extract triples from the returned tome (sieve via
  //      something we don't have on the Worker; or fall back to
  //      asking the LLM to re-extract).
  //   4. Compute drift per axis per docs/SLIDER_CONTRACT.md.
  //   5. Build RenderResult and cache it.
  return json(
    {
      error: "not implemented",
      status: "phase-e-scaffold",
      cache_key: key,
      quantized_sliders: quantized,
      next: {
        action: "Implement render pipeline in worker/src/routes/render.ts",
        spec: "See docs/SLIDER_CONTRACT.md for axis semantics + drift formulas.",
        depends_on: [
          "Decide: server-side drift measurement (needs sieve in TS) " +
            "or client-side (browser already has WASM core).",
          "Wire RENDER_CACHE KV namespace in wrangler.toml.",
        ],
      },
    },
    501,
  );
}

// Suppress unused-warnings until EXECUTE wires these in.
void putCached;
void DEFAULT_TTL_SECONDS;
