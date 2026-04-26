// /api/render — Slider-conditioned tome rendering.
//
// POST {triples, slider_position, force_render?}
// → RenderResult per worker/src/cache/bin_cache.ts. Cache-first; LLM
// call only on cache miss. Anthropic is the LLM provider; the render
// path mirrors sum_engine_internal.ensemble.slider_renderer.render
// in shape — same cache key, same canonical-vs-LLM branch, same
// quantization rules.
//
// What this Worker DOES NOT do (yet):
//   - Re-extraction of triples from the rendered tome. The Python
//     bench is the canonical source for fact-preservation metrics.
//     The Worker returns the tome and lets the caller decide.
//   - Per-axis drift measurement. The contract bench produces those
//     numbers ahead of time; live renders just expose the rendered
//     tome and the cache_status.
// Both are deferred to a future revision when the verifier substrate
// can run on Worker (or the demo proxies to a Python service).

import type { Env } from "../index";
import {
  deriveCacheKey,
  getCached,
  putCached,
  type RenderResult,
} from "../cache/bin_cache";
import {
  applyDensity,
  buildSystemPrompt,
  deterministicTome,
  requiresExtrapolator,
  type SlidersForPrompt,
} from "../render/axis_prompts";

interface RenderRequest {
  triples: Array<[string, string, string]>;
  slider_position: SlidersForPrompt;
  force_render?: boolean;
  cache_ttl_seconds?: number;
}

const DEFAULT_TTL_SECONDS = 24 * 60 * 60;
const MAX_OUTPUT_TOKENS = 2048;
const ANTHROPIC_DEFAULT = "claude-haiku-4-5-20251001";

function snapToBin(value: number, bins = 5): number {
  if (value < 0 || value > 1) throw new Error(`slider value out of [0, 1]: ${value}`);
  const idx = Math.min(Math.floor(value * bins), bins - 1);
  return (idx + 0.5) / bins;
}

function quantizeSliders(s: SlidersForPrompt): RenderResult["quantized_sliders"] {
  if (s.density < 0 || s.density > 1) {
    throw new Error(`density out of [0, 1]: ${s.density}`);
  }
  return {
    density: s.density,
    length: snapToBin(s.length),
    formality: snapToBin(s.formality),
    audience: snapToBin(s.audience),
    perspective: snapToBin(s.perspective),
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

function formatTriplesForLLM(triples: Array<[string, string, string]>): string {
  const lines = triples.map((t, i) => `${i + 1}. (${t[0]}, ${t[1]}, ${t[2]})`);
  return `FACTS:\n${lines.join("\n")}`;
}

async function callAnthropic(
  env: Env,
  systemPrompt: string,
  userPrompt: string,
): Promise<string> {
  if (!env.ANTHROPIC_API_KEY) {
    throw new Error("ANTHROPIC_API_KEY not set; render requires an LLM provider");
  }
  const base = env.CF_AI_GATEWAY_BASE
    ? `${env.CF_AI_GATEWAY_BASE.replace(/\/$/, "")}/anthropic/v1/messages`
    : "https://api.anthropic.com/v1/messages";
  const model = env.SUM_DEFAULT_MODEL_ANTHROPIC ?? ANTHROPIC_DEFAULT;

  const res = await fetch(base, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model,
      max_tokens: MAX_OUTPUT_TOKENS,
      system: systemPrompt,
      messages: [{ role: "user", content: userPrompt }],
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`anthropic ${res.status}: ${text.slice(0, 500)}`);
  }

  const data = (await res.json()) as { content?: Array<{ type: string; text?: string }> };
  const block = (data.content ?? []).find((b) => b.type === "text");
  if (!block?.text) throw new Error("anthropic: empty completion");
  return block.text;
}

async function sha256Hex32(s: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
    .slice(0, 16);
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

  const tStart = Date.now();
  let quantized: RenderResult["quantized_sliders"];
  try {
    quantized = quantizeSliders(body.slider_position);
  } catch (e) {
    return json({ error: (e as Error).message }, 400);
  }

  const key = await deriveCacheKey(body.triples, quantized);

  // Cache-first.
  if (!body.force_render) {
    const cached = await getCached(env, key);
    if (cached) {
      return json({
        ...cached,
        cache_status: "hit",
        wall_clock_ms: Date.now() - tStart,
      });
    }
  }

  // Apply density (deterministic axiom subset).
  const keptTriples = applyDensity(body.triples, quantized.density);

  let tome: string;
  let llmCallsMade = 0;
  if (!requiresExtrapolator(quantized)) {
    // Canonical path: deterministic tome from kept triples; no LLM.
    tome = deterministicTome(keptTriples);
  } else {
    const systemPrompt = buildSystemPrompt(quantized);
    const userPrompt = formatTriplesForLLM(keptTriples);
    try {
      tome = await callAnthropic(env, systemPrompt, userPrompt);
      llmCallsMade = 1;
    } catch (e) {
      return json(
        { error: `render failed: ${(e as Error).message}`, cache_key: key },
        502,
      );
    }
  }

  const renderId = await sha256Hex32(key + tome);

  // Build the result. NB: Worker's RenderResult does NOT populate
  // reextracted_triples / claimed_triples / drift — those come from
  // the Python bench. Returns empty arrays / placeholder drift so
  // the schema shape matches across runtimes.
  const result: RenderResult = {
    tome,
    triples_used: keptTriples,
    drift: [],
    cache_status: "miss",
    llm_calls_made: llmCallsMade,
    wall_clock_ms: Date.now() - tStart,
    quantized_sliders: quantized,
    render_id: renderId,
  };

  // Best-effort cache write.
  const ttl = body.cache_ttl_seconds ?? DEFAULT_TTL_SECONDS;
  await putCached(env, key, result, ttl);

  return json(result);
}
