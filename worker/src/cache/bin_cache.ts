import { compareTriples } from "../unicode_order";
// Bin-quantized KV cache for slider renders.
//
// Modelled on sum_engine_internal.ensemble.slider_renderer.SliderCache
// (Python protocol), but Worker-only: nothing shares a cache across the
// two runtimes, and the keys are not compatible (see deriveCacheKey).

import type { Env } from "../index";

// ─── Mirrors the Python RenderResult dataclass ────────────────────

export interface AxisDrift {
  axis: "density" | "length" | "formality" | "audience" | "perspective";
  value: number;
  threshold: number;
  classification: "ok" | "warn" | "fail";
  explanation?: string;
}

export interface RenderResult {
  tome: string;
  triples_used: Array<[string, string, string]>;
  drift: AxisDrift[];
  cache_status: "hit" | "miss" | "bypass";
  llm_calls_made: number;
  wall_clock_ms: number;
  quantized_sliders: {
    density: number;
    length: number;
    formality: number;
    audience: number;
    perspective: number;
  };
  render_id: string;
  // v0.9.A: signed receipt for the render. Optional in the type so
  // pre-receipt cache entries (from older deploys) still deserialise;
  // present on a fresh render when signing is configured and succeeds (a
  // render whose signing fails is returned without one and not cached).
  render_receipt?: unknown;
}

// ─── Cache contract ───────────────────────────────────────────────

/**
 * Cache key derivation: sha256(JSON.stringify({sliders, triples}))[:32]
 * over the code-point-sorted input triples. Same shape as
 * sum_engine_internal.ensemble.slider_renderer.cache_key but NOT the same
 * bytes in general: JSON.stringify writes density 1 as `1` (Python `1.0`),
 * 1e-5 as `0.00001` (Python `1e-05`) and non-ASCII text raw (Python escapes
 * it). routes/render.ts stores the render in KV under
 * `render-v2:<key>:<provider>`, where <provider> is `anthropic` or `openai`
 * on the LLM path (a cache hit there keeps it) and `canonical` on the canonical path.
 */
export async function deriveCacheKey(
  triples: Array<[string, string, string]>,
  quantizedSliders: RenderResult["quantized_sliders"],
): Promise<string> {
  const sortedTriples = [...triples].sort(compareTriples);
  // Construct with keys in alphabetical order so JSON.stringify
  // preserves the same ordering as Python's sort_keys=True.
  const sliders = {
    audience: quantizedSliders.audience,
    density: quantizedSliders.density,
    formality: quantizedSliders.formality,
    length: quantizedSliders.length,
    perspective: quantizedSliders.perspective,
  };
  const payload = { sliders, triples: sortedTriples };
  const canonical = JSON.stringify(payload);
  const enc = new TextEncoder().encode(canonical);
  const hashBuf = await crypto.subtle.digest("SHA-256", enc);
  return Array.from(new Uint8Array(hashBuf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
    .slice(0, 32);
}

/**
 * Get-or-miss. Returns RenderResult on hit; null on miss. Never
 * throws — cache failures fall through to a re-render (the caller
 * decides how to handle).
 */
export async function getCached(
  env: Env,
  key: string,
): Promise<RenderResult | null> {
  if (!env.RENDER_CACHE) return null;
  try {
    const raw = await env.RENDER_CACHE.get(key);
    if (raw === null) return null;
    return JSON.parse(raw) as RenderResult;
  } catch {
    return null;
  }
}

/**
 * Put with TTL. Best-effort; failures swallowed (caller already has
 * the value to return to the user).
 */
export async function putCached(
  env: Env,
  key: string,
  value: RenderResult,
  ttlSeconds: number,
): Promise<void> {
  if (!env.RENDER_CACHE) return;
  try {
    await env.RENDER_CACHE.put(key, JSON.stringify(value), {
      expirationTtl: ttlSeconds,
    });
  } catch {
    // best-effort
  }
}
