// Bin-quantized KV cache for slider renders.
//
// Mirrors the contract of sum_engine_internal.ensemble.slider_renderer.
// SliderCache (Python protocol). Two backends compatible at the key
// level: this Worker-side KV-backed implementation, and the Python
// in-memory implementation used by tests. Same key derivation; same
// stored value shape; cache-coherence between them is guaranteed by
// the content-addressed key.
//
// SCAFFOLD STATE: type contracts only. Logic ships in EXECUTE.

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
}

// ─── Cache contract ───────────────────────────────────────────────

/**
 * Cache key derivation. MUST match
 * sum_engine_internal.ensemble.slider_renderer.cache_key byte-for-byte
 * so a Python-rendered RenderResult can be served by the Worker-side
 * cache and vice versa. Same JSON canonicalisation, same SHA-256,
 * same 32-char truncation.
 *
 * STATE 4 deliverable. Until then, throws to fail loud rather than
 * silently produce a key that diverges from the Python side.
 */
export async function deriveCacheKey(
  triples: Array<[string, string, string]>,
  quantizedSliders: RenderResult["quantized_sliders"],
): Promise<string> {
  // Sort triples lexicographically (matches Python `sorted()`).
  const sortedTriples = [...triples].sort();
  const payload = {
    triples: sortedTriples,
    sliders: quantizedSliders,
  };
  const canonical = JSON.stringify(payload);  // not JCS — Python uses json.dumps with sort_keys
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
