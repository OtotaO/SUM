// slider — first registered transform on the Worker side.
//
// Mirrors sum_engine_internal/transforms/slider.py byte-for-byte
// on the three canonicalize_* functions so a transform receipt
// produced by Python verifies in this Worker (and vice versa).
//
// v0 scope (T1b — this PR):
//   - Canonical-path render (all LLM axes centred → deterministic).
//
// Deferred to T1c (LLM-axis dispatch on Worker side):
//   - LLM-axis renders. For now, the Worker's LLM-axis path remains
//     the legacy POST /api/render (with sum.render_receipt.v1).
//     POST /api/transform with transform="slider" + off-centre axes
//     returns HTTP 501 pointing at /api/render.

import canonicalize from "canonicalize";
import type {
  Transform,
  TransformEnv,
  TransformResult,
} from "./_base";

// Quantization mirrors worker/src/routes/render.ts::snapToBin
// + sum_engine_internal/transforms/slider.py::_snap_to_bin.
const BIN_COUNT = 5;
const LLM_AXES = ["length", "formality", "audience", "perspective"] as const;

function snapToBin(value: number, bins = BIN_COUNT): number {
  if (value < 0 || value > 1) {
    throw new Error(`slider value out of [0, 1]: ${value}`);
  }
  const idx = Math.min(Math.floor(value * bins), bins - 1);
  return (idx + 0.5) / bins;
}

interface QuantizedParameters {
  density: number;
  length: number;
  formality: number;
  audience: number;
  perspective: number;
}

function quantize(params: unknown): QuantizedParameters {
  if (!params || typeof params !== "object") {
    throw new Error("slider parameters: must be an object");
  }
  const p = params as Record<string, unknown>;
  const density = p.density;
  if (typeof density !== "number") {
    throw new Error("slider parameters: missing or non-numeric 'density'");
  }
  if (density < 0 || density > 1) {
    throw new Error(`density out of [0, 1]: ${density}`);
  }
  const out: QuantizedParameters = {
    density,
    length: 0,
    formality: 0,
    audience: 0,
    perspective: 0,
  };
  for (const axis of LLM_AXES) {
    const v = p[axis];
    if (typeof v !== "number") {
      throw new Error(`slider parameters: missing or non-numeric '${axis}'`);
    }
    out[axis] = snapToBin(v);
  }
  return out;
}

function requiresExtrapolator(q: QuantizedParameters): boolean {
  // LLM dispatch needed iff any non-density axis is off-centre.
  return LLM_AXES.some((axis) => q[axis] !== 0.5);
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/**
 * Apply density: keep the leading `ceil(density * len)` triples by
 * SHA-256-stable deterministic order. Empty input yields empty
 * output. Density 0 yields exactly one triple (the prefix);
 * density 1 yields all triples. Mirrors the Python helper plus
 * worker/src/render/axis_prompts.ts::applyDensity.
 */
async function applyDensity(
  triples: Array<[string, string, string]>,
  density: number,
): Promise<Array<[string, string, string]>> {
  if (triples.length === 0) return [];
  // Sort by SHA-256 of joined triple — same key as Python.
  const withHash: Array<[string, [string, string, string]]> = [];
  for (const t of triples) {
    const h = await sha256Hex(new TextEncoder().encode(t.join("|")));
    withHash.push([h, t]);
  }
  withHash.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  const sorted = withHash.map(([, t]) => t);
  if (density <= 0) return sorted.slice(0, 1);
  // 0.9999 epsilon mirrors Python int(len * density + 0.9999) ceiling.
  const keep = Math.max(1, Math.floor(sorted.length * density + 0.9999));
  return sorted.slice(0, keep);
}

/**
 * Pure-algorithmic prose composition. Mirrors the Python helper
 * plus worker/src/render/axis_prompts.ts::deterministicTome.
 */
function deterministicTome(triples: Array<[string, string, string]>): string {
  if (triples.length === 0) return "";
  const sentences = triples.map(([s, p, o]) => {
    const article = s.charAt(0).toUpperCase() === s.charAt(0) ? "" : "The";
    const pred = p.replace(/_/g, " ");
    return `${article} ${s} ${pred} ${o}.`.replace(/  /g, " ").trim();
  });
  return sentences.join(" ");
}

function sortComponentwise(
  triples: Array<[string, string, string]>,
): Array<[string, string, string]> {
  return [...triples].sort((a, b) => {
    for (let i = 0; i < 3; i++) {
      if (a[i] < b[i]) return -1;
      if (a[i] > b[i]) return 1;
    }
    return 0;
  });
}

export const SLIDER_TRANSFORM: Transform = {
  name: "slider",
  requiresLLM: true, // may require; canonical-path doesn't.
  digitalSourceType: "trainedAlgorithmicMedia",

  canonicalizeParameters(params: unknown): Uint8Array {
    const q = quantize(params);
    const canonical = canonicalize(q);
    if (typeof canonical !== "string") {
      throw new Error("canonicalize returned undefined for slider params");
    }
    return new TextEncoder().encode(canonical);
  },

  canonicalizeInput(rawInput: unknown): Uint8Array {
    if (!rawInput || typeof rawInput !== "object" || !("triples" in rawInput)) {
      throw new Error("slider input: expected dict with 'triples' key");
    }
    const triples = (rawInput as { triples: unknown[] }).triples.map((t) => {
      const arr = t as unknown[];
      return [String(arr[0]), String(arr[1]), String(arr[2])] as [string, string, string];
    });
    const sorted = sortComponentwise(triples);
    const canonical = canonicalize(sorted);
    if (typeof canonical !== "string") {
      throw new Error("canonicalize returned undefined for slider input");
    }
    return new TextEncoder().encode(canonical);
  },

  canonicalizeOutput(output: unknown): Uint8Array {
    if (typeof output !== "string") {
      throw new Error("slider output: expected string (tome)");
    }
    return new TextEncoder().encode(output);
  },

  async apply(
    input: unknown,
    parameters: Record<string, unknown>,
    _env: TransformEnv,
  ): Promise<TransformResult> {
    if (!input || typeof input !== "object" || !("triples" in input)) {
      throw new Error("slider input: expected dict with 'triples' key");
    }
    const triples = (input as { triples: unknown[] }).triples.map((t) => {
      const arr = t as unknown[];
      return [String(arr[0]), String(arr[1]), String(arr[2])] as [string, string, string];
    });

    const quantized = quantize(parameters);
    const kept = await applyDensity(triples, quantized.density);

    if (requiresExtrapolator(quantized)) {
      // T1b deliberately does not wire the LLM dispatch into the
      // transform-registry path. The legacy POST /api/render still
      // serves LLM-axis renders (with sum.render_receipt.v1).
      // Closing this gap is T1c.
      throw new Error(
        "slider transform v0 (T1b) supports canonical-path renders " +
          "only. For LLM-axis renders (any of length/formality/audience/" +
          "perspective != 0.5), use POST /api/render which produces " +
          "sum.render_receipt.v1. LLM-axis dispatch via the transform " +
          "registry lands in T1c.",
      );
    }

    const tome = deterministicTome(kept);
    return {
      output: tome,
      modelUsed: "canonical-deterministic-v0",
      provider: "canonical-path",
      digitalSourceType: "algorithmicMedia",
      llmCallsMade: 0,
      extra: {
        quantizedParameters: quantized,
        triplesUsed: kept,
      },
    };
  },
};
