// /api/render — Slider-conditioned tome rendering.
//
// POST {triples, slider_position, provider?, force_render?}
// → RenderResult per worker/src/cache/bin_cache.ts. Cache-first; LLM
// call only on cache miss. Anthropic and OpenAI are both supported;
// the render path mirrors sum_engine_internal.ensemble.slider_renderer
// .render in shape — same cache key, same canonical-vs-LLM branch,
// same quantization rules. Provider selection:
//   - body.provider == "openai"    → callOpenAI (requires OPENAI_API_KEY)
//   - body.provider == "anthropic" → callAnthropic (requires ANTHROPIC_API_KEY)
//   - body.provider absent         → Anthropic if configured, else OpenAI
// The signed receipt's `provider` field reflects what actually served,
// including whether the call went via CF AI Gateway.
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
import {
  DIGITAL_SOURCE_TYPE_AI,
  DIGITAL_SOURCE_TYPE_DETERMINISTIC,
  hashTome,
  hashTriples,
  signReceipt,
  type ReceiptPayload,
} from "../receipt/sign";
import type { JWK } from "jose";
import { checkRateLimit, classifyScope, rateLimitedResponse } from "../rate_limit";
import { admitLLM } from "../llm_budget";
import { readBoundedJSON, requireObject, validateTriples, validateSliders, RequestError, errorResponse, providerJSON } from "../request_limits";

type ProviderChoice = "anthropic" | "openai";

interface RenderRequest {
  triples: Array<[string, string, string]>;
  slider_position: SlidersForPrompt;
  /** Optional provider override. Absent ⇒ Anthropic if configured,
   * else OpenAI. Explicit choice fails fast if the matching key is
   * not configured — no silent provider substitution, because the
   * receipt's `provider` field would no longer match the caller's
   * intent.
   *
   * Cache key includes the provider, so OpenAI and Anthropic renders
   * of the same (triples, sliders) do not collide. */
  provider?: ProviderChoice;
  force_render?: boolean;
  cache_ttl_seconds?: number;
}

const DEFAULT_TTL_SECONDS = 24 * 60 * 60;
const MAX_OUTPUT_TOKENS = 2048;
const ANTHROPIC_DEFAULT = "claude-haiku-4-5-20251001";
const OPENAI_DEFAULT = "gpt-4o-mini";

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

interface LLMRenderResult {
  tome: string;
  /** What actually served the call. Comes from the API response's
   *  `model` field, not the requested model — Anthropic and OpenAI
   *  may resolve a tag to a more specific snapshot id. The receipt
   *  signs THIS value so the audit trail matches reality. */
  model_used: string;
  provider:
    | "anthropic"
    | "openai"
    | "cf-ai-gateway-anthropic"
    | "cf-ai-gateway-openai"
    | "canonical-path";
}

async function callAnthropic(
  env: Env,
  systemPrompt: string,
  userPrompt: string,
  userKey?: string,
): Promise<LLMRenderResult> {
  // User-supplied key takes precedence (BYO-keys mode); falls back to
  // operator's env var (operator-funded mode). Empty user-key strings
  // are treated as absent so a stray empty header doesn't break the
  // operator-funded path.
  const byoKey = userKey?.trim();
  const apiKey = byoKey || env.ANTHROPIC_API_KEY?.trim();
  if (!apiKey) {
    throw new Error("ANTHROPIC_API_KEY not set on Worker and no X-Render-LLM-Key-Anthropic header supplied");
  }
  // CF AI Gateway routing is only honoured for operator-funded calls;
  // a BYO-key user shouldn't be silently proxied through the
  // operator's gateway (would mix metrics and could double-bill).
  const usingGateway = Boolean(env.CF_AI_GATEWAY_BASE) && !byoKey;
  const base = usingGateway
    ? `${env.CF_AI_GATEWAY_BASE!.replace(/\/$/, "")}/anthropic/v1/messages`
    : "https://api.anthropic.com/v1/messages";
  const requestedModel = env.SUM_DEFAULT_MODEL_ANTHROPIC ?? ANTHROPIC_DEFAULT;

  const data = await providerJSON(base, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: requestedModel,
      max_tokens: MAX_OUTPUT_TOKENS,
      system: systemPrompt,
      messages: [{ role: "user", content: userPrompt }],
    }),
  }) as {
    content?: Array<{ type: string; text?: string }>;
    model?: string;
  };

  const block = (data.content ?? []).find((b) => b.type === "text");
  if (!block?.text) throw new Error("anthropic: empty completion");
  // Honest model identifier: prefer the API's reported model (which
  // may be a more specific snapshot id than the requested tag); fall
  // back to the requested model with `_inferred` suffix when absent
  // so the inference itself is visible in the signed receipt.
  const modelUsed = data.model ?? `${requestedModel}_inferred`;
  return {
    tome: block.text,
    model_used: modelUsed,
    provider: usingGateway ? "cf-ai-gateway-anthropic" : "anthropic",
  };
}

async function callOpenAI(
  env: Env,
  systemPrompt: string,
  userPrompt: string,
  userKey?: string,
): Promise<LLMRenderResult> {
  // Same BYO-key precedence as callAnthropic; see its comments.
  const byoKey = userKey?.trim();
  const apiKey = byoKey || env.OPENAI_API_KEY?.trim();
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY not set on Worker and no X-Render-LLM-Key-OpenAI header supplied");
  }
  const usingGateway = Boolean(env.CF_AI_GATEWAY_BASE) && !byoKey;
  const base = usingGateway
    ? `${env.CF_AI_GATEWAY_BASE!.replace(/\/$/, "")}/openai/chat/completions`
    : "https://api.openai.com/v1/chat/completions";
  const requestedModel = env.SUM_DEFAULT_MODEL_OPENAI ?? OPENAI_DEFAULT;

  const data = await providerJSON(base, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: requestedModel,
      max_tokens: MAX_OUTPUT_TOKENS,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
    }),
  }) as {
    choices?: Array<{ message?: { content?: string } }>;
    model?: string;
  };

  const text = data.choices?.[0]?.message?.content;
  if (!text) throw new Error("openai: empty completion");
  const modelUsed = data.model ?? `${requestedModel}_inferred`;
  return {
    tome: text,
    model_used: modelUsed,
    provider: usingGateway ? "cf-ai-gateway-openai" : "openai",
  };
}

function resolveProvider(
  env: Env,
  requested?: ProviderChoice,
  userKeys?: { anthropic?: string; openai?: string },
): ProviderChoice {
  if (requested === "openai") return "openai";
  if (requested === "anthropic") return "anthropic";
  // Default-provider resolution considers BOTH user-supplied keys and
  // operator env vars as "configured." A BYO-keys user who supplied
  // an Anthropic key gets the anthropic path even if the operator
  // has no Anthropic env var.
  const hasAnthropic = Boolean(userKeys?.anthropic?.trim() || env.ANTHROPIC_API_KEY?.trim());
  const hasOpenAI = Boolean(userKeys?.openai?.trim() || env.OPENAI_API_KEY?.trim());
  if (hasAnthropic) return "anthropic";
  if (hasOpenAI) return "openai";
  throw new Error(
    "render requires at least one LLM provider configured " +
      "(ANTHROPIC_API_KEY or OPENAI_API_KEY)",
  );
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
    const value = await readBoundedJSON(request);
    requireObject(value);
    validateTriples(value.triples);
    validateSliders(value.slider_position);
    if (value.provider !== undefined && value.provider !== "anthropic" && value.provider !== "openai") throw new RequestError("provider must be anthropic or openai");
    if (value.force_render !== undefined && typeof value.force_render !== "boolean") throw new RequestError("force_render must be boolean");
    if (value.cache_ttl_seconds !== undefined && (typeof value.cache_ttl_seconds !== "number" || !Number.isInteger(value.cache_ttl_seconds) || value.cache_ttl_seconds < 60 || value.cache_ttl_seconds > DEFAULT_TTL_SECONDS)) throw new RequestError("cache_ttl_seconds must be an integer from 60 to 86400");
    body = value as unknown as RenderRequest;
  } catch (error) { return errorResponse(error); }

  const tStart = Date.now();
  let quantized: RenderResult["quantized_sliders"];
  try {
    quantized = quantizeSliders(body.slider_position);
  } catch (e) {
    return json({ error: (e as Error).message }, 400);
  }

  // BYO-keys mode: user-supplied keys via request headers take
  // precedence over the operator's env vars inside callAnthropic /
  // callOpenAI. The receipt's `provider` field still reflects what
  // actually served (`anthropic` / `openai`); the Worker never
  // persists the user-supplied key.
  const userKeys = {
    anthropic: request.headers.get("x-render-llm-key-anthropic")?.trim() || undefined,
    openai: request.headers.get("x-render-llm-key-openai")?.trim() || undefined,
  };

  // Canonical rendering needs neither provider credentials nor a paid quota.
  const needsLLM = requiresExtrapolator(quantized);
  let providerChoice: ProviderChoice | undefined;
  if (needsLLM) {
    try {
      providerChoice = resolveProvider(env, body.provider, userKeys);
      if (!(userKeys[providerChoice] || (providerChoice === "openai" ? env.OPENAI_API_KEY?.trim() : env.ANTHROPIC_API_KEY?.trim()))) {
        return json({ error: `no credential configured for ${providerChoice}` }, 503);
      }
    } catch (e) { return json({ error: (e as Error).message }, 503); }
  }
  const selectedUserKey = providerChoice ? userKeys[providerChoice] : undefined;
  if (!needsLLM && env.RENDER_CACHE) {
    const rl = await checkRateLimit(request, env.RENDER_CACHE, "canonical");
    if (!rl.allowed) return rateLimitedResponse(rl);
  }

  // Versioned provider namespace avoids reusing old ambiguous default-provider
  // cache entries after provider configuration changes. Receipts stay immutable.
  const baseKey = await deriveCacheKey(body.triples, quantized);
  const key = `render-v2:${baseKey}:${providerChoice ?? "canonical"}`;

  // Cache-first. The cached value already carries its render_receipt
  // (signed at the time of the original miss render); HIT path
  // re-serves the same receipt verbatim so the kid + signed_at +
  // tome_hash all remain consistent with the issuance.
  if (!body.force_render) {
    const cached = await getCached(env, key);
    if (cached) {
      return json({
        ...cached,
        cache_status: "hit",
        llm_calls_made: 0,
        wall_clock_ms: Date.now() - tStart,
      });
    }
  }

  // Apply density (deterministic axiom subset).
  const keptTriples = applyDensity(body.triples, quantized.density);

  let tome: string;
  let llmCallsMade = 0;
  // Honest provenance: track exactly which model + provider produced
  // the tome. Receipt signs THESE values, never the configured-default
  // values (which can drift from reality on provider fallback or model
  // snapshot resolution).
  let modelUsed: string = "canonical-deterministic-v0";
  let providerUsed: LLMRenderResult["provider"] = "canonical-path";
  if (!needsLLM) {
    // Canonical path: deterministic tome from kept triples; no LLM.
    tome = deterministicTome(keptTriples);
  } else {
    const systemPrompt = buildSystemPrompt(quantized);
    const userPrompt = formatTriplesForLLM(keptTriples);
    const scope = classifyScope("render", selectedUserKey) as "llm-axis-demo" | "llm-axis-byok";
    const admission = await admitLLM(request, env, scope);
    if (admission instanceof Response) return admission;
    try {
      const llmResult =
        providerChoice === "openai"
          ? await callOpenAI(env, systemPrompt, userPrompt, selectedUserKey)
          : await callAnthropic(env, systemPrompt, userPrompt, selectedUserKey);
      tome = llmResult.tome;
      modelUsed = llmResult.model_used;
      providerUsed = llmResult.provider;
      llmCallsMade = 1;
    } catch (e) {
      return json(
        { error: `render failed: ${(e as Error).message}`, cache_key: key },
        502,
      );
    } finally { await admission.release(); }
  }

  const renderId = await sha256Hex32(key + tome);

  // v0.9.A — sign a render receipt if a signing JWK is configured.
  // Absence of the signing config is non-fatal: receipt is omitted,
  // render still returns. Lets the deploy stage roll out without
  // the receipt path being a hard dependency.
  let renderReceipt: unknown = undefined;
  if (env.RENDER_RECEIPT_SIGNING_JWK && env.RENDER_RECEIPT_SIGNING_KID) {
    try {
      const signingJWK: JWK = JSON.parse(env.RENDER_RECEIPT_SIGNING_JWK);
      const triplesHash = await hashTriples(keptTriples);
      const tomeHash = await hashTome(tome);
      const payload: ReceiptPayload = {
        render_id: renderId,
        sliders_quantized: quantized,
        triples_hash: triplesHash,
        tome_hash: tomeHash,
        // Honest provenance: what actually served, not what was
        // configured. See LLMRenderResult.model_used.
        model: modelUsed,
        provider: providerUsed,
        signed_at: new Date().toISOString(),
        digital_source_type:
          providerUsed === "canonical-path"
            ? DIGITAL_SOURCE_TYPE_DETERMINISTIC
            : DIGITAL_SOURCE_TYPE_AI,
      };
      renderReceipt = await signReceipt(payload, signingJWK, env.RENDER_RECEIPT_SIGNING_KID);
    } catch (e) {
      // Signing failure is non-fatal — log and continue without receipt.
      console.error("render_receipt signing failed:", (e as Error).message);
    }
  }

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
    render_receipt: renderReceipt,
  };

  // Best-effort cache write.
  const ttl = body.cache_ttl_seconds ?? DEFAULT_TTL_SECONDS;
  await putCached(env, key, result, ttl);

  return json(result);
}
