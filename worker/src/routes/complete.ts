// /api/complete — LLM proxy for the SUM hosted demo.
//
// Ported from single_file_demo/functions/api/complete.ts (Pages
// Function) to a Workers-native route. Semantics are identical:
// POST { prompt: string, model?: string } → { completion, source, model }.
// Anthropic is preferred; OpenAI is the fallback when ANTHROPIC_API_KEY
// is absent. Optional CF_AI_GATEWAY_BASE routes both through Cloudflare
// AI Gateway for caching + observability without changing the call
// shape.
//
// Why this lives on the Worker side rather than the demo's JS:
//   * the LLM API key never touches the user's browser;
//   * the demo stays single-file, zero-dep, and CSP-tight (connect-src
//     'self' covers both the static HTML and the same-origin proxy).
//
// See README §"Hosted-demo LLM proxy" for the three-leg fallback chain
// (artifact runtime → this Worker → naive tokeniser) the demo's
// extractTriples() walks at runtime.

import type { Env } from "../index";

const ANTHROPIC_DEFAULT = "claude-haiku-4-5-20251001";
const OPENAI_DEFAULT = "gpt-4o-mini";
const MAX_PROMPT_CHARS = 40_000;
const MAX_OUTPUT_TOKENS = 2048;

interface CompleteRequest {
  prompt: string;
  model?: string;
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

async function callAnthropic(env: Env, prompt: string, model: string): Promise<string> {
  const base = env.CF_AI_GATEWAY_BASE
    ? `${env.CF_AI_GATEWAY_BASE.replace(/\/$/, "")}/anthropic/v1/messages`
    : "https://api.anthropic.com/v1/messages";

  const res = await fetch(base, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": env.ANTHROPIC_API_KEY!,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model,
      max_tokens: MAX_OUTPUT_TOKENS,
      messages: [{ role: "user", content: prompt }],
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

async function callOpenAI(env: Env, prompt: string, model: string): Promise<string> {
  const base = env.CF_AI_GATEWAY_BASE
    ? `${env.CF_AI_GATEWAY_BASE.replace(/\/$/, "")}/openai/chat/completions`
    : "https://api.openai.com/v1/chat/completions";

  const res = await fetch(base, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${env.OPENAI_API_KEY!}`,
    },
    body: JSON.stringify({
      model,
      max_tokens: MAX_OUTPUT_TOKENS,
      messages: [{ role: "user", content: prompt }],
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`openai ${res.status}: ${text.slice(0, 500)}`);
  }

  const data = (await res.json()) as { choices?: Array<{ message?: { content?: string } }> };
  const text = data.choices?.[0]?.message?.content;
  if (!text) throw new Error("openai: empty completion");
  return text;
}

export async function handleComplete(request: Request, env: Env): Promise<Response> {
  if (request.method !== "POST") {
    return json({ error: "method not allowed; use POST" }, 405);
  }

  let body: CompleteRequest;
  try {
    body = (await request.json()) as CompleteRequest;
  } catch {
    return json({ error: "invalid JSON body" }, 400);
  }

  const prompt = body?.prompt;
  if (typeof prompt !== "string" || !prompt.trim()) {
    return json({ error: "missing or empty 'prompt'" }, 400);
  }
  if (prompt.length > MAX_PROMPT_CHARS) {
    return json({ error: `prompt exceeds ${MAX_PROMPT_CHARS} chars` }, 413);
  }

  const useAnthropic = Boolean(env.ANTHROPIC_API_KEY);
  const useOpenAI = !useAnthropic && Boolean(env.OPENAI_API_KEY);

  if (!useAnthropic && !useOpenAI) {
    // 503 signals the demo's JS to fall through to the naive tokeniser
    // rather than surfacing an error — this is expected behaviour on a
    // demo that hasn't been configured with LLM secrets yet.
    return json(
      {
        error:
          "no LLM provider configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY " +
          "via `wrangler secret put` in the worker/ directory.",
      },
      503,
    );
  }

  try {
    if (useAnthropic) {
      const model = body.model ?? env.SUM_DEFAULT_MODEL_ANTHROPIC ?? ANTHROPIC_DEFAULT;
      const completion = await callAnthropic(env, prompt, model);
      return json({ completion, source: "anthropic", model });
    }
    const model = body.model ?? env.SUM_DEFAULT_MODEL_OPENAI ?? OPENAI_DEFAULT;
    const completion = await callOpenAI(env, prompt, model);
    return json({ completion, source: "openai", model });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return json({ error: `upstream failure: ${msg}` }, 502);
  }
}
