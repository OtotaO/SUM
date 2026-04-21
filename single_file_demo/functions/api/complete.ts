// Cloudflare Pages Function — LLM proxy for the SUM single-file demo.
//
// When the demo is served outside a Claude artifact runtime (plain
// browser, hosted Pages deployment, curl from a script), there is no
// `window.claude.complete`. This function provides the fallback: a
// same-origin POST /api/complete endpoint that proxies to a real LLM
// provider. The demo calls it between the artifact path and the naive
// tokenizer path; see index.html → tryPagesFunctionExtract.
//
// Environment bindings (set in Cloudflare Pages dashboard → Settings →
// Environment variables):
//
//   ANTHROPIC_API_KEY     preferred. Enables Claude as the extractor.
//   OPENAI_API_KEY        fallback if ANTHROPIC not set.
//   CF_AI_GATEWAY_BASE    optional. If set, provider calls route through
//                         Cloudflare AI Gateway for caching + logs.
//                         Example: https://gateway.ai.cloudflare.com/v1/
//                                  {account_id}/{gateway_id}
//   SUM_DEFAULT_MODEL     optional. Overrides the default model id per
//                         provider. Default: claude-haiku-4-5-20251001
//                         (Anthropic) or gpt-4o-mini (OpenAI).
//
// Never commit these values. They live only in the Pages environment.
//
// Request:  POST /api/complete  { "prompt": "<SVO extraction prompt>" }
// Response: 200 { "completion": "...", "source": "anthropic" | "openai" }
//           4xx/5xx { "error": "..." }

interface Env {
  ANTHROPIC_API_KEY?: string;
  OPENAI_API_KEY?: string;
  CF_AI_GATEWAY_BASE?: string;
  SUM_DEFAULT_MODEL?: string;
}

interface CompleteRequest {
  prompt: string;
  model?: string;
}

// Minimal Pages Function context shape. Cloudflare's build pulls in the
// full type from @cloudflare/workers-types when available; this local
// alias keeps the file compiling even if those types aren't installed
// (the demo has no package.json today).
type PagesCtx<E> = { request: Request; env: E };
type PagesHandler<E> = (ctx: PagesCtx<E>) => Promise<Response>;

const ANTHROPIC_DEFAULT_MODEL = "claude-haiku-4-5-20251001";
const OPENAI_DEFAULT_MODEL = "gpt-4o-mini";
const MAX_PROMPT_CHARS = 40_000;
const MAX_OUTPUT_TOKENS = 2048;

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

async function callAnthropic(
  env: Env,
  prompt: string,
  model: string,
): Promise<string> {
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

  const data = (await res.json()) as {
    content?: Array<{ type: string; text?: string }>;
  };
  const block = (data.content ?? []).find((b) => b.type === "text");
  if (!block?.text) throw new Error("anthropic: empty completion");
  return block.text;
}

async function callOpenAI(
  env: Env,
  prompt: string,
  model: string,
): Promise<string> {
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

  const data = (await res.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };
  const text = data.choices?.[0]?.message?.content;
  if (!text) throw new Error("openai: empty completion");
  return text;
}

export const onRequestPost: PagesHandler<Env> = async (ctx) => {
  let body: CompleteRequest;
  try {
    body = (await ctx.request.json()) as CompleteRequest;
  } catch {
    return json({ error: "invalid JSON body" }, 400);
  }

  const prompt = body?.prompt;
  if (typeof prompt !== "string" || !prompt.trim()) {
    return json({ error: "missing or empty 'prompt'" }, 400);
  }
  if (prompt.length > MAX_PROMPT_CHARS) {
    return json(
      { error: `prompt exceeds ${MAX_PROMPT_CHARS} chars` },
      413,
    );
  }

  const env = ctx.env;
  const useAnthropic = Boolean(env.ANTHROPIC_API_KEY);
  const useOpenAI = !useAnthropic && Boolean(env.OPENAI_API_KEY);

  if (!useAnthropic && !useOpenAI) {
    return json(
      {
        error:
          "no provider configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY " +
          "in Pages environment variables.",
      },
      503,
    );
  }

  try {
    if (useAnthropic) {
      const model = body.model ?? env.SUM_DEFAULT_MODEL ?? ANTHROPIC_DEFAULT_MODEL;
      const completion = await callAnthropic(env, prompt, model);
      return json({ completion, source: "anthropic", model });
    }
    const model = body.model ?? env.SUM_DEFAULT_MODEL ?? OPENAI_DEFAULT_MODEL;
    const completion = await callOpenAI(env, prompt, model);
    return json({ completion, source: "openai", model });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return json({ error: `upstream failure: ${msg}` }, 502);
  }
};

// Reject non-POST methods with 405 so accidental GETs return a clear
// signal instead of Cloudflare's default HTML 404.
export const onRequest: PagesHandler<Env> = async () =>
  json({ error: "method not allowed; use POST" }, 405);
