// POST /api/transform — generic transform-registry dispatch.
//
// The new substrate from PR #210–215. Replaces nothing; complements
// the existing /api/render, which keeps producing
// sum.render_receipt.v1 for slider LLM-axis renders and stays the
// recommended path until T1c wires LLM dispatch through this route.
//
// Request shape:
//   POST /api/transform
//   {
//     "transform": "slider" | (future: "extract" | "compose" | …),
//     "input": { … transform-specific … },
//     "parameters": { … transform-specific … }
//   }
//
// Response shape on success:
//   200 OK
//   {
//     "output": <transform-specific>,
//     "transform_id": "<sha256-trunc-16>",
//     "wall_clock_ms": <int>,
//     "model": "<echoed model>",
//     "provider": "<resolved provider>",
//     "transform_receipt": <sum.transform_receipt.v1 | absent>
//   }
//
// The `transform_receipt` field is absent when the Worker has no
// signing config (RENDER_RECEIPT_SIGNING_JWK / _KID) — same
// behaviour as /api/render.

import type { Env } from "../index";
import type { JWK } from "jose";

import {
  buildPayload,
  canonicalHash,
  signTransformReceipt,
} from "../receipt/transform_sign";
import { getTransform, hasTransform, listTransforms } from "../transforms/_registry";
import type { TransformEnv } from "../transforms/_base";

interface TransformRequest {
  transform?: string;
  input?: unknown;
  parameters?: Record<string, unknown>;
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

export async function handleTransform(
  request: Request,
  env: Env,
): Promise<Response> {
  if (request.method !== "POST") {
    return json({ error: "method not allowed; use POST" }, 405);
  }

  let body: TransformRequest;
  try {
    body = (await request.json()) as TransformRequest;
  } catch {
    return json({ error: "invalid JSON body" }, 400);
  }

  if (!body.transform || typeof body.transform !== "string") {
    return json(
      {
        error:
          "missing 'transform' field; expected one of: " +
          listTransforms().join(", "),
      },
      400,
    );
  }
  if (!hasTransform(body.transform)) {
    return json(
      {
        error: `unknown transform: ${body.transform}; known: ${
          listTransforms().join(", ")
        }`,
      },
      400,
    );
  }
  if (body.input === undefined || body.input === null) {
    return json({ error: "missing 'input' field" }, 400);
  }
  const parameters = body.parameters ?? {};

  const transform = getTransform(body.transform);

  // Build the TransformEnv from request headers + Worker env vars.
  // Same BYO-key precedence as the legacy /api/render path.
  const transformEnv: TransformEnv = {
    anthropicApiKey:
      request.headers.get("x-render-llm-key-anthropic") ??
      env.ANTHROPIC_API_KEY,
    openaiApiKey:
      request.headers.get("x-render-llm-key-openai") ??
      env.OPENAI_API_KEY,
    cfAiGatewayBase: env.CF_AI_GATEWAY_BASE,
    defaultAnthropicModel: env.SUM_DEFAULT_MODEL_ANTHROPIC,
    defaultOpenaiModel: env.SUM_DEFAULT_MODEL_OPENAI,
  };
  if (env.RENDER_RECEIPT_SIGNING_JWK && env.RENDER_RECEIPT_SIGNING_KID) {
    try {
      transformEnv.signingJWK = JSON.parse(env.RENDER_RECEIPT_SIGNING_JWK) as JWK;
      transformEnv.kid = env.RENDER_RECEIPT_SIGNING_KID;
    } catch {
      // Misconfigured JWK — leave unset; receipt will be omitted.
    }
  }

  const tStart = Date.now();
  let result;
  try {
    result = await transform.apply(body.input, parameters, transformEnv);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    // NotImplementedError-style errors (slider's LLM-axis path) map to
    // 501 Not Implemented to distinguish from generic errors.
    if (msg.includes("T1b") || msg.includes("T1c")) {
      return json({ error: `transform not yet implemented: ${msg}` }, 501);
    }
    return json({ error: `transform failed: ${msg}` }, 400);
  }
  const wallClockMs = Date.now() - tStart;

  // Compute the four hashes for the receipt payload. These canonicalisers
  // are the same byte-shape as the Python equivalents — cross-runtime
  // K-matrix fixtures pin the byte-equivalence.
  const parametersBytes = transform.canonicalizeParameters(parameters);
  const inputBytes = transform.canonicalizeInput(body.input);
  const outputBytes = transform.canonicalizeOutput(result.output);

  const parametersHash = await canonicalHash(parametersBytes);
  const inputHash = await canonicalHash(inputBytes);
  const outputHash = await canonicalHash(outputBytes);

  let transformReceipt: unknown = undefined;
  let transformId: string;
  if (transformEnv.signingJWK && transformEnv.kid) {
    try {
      const payload = await buildPayload({
        transform: body.transform,
        parametersHash,
        inputHash,
        outputHash,
        model: result.modelUsed,
        provider: result.provider,
        digitalSourceType: result.digitalSourceType,
      });
      transformReceipt = await signTransformReceipt(
        payload,
        transformEnv.signingJWK,
        transformEnv.kid,
      );
      transformId = payload.transform_id;
    } catch (e) {
      console.error("transform_receipt signing failed:", (e as Error).message);
      // Derive transform_id even when signing failed so the
      // caller still gets a stable id back.
      const { deriveTransformId } = await import("../receipt/transform_sign");
      transformId = await deriveTransformId(
        body.transform,
        parametersHash,
        inputHash,
        outputHash,
      );
    }
  } else {
    const { deriveTransformId } = await import("../receipt/transform_sign");
    transformId = await deriveTransformId(
      body.transform,
      parametersHash,
      inputHash,
      outputHash,
    );
  }

  return json({
    output: result.output,
    transform_id: transformId,
    wall_clock_ms: wallClockMs,
    model: result.modelUsed,
    provider: result.provider,
    digital_source_type: result.digitalSourceType,
    llm_calls_made: result.llmCallsMade,
    extra: result.extra ?? {},
    transform_receipt: transformReceipt,
  });
}
