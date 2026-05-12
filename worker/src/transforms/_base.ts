// Transform protocol — TypeScript port mirroring
// sum_engine_internal/transforms/_base.py
//
// Contract: `(bundle | text) × transform × parameters → signed
// artifact`. Each registered transform implements this interface;
// the dispatch surface (POST /api/transform) calls
// `transform.apply()` then signs a sum.transform_receipt.v1
// envelope over the (parameters_hash, input_hash, output_hash,
// model, provider) tuple.
//
// Cross-runtime guarantee: a Python implementation of the same
// transform name MUST produce byte-identical canonicalize_*
// outputs as this TypeScript implementation. The K-matrix fixture
// extension under fixtures/transform_receipts/ locks this.

import type { JWK } from "jose";

export type Provider =
  | "anthropic"
  | "openai"
  | "cf-ai-gateway-anthropic"
  | "cf-ai-gateway-openai"
  | "canonical-path";

export type DigitalSourceType =
  | "trainedAlgorithmicMedia"
  | "algorithmicMedia";

/**
 * Capabilities passed to every transform. The dispatch surface
 * populates this from request headers (BYO-keys) and Worker env
 * vars (operator-funded). User-supplied keys take precedence —
 * same rule as the legacy /api/render route.
 */
export interface TransformEnv {
  // Receipt signing — required to produce a signed receipt.
  // Absence is non-fatal: the transform still runs and returns
  // output, but the caller gets no `transform_receipt` field.
  signingJWK?: JWK;
  kid?: string;

  // LLM capabilities (optional; only used by requires_llm transforms).
  anthropicApiKey?: string;
  openaiApiKey?: string;
  cfAiGatewayBase?: string;
  defaultAnthropicModel?: string;
  defaultOpenaiModel?: string;

  // Provider preference (optional).
  preferredProvider?: "anthropic" | "openai";
}

/**
 * What a transform returns to the dispatch surface. The
 * `output` field shape varies per transform; the dispatcher
 * uses `transform.canonicalize_output(output)` to produce the
 * bytes that get hashed into the receipt's output_hash.
 */
export interface TransformResult {
  output: unknown;
  /** Model the API echoed back; "canonical-deterministic-v0" for
   *  non-LLM transforms. Honest-provenance: never the
   *  configured-default. */
  modelUsed: string;
  /** What actually served the call. */
  provider: Provider;
  digitalSourceType: DigitalSourceType;
  llmCallsMade: number;
  /** Per-transform optional auxiliary data; not signed into the
   *  receipt. Surface for the HTTP response body. */
  extra?: Record<string, unknown>;
}

/**
 * The interface every registered transform satisfies.
 *
 * Implementations are typically simple object literals — see
 * `transforms/slider.ts` for the canonical example. The registry
 * stores instances (not constructors), making per-deployment
 * parametrisation trivial without subclassing.
 */
export interface Transform {
  readonly name: string;
  readonly requiresLLM: boolean;
  readonly digitalSourceType: DigitalSourceType;

  /**
   * JCS-canonical bytes of the parameters object. The receipt's
   * `parameters_hash` is `sha256-` + hex of this output. Per the
   * spec, sort keys alphabetically; render numeric values with
   * the transform's documented precision rule (the slider
   * transform, for example, rounds non-density axes to bin
   * centres before hashing).
   */
  canonicalizeParameters(params: unknown): Uint8Array;

  /**
   * Canonical bytes of the input. Each transform pins the
   * accepted input shape (CanonicalBundle for slider/compose;
   * text or triples for extract).
   */
  canonicalizeInput(rawInput: unknown): Uint8Array;

  /**
   * Canonical bytes of the output. Each transform pins its own
   * canonicalisation (tome → UTF-8; tag-set → JCS of sorted
   * unique triples; merged bundle → state_integer ‖ tome bytes).
   */
  canonicalizeOutput(output: unknown): Uint8Array;

  /**
   * Run the transform. Async because LLM-mediated transforms
   * call out to the network; pure-algorithmic transforms can
   * return immediately.
   */
  apply(
    input: unknown,
    parameters: Record<string, unknown>,
    env: TransformEnv,
  ): Promise<TransformResult>;
}
