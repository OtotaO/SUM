// Sign sum.transform_receipt.v1 envelopes.
//
// Mirrors sum_engine_internal/transform_receipt/sign.py. Same JOSE
// envelope core (canonicalize → EdDSA / Ed25519 detached JWS) as
// the render-receipt path; the only differences are the schema
// identifier and the payload fields.
//
// Why a separate file from receipt/sign.ts: the render-receipt
// surface stays untouched. New transforms produce transform
// receipts; legacy /api/render keeps producing render receipts.
// Existing render receipts remain valid forever.

import { CompactSign, importJWK, type JWK } from "jose";
import canonicalize from "canonicalize";

import type {
  DigitalSourceType,
  Provider,
} from "../transforms/_base";

export const TRANSFORM_RECEIPT_SCHEMA = "sum.transform_receipt.v1";

export interface TransformReceiptPayload {
  transform_id: string;
  transform: string;
  parameters_hash: string;
  input_hash: string;
  output_hash: string;
  model: string;
  provider: Provider;
  signed_at: string;
  digital_source_type: DigitalSourceType;
  /** Optional T4 field. Omitted entirely when null. */
  source_chain_hash?: string;
}

export interface TransformReceipt {
  schema: typeof TRANSFORM_RECEIPT_SCHEMA;
  kid: string;
  payload: TransformReceiptPayload;
  jws: string;
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", bytes);
  return (
    "sha256-" +
    Array.from(new Uint8Array(buf))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("")
  );
}

/**
 * Compute the `"sha256-<hex>"` for already-canonical bytes. Single
 * shape across parameters_hash, input_hash, output_hash,
 * source_chain_hash.
 */
export async function canonicalHash(canonicalBytes: Uint8Array): Promise<string> {
  return sha256Hex(canonicalBytes);
}

/**
 * Derive transform_id from the four hash inputs. Stable across
 * runs and runtimes (same SHA-256 over the same UTF-8 bytes).
 */
export async function deriveTransformId(
  transform: string,
  parametersHash: string,
  inputHash: string,
  outputHash: string,
): Promise<string> {
  const enc = new TextEncoder();
  const chunks = [
    enc.encode(transform),
    enc.encode("|"),
    enc.encode(parametersHash),
    enc.encode("|"),
    enc.encode(inputHash),
    enc.encode("|"),
    enc.encode(outputHash),
  ];
  let totalLen = 0;
  for (const c of chunks) totalLen += c.length;
  const merged = new Uint8Array(totalLen);
  let off = 0;
  for (const c of chunks) {
    merged.set(c, off);
    off += c.length;
  }
  const buf = await crypto.subtle.digest("SHA-256", merged);
  const hex = Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  return hex.slice(0, 16);
}

/** ISO-8601 UTC with millisecond precision; matches Python's
 *  build_payload output byte-for-byte (modulo the current time). */
export function isoNowMs(): string {
  return new Date().toISOString();
}

export interface BuildPayloadArgs {
  transform: string;
  parametersHash: string;
  inputHash: string;
  outputHash: string;
  model: string;
  provider: Provider;
  digitalSourceType: DigitalSourceType;
  sourceChainHash?: string;
  signedAt?: string;
}

export async function buildPayload(
  args: BuildPayloadArgs,
): Promise<TransformReceiptPayload> {
  const transformId = await deriveTransformId(
    args.transform,
    args.parametersHash,
    args.inputHash,
    args.outputHash,
  );
  const payload: TransformReceiptPayload = {
    transform_id: transformId,
    transform: args.transform,
    parameters_hash: args.parametersHash,
    input_hash: args.inputHash,
    output_hash: args.outputHash,
    model: args.model,
    provider: args.provider,
    signed_at: args.signedAt ?? isoNowMs(),
    digital_source_type: args.digitalSourceType,
  };
  if (args.sourceChainHash) {
    payload.source_chain_hash = args.sourceChainHash;
  }
  return payload;
}

/**
 * Sign the transform-receipt payload with the Worker's signing
 * JWK. Returns the receipt block ready to embed in a /api/transform
 * response.
 *
 * Same JWS shape as render receipts: detached, EdDSA / Ed25519,
 * protected header `{alg, kid, b64: false, crit: ['b64']}`.
 */
export async function signTransformReceipt(
  payload: TransformReceiptPayload,
  signingJWK: JWK,
  kid: string,
): Promise<TransformReceipt> {
  const key = await importJWK(signingJWK, "EdDSA");
  const canonicalStr = canonicalize(payload);
  if (typeof canonicalStr !== "string") {
    throw new Error("canonicalize returned undefined for transform receipt payload");
  }
  const canonicalBytes = new TextEncoder().encode(canonicalStr);
  const jws = await new CompactSign(canonicalBytes)
    .setProtectedHeader({ alg: "EdDSA", kid, b64: false, crit: ["b64"] })
    .sign(key);

  // Detach the payload from the compact form per RFC 7515 §A.5:
  // "<protected>.<empty>.<signature>" — middle segment empty
  // because the canonical bytes are the detached payload.
  const parts = jws.split(".");
  if (parts.length !== 3) {
    throw new Error(`unexpected JWS shape: ${parts.length} segments`);
  }
  const detachedJws = `${parts[0]}..${parts[2]}`;

  return {
    schema: TRANSFORM_RECEIPT_SCHEMA,
    kid,
    payload,
    jws: detachedJws,
  };
}
