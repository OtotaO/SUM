// Render receipt signing — Phase E.1 v0.9.A.
//
// Each /api/render response carries a `render_receipt` block: a
// detached JWS (RFC 7515 §A.5) with EdDSA / Ed25519 signature over
// the JCS-canonicalised payload (RFC 8785). Consumers can:
//
//   1. Fetch the public key from /.well-known/jwks.json (RFC 7517).
//   2. JCS-canonicalise the receipt's `payload` object.
//   3. Verify with `jose.flattenedVerify` against the JWK whose
//      `kid` matches the receipt's `kid`.
//
// The receipt attests "this Worker rendered this tome from these
// triples at these slider positions at this time." It does NOT
// attest factual truth of the tome — that's what the Python bench
// + NLI audit verifies separately. The `digital_source_type` field
// uses C2PA terminology (`trainedAlgorithmicMedia`) so consumers
// know unambiguously the tome is AI-generated.

import { CompactSign, importJWK, type JWK } from "jose";
import canonicalize from "canonicalize";

export const RECEIPT_SCHEMA = "sum.render_receipt.v1";
export const DIGITAL_SOURCE_TYPE_AI = "trainedAlgorithmicMedia";

export interface ReceiptPayload {
  render_id: string;
  sliders_quantized: {
    density: number;
    length: number;
    formality: number;
    audience: number;
    perspective: number;
  };
  triples_hash: string;
  tome_hash: string;
  model: string;
  signed_at: string;
  digital_source_type: typeof DIGITAL_SOURCE_TYPE_AI;
}

export interface RenderReceipt {
  schema: typeof RECEIPT_SCHEMA;
  kid: string;
  payload: ReceiptPayload;
  jws: string;
}

/** Hex-encoded sha256 of the input bytes, prefixed `sha256-`. */
async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", bytes);
  return (
    "sha256-" +
    Array.from(new Uint8Array(buf))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("")
  );
}

export async function hashTriples(
  triples: Array<[string, string, string]>,
): Promise<string> {
  // Sort + JCS for byte-stable cross-runtime hashing. Python's
  // sorted(tuple(t) for t in triples) + jcs.canonicalize must
  // produce the same bytes for the same input.
  const sorted = [...triples].sort((a, b) => {
    const ak = `${a[0]}||${a[1]}||${a[2]}`;
    const bk = `${b[0]}||${b[1]}||${b[2]}`;
    return ak < bk ? -1 : ak > bk ? 1 : 0;
  });
  const canonical = canonicalize(sorted);
  if (typeof canonical !== "string") throw new Error("canonicalize returned undefined");
  return sha256Hex(new TextEncoder().encode(canonical));
}

export async function hashTome(tome: string): Promise<string> {
  return sha256Hex(new TextEncoder().encode(tome));
}

/**
 * Sign the receipt payload with the Worker's signing JWK. Returns
 * the receipt block ready to embed in the /api/render response.
 *
 * The JWS is detached: protected header `{alg, kid, b64: false,
 * crit: ['b64']}`, empty middle segment, signature segment carries
 * Ed25519 over the JCS-canonical bytes of `payload`. Consumers
 * verify by re-canonicalising `payload` from the response they
 * received and calling `jose.flattenedVerify({protected, payload,
 * signature})` against the matching JWK from /.well-known/jwks.json.
 */
export async function signReceipt(
  payload: ReceiptPayload,
  signingJWK: JWK,
  kid: string,
): Promise<RenderReceipt> {
  const key = await importJWK(signingJWK, "EdDSA");
  const canonicalStr = canonicalize(payload);
  if (typeof canonicalStr !== "string") {
    throw new Error("canonicalize returned undefined for payload");
  }
  const canonicalBytes = new TextEncoder().encode(canonicalStr);
  const jws = await new CompactSign(canonicalBytes)
    .setProtectedHeader({ alg: "EdDSA", kid, b64: false, crit: ["b64"] })
    .sign(key);
  return { schema: RECEIPT_SCHEMA, kid, payload, jws };
}
