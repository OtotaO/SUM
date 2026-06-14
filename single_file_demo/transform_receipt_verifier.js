// SUM transform-receipt verifier (T1d).
//
// Verifies sum.transform_receipt.v1 envelopes. Mirrors the
// six-step algorithm + forward-compat levers from receipt_verifier.js
// (which handles sum.render_receipt.v1); the only difference is the
// schema string and the payload-field semantics.
//
// Why a parallel module rather than a refactor: receipt_verifier.js
// is already battle-tested against the K1–K4 + A1–A6 fixture matrix
// for the render-receipt format. Refactoring it to share with the
// transform-receipt path risks regressing those fixtures. A clean
// duplication keeps the render-receipt path untouched; if a future
// PR wants to extract a shared core, the two files share the same
// algorithm by construction so the merge is mechanical.
//
// Cross-runtime contract: Python's verify_transform_receipt produces
// byte-identical accept/reject + error_class outcomes against the
// same fixture set. The K-matrix extension for transform receipts
// lives at fixtures/transform_receipts/ (when seeded by T1d-follow-up).

import { flattenedVerify, canonicalize } from "./vendor/sum-verify-deps.js";

export const SUPPORTED_SCHEMA = "sum.transform_receipt.v1";

export const KNOWN_CRIT_EXTENSIONS = new Set(["b64"]);

export const ERROR_CLASSES = Object.freeze({
  MALFORMED_RECEIPT: "malformed_receipt",
  MALFORMED_JWS: "malformed_jws",
  MALFORMED_JWKS: "malformed_jwks",
  UNKNOWN_KID: "unknown_kid",
  KID_MISMATCH: "kid_mismatch",
  SCHEMA_UNKNOWN: "schema_unknown",
  CRIT_UNKNOWN_EXTENSION: "crit_unknown_extension",
  HEADER_INVARIANT_VIOLATED: "header_invariant_violated",
  SIGNATURE_INVALID: "signature_invalid",
  UNSUPPORTED_ALG: "unsupported_alg",
  SIGNED_AT_OUT_OF_WINDOW: "signed_at_out_of_window",
});

export const SUPPORTED_SIGNATURE_ALGORITHMS = new Set(["EdDSA"]);

export class VerifyError extends Error {
  constructor(errorClass, message) {
    super(message);
    this.errorClass = errorClass;
    this.name = "VerifyError";
  }
}

function b64urlDecodeToBytes(s) {
  const pad = "=".repeat((4 - (s.length % 4)) % 4);
  const std = (s + pad).replace(/-/g, "+").replace(/_/g, "/");
  if (typeof atob === "function") {
    const bin = atob(std);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  return Uint8Array.from(Buffer.from(s, "base64url"));
}

async function importEd25519Jwk(jwk) {
  if (jwk.kty !== "OKP" || jwk.crv !== "Ed25519") {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_JWKS,
      `expected OKP/Ed25519 JWK, got kty=${jwk.kty} crv=${jwk.crv}`,
    );
  }
  return crypto.subtle.importKey(
    "jwk",
    jwk,
    { name: "Ed25519" },
    false,
    ["verify"],
  );
}

/**
 * Verify a sum.transform_receipt.v1 envelope. Six-step algorithm
 * identical to receipt_verifier.js::verifyReceipt; differs only in
 * SUPPORTED_SCHEMA.
 *
 * @param {object} receipt
 * @param {object} jwks
 * @returns {Promise<{verified: true, kid: string, protectedHeader: object, payload: object}>}
 * @throws {VerifyError}
 */
/**
 * Verify a sum.transform_receipt.v1 envelope.
 *
 * Optional replay-defense window: pass `{maxAgeSeconds, maxFutureSkewSeconds}`
 * in the third argument to reject receipts whose `payload.signed_at`
 * is outside the acceptance window. Default (omitted) does NOT
 * enforce — long-lived archival receipts remain valid.
 *
 * @param {object} receipt
 * @param {object} jwks
 * @param {object} [opts]
 * @param {number} [opts.maxAgeSeconds] When set, reject receipts older
 *   than this many seconds (or further in the future than maxFutureSkewSeconds).
 * @param {number} [opts.maxFutureSkewSeconds=60] Clock-skew tolerance.
 */
export async function verifyTransformReceipt(receipt, jwks, opts) {
  const { maxAgeSeconds = null, maxFutureSkewSeconds = 60 } = opts || {};
  // ---- Step 0: shape gate ----
  if (!receipt || typeof receipt !== "object") {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_RECEIPT,
      "receipt is not an object",
    );
  }
  if (receipt.schema !== SUPPORTED_SCHEMA) {
    throw new VerifyError(
      ERROR_CLASSES.SCHEMA_UNKNOWN,
      `unsupported receipt schema: ${receipt.schema} ` +
        `(this verifier handles ${SUPPORTED_SCHEMA})`,
    );
  }
  const { kid, payload, jws } = receipt;
  if (typeof kid !== "string" || !kid) {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_RECEIPT,
      "receipt.kid missing or empty",
    );
  }
  if (!payload || typeof payload !== "object") {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_RECEIPT,
      "receipt.payload missing or non-object",
    );
  }
  if (typeof jws !== "string" || !jws) {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_RECEIPT,
      "receipt.jws missing or empty",
    );
  }

  // ---- Step 1: kid lookup ----
  // Validate JWKS shape first: an array's `.keys` is Array.prototype.keys (a
  // function), so `(arrayJwks?.keys || []).find` throws TypeError — fail closed
  // with a clean class instead, matching the Python core.
  if (jwks === null || typeof jwks !== "object" || Array.isArray(jwks) || !Array.isArray(jwks.keys)) {
    throw new VerifyError(ERROR_CLASSES.MALFORMED_JWKS, "jwks must be an object with a 'keys' array");
  }
  const key = jwks.keys.find((k) => k && typeof k === "object" && k.kid === kid);
  if (!key) {
    throw new VerifyError(
      ERROR_CLASSES.UNKNOWN_KID,
      `no key in JWKS for kid=${kid}`,
    );
  }

  // ---- Step 2: JCS canonicalize payload ----
  const canonicalText = canonicalize(payload);
  if (canonicalText === undefined || canonicalText === null) {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_RECEIPT,
      "payload could not be JCS-canonicalized",
    );
  }
  const canonicalBytes = new TextEncoder().encode(canonicalText);

  // ---- Step 3: split detached JWS ----
  const parts = jws.split(".");
  if (parts.length !== 3) {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_JWS,
      `JWS must have exactly 3 segments, got ${parts.length}`,
    );
  }
  const [proto, middle, signature] = parts;
  if (middle !== "") {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_JWS,
      "detached JWS middle segment must be empty (RFC 7515 §A.5)",
    );
  }

  // ---- Step 3.5: protected header forward-compat check ----
  let header;
  try {
    const headerJson = new TextDecoder().decode(b64urlDecodeToBytes(proto));
    header = JSON.parse(headerJson);
  } catch (e) {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_JWS,
      `protected header is not valid JSON: ${e.message}`,
    );
  }
  if (!header || typeof header !== "object") {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_JWS,
      "protected header is not an object",
    );
  }
  if (header.alg && !SUPPORTED_SIGNATURE_ALGORITHMS.has(header.alg)) {
    throw new VerifyError(
      ERROR_CLASSES.UNSUPPORTED_ALG,
      `unsupported alg ${header.alg}; this verifier accepts ${
        [...SUPPORTED_SIGNATURE_ALGORITHMS].join(", ")
      }`,
    );
  }
  if (Array.isArray(header.crit)) {
    for (const ext of header.crit) {
      if (!KNOWN_CRIT_EXTENSIONS.has(ext)) {
        throw new VerifyError(
          ERROR_CLASSES.CRIT_UNKNOWN_EXTENSION,
          `unknown crit extension: ${ext} ` +
            `(this verifier handles ${[...KNOWN_CRIT_EXTENSIONS].join(", ")})`,
        );
      }
    }
  }

  // ---- Steps 4 + 5: verify the JWS ----
  let importedKey;
  try {
    importedKey = await importEd25519Jwk(key);
  } catch (e) {
    if (e instanceof VerifyError) throw e;
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_JWKS,
      `JWK import failed: ${e.message}`,
    );
  }

  let result;
  try {
    result = await flattenedVerify(
      { protected: proto, payload: canonicalBytes, signature },
      importedKey,
    );
  } catch (e) {
    throw new VerifyError(
      ERROR_CLASSES.SIGNATURE_INVALID,
      `signature verification failed: ${e.code || e.message}`,
    );
  }

  // ---- Step 6: protected-header invariants ----
  // jose's flattenedVerify returns `protectedHeader` already parsed
  // into an object — same as receipt_verifier.js consumes it. (Earlier
  // versions of this verifier mistakenly TextDecoded the field as if
  // it were the raw bytes; the cross-runtime fixture set in
  // fixtures/transform_receipts/ caught it.)
  const protectedHeader = result.protectedHeader;
  if (protectedHeader.alg !== "EdDSA") {
    throw new VerifyError(
      ERROR_CLASSES.HEADER_INVARIANT_VIOLATED,
      `expected alg=EdDSA, got alg=${protectedHeader.alg}`,
    );
  }
  if (protectedHeader.kid !== kid) {
    throw new VerifyError(
      ERROR_CLASSES.KID_MISMATCH,
      `protected.kid=${protectedHeader.kid} != receipt.kid=${kid}`,
    );
  }
  if (protectedHeader.b64 !== false) {
    throw new VerifyError(
      ERROR_CLASSES.HEADER_INVARIANT_VIOLATED,
      "expected b64:false in protected header",
    );
  }

  // ---- Step 7: optional replay-window check ----
  // Default (maxAgeSeconds=null) does NOT enforce. When the caller
  // opts in, reject receipts whose payload.signed_at is outside the
  // acceptance window. This is the policy layer; the signature is
  // already verified above. See docs/TRANSFORM_RECEIPT_FORMAT.md §6.2.
  if (maxAgeSeconds !== null) {
    enforceSignedAtWindow(payload, maxAgeSeconds, maxFutureSkewSeconds);
  }

  return {
    verified: true,
    kid,
    protectedHeader,
    payload,
  };
}

function parseSignedAt(s) {
  // Parse ISO-8601 with trailing Z into UTC ms-since-epoch. JS Date
  // handles "Z" suffix; reject if NaN.
  const ms = Date.parse(s);
  if (Number.isNaN(ms)) {
    throw new Error(`signed_at ${JSON.stringify(s)} not parseable as ISO-8601`);
  }
  return ms;
}

function enforceSignedAtWindow(payload, maxAgeSeconds, maxFutureSkewSeconds) {
  const signedAt = payload && payload.signed_at;
  if (typeof signedAt !== "string") {
    throw new VerifyError(
      ERROR_CLASSES.SIGNED_AT_OUT_OF_WINDOW,
      `max_age_seconds=${maxAgeSeconds} requested but payload.signed_at is missing or non-string (${JSON.stringify(signedAt)}); failing closed`,
    );
  }
  let signedAtMs;
  try {
    signedAtMs = parseSignedAt(signedAt);
  } catch (e) {
    throw new VerifyError(
      ERROR_CLASSES.SIGNED_AT_OUT_OF_WINDOW,
      `max_age_seconds=${maxAgeSeconds} requested but ${e.message}`,
    );
  }
  const nowMs = Date.now();
  const ageSeconds = (nowMs - signedAtMs) / 1000;
  if (ageSeconds > maxAgeSeconds) {
    throw new VerifyError(
      ERROR_CLASSES.SIGNED_AT_OUT_OF_WINDOW,
      `receipt signed_at=${signedAt} is ${ageSeconds.toFixed(0)}s old; max_age_seconds=${maxAgeSeconds}`,
    );
  }
  if (-ageSeconds > maxFutureSkewSeconds) {
    throw new VerifyError(
      ERROR_CLASSES.SIGNED_AT_OUT_OF_WINDOW,
      `receipt signed_at=${signedAt} is ${(-ageSeconds).toFixed(0)}s in the future; max_future_skew_seconds=${maxFutureSkewSeconds}`,
    );
  }
}
