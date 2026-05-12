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
export async function verifyTransformReceipt(receipt, jwks) {
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
  const key = (jwks?.keys || []).find((k) => k.kid === kid);
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
  const protectedHeader = JSON.parse(
    new TextDecoder().decode(result.protectedHeader),
  );
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

  return {
    verified: true,
    kid,
    protectedHeader,
    payload,
  };
}
