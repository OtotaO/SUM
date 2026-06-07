// SUM meaning-risk + perspective receipt verifier (Node/browser).
//
// Verifies sum.meaning_risk_receipt.v1 and sum.perspective_risk_receipt.v1
// envelopes — the receipts that carry SUM's differentiating moat
// (rewriting-robust semantic-loss bounding, per-perspective). Until now
// these verified ONLY in Python; this is the second runtime, so the
// cross-runtime claim is true for the new receipts, not just render /
// transform.
//
// Same six-step JOSE-over-JCS algorithm as transform_receipt_verifier.js
// (Ed25519 over JCS-canonical bytes, detached JWS, b64:false), PLUS the
// disclosure invariant these receipts add: not_covered must be a
// non-empty array and disclosure a non-empty string (mirrors Python's
// MeaningReceiptDisclosureError) — a signed-but-disclosure-free receipt
// reads as a bare bound and is rejected.
//
// SCOPE (honest): this is Stage A — signature + schema + header +
// disclosure. Stage B (replay: recompute the per-pair losses with the
// named scorer and re-derive every bound) is NOT done here, because it
// requires running the scorer over the source/transform pairs, which is
// the producer's Python/model surface. A verified signature here proves
// the receipt is authentic and self-disclosing; replay remains a
// side-band Python check. The receipt FORMAT verifies cross-runtime;
// the meaning re-computation does not (and says so).

import { flattenedVerify, canonicalize } from "./vendor/sum-verify-deps.js";

export const MEANING_RISK_SCHEMA = "sum.meaning_risk_receipt.v1";
export const PERSPECTIVE_SCHEMA = "sum.perspective_risk_receipt.v1";

export const KNOWN_CRIT_EXTENSIONS = new Set(["b64"]);
export const SUPPORTED_SIGNATURE_ALGORITHMS = new Set(["EdDSA"]);

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
  DISCLOSURE_MISSING: "disclosure_missing",
});

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
  return crypto.subtle.importKey("jwk", jwk, { name: "Ed25519" }, false, ["verify"]);
}

/**
 * Verify a SUM meaning-family signed envelope (Stage A + disclosure).
 *
 * @param {object} receipt          {schema, kid, payload, jws}
 * @param {object} jwks             {keys: [...]}
 * @param {string} supportedSchema  the schema this call accepts
 * @returns {Promise<{verified:true, kid:string, protectedHeader:object, payload:object}>}
 * @throws {VerifyError}
 */
export async function verifyMeaningEnvelope(receipt, jwks, supportedSchema) {
  // ---- Step 0: shape gate ----
  if (!receipt || typeof receipt !== "object") {
    throw new VerifyError(ERROR_CLASSES.MALFORMED_RECEIPT, "receipt is not an object");
  }
  if (receipt.schema !== supportedSchema) {
    throw new VerifyError(
      ERROR_CLASSES.SCHEMA_UNKNOWN,
      `unsupported receipt schema: ${receipt.schema} (this verifier handles ${supportedSchema})`,
    );
  }
  const { kid, payload, jws } = receipt;
  if (typeof kid !== "string" || !kid) {
    throw new VerifyError(ERROR_CLASSES.MALFORMED_RECEIPT, "receipt.kid missing or empty");
  }
  if (!payload || typeof payload !== "object") {
    throw new VerifyError(ERROR_CLASSES.MALFORMED_RECEIPT, "receipt.payload missing or non-object");
  }
  if (typeof jws !== "string" || !jws) {
    throw new VerifyError(ERROR_CLASSES.MALFORMED_RECEIPT, "receipt.jws missing or empty");
  }

  // ---- Step 1: kid lookup ----
  const key = (jwks?.keys || []).find((k) => k.kid === kid);
  if (!key) {
    throw new VerifyError(ERROR_CLASSES.UNKNOWN_KID, `no key in JWKS for kid=${kid}`);
  }

  // ---- Step 2: JCS canonicalize ----
  const canonicalText = canonicalize(payload);
  if (canonicalText === undefined || canonicalText === null) {
    throw new VerifyError(ERROR_CLASSES.MALFORMED_RECEIPT, "payload could not be JCS-canonicalized");
  }
  const canonicalBytes = new TextEncoder().encode(canonicalText);

  // ---- Step 3: split detached JWS ----
  const parts = jws.split(".");
  if (parts.length !== 3) {
    throw new VerifyError(ERROR_CLASSES.MALFORMED_JWS, `JWS must have exactly 3 segments, got ${parts.length}`);
  }
  const [proto, middle, signature] = parts;
  if (middle !== "") {
    throw new VerifyError(ERROR_CLASSES.MALFORMED_JWS, "detached JWS middle segment must be empty (RFC 7515 §A.5)");
  }

  // ---- Step 3.5: protected-header forward-compat ----
  let header;
  try {
    header = JSON.parse(new TextDecoder().decode(b64urlDecodeToBytes(proto)));
  } catch (e) {
    throw new VerifyError(ERROR_CLASSES.MALFORMED_JWS, `protected header is not valid JSON: ${e.message}`);
  }
  if (!header || typeof header !== "object") {
    throw new VerifyError(ERROR_CLASSES.MALFORMED_JWS, "protected header is not an object");
  }
  if (header.alg && !SUPPORTED_SIGNATURE_ALGORITHMS.has(header.alg)) {
    throw new VerifyError(
      ERROR_CLASSES.UNSUPPORTED_ALG,
      `unsupported alg ${header.alg}; this verifier accepts ${[...SUPPORTED_SIGNATURE_ALGORITHMS].join(", ")}`,
    );
  }
  if (Array.isArray(header.crit)) {
    for (const ext of header.crit) {
      if (!KNOWN_CRIT_EXTENSIONS.has(ext)) {
        throw new VerifyError(
          ERROR_CLASSES.CRIT_UNKNOWN_EXTENSION,
          `unknown crit extension: ${ext} (this verifier handles ${[...KNOWN_CRIT_EXTENSIONS].join(", ")})`,
        );
      }
    }
  }

  // ---- Steps 4 + 5: cryptographic verify ----
  let importedKey;
  try {
    importedKey = await importEd25519Jwk(key);
  } catch (e) {
    if (e instanceof VerifyError) throw e;
    throw new VerifyError(ERROR_CLASSES.MALFORMED_JWKS, `JWK import failed: ${e.message}`);
  }
  let result;
  try {
    result = await flattenedVerify(
      { protected: proto, payload: canonicalBytes, signature },
      importedKey,
    );
  } catch (e) {
    throw new VerifyError(ERROR_CLASSES.SIGNATURE_INVALID, `signature verification failed: ${e.code || e.message}`);
  }

  // ---- Step 6: protected-header invariants ----
  const protectedHeader = result.protectedHeader;
  if (protectedHeader.alg !== "EdDSA") {
    throw new VerifyError(ERROR_CLASSES.HEADER_INVARIANT_VIOLATED, `expected alg=EdDSA, got alg=${protectedHeader.alg}`);
  }
  if (protectedHeader.kid !== kid) {
    throw new VerifyError(ERROR_CLASSES.KID_MISMATCH, `protected.kid=${protectedHeader.kid} != receipt.kid=${kid}`);
  }
  if (protectedHeader.b64 !== false) {
    throw new VerifyError(ERROR_CLASSES.HEADER_INVARIANT_VIOLATED, "expected b64:false in protected header");
  }

  // ---- Step 7: disclosure invariants (the meaning-family addition) ----
  // A meaning-family receipt bounds a NAMED proxy while declaring what it
  // cannot cover. A signed-but-disclosure-free receipt reads as a bare
  // bound — reject it. Mirrors Python MeaningReceiptDisclosureError.
  if (!Array.isArray(payload.not_covered) || payload.not_covered.length === 0) {
    throw new VerifyError(
      ERROR_CLASSES.DISCLOSURE_MISSING,
      `payload.not_covered must be a non-empty array; got ${JSON.stringify(payload.not_covered)}`,
    );
  }
  if (typeof payload.disclosure !== "string" || payload.disclosure.trim() === "") {
    throw new VerifyError(
      ERROR_CLASSES.DISCLOSURE_MISSING,
      `payload.disclosure must be a non-empty string; got ${JSON.stringify(payload.disclosure)}`,
    );
  }

  return { verified: true, kid, protectedHeader, payload };
}

export function verifyMeaningRiskReceipt(receipt, jwks) {
  return verifyMeaningEnvelope(receipt, jwks, MEANING_RISK_SCHEMA);
}

export function verifyPerspectiveReceipt(receipt, jwks) {
  return verifyMeaningEnvelope(receipt, jwks, PERSPECTIVE_SCHEMA);
}
