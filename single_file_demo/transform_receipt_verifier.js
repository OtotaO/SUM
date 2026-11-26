// SUM transform-receipt verifier (T1d).
//
// Verifies sum.transform_receipt.v1 envelopes. Mirrors the
// six-step algorithm + forward-compat levers from receipt_verifier.js
// (which handles sum.render_receipt.v1); the only difference is the
// schema string and the payload-field semantics.
//
// Why a parallel module rather than a refactor: receipt_verifier.js
// is already battle-tested against the render-receipt fixture matrix
// (run under Node in CI; the K1–K4 + A1–A8 harnesses cover bundles). Refactoring it to share with the
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

// Payload fields REQUIRED by sum.transform_receipt.v1. `receipt.schema` sits OUTSIDE the
// JWS and is therefore attacker-editable: relabelling another receipt
// family to this schema makes the schema check above pass on a payload
// this verifier has never validated. A genuine signature over a DIFFERENT
// family's payload is still a genuine signature, so the crypto alone does
// not close this. Same bug class as JWT alg-confusion.
//
// Kept byte-for-byte in step with the Python REQUIRED_PAYLOAD_FIELDS in
// sum_engine_internal/transform_receipt/verifier.py. Cross-runtime accept/reject parity is the
// contract the trust triangle asserts; a divergence here is a defect.
export const REQUIRED_PAYLOAD_FIELDS = Object.freeze([
  "transform",
  "transform_id",
  "input_hash",
  "output_hash",
  "parameters_hash",
  "model",
  "provider",
  "signed_at",
  "digital_source_type",
]);

function checkPayloadShape(payload) {
  const missing = REQUIRED_PAYLOAD_FIELDS.filter(
    (f) => !Object.prototype.hasOwnProperty.call(payload, f),
  );
  if (missing.length > 0) {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_RECEIPT,
      `payload declares schema ${SUPPORTED_SCHEMA} but is missing ` +
        `required field(s) ${JSON.stringify(missing)}: refusing to verify ` +
        `a payload of another receipt family ` +
        `(schema is not covered by the signature)`,
    );
  }
}


export const KNOWN_CRIT_EXTENSIONS = new Set(["b64"]);

// The protected-header rule, the same in jose_envelope.py: size limits (the
// libraries' own limits differ, joserfc's by version and jose has none), the
// type joserfc requires of each registered JWS header name (RFC 7515 section
// 4.1, plus b64 from RFC 7797; alg is checked against the algorithm registry
// instead), and a strict UTF-8 decoder that keeps a BOM (ignoreBOM) so
// JSON.parse refuses it, as Python's decode and json.loads do. The default
// decoder dropped a BOM and turned invalid bytes into U+FFFD.
export const MAX_PROTECTED_HEADER_B64_CHARS = 512;
export const MAX_PAYLOAD_JCS_BYTES = 128000;
const isString = (v) => typeof v === "string";
const isHttpUrl = (v) => isString(v) && (v.startsWith("http://") || v.startsWith("https://"));
const isStringArray = (v) => Array.isArray(v) && v.every(isString);
const REGISTERED_HEADER_TYPES = [
  ["jku", isHttpUrl, "an http or https URL string"],
  ["jwk", (v) => v !== null && typeof v === "object" && !Array.isArray(v), "an object"],
  ["kid", isString, "a string"],
  ["x5u", isHttpUrl, "an http or https URL string"],
  ["x5c", isStringArray, "an array of strings"],
  ["x5t", isString, "a string"],
  ["x5t#S256", isString, "a string"],
  ["typ", isString, "a string"],
  ["cty", isString, "a string"],
  ["crit", isStringArray, "an array of strings"],
  ["b64", (v) => typeof v === "boolean", "a boolean"],
];
const STRICT_UTF8 = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });

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

// A JWS signature segment and a JWK `x` must be the canonical unpadded
// base64url of their bytes: the right length, base64url characters only (no
// "=", as RFC 7515 section 2 requires), and the unused low bits of the last
// character zero. atob and WebCrypto ignore those bits and padding. The Python
// verifier applies the same rule (jose_envelope._is_canonical_b64url), so both
// runtimes give the same verdict on a padded or respelled signature or key.
const B64URL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
function isCanonicalB64url(s, nBytes) {
  const len = Math.ceil((nBytes * 4) / 3);
  const spareBits = len * 6 - nBytes * 8;
  return typeof s === "string" && s.length === len && /^[A-Za-z0-9_-]+$/.test(s) &&
    B64URL_ALPHABET.indexOf(s[len - 1]) % (1 << spareBits) === 0;
}

// The protected header segment follows the same rule at any length: base64url
// characters only, no "=", a length that is not 1 mod 4, spare low bits zero.
function isCanonicalB64urlSegment(s) {
  if (typeof s !== "string" || !/^[A-Za-z0-9_-]+$/.test(s) || s.length % 4 === 1) return false;
  const spareBits = [0, 0, 4, 2][s.length % 4];
  return B64URL_ALPHABET.indexOf(s[s.length - 1]) % (1 << spareBits) === 0;
}

// Why this JWK may not verify an EdDSA signature, or null if it may. The same
// rule as _jwk_member_error in the Python verifier, applied before import so
// both runtimes reject the same keys with the same class: WebCrypto refused a
// private key, key_ops without "verify" and use other than "sig", while
// joserfc accepted a private key and refused some of the others (key_ops
// without "verify", use "enc", another alg) only at verify time.
// alg may be "EdDSA" or its RFC 9864 name "Ed25519", which WebCrypto writes
// when it exports an Ed25519 key; the protected header must still name EdDSA.
function jwkMemberError(jwk) {
  const has = (k) => Object.prototype.hasOwnProperty.call(jwk, k);
  if (has("d")) return "carries the private key member d; a verification key set holds public keys only";
  if (has("key_ops")) {
    const ops = jwk.key_ops;
    if (!Array.isArray(ops) || !ops.every((o) => typeof o === "string") ||
        new Set(ops).size !== ops.length || !ops.includes("verify")) {
      return 'key_ops must be an array of distinct strings that includes "verify"';
    }
  }
  if (has("use") && jwk.use !== "sig") return 'use must be "sig"';
  if (has("alg") && jwk.alg !== "EdDSA" && jwk.alg !== "Ed25519") return 'alg must be "EdDSA" or "Ed25519"';
  return null;
}

async function importEd25519Jwk(jwk) {
  if (jwk.kty !== "OKP" || jwk.crv !== "Ed25519") {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_JWKS,
      `expected OKP/Ed25519 JWK, got kty=${jwk.kty} crv=${jwk.crv}`,
    );
  }
  if (!isCanonicalB64url(jwk.x, 32)) {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_JWKS,
      "JWK x is not the canonical base64url of a 32-byte Ed25519 public key",
    );
  }
  const memberError = jwkMemberError(jwk);
  if (memberError !== null) {
    throw new VerifyError(ERROR_CLASSES.MALFORMED_JWKS, `JWK ${memberError}`);
  }
  // Import the public key from x alone, so no implementation can build it
  // from another member; the key that verifies is the x a pin names.
  return crypto.subtle.importKey(
    "jwk",
    { kty: "OKP", crv: "Ed25519", x: jwk.x },
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
  if (!receipt || typeof receipt !== "object" || Array.isArray(receipt)) {
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
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
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

  if (key.kty !== "OKP" || key.crv !== "Ed25519") {
    // Checked here, at step 1, as in Python, so a key of another type is
    // malformed_jwks whatever else is wrong with the receipt.
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_JWKS,
      `expected OKP/Ed25519 JWK for kid=${kid}, got kty=${key.kty} crv=${key.crv}`,
    );
  }

  // ---- Step 2: JCS canonicalize payload ----
  // canonicalize throws a plain Error on a lone surrogate or a non-finite
  // number (JSON.parse reads 1e999 as Infinity). Python rejects both as
  // malformed_receipt, so name the class here too.
  let canonicalText;
  try {
    canonicalText = canonicalize(payload);
  } catch (e) {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_RECEIPT,
      `payload could not be JCS-canonicalized: ${e.message}`,
    );
  }
  if (canonicalText === undefined || canonicalText === null) {
    throw new VerifyError(ERROR_CLASSES.MALFORMED_RECEIPT, "payload could not be JCS-canonicalized");
  }
  const canonicalBytes = new TextEncoder().encode(canonicalText);
  if (canonicalBytes.length > MAX_PAYLOAD_JCS_BYTES) {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_RECEIPT,
      `payload is ${canonicalBytes.length} bytes after JCS canonicalization; the limit is ${MAX_PAYLOAD_JCS_BYTES}`,
    );
  }

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
  if (!isCanonicalB64urlSegment(proto)) {
    throw new VerifyError(ERROR_CLASSES.MALFORMED_JWS, "protected header is not canonical unpadded base64url");
  }
  if (proto.length > MAX_PROTECTED_HEADER_B64_CHARS) {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_JWS,
      `protected header is ${proto.length} base64url characters; the limit is ${MAX_PROTECTED_HEADER_B64_CHARS}`,
    );
  }
  try {
    const headerJson = STRICT_UTF8.decode(b64urlDecodeToBytes(proto));
    header = JSON.parse(headerJson);
  } catch (e) {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_JWS,
      `protected header is not valid JSON: ${e.message}`,
    );
  }
  // Array.isArray is load-bearing: typeof [] === "object", so a JSON array
  // header slipped past this check and failed later as signature_invalid,
  // diverging from Python's malformed_jws. RFC 7515 §4 requires an object.
  if (!header || typeof header !== "object" || Array.isArray(header)) {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_JWS,
      "protected header is not an object",
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
  // `header.alg && …` would SKIP the registry check whenever alg is absent or
  // falsy (missing, "", 0, false, null), so the anti-downgrade control could be
  // bypassed by simply omitting the claim; those cases then failed later as
  // signature_invalid where Python says unsupported_alg. Require a string, as
  // receipt_verifier.js and the Python core both do.
  if (typeof header.alg !== "string" || !SUPPORTED_SIGNATURE_ALGORITHMS.has(header.alg)) {
    throw new VerifyError(
      ERROR_CLASSES.UNSUPPORTED_ALG,
      `unsupported alg ${header.alg}; this verifier accepts ${
        [...SUPPORTED_SIGNATURE_ALGORITHMS].join(", ")
      }`,
    );
  }

  // ---- Step 3.7: registered header member types ----
  for (const [name, hasType, wanted] of REGISTERED_HEADER_TYPES) {
    if (Object.prototype.hasOwnProperty.call(header, name) && !hasType(header[name])) {
      throw new VerifyError(ERROR_CLASSES.MALFORMED_JWS, `protected header ${name} must be ${wanted}`);
    }
  }

  // ---- Step 3.8: unencoded-payload invariants, before the signature ----
  // The same check as step 6, made before verifying, as in Python: jose builds
  // the signing input from b64, and this verifier always passes the raw bytes,
  // so another b64 value would otherwise surface as signature_invalid.
  if (header.b64 !== false) {
    throw new VerifyError(
      ERROR_CLASSES.HEADER_INVARIANT_VIOLATED,
      `expected b64=false (detached payload encoding), got b64=${JSON.stringify(header.b64)}`,
    );
  }
  if (!Array.isArray(header.crit) || !header.crit.includes("b64")) {
    throw new VerifyError(
      ERROR_CLASSES.HEADER_INVARIANT_VIOLATED,
      `expected crit array containing "b64", got ${JSON.stringify(header.crit)}`,
    );
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

  if (!isCanonicalB64url(signature, 64)) {
    throw new VerifyError(
      ERROR_CLASSES.SIGNATURE_INVALID,
      "signature is not the canonical base64url of a 64-byte Ed25519 signature",
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

  // AFTER the signature is proven, mirroring the Python ordering: it keeps
  // the malformed_jws / signature_invalid precedence intact and does not
  // leak payload shape to a caller without a valid signature.
  checkPayloadShape(payload);

  return {
    verified: true,
    kid,
    protectedHeader,
    payload,
  };
}

// The receipt timestamp form (RENDER_RECEIPT_FORMAT.md §6.1), a strict profile
// of RFC 3339: uppercase T and Z, seconds required (00 to 59), an optional
// fraction of 1 to 9 digits truncated to milliseconds, Z or +-HH:MM.
// Shared line for line with _instant_ms in the Python verifier and with
// parseInstantMs in receipt_verifier.js (kept per file, like the other helpers).
const INSTANT = /^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]{1,9}))?(Z|([+-])([0-9]{2}):([0-9]{2}))$/;

function daysFromCivil(y, m, d) {
  y -= m <= 2 ? 1 : 0;
  const era = Math.floor((y >= 0 ? y : y - 399) / 400);
  const yoe = y - era * 400;
  const doy = Math.floor((153 * (m + (m > 2 ? -3 : 9)) + 2) / 5) + d - 1;
  const doe = yoe * 365 + Math.floor(yoe / 4) - Math.floor(yoe / 100) + doy;
  return era * 146097 + doe - 719468;
}

// Epoch milliseconds for an instant in the receipt timestamp form, or null. Exported so
// test_transform_receipt_verify.js can run the shared instant table on this copy.
export function parseInstantMs(value) {
  if (typeof value !== "string") return null;
  const m = INSTANT.exec(value);
  if (!m) return null;
  const [y, mo, d, h, mi, s] = [1, 2, 3, 4, 5, 6].map((i) => Number(m[i]));
  const ms = Number((m[7] || "").padEnd(3, "0").slice(0, 3));
  const leap = y % 4 === 0 && (y % 100 !== 0 || y % 400 === 0);
  const daysInMonth = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  if (!(mo >= 1 && mo <= 12 && d >= 1 && d <= daysInMonth[mo - 1] && h <= 23 && mi <= 59 && s <= 59)) {
    return null;
  }
  let offset = 0;
  if (m[8] !== "Z") {
    const oh = Number(m[10]);
    const om = Number(m[11]);
    if (oh > 23 || om > 59) return null;
    offset = (oh * 60 + om) * (m[9] === "+" ? 1 : -1);
  }
  return (((daysFromCivil(y, mo, d) * 24 + h) * 60 + mi) * 60 + s) * 1000 + ms - offset * 60000;
}

function parseSignedAt(s) {
  const ms = parseInstantMs(s);
  if (ms === null) {
    throw new Error(`signed_at ${JSON.stringify(s)} is not in the receipt timestamp form (RENDER_RECEIPT_FORMAT.md §6.1)`);
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
