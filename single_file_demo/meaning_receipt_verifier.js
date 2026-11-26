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
// disclosure. Stage B, as RECEIPT_FAMILY_SPEC section 4 defines it, is
// re-hashing the committed integer-micro loss vector and re-running the
// conformal certifier over it; it runs NO scorer. Stage B is not done here
// because no JS port of the bound kernel exists and JS lacks the primitives
// it needs (no fsum, no lgamma; Clopper-Pearson needs the latter). It is
// `python -m sum_verify <receipt> --losses <file>`. A verified signature
// here proves the receipt is authentic and self-disclosing; the bound is
// attested only after Stage B. The receipt FORMAT verifies cross-runtime;
// the bound replay does not yet (and says so).

import { flattenedVerify, canonicalize } from "./vendor/sum-verify-deps.js";

export const MEANING_RISK_SCHEMA = "sum.meaning_risk_receipt.v1";
export const PERSPECTIVE_SCHEMA = "sum.perspective_risk_receipt.v1";

// Fields that make a payload a member of its declared family rather than a
// sibling. `schema` sits OUTSIDE the JWS and is attacker-editable, so a
// genuine signature over another family's payload still verifies; PR #444
// closed this for render and transform and left meaning, perspective and
// chain open. This verifier is GENERIC over two schemas, so the required set
// is keyed by schema: perspective carries groups /
// marginal_risk_upper_bound_micro where meaning carries
// risk_upper_bound_micro, and a single shared list would false-reject one of
// them. Mirrors sum_verify/_meaning.py REQUIRED_PAYLOAD_FIELDS for the
// meaning family.
export const REQUIRED_PAYLOAD_FIELDS = Object.freeze({
  [MEANING_RISK_SCHEMA]: Object.freeze([
    "corpus_id", "scorer", "n", "method",
    "risk_upper_bound_micro", "delta_micro", "loss_definition",
  ]),
  [PERSPECTIVE_SCHEMA]: Object.freeze([
    "corpus_id", "scorer", "n", "method", "groups",
    "marginal_risk_upper_bound_micro", "delta_micro", "loss_definition",
  ]),
});

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
  return crypto.subtle.importKey("jwk", { kty: "OKP", crv: "Ed25519", x: jwk.x }, { name: "Ed25519" }, false, ["verify"]);
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
  if (!receipt || typeof receipt !== "object" || Array.isArray(receipt)) {
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
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new VerifyError(ERROR_CLASSES.MALFORMED_RECEIPT, "receipt.payload missing or non-object");
  }
  if (typeof jws !== "string" || !jws) {
    throw new VerifyError(ERROR_CLASSES.MALFORMED_RECEIPT, "receipt.jws missing or empty");
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
    throw new VerifyError(ERROR_CLASSES.UNKNOWN_KID, `no key in JWKS for kid=${kid}`);
  }

  if (key.kty !== "OKP" || key.crv !== "Ed25519") {
    // Checked here, at step 1, as in Python, so a key of another type is
    // malformed_jwks whatever else is wrong with the receipt.
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_JWKS,
      // A member that is not a string is named by its type: an object
      // whose toString is not callable would throw while being printed.
      `expected OKP/Ed25519 JWK for kid=${kid}, got kty=${typeof key.kty === "string" ? key.kty : `(${typeof key.kty})`} crv=${typeof key.crv === "string" ? key.crv : `(${typeof key.crv})`}`,
    );
  }

  // ---- Step 2: JCS canonicalize ----
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
    throw new VerifyError(ERROR_CLASSES.MALFORMED_JWS, `JWS must have exactly 3 segments, got ${parts.length}`);
  }
  const [proto, middle, signature] = parts;
  if (middle !== "") {
    throw new VerifyError(ERROR_CLASSES.MALFORMED_JWS, "detached JWS middle segment must be empty (RFC 7515 §A.5)");
  }

  // ---- Step 3.5: protected-header forward-compat ----
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
    header = JSON.parse(STRICT_UTF8.decode(b64urlDecodeToBytes(proto)));
  } catch (e) {
    throw new VerifyError(ERROR_CLASSES.MALFORMED_JWS, `protected header is not valid JSON: ${e.message}`);
  }
  // Array.isArray is load-bearing: typeof [] === "object", so a JSON array
  // header slipped past this check and failed later as signature_invalid,
  // diverging from Python's malformed_jws. RFC 7515 §4 requires an object.
  if (!header || typeof header !== "object" || Array.isArray(header)) {
    throw new VerifyError(ERROR_CLASSES.MALFORMED_JWS, "protected header is not an object");
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
  // `header.alg && …` would SKIP the registry check whenever alg is absent or
  // falsy (missing, "", 0, false, null), so the anti-downgrade control could be
  // bypassed by simply omitting the claim; those cases then failed later as
  // signature_invalid where Python says unsupported_alg. Require a string, as
  // receipt_verifier.js and the Python core both do.
  if (typeof header.alg !== "string" || !SUPPORTED_SIGNATURE_ALGORITHMS.has(header.alg)) {
    throw new VerifyError(
      ERROR_CLASSES.UNSUPPORTED_ALG,
      `unsupported alg ${header.alg}; this verifier accepts ${[...SUPPORTED_SIGNATURE_ALGORITHMS].join(", ")}`,
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

  // ---- Steps 4 + 5: cryptographic verify ----
  let importedKey;
  try {
    importedKey = await importEd25519Jwk(key);
  } catch (e) {
    if (e instanceof VerifyError) throw e;
    throw new VerifyError(ERROR_CLASSES.MALFORMED_JWKS, `JWK import failed: ${e.message}`);
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
  // `trim()` alone does not strip U+200B (zero-width space) or other Cf/Cc
  // code points, so a disclosure of "​" would pass while rendering blank.
  // Require at least one VISIBLE character — matches Python's _has_visible_text
  // so the trust triangle (Python <-> Node <-> Browser) agrees on rejection.
  const _disclosureVisible =
    typeof payload.disclosure === "string" &&
    payload.disclosure.replace(/[\s\p{Cf}\p{Cc}\p{Zs}\p{Zl}\p{Zp}]/gu, "") !== "";
  if (!_disclosureVisible) {
    throw new VerifyError(
      ERROR_CLASSES.DISCLOSURE_MISSING,
      `payload.disclosure must be a non-empty string with visible text; got ${JSON.stringify(payload.disclosure)}`,
    );
  }

  // Receipt-family shape gate, AFTER the signature is proven, mirroring the
  // Python ordering: it preserves the malformed_jws / signature_invalid
  // precedence and does not leak payload shape to a caller without a valid
  // signature.
  const required = REQUIRED_PAYLOAD_FIELDS[supportedSchema];
  if (required) {
    const missingFields = required.filter(
      (f) => !Object.prototype.hasOwnProperty.call(payload, f),
    );
    if (missingFields.length > 0) {
      throw new VerifyError(
        ERROR_CLASSES.MALFORMED_RECEIPT,
        `payload declares schema ${supportedSchema} but is missing required ` +
          `field(s) ${JSON.stringify(missingFields)}: refusing to verify a ` +
          `payload of another receipt family ` +
          `(schema is not covered by the signature)`,
      );
    }
  }

  return { verified: true, kid, protectedHeader, payload };
}

export function verifyMeaningRiskReceipt(receipt, jwks) {
  return verifyMeaningEnvelope(receipt, jwks, MEANING_RISK_SCHEMA);
}

export function verifyPerspectiveReceipt(receipt, jwks) {
  return verifyMeaningEnvelope(receipt, jwks, PERSPECTIVE_SCHEMA);
}
