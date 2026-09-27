// SUM render-receipt verifier (Phase E.1 v0.9.B).
//
// Implements the six-step verifier algorithm from
// docs/RENDER_RECEIPT_FORMAT.md §2.1, plus the two forward-compat
// levers from §1.4 (schema check, RFC 7515 §4.1.11 crit-extension
// fail-closed). Uses the vendored ESM bundle at
// vendor/sum-verify-deps.js — no CDN, no network at page load.
//
// Same module is consumed by:
//   - test_render_receipt_verify.js  (Node smoke against the
//     fixture set under fixtures/render_receipts/).
//   - index.html                     (in-page Verify-last-render
//     button next to the rendered tome).
//   - any third-party verifier UI    (this file is the public
//     surface for the v0.9.B trust loop).

import { flattenedVerify, canonicalize } from "./vendor/sum-verify-deps.js";

export const SUPPORTED_SCHEMA = "sum.render_receipt.v1";

// Payload fields REQUIRED by sum.render_receipt.v1. `receipt.schema` sits OUTSIDE the
// JWS and is therefore attacker-editable: relabelling another receipt
// family to this schema makes the schema check above pass on a payload
// this verifier has never validated. A genuine signature over a DIFFERENT
// family's payload is still a genuine signature, so the crypto alone does
// not close this. Same bug class as JWT alg-confusion.
//
// Kept byte-for-byte in step with the Python REQUIRED_PAYLOAD_FIELDS in
// sum_engine_internal/render_receipt/verifier.py. Cross-runtime accept/reject parity is the
// contract the trust triangle asserts; a divergence here is a defect.
export const REQUIRED_PAYLOAD_FIELDS = Object.freeze([
  "render_id",
  "sliders_quantized",
  "triples_hash",
  "tome_hash",
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


// crit extensions this verifier knows how to handle. RFC 7515
// §4.1.11: a verifier MUST reject closed on critical extensions it
// doesn't understand. b64=false is the unencoded-payload semantics
// from RFC 7797 — the only critical extension v1 receipts use.
export const KNOWN_CRIT_EXTENSIONS = new Set(["b64"]);

// Error classes (runtime-neutral; mirrored by the v0.9.C Python
// verifier). See fixtures/render_receipts/README.md.
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
  REVOKED_KID: "revoked_kid",
  UNSUPPORTED_ALG: "unsupported_alg",
  SIGNED_AT_OUT_OF_WINDOW: "signed_at_out_of_window",
});

// G3 crypto-agility: signature algorithms accepted under `current`
// in the in-tree algorithm registry (docs/ALGORITHM_REGISTRY.md).
// A `alg` claim outside this set is rejected with `unsupported_alg`,
// distinct from `header_invariant_violated`. Mirrors
// SUPPORTED_SIGNATURE_ALGORITHMS in the Python verifier.
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
  // Node fallback (smoke test path)
  return Uint8Array.from(Buffer.from(s, "base64url"));
}

async function importEd25519Jwk(jwk) {
  // EdDSA / Ed25519 (OKP) JWK import via SubtleCrypto. Supported in
  // Node ≥20, Chrome 113+, Firefox 129+, Safari 17+. Older
  // browsers fail at this step with a precise error message. (Node
  // gained Ed25519 in 18.4, but the vendored canonicalize@5 calls
  // String.prototype.isWellFormed, which Node has only from 20.
  // Node 20 is verified; canonicalize 5 itself declares Node >= 22.)
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

const REVOKED_KIDS_SCHEMA = "sum.revoked_kids.v1";
// RFC 3339 instant, seconds required, fraction truncated to milliseconds.
// Shared line for line with _instant_ms in the Python verifier.
const INSTANT = /^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]{1,9}))?(Z|([+-])([0-9]{2}):([0-9]{2}))$/;

function daysFromCivil(y, m, d) {
  y -= m <= 2 ? 1 : 0;
  const era = Math.floor((y >= 0 ? y : y - 399) / 400);
  const yoe = y - era * 400;
  const doy = Math.floor((153 * (m + (m > 2 ? -3 : 9)) + 2) / 5) + d - 1;
  const doe = yoe * 365 + Math.floor(yoe / 4) - Math.floor(yoe / 100) + doy;
  return era * 146097 + doe - 719468;
}

// Epoch milliseconds for an RFC 3339 instant, or null if malformed.
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

// Accept the entry list or the served sum.revoked_kids.v1 document; fail
// closed on anything else, and on any entry that is not an object.
function revocationEntries(revokedKids) {
  let list = revokedKids;
  if (list && typeof list === "object" && !Array.isArray(list)) {
    // The served document, or the bare {"revoked": [...]} form that
    // RENDER_RECEIPT_FORMAT §6.1 describes; any other object fails closed.
    const schema = list.schema === undefined ? REVOKED_KIDS_SCHEMA : list.schema;
    if (schema === REVOKED_KIDS_SCHEMA && Array.isArray(list.revoked)) {
      list = list.revoked;
    } else {
      throw new VerifyError(
        ERROR_CLASSES.REVOKED_KID,
        "revoked_kids must be a list of revocation entries or the " +
          "sum.revoked_kids.v1 document served at " +
          "/.well-known/revoked-kids.json; failing closed",
      );
    }
  }
  if (!Array.isArray(list)) {
    throw new VerifyError(
      ERROR_CLASSES.REVOKED_KID,
      `revoked_kids must be a list of revocation entries, got ${typeof list}; failing closed`,
    );
  }
  for (const entry of list) {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      throw new VerifyError(
        ERROR_CLASSES.REVOKED_KID,
        "revocation list contains a non-object entry; failing closed",
      );
    }
    if (typeof entry.kid !== "string" || entry.kid === "") {
      throw new VerifyError(
        ERROR_CLASSES.REVOKED_KID,
        "revocation list entry has no non-empty string kid; failing closed",
      );
    }
  }
  return list;
}

/**
 * Check the receipt's kid against a revocation list and reject with
 * REVOKED_KID when signed_at is at or after effective_revocation_at.
 * Mirrors sum_engine_internal.render_receipt.verifier._check_revoked_kid:
 * the same accepted input shapes, the same instant grammar and the same
 * millisecond arithmetic (cases: Tests/fixtures/revocation_instants.json).
 *
 * @param {object} receipt
 * @param {Array<{kid: string, effective_revocation_at: string, reason?: string}>|{schema?: string, revoked: Array}} revokedKids
 *   the entry list, or the sum.revoked_kids.v1 document; anything else fails closed
 */
function checkRevokedKid(receipt, revokedKids) {
  const entries = revocationEntries(revokedKids);
  const kid = receipt && receipt.kid;
  if (typeof kid !== "string") return;
  const payload = receipt && receipt.payload;
  const signedAt = payload && typeof payload === "object" ? payload.signed_at : undefined;

  for (const entry of entries) {
    if (entry.kid !== kid) continue;
    const effectiveAt = entry.effective_revocation_at;
    const effective = parseInstantMs(effectiveAt);
    if (effective === null) {
      throw new VerifyError(
        ERROR_CLASSES.REVOKED_KID,
        `kid ${JSON.stringify(kid)} appears on revocation list with malformed ` +
          `effective_revocation_at=${JSON.stringify(effectiveAt)}; failing closed`,
      );
    }
    const signed = parseInstantMs(signedAt);
    if (signed === null) {
      throw new VerifyError(
        ERROR_CLASSES.REVOKED_KID,
        `kid ${JSON.stringify(kid)} on revocation list and receipt has no ` +
          `parseable signed_at; failing closed`,
      );
    }
    // Compare instants; a string compare put "...16.849Z" before "...16Z".
    if (signed >= effective) {
      throw new VerifyError(
        ERROR_CLASSES.REVOKED_KID,
        `kid ${JSON.stringify(kid)} revoked effective ${effectiveAt}; ` +
          `receipt signed at ${signedAt} (>= effective time)`,
      );
    }
    // signed_at < effective_at: legitimate historical receipt;
    // continue verification.
    return;
  }
}

/**
 * Verify a SUM render receipt against a JWKS.
 *
 * @param {object} receipt - { schema, kid, payload, jws }
 * @param {object} jwks    - { keys: [...] }
 * @param {Array=} revokedKids - Optional G3 revocation list. When
 *   provided, kids on the list with signed_at >= effective_revocation_at
 *   are rejected with REVOKED_KID. Pass undefined or null to skip
 *   revocation entirely (default; backwards-compat with v0.9.C).
 * @returns {Promise<{ verified: true, kid: string, protectedHeader: object, payload: object }>}
 * @throws  {VerifyError}  - on any failure, with .errorClass set to one of ERROR_CLASSES.
 */
/**
 * Verify a sum.render_receipt.v1 envelope.
 *
 * Optional replay-defense window: pass `{maxAgeSeconds, maxFutureSkewSeconds}`
 * in the fourth argument to reject receipts whose `payload.signed_at`
 * is outside the acceptance window. Default (omitted) does NOT
 * enforce. See docs/RENDER_RECEIPT_FORMAT.md §6.2.
 *
 * @param {object} receipt
 * @param {object} jwks
 * @param {Array|null} [revokedKids] G3 revocation list (optional).
 * @param {object} [opts]
 * @param {number} [opts.maxAgeSeconds] Max age in seconds (opt-in).
 * @param {number} [opts.maxFutureSkewSeconds=60] Clock-skew tolerance.
 */
export async function verifyReceipt(receipt, jwks, revokedKids, opts) {
  const { maxAgeSeconds = null, maxFutureSkewSeconds = 60 } = opts || {};
  // ---- G3 revocation gate (runs BEFORE crypto verify) ----
  // A kid that was both revoked AND tampered surfaces as
  // `revoked_kid` (more actionable for an operator — points at
  // "rotate + revoke" rather than "investigate the signature").
  if (revokedKids != null) {
    checkRevokedKid(receipt, revokedKids);
  }

  // ---- Step 0 (shape gate) ----
  if (!receipt || typeof receipt !== "object" || Array.isArray(receipt)) {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_RECEIPT,
      "receipt is not an object",
    );
  }

  // ---- Step 0.5 (forward-compat: schema) ----
  // Per RENDER_RECEIPT_FORMAT.md §1.4, a v1-aware verifier MUST
  // reject receipts with an unknown schema identifier. This is
  // future-proofing for v2.
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

  // ---- Step 1: kid lookup in JWKS ----
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

  // ---- Step 2: JCS-canonicalize payload ----
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

  // ---- Step 3.5 (forward-compat): inspect protected header BEFORE verify ----
  // The crit-extension rule per RFC 7515 §4.1.11: a verifier that
  // doesn't understand a critical extension MUST reject. We can
  // read the header bytes without verifying — they're encoded in
  // the protected segment, not derived from the signature. Doing
  // this BEFORE signature verification means a future crit
  // extension surfaces as crit_unknown_extension (the spec's
  // intended fail-closed class), not as signature_invalid.
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
  // Valid JSON is not enough: RFC 7515 §4 requires an object. `null` would
  // throw a raw TypeError on the property access below, and an array would
  // slip past a bare typeof check (typeof [] === "object") to fail later as
  // signature_invalid — both escaping the declared class. Matches the Python
  // guard in jose_envelope.py.
  if (!header || typeof header !== "object" || Array.isArray(header)) {
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_JWS,
      `protected header must be a JSON object, got ${
        header === null ? "null" : Array.isArray(header) ? "array" : typeof header
      }`,
    );
  }
  if (Array.isArray(header.crit)) {
    for (const ext of header.crit) {
      if (!KNOWN_CRIT_EXTENSIONS.has(ext)) {
        throw new VerifyError(
          ERROR_CLASSES.CRIT_UNKNOWN_EXTENSION,
          `protected header crit contains unsupported extension: ${ext}`,
        );
      }
    }
  }

  // ---- Step 3.6: G3 alg-registry check (crypto-agility) ----
  // Cross-check the protected header's `alg` against the in-tree
  // algorithm registry BEFORE signature verification. Mirrors the
  // Python verifier's pre-jose alg check; defends against the
  // JWT/JWS classic "alg downgrade" / "alg confusion" attack
  // pattern. See docs/ALGORITHM_REGISTRY.md.
  if (typeof header.alg !== "string" || !SUPPORTED_SIGNATURE_ALGORITHMS.has(header.alg)) {
    throw new VerifyError(
      ERROR_CLASSES.UNSUPPORTED_ALG,
      `protected header alg=${JSON.stringify(header.alg)} is not in the ` +
        `supported algorithm registry ` +
        `(${JSON.stringify([...SUPPORTED_SIGNATURE_ALGORITHMS])}); see ` +
        `docs/ALGORITHM_REGISTRY.md`,
    );
  }

  // ---- Step 4: import the key ----
  let cryptoKey;
  try {
    cryptoKey = await importEd25519Jwk(key);
  } catch (e) {
    if (e instanceof VerifyError) throw e;
    throw new VerifyError(
      ERROR_CLASSES.MALFORMED_JWKS,
      `JWKS key for kid=${kid} could not be imported: ${e.message}`,
    );
  }

  // ---- Step 5: cryptographic verify ----
  const flattened = {
    protected: proto,
    payload: canonicalBytes,
    signature,
  };

  let result;
  try {
    result = await flattenedVerify(flattened, cryptoKey);
  } catch (e) {
    // jose throws with .code = "ERR_JWS_SIGNATURE_VERIFICATION_FAILED"
    // on signature failure. Other crypto errors (bad encoding,
    // unsupported alg, etc.) come through with different codes;
    // surface them as signature_invalid since the receipt cannot
    // be trusted regardless.
    throw new VerifyError(
      ERROR_CLASSES.SIGNATURE_INVALID,
      `signature verification failed: ${e.code || e.message}`,
    );
  }

  // ---- Step 6: assert protected header invariants ----
  const ph = result.protectedHeader;
  if (ph.alg !== "EdDSA") {
    throw new VerifyError(
      ERROR_CLASSES.HEADER_INVARIANT_VIOLATED,
      `expected alg=EdDSA, got ${ph.alg}`,
    );
  }
  if (ph.kid !== receipt.kid) {
    throw new VerifyError(
      ERROR_CLASSES.KID_MISMATCH,
      `protected header kid=${ph.kid} != receipt.kid=${receipt.kid}`,
    );
  }
  if (ph.b64 !== false) {
    throw new VerifyError(
      ERROR_CLASSES.HEADER_INVARIANT_VIOLATED,
      `expected b64=false (detached payload encoding), got b64=${ph.b64}`,
    );
  }
  if (!Array.isArray(ph.crit) || !ph.crit.includes("b64")) {
    throw new VerifyError(
      ERROR_CLASSES.HEADER_INVARIANT_VIOLATED,
      `expected crit array containing "b64", got ${JSON.stringify(ph.crit)}`,
    );
  }

  // ---- Step 7: optional replay-window check ----
  // Default (maxAgeSeconds=null) does NOT enforce. See
  // docs/RENDER_RECEIPT_FORMAT.md §6.2 for the receiver-policy
  // contract. Cryptographic integrity is already proven above; this
  // is a policy-layer rejection ("the receipt is genuine but I don't
  // accept it at this age").
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
    protectedHeader: ph,
    payload,
  };
}

function parseSignedAtRender(s) {
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
    signedAtMs = parseSignedAtRender(signedAt);
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
