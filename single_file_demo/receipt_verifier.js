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
  // Node fallback (smoke test path)
  return Uint8Array.from(Buffer.from(s, "base64url"));
}

async function importEd25519Jwk(jwk) {
  // EdDSA / Ed25519 (OKP) JWK import via SubtleCrypto. Supported in
  // Node ≥18.4, Chrome 113+, Firefox 129+, Safari 17+. Older
  // browsers fail at this step with a precise error message.
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
 * Check the receipt's kid against a G3 revocation list. Throws
 * VerifyError(REVOKED_KID) if the kid is on the list AND the
 * receipt's signed_at is at or after the revocation's
 * effective_revocation_at. See docs/RENDER_RECEIPT_FORMAT.md §6.1.
 *
 * Mirrors sum_engine_internal.render_receipt._check_revoked_kid
 * exactly so cross-runtime fixtures produce byte-identical
 * outcomes.
 *
 * @param {object} receipt
 * @param {Array<{kid: string, effective_revocation_at: string, reason?: string}>} revokedKids
 */
function checkRevokedKid(receipt, revokedKids) {
  const kid = receipt && receipt.kid;
  if (typeof kid !== "string") return;
  const payload = (receipt && receipt.payload) || {};
  const signedAt = payload.signed_at;

  for (const entry of revokedKids) {
    if (!entry || typeof entry !== "object") continue;
    if (entry.kid !== kid) continue;
    const effectiveAt = entry.effective_revocation_at;
    if (typeof effectiveAt !== "string") {
      throw new VerifyError(
        ERROR_CLASSES.REVOKED_KID,
        `kid ${JSON.stringify(kid)} appears on revocation list with malformed ` +
          `effective_revocation_at=${JSON.stringify(effectiveAt)}; failing closed`,
      );
    }
    if (typeof signedAt !== "string") {
      throw new VerifyError(
        ERROR_CLASSES.REVOKED_KID,
        `kid ${JSON.stringify(kid)} on revocation list and receipt has no ` +
          `parseable signed_at; failing closed`,
      );
    }
    // ISO-8601 UTC strings compare correctly via lex-order.
    if (signedAt >= effectiveAt) {
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
export async function verifyReceipt(receipt, jwks, revokedKids) {
  // ---- G3 revocation gate (runs BEFORE crypto verify) ----
  // A kid that was both revoked AND tampered surfaces as
  // `revoked_kid` (more actionable for an operator — points at
  // "rotate + revoke" rather than "investigate the signature").
  if (revokedKids != null) {
    checkRevokedKid(receipt, revokedKids);
  }

  // ---- Step 0 (shape gate) ----
  if (!receipt || typeof receipt !== "object") {
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

  // ---- Step 1: kid lookup in JWKS ----
  const key = (jwks?.keys || []).find((k) => k.kid === kid);
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

  return {
    verified: true,
    kid,
    protectedHeader: ph,
    payload,
  };
}
