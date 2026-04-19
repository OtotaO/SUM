/**
 * ProvenanceRecord — JavaScript port of internal/infrastructure/provenance.py.
 *
 * Byte-identical prov_id computation across Python and JavaScript. The
 * content-addressability contract:
 *
 *     prov_id = "prov:" + hex(sha256(JCS(record)))[:32]
 *
 * holds in both runtimes because (a) the JCS canonicalizer is
 * byte-identical (verified by scripts/verify_jcs_byte_identity.py) and
 * (b) SHA-256 is universal. A ProvenanceRecord minted in Python produces
 * the same prov_id as the JS port produces on the same fields — which
 * means the AkashicLedger (Python) and the single-file demo artifact
 * (JS, browser) agree on axiom-evidence identity without any schema
 * negotiation.
 *
 * Scope matches provenance.py exactly. Field invariants are validated on
 * construction; the same invariants that raise InvalidProvenanceError in
 * Python raise new Error() here with message-substring matches so
 * cross-runtime error paths stay debuggable.
 *
 * No dependencies. Pure ES2015+. Works in Node ≥ 14 and every modern
 * browser. SHA-256 uses Node's ``crypto.createHash`` when available, or
 * falls back to the async ``crypto.subtle.digest`` (browser path, which
 * returns a Promise and therefore makes ``computeProvId`` an async
 * function in that environment — see ``computeProvIdAsync`` for the
 * browser API).
 */
'use strict';

const { canonicalize } = require('./jcs');

const PROVENANCE_SCHEMA_VERSION = '1.0.0';
const EXCERPT_MAX_CHARS = 200;

const ALLOWED_URI_PREFIXES = [
  'sha256:',
  'doi:',
  'https://',
  'urn:sum:source:',
];

class InvalidProvenanceError extends Error {
  constructor(message) {
    super(message);
    this.name = 'InvalidProvenanceError';
  }
}

/**
 * Validate a source URI against the allowed schemes.
 *
 * Bare ``http://`` (no signature, no content hash) is explicitly
 * rejected — see provenance.py for the reasoning (cargo-cult
 * provenance is worse than no provenance).
 *
 * @param {string} uri
 * @throws {InvalidProvenanceError}
 */
function validateSourceUri(uri) {
  if (typeof uri !== 'string' || uri === '') {
    throw new InvalidProvenanceError('source_uri must be a non-empty string');
  }
  for (const prefix of ALLOWED_URI_PREFIXES) {
    if (uri.startsWith(prefix)) {
      if (prefix === 'sha256:') {
        const body = uri.slice('sha256:'.length);
        if (body.length !== 64 || !/^[0-9a-f]{64}$/.test(body)) {
          throw new InvalidProvenanceError(
            'sha256: URI body must be 64 lowercase hex chars'
          );
        }
      }
      return;
    }
  }
  throw new InvalidProvenanceError(
    `source_uri scheme not supported: ${JSON.stringify(uri)}. ` +
    `Allowed: ${ALLOWED_URI_PREFIXES.join(', ')}. ` +
    `A bare http:// URL is not acceptable — fetch, hash, and use ` +
    `sha256:<hex> instead.`
  );
}

/**
 * Construct a validated ProvenanceRecord. All fields required except
 * schema_version, which defaults to the module-level constant.
 *
 * @param {object} params
 * @param {string} params.source_uri
 * @param {number} params.byte_start
 * @param {number} params.byte_end
 * @param {string} params.extractor_id
 * @param {string} params.timestamp
 * @param {string} params.text_excerpt
 * @param {string} [params.schema_version]
 * @returns {object}  frozen object, cross-runtime-canonicalizable
 */
function provenanceRecord({
  source_uri,
  byte_start,
  byte_end,
  extractor_id,
  timestamp,
  text_excerpt,
  schema_version = PROVENANCE_SCHEMA_VERSION,
}) {
  validateSourceUri(source_uri);
  if (!Number.isInteger(byte_start) || byte_start < 0) {
    throw new InvalidProvenanceError(
      `byte_start must be >= 0 integer, got ${byte_start}`
    );
  }
  if (!Number.isInteger(byte_end) || byte_end <= byte_start) {
    throw new InvalidProvenanceError(
      `byte_end (${byte_end}) must be strictly greater than ` +
      `byte_start (${byte_start}) — empty ranges are not evidence`
    );
  }
  if (typeof extractor_id !== 'string' || extractor_id.trim() === '') {
    throw new InvalidProvenanceError('extractor_id must be non-empty');
  }
  if (typeof timestamp !== 'string' || timestamp.trim() === '') {
    throw new InvalidProvenanceError('timestamp must be non-empty');
  }
  if (typeof text_excerpt !== 'string') {
    throw new InvalidProvenanceError('text_excerpt must be a string');
  }
  if (text_excerpt.length > EXCERPT_MAX_CHARS) {
    throw new InvalidProvenanceError(
      `text_excerpt exceeds ${EXCERPT_MAX_CHARS} chars ` +
      `(got ${text_excerpt.length}); truncate at the caller`
    );
  }

  const record = Object.freeze({
    source_uri,
    byte_start,
    byte_end,
    extractor_id,
    timestamp,
    text_excerpt,
    schema_version,
  });
  return record;
}

/**
 * Synchronous prov_id computation using Node's built-in crypto.
 * Falls through to throwing in browser environments — use
 * ``computeProvIdAsync`` there.
 *
 * @param {object} record
 * @returns {string}  "prov:" + 32 hex chars
 */
function computeProvId(record) {
  const canonical = canonicalize(record);
  // Node.js path
  const { createHash } = require('crypto');
  const digest = createHash('sha256').update(canonical).digest('hex');
  return 'prov:' + digest.slice(0, 32);
}

/**
 * Async prov_id computation using WebCrypto. Works in both Node (where
 * globalThis.crypto.subtle is available since Node 19) and browsers.
 *
 * @param {object} record
 * @returns {Promise<string>}
 */
async function computeProvIdAsync(record) {
  const canonical = canonicalize(record);
  const subtle = (globalThis.crypto && globalThis.crypto.subtle)
    ? globalThis.crypto.subtle
    : null;
  if (!subtle) {
    // Fallback to sync path via Node crypto.
    return computeProvId(record);
  }
  const digestBuf = await subtle.digest('SHA-256', canonical);
  const bytes = new Uint8Array(digestBuf);
  let hex = '';
  for (const b of bytes) hex += b.toString(16).padStart(2, '0');
  return 'prov:' + hex.slice(0, 32);
}

/**
 * Helper: content-addressable URI for a text block.
 * Mirrors ``sha256_uri_for_text`` in provenance.py.
 *
 * @param {string} text
 * @returns {string}  "sha256:<64-hex>"
 */
function sha256UriForText(text) {
  const { createHash } = require('crypto');
  const encoder = new TextEncoder();
  const hex = createHash('sha256')
    .update(Buffer.from(encoder.encode(text)))
    .digest('hex');
  return 'sha256:' + hex;
}

module.exports = {
  PROVENANCE_SCHEMA_VERSION,
  EXCERPT_MAX_CHARS,
  InvalidProvenanceError,
  provenanceRecord,
  validateSourceUri,
  computeProvId,
  computeProvIdAsync,
  sha256UriForText,
};
