// The review packet is an unsigned container. Only render.receipt is signed.
// Pure functions shared by the browser and offline Node regression tests.
import { canonicalize } from './vendor/sum-verify-deps.js';
import { verifyReceipt, VerifyError } from './receipt_verifier.js';

export const REVIEW_SCHEMA = 'sum.review_packet.v1';
export const MAX_REVIEW_CHARS = 100000;
export const REVIEW_SCOPE = 'Literal sentence comparison with lexical suggestions. Paraphrases, entailment, factual accuracy and meaning preservation are not measured. Human decisions and the original source are unsigned.';

export async function hashText(text) {
  const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return 'sha256-' + Array.from(new Uint8Array(bytes), b => b.toString(16).padStart(2, '0')).join('');
}

export function sourceSpans(text) {
  if (typeof text !== 'string' || text.length > MAX_REVIEW_CHARS) throw new Error('Text must be at most 100,000 characters.');
  // Offsets count UTF-16 code units, matching textarea.setSelectionRange.
  // The complete text is retained even when sentence boundaries are ambiguous.
  return Array.from(text.matchAll(/[^.!?\n]+(?:[.!?]+(?=\s|$)|$)|[^\n]+/gu), (m, i) => {
    const leading = m[0].length - m[0].trimStart().length;
    const value = m[0].trim();
    return { id: `s${i + 1}`, start: m.index + leading, end: m.index + leading + value.length, text: value };
  }).filter(s => s.text);
}

const tokens = text => new Set(text.toLocaleLowerCase('en').match(/[\p{L}\p{N}]+/gu) || []);
function overlap(a, b) {
  const left = tokens(a), right = tokens(b);
  const common = [...left].filter(t => right.has(t)).length;
  return common / Math.max(1, left.size + right.size - common);
}

export function compareTexts(source, output) {
  const originals = sourceSpans(source), rewrites = sourceSpans(output);
  // Bound pairwise advisory work; never silently truncate the source itself.
  if (originals.length > 300 || rewrites.length > 300) throw new Error('Review supports up to 300 sentence/line spans per text. Split this document into sections.');
  const used = new Set(), rows = [];
  for (const span of originals) {
    const exact = rewrites.find(s => !used.has(s.id) && s.text === span.text);
    if (exact) used.add(exact.id);
    rows.push({ id: `source-${span.id}`, kind: exact ? 'verbatim' : 'source-unmatched', source: span, output: exact || null, decision: 'unreviewed' });
  }
  for (const row of rows.filter(r => !r.output)) {
    let candidate = null, best = 0.2;
    for (const span of rewrites.filter(s => !used.has(s.id))) {
      const score = overlap(row.source.text, span.text);
      if (score > best) { candidate = span; best = score; }
    }
    if (candidate) {
      row.kind = 'changed-candidate';
      row.output = candidate;
      used.add(candidate.id);
    }
  }
  for (const span of rewrites.filter(s => !used.has(s.id))) {
    rows.push({ id: `output-${span.id}`, kind: 'output-unmatched', source: null, output: span, decision: 'unreviewed' });
  }
  return { method: 'literal-spans-v1', offset_unit: 'utf16-code-unit', scope: REVIEW_SCOPE, rows };
}

export function publicJwks(jwks) {
  return { keys: (Array.isArray(jwks?.keys) ? jwks.keys : []).filter(k => k && k.kty === 'OKP' && k.crv === 'Ed25519' && typeof k.x === 'string').map(k => {
    const publicKey = { kty: k.kty, crv: k.crv, x: k.x };
    for (const field of ['kid', 'alg', 'use']) if (typeof k[field] === 'string') publicKey[field] = k[field];
    return publicKey;
  }) };
}

function compareCodepoints(left, right) {
  const a = Array.from(left, c => c.codePointAt(0));
  const b = Array.from(right, c => c.codePointAt(0));
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return a.length - b.length;
}

export async function checkRenderBinding(render, output, jwks) {
  const verified = await verifyReceipt(render.receipt, jwks);
  if (!Array.isArray(render.triples) || render.triples.some(t => !Array.isArray(t) || t.length !== 3 || t.some(v => typeof v !== 'string'))) throw new Error('Malformed render triples.');
  const sorted = render.triples.map(t => [...t]).sort((a, b) => {
    // Python tuple ordering compares Unicode code points, including astral
    // characters. JavaScript's ordinary `<` instead compares UTF-16 units.
    for (let i = 0; i < 3; i++) { const order = compareCodepoints(a[i], b[i]); if (order) return order; }
    return 0;
  });
  if (await hashText(output) !== verified.payload.tome_hash) throw new Error('Output bytes do not match the signed tome hash.');
  if (await hashText(canonicalize(sorted)) !== verified.payload.triples_hash) throw new Error('Render triples do not match the signed triples hash.');
  if (canonicalize(render.sliders) !== canonicalize(verified.payload.sliders_quantized)) throw new Error('Slider settings do not match the signed receipt.');
  return { signature: 'verified', output_binding: 'verified', triples_binding: 'verified', kid: verified.kid,
    source_binding: 'not-signed', reviewer_identity: 'not-verified', signer_identity: 'not-established', revocation: 'not-checked', freshness: 'not-checked', meaning: 'not-measured' };
}

export async function makeReviewPacket({ source, output, review, render = null, jwks = null }) {
  return { schema: REVIEW_SCHEMA, created_at: new Date().toISOString(),
    source: { text: source, hash: await hashText(source) },
    output: { text: output, hash: await hashText(output) }, review: structuredClone(review),
    render: render ? structuredClone(render) : null, jwks: jwks ? publicJwks(jwks) : null,
    disclosure: REVIEW_SCOPE,
    verification_guide: 'Open this packet in the SUM workbench packet verifier, or import verifyReviewPacket from single_file_demo/review_packet.js in Node. It recomputes text hashes and review spans. When a receipt and public keys are included, it verifies Ed25519 plus exact output, selected triples and slider bindings. The container, source, key ownership, review decisions and reviewer identity are not signed. The public keys inside the packet are unsigned too, so a valid signature alone does not show who signed: pass the keys the issuing site publishes, as verifyReviewPacket(packet, { siteJwks }) in Node; without them a packet with a receipt reports key_pin not-checked. Independently establish revocation/freshness policy. No account or API key is needed. Keep the packet private if its source contains private material.' };
}

export async function verifyReviewPacket(packet, options = {}) {
  const siteJwks = options?.siteJwks;
  if (packet?.schema !== REVIEW_SCHEMA) throw new Error('Unsupported review packet schema.');
  for (const field of ['source', 'output']) {
    if (typeof packet[field]?.text !== 'string' || packet[field].text.length > MAX_REVIEW_CHARS) throw new Error(`Invalid ${field} text.`);
    if (await hashText(packet[field].text) !== packet[field].hash) throw new Error(`${field} text hash mismatch.`);
  }
  const expected = compareTexts(packet.source.text, packet.output.text);
  if (packet.review?.method !== expected.method || packet.review?.offset_unit !== expected.offset_unit || packet.review?.scope !== expected.scope) throw new Error('Unsupported review method or disclosure.');
  if (!Array.isArray(packet.review?.rows) || packet.review.rows.length !== expected.rows.length) throw new Error('Review spans do not match the supplied texts.');
  for (let i = 0; i < expected.rows.length; i++) {
    const { decision, ...row } = packet.review.rows[i];
    if (!['unreviewed', 'accepted', 'needs-change'].includes(decision) || canonicalize(row) !== canonicalize((({ decision, ...r }) => r)(expected.rows[i]))) throw new Error('Review span or decision is invalid.');
  }
  const result = { text_hashes: 'consistent-not-authenticated', review_spans: 'recomputed', human_decisions: 'unsigned', signature: 'absent', source_binding: 'not-signed', meaning: 'not-measured' };
  if (packet.render) {
    if (!packet.jwks) throw new Error('Public keys are missing for the attached render receipt.');
    if (typeof packet.jwks !== 'object' || Array.isArray(packet.jwks) || !Array.isArray(packet.jwks.keys)) throw new Error('Packet public keys must be an object with a keys array.');
    Object.assign(result, await checkRenderBinding(packet.render, packet.output.text, packet.jwks).catch(e => { throw receiptFailure(e, packet.render?.receipt); }));
    Object.assign(result, pinPacketKey(packet.jwks, siteJwks, result.kid));
  } else result.key_pin = 'not-applicable';
  return result;
}

// Key pin. The packet, including its jwks, is unsigned: anyone can make an
// Ed25519 key, sign a render receipt over their own output and carry the
// public key in the packet, and the signature then verifies. The pin compares
// the verifying key with the keys this site publishes at
// /.well-known/jwks.json. Key IDs are compared as exact strings: no trimming,
// no case folding. Key material is (kty, crv, x), and an Ed25519 x is accepted
// only in its one canonical spelling, so equal strings mean equal key bytes.
// key_pin is 'site-key', 'not-site-key', 'not-checked' (no site keys
// supplied) or 'not-applicable' (no render receipt).

// Public keys whose private keys are published, so a signature under one
// authenticates nobody: the all-zero 32-byte Ed25519 seed (finding M1) and
// the five test vectors of RFC 8032 section 7.1 (TEST 1 is the key in the
// example JWKS of docs/RENDER_RECEIPT_FORMAT.md).
export const PUBLIC_TEST_KEY_X = 'O2onvM62pC1io6jQKm8Nc2UyFXcd4kOmOsBIoYtZ2ik';
export const PUBLIC_TEST_KEYS_X = Object.freeze([PUBLIC_TEST_KEY_X,
  '11qYAYKxCrfVS_7TyWQHOg7hcvPapiMlrwIaaPcHURo', 'PUAXw-hDiVqStwqnTRt-vJyYLM8uxJaMwM1V8Sr0Zgw', '_FHNjmIYoaONpH7QAjDwWAgW7RO6MwOsXeuRFUiQgCU',
  'J4EX_BRMcjQPZ9DyMW6Dhs7_vyskKMnFH-98WX8dQm4', '7Bcrk61eVjv0kyxw4SRQNMNUZ-8u_U1k6_gZaDRn4r8']);

// A key ID in a packet is chosen by whoever made the packet, and the page
// shows it in status text. A newline, a run of spaces or a no-break, em or
// ideographic space can put text from the packet at the start of a line of
// its own (such as a forged site-key sentence), a bidi control or a
// right-to-left letter reorders the text around it, and zero-width or
// lookalike characters make another key ID look like the site's. So a key ID
// is always shown quoted, with every space and every character outside
// printable ASCII as a \uXXXX escape, so no words from the packet are shown
// with spaces between them. The quoted text is at most 40 characters long,
// escapes included, so it fits a phone-width line.
const MAX_SHOWN_KID = 40;
const unitEscape = c => '\\u' + c.charCodeAt(0).toString(16).padStart(4, '0');
export function quoteKid(kid) {
  if (typeof kid !== 'string') return '(no key ID)';
  const chars = Array.from(kid);
  let shown = '', count = 0;
  for (const c of chars) {
    const escaped = JSON.stringify(c).slice(1, -1).replace(/[^\x21-\x7e]/g, unitEscape);
    if (shown.length + escaped.length > MAX_SHOWN_KID) break;
    shown += escaped; count++;
  }
  return count < chars.length ? `"${shown}" (first ${count} of ${chars.length} characters)` : `"${shown}"`;
}

// receipt_verifier.js puts packet fields (key ID, schema, key type, header
// values) into its error messages unchanged. A receipt failure is reported
// from its error class instead, with the key ID shown by quoteKid. It is
// still a VerifyError with the same errorClass, and the verifier's error is
// kept as its cause.
const RECEIPT_FAILURES = {
  malformed_receipt: 'the render receipt is malformed',
  schema_unknown: 'the render receipt schema is not sum.render_receipt.v1',
  malformed_jwks: 'the public key for this key ID could not be imported as an Ed25519 key',
  unknown_kid: 'there is no public key for this key ID',
  malformed_jws: 'the receipt signature (JWS) is malformed',
  crit_unknown_extension: 'the signed header lists a critical extension this verifier does not support',
  unsupported_alg: 'the signed header names an unsupported signature algorithm',
  header_invariant_violated: 'the signed header does not follow the receipt format',
  kid_mismatch: 'the signed header names a different key ID',
  signature_invalid: 'signature verification failed',
};
export function receiptFailure(error, receipt) {
  if (typeof error?.errorClass !== 'string') return error;
  const errorClass = Object.hasOwn(RECEIPT_FAILURES, error.errorClass) ? error.errorClass : null;
  const kid = typeof receipt?.kid === 'string' ? ` Key ID: ${quoteKid(receipt.kid)}.` : '';
  const failure = new VerifyError(error.errorClass, `Receipt check failed: ${errorClass ? `${RECEIPT_FAILURES[errorClass]} (${errorClass})` : 'the receipt did not verify'}.${kid}`);
  return Object.assign(failure, { cause: error });
}

// An Ed25519 public key is 32 bytes: exactly 43 base64url characters, no
// padding, and the two spare low bits of the last character are zero
// (RFC 8037, RFC 7515 section 2). WebCrypto also imports other spellings of
// the same bytes (spare bits set; in Node also padding, the standard alphabet
// or whitespace), which a string comparison would treat as different keys.
const B64URL = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';
const isCanonicalEd25519X = x => typeof x === 'string' && /^[A-Za-z0-9_-]{43}$/.test(x) && B64URL.indexOf(x[42]) % 4 === 0;

function keysById(jwks, label) {
  if (!jwks || typeof jwks !== 'object' || Array.isArray(jwks) || !Array.isArray(jwks.keys)) throw new Error(`${label} must be an object with a keys array.`);
  const byKid = new Map();
  for (const k of jwks.keys) {
    // Entries without a string kid can never be selected for a receipt.
    if (!k || typeof k !== 'object' || typeof k.kid !== 'string') continue;
    if (k.kty === 'OKP' && k.crv === 'Ed25519' && !isCanonicalEd25519X(k.x)) {
      throw new Error(`${label}: the key under key ID ${quoteKid(k.kid)} is not a well-formed Ed25519 public key (x must be exactly 43 base64url characters, without padding, encoding 32 bytes). Not verified.`);
    }
    if (!byKid.has(k.kid)) byKid.set(k.kid, new Set());
    byKid.get(k.kid).add(JSON.stringify([k.kty, k.crv, k.x]));
  }
  return byKid;
}

/**
 * key_warning for the key that verified a receipt: the first entry with this
 * kid, the same selection rule as verifyReceipt.
 * @returns {{key_warning?: string}}
 */
export function keyWarning(jwks, kid) {
  const used = jwks?.keys?.find?.(k => k && typeof k === 'object' && k.kid === kid);
  return PUBLIC_TEST_KEYS_X.includes(used?.x) ? { key_warning: 'public-test-vector-key' } : {};
}

/**
 * Pin the key that verified a render receipt to this site's published keys.
 * Throws on an ambiguous packet key ID or a key-ID collision with the site.
 * @param {object} packetJwks keys carried by the packet (unsigned)
 * @param {object|null|undefined} siteJwks this site's /.well-known/jwks.json;
 *   null or undefined means not supplied, and the pin is 'not-checked'
 * @param {string} kid the verified receipt's key ID
 * @returns {{key_pin: string, key_warning?: string}}
 */
export function pinPacketKey(packetJwks, siteJwks, kid) {
  const packetKeys = keysById(packetJwks, 'Packet public keys');
  for (const [id, material] of packetKeys) {
    if (material.size > 1) throw new Error(`Ambiguous key ID: the packet carries different keys under key ID ${quoteKid(id)}. Not verified.`);
  }
  if (typeof kid !== 'string' || !packetKeys.has(kid)) throw new Error(`The packet carries no key for key ID ${quoteKid(kid)}.`);
  const warning = keyWarning(packetJwks, kid);
  if (siteJwks === undefined || siteJwks === null) return { key_pin: 'not-checked', ...warning };
  const siteKeys = keysById(siteJwks, 'Site public keys');
  for (const [id, [material]] of packetKeys) {
    if (siteKeys.has(id) && !siteKeys.get(id).has(material)) throw new Error(`Key ID collision: the packet reuses key ID ${quoteKid(id)}, which this site's public keys list for a different key. Do not rely on this packet's signature.`);
  }
  const [material] = packetKeys.get(kid);
  return { key_pin: siteKeys.get(kid)?.has(material) ? 'site-key' : 'not-site-key', ...warning };
}
