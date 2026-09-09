// The review packet is an unsigned container. Only render.receipt is signed.
// Pure functions shared by the browser and offline Node regression tests.
import { canonicalize } from './vendor/sum-verify-deps.js';
import { verifyReceipt } from './receipt_verifier.js';

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
    verification_guide: 'Open this packet in the SUM workbench packet verifier, or import verifyReviewPacket from single_file_demo/review_packet.js in Node. It recomputes text hashes and review spans. When a receipt and public keys are included, it verifies Ed25519 plus exact output, selected triples and slider bindings. The container, source, key ownership, review decisions and reviewer identity are not signed. Independently establish issuer key trust and revocation/freshness policy. No account or API key is needed. Keep the packet private if its source contains private material.' };
}

export async function verifyReviewPacket(packet) {
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
    Object.assign(result, await checkRenderBinding(packet.render, packet.output.text, packet.jwks));
  }
  return result;
}
