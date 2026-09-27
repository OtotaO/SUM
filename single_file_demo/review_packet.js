// The review packet is an unsigned container. Only render.receipt is signed.
// Pure functions shared by the browser and offline Node regression tests.
import { canonicalize } from './vendor/sum-verify-deps.js';
import { verifyReceipt } from './receipt_verifier.js';

export const REVIEW_SCHEMA = 'sum.review_packet.v1';
export const MAX_REVIEW_CHARS = 100000;
export const MAX_REVIEW_SPANS = 300;
// literal-spans-v1 and literal-spans-v2 are FROZEN: their splitters, pairing
// and scope strings must never change, because packets exported with them are
// verified by recomputing through them. Pinned packets in
// test_review_packet.mjs fail if either changes. A different rule needs a new
// method name.
export const REVIEW_SCOPE = 'Literal sentence comparison with lexical suggestions. Paraphrases, entailment, factual accuracy and meaning preservation are not measured. Human decisions and the original source are unsigned.';
export const REVIEW_SCOPE_V2 = 'Literal sentence comparison with word-level string evidence recomputed from the texts. Paraphrases, entailment, factual accuracy and meaning preservation are not measured. Human decisions and the original source are unsigned.';
export const REVIEW_METHOD_V1 = 'literal-spans-v1';
export const REVIEW_METHOD_V2 = 'literal-spans-v2';
export const REVIEW_METHOD = REVIEW_METHOD_V2; // what new comparisons and exports use

export async function hashText(text) {
  const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return 'sha256-' + Array.from(new Uint8Array(bytes), b => b.toString(16).padStart(2, '0')).join('');
}

function checkLength(text) {
  if (typeof text !== 'string' || text.length > MAX_REVIEW_CHARS) throw new Error('Text must be at most 100,000 characters.');
}

// literal-spans-v1 splitter (frozen). Known defects, kept for verification of
// existing packets: a "." not followed by whitespace makes the fallback swallow
// the rest of the line, and "Dr." ends a sentence.
export function sourceSpans(text) {
  checkLength(text);
  // Offsets count UTF-16 code units, matching textarea.setSelectionRange.
  // The complete text is retained even when sentence boundaries are ambiguous.
  return Array.from(text.matchAll(/[^.!?\n]+(?:[.!?]+(?=\s|$)|$)|[^\n]+/gu), (m, i) => {
    const leading = m[0].length - m[0].trimStart().length;
    const value = m[0].trim();
    return { id: `s${i + 1}`, start: m.index + leading, end: m.index + leading + value.length, text: value };
  }).filter(s => s.text);
}

// literal-spans-v2 splitter (frozen). A passage ends at . ! or ? (plus closing
// quotes or brackets) followed by whitespace or the end of the text, unless the
// "." is a single one after a listed abbreviation or a single letter, or the
// next non-space character is a lowercase letter or a digit. A passage also
// ends after the Chinese and Japanese marks 。！？ (plus closing quotes or
// brackets), with or without a following space. A newline always ends a
// passage, and nothing swallows the rest of a line. Linear in the text length.
export const ABBREVIATIONS = new Set(['dr', 'mr', 'mrs', 'ms', 'mx', 'prof', 'st', 'jr', 'sr', 'inc', 'ltd', 'co', 'corp', 'llc', 'plc',
  'no', 'nos', 'vs', 'v', 'etc', 'e.g', 'i.e', 'cf', 'al', 'approx', 'art', 'sec', 'para', 'fig', 'vol', 'ch', 'p', 'pp', 'ed', 'eds',
  'u.s', 'u.k', 'e.u', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec', 'mon', 'tue', 'wed',
  'thu', 'fri', 'sat', 'sun', 'est', 'dept', 'univ', 'gen', 'gov', 'rev', 'hon', 'ave', 'rd', 'blvd', 'mt', 'ft', 'oz', 'lb', 'lbs']);
const V2_END = /[.!?]+["'”’)\]]*(?=\s|$)|[。！？]+[」』）"'”’)\]]*/g;
const WORD_CP = /[\p{L}\p{N}.]/u;
export function sourceSpansV2(text) {
  checkLength(text);
  const spans = [];
  const push = (a, b) => {
    const raw = text.slice(a, b);
    const lead = raw.length - raw.trimStart().length;
    const value = raw.trim();
    if (value) spans.push({ id: `s${spans.length + 1}`, start: a + lead, end: a + lead + value.length, text: value });
  };
  let lineStart = 0;
  for (const line of text.split('\n')) {
    let start = lineStart;
    const re = new RegExp(V2_END.source, 'g');
    let m;
    while ((m = re.exec(line))) {
      const endInLine = m.index + m[0].length;
      if (!/^[.!?]/.test(m[0])) { push(start, lineStart + endInLine); start = lineStart + endInLine; continue; }
      // The word before the mark: the longest run of letters, digits and "." ending there.
      let k = m.index;
      while (k > 0) {
        const low = line.charCodeAt(k - 1), pair = k > 1 && low >= 0xdc00 && low <= 0xdfff && line.charCodeAt(k - 2) >= 0xd800 && line.charCodeAt(k - 2) <= 0xdbff;
        const ch = pair ? line.slice(k - 2, k) : line[k - 1];
        if (!WORD_CP.test(ch)) break;
        k -= ch.length;
      }
      const word = line.slice(k, m.index).toLocaleLowerCase('en');
      let f = endInLine;
      while (f < line.length && /\s/.test(line[f])) f++;
      const next = f < line.length ? line[f] : '';
      const isAbbrev = m[0] === '.' && (ABBREVIATIONS.has(word) || /^\p{L}$/u.test(word));
      const continues = next !== '' && /[\p{Ll}\p{N}]/u.test(next);
      if (isAbbrev || continues) continue;
      push(start, lineStart + endInLine);
      start = lineStart + endInLine;
    }
    push(start, lineStart + line.length);
    lineStart += line.length + 1;
  }
  return spans;
}

// Pairing words. v1 (frozen): runs of letters and digits. v2 (frozen): the
// same, except that each Chinese or Japanese character is a word of its own.
const bagV1 = text => new Set(text.toLocaleLowerCase('en').match(/[\p{L}\p{N}]+/gu) || []);
const CJK_CLASS = '\\p{Script=Han}\\p{Script=Hiragana}\\p{Script=Katakana}';
const BAG_V2 = new RegExp(`[${CJK_CLASS}]|(?:(?![${CJK_CLASS}])[\\p{L}\\p{N}])+`, 'gu');
const bagV2 = text => new Set(text.toLocaleLowerCase('en').match(BAG_V2) || []);

const METHODS = {
  [REVIEW_METHOD_V1]: { split: sourceSpans, scope: REVIEW_SCOPE, bag: bagV1 },
  [REVIEW_METHOD_V2]: { split: sourceSpansV2, scope: REVIEW_SCOPE_V2, bag: bagV2 },
};

function overlap(left, right) {
  let common = 0;
  for (const t of left) if (right.has(t)) common++;
  return common / Math.max(1, left.size + right.size - common);
}

export function compareTexts(source, output, method = REVIEW_METHOD) {
  const spec = METHODS[method];
  if (!spec) throw new Error('Unsupported review method.');
  const originals = spec.split(source), rewrites = spec.split(output);
  // Bound pairwise work; never silently truncate the source itself.
  if (originals.length > MAX_REVIEW_SPANS || rewrites.length > MAX_REVIEW_SPANS) throw new Error('Review supports up to 300 sentence or line passages per text. Split this document into sections.');
  const used = new Set(), rows = [];
  const bags = new Map();
  const bagOf = span => { if (!bags.has(span)) bags.set(span, spec.bag(span.text)); return bags.get(span); };
  for (const span of originals) {
    const exact = rewrites.find(s => !used.has(s.id) && s.text === span.text);
    if (exact) used.add(exact.id);
    rows.push({ id: `source-${span.id}`, kind: exact ? 'verbatim' : 'source-unmatched', source: span, output: exact || null, decision: 'unreviewed' });
  }
  for (const row of rows.filter(r => !r.output)) {
    let candidate = null, best = 0.2;
    for (const span of rewrites.filter(s => !used.has(s.id))) {
      const score = overlap(bagOf(row.source), bagOf(span));
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
  return { method, offset_unit: 'utf16-code-unit', scope: spec.scope, rows };
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
    disclosure: typeof review?.scope === 'string' ? review.scope : REVIEW_SCOPE,
    verification_guide: 'Open this packet in the SUM workbench packet verifier, or import verifyReviewPacket from single_file_demo/review_packet.js in Node. It recomputes text hashes and review spans. When a receipt and public keys are included, it verifies Ed25519 plus exact output, selected triples and slider bindings. The container, source, key ownership, review decisions and reviewer identity are not signed. Independently establish issuer key trust and revocation/freshness policy. No account or API key is needed. Keep the packet private if its source contains private material.' };
}

export async function verifyReviewPacket(packet) {
  if (packet?.schema !== REVIEW_SCHEMA) throw new Error('Unsupported review packet schema.');
  for (const field of ['source', 'output']) {
    if (typeof packet[field]?.text !== 'string' || packet[field].text.length > MAX_REVIEW_CHARS) throw new Error(`Invalid ${field} text.`);
    if (await hashText(packet[field].text) !== packet[field].hash) throw new Error(`${field} text hash mismatch.`);
  }
  // Recompute with the splitter the packet names, and only a known one:
  // literal-spans-v1 through the frozen v1 splitter, literal-spans-v2 through v2.
  const method = packet.review?.method;
  if (typeof method !== 'string' || !Object.hasOwn(METHODS, method)) throw new Error('Unsupported review method or disclosure.');
  const expected = compareTexts(packet.source.text, packet.output.text, method);
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
