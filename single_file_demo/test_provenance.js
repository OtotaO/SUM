/**
 * Conformance tests for single_file_demo/provenance.js.
 *
 * Mirrors the invariant-check portion of Tests/test_provenance_m1.py.
 * Cross-runtime prov_id byte-identity is covered separately by
 * scripts/verify_prov_id_cross_runtime.py; this file is the JS-local
 * guard.
 *
 * Run: node single_file_demo/test_provenance.js
 * Exit 0 all-pass, 1 any failure.
 */
'use strict';

const {
  provenanceRecord,
  validateSourceUri,
  computeProvId,
  sha256UriForText,
  PROVENANCE_SCHEMA_VERSION,
  EXCERPT_MAX_CHARS,
} = require('./provenance');

let passed = 0;
let failed = 0;

function ok(label) { passed++; }
function bad(label, detail) {
  failed++;
  console.error(`FAIL ${label}: ${detail}`);
}

function expectThrow(label, fn, expectedMatch) {
  try {
    fn();
    bad(label, 'expected throw');
  } catch (e) {
    if (!expectedMatch || e.message.includes(expectedMatch)) {
      ok(label);
    } else {
      bad(label, `wrong error: ${e.message}`);
    }
  }
}

function expectEq(label, actual, expected) {
  if (actual === expected) ok(label);
  else bad(label, `expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
}

// ─── ProvenanceRecord invariants ──────────────────────────────────

const validSha = 'sha256:' + 'a'.repeat(64);

const rec = provenanceRecord({
  source_uri: validSha,
  byte_start: 0,
  byte_end: 17,
  extractor_id: 'sum.test',
  timestamp: '2026-04-19T00:00:00+00:00',
  text_excerpt: 'Alice likes cats.',
});
expectEq('happy path schema_version', rec.schema_version, PROVENANCE_SCHEMA_VERSION);

expectThrow('byte_end reversed', () => provenanceRecord({
  source_uri: validSha, byte_start: 10, byte_end: 5,
  extractor_id: 'sum.test',
  timestamp: '2026-04-19T00:00:00+00:00',
  text_excerpt: 'x',
}), 'byte_end');

expectThrow('empty range', () => provenanceRecord({
  source_uri: validSha, byte_start: 0, byte_end: 0,
  extractor_id: 'sum.test',
  timestamp: '2026-04-19T00:00:00+00:00',
  text_excerpt: '',
}), 'byte_end');

expectThrow('empty extractor', () => provenanceRecord({
  source_uri: validSha, byte_start: 0, byte_end: 1,
  extractor_id: '',
  timestamp: '2026-04-19T00:00:00+00:00',
  text_excerpt: 'x',
}), 'extractor_id');

expectThrow('excerpt over limit', () => provenanceRecord({
  source_uri: validSha, byte_start: 0, byte_end: 1,
  extractor_id: 'sum.test',
  timestamp: '2026-04-19T00:00:00+00:00',
  text_excerpt: 'x'.repeat(EXCERPT_MAX_CHARS + 1),
}), 'text_excerpt');

// ─── Source URI validation ────────────────────────────────────────

validateSourceUri('sha256:' + '0'.repeat(64)); ok('sha256 accepted');
expectThrow('sha256 short', () => validateSourceUri('sha256:abc'), '64 lowercase hex');
expectThrow('sha256 uppercase', () => validateSourceUri('sha256:' + 'A'.repeat(64)), '64 lowercase hex');
validateSourceUri('doi:10.1234/xyz'); ok('doi accepted');
validateSourceUri('https://example.org/paper'); ok('https accepted');
validateSourceUri('urn:sum:source:seed_v1'); ok('urn:sum:source accepted');
expectThrow('bare http rejected', () => validateSourceUri('http://example.com/page'), 'not supported');
expectThrow('file rejected', () => validateSourceUri('file:///tmp/x.txt'), 'not supported');

// ─── prov_id content-addressability ───────────────────────────────

function _rec(overrides = {}) {
  return provenanceRecord({
    source_uri: 'sha256:' + 'b'.repeat(64),
    byte_start: 0,
    byte_end: 17,
    extractor_id: 'sum.test',
    timestamp: '2026-04-19T00:00:00+00:00',
    text_excerpt: 'Alice likes cats.',
    ...overrides,
  });
}

expectEq('identical records same prov_id', computeProvId(_rec()), computeProvId(_rec()));

if (computeProvId(_rec()) !== computeProvId(_rec({ byte_end: 18 }))) ok('different byte range different id');
else bad('different byte range different id', 'ids matched');

if (computeProvId(_rec()) !== computeProvId(_rec({ extractor_id: 'sum.other' }))) ok('different extractor different id');
else bad('different extractor different id', 'ids matched');

const pid = computeProvId(_rec());
if (pid.startsWith('prov:') && pid.length === 'prov:'.length + 32) ok('prov_id format');
else bad('prov_id format', `got ${pid}`);

// ─── sha256_uri_for_text helper ───────────────────────────────────

expectEq('same text same uri', sha256UriForText('hello'), sha256UriForText('hello'));
if (sha256UriForText('hello') !== sha256UriForText('world')) ok('different text different uri');
else bad('different text different uri', 'collision');

// Known-value: sha256('hello') = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
expectEq('known hash', sha256UriForText('hello'),
  'sha256:2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824');

// ─── Summary ──────────────────────────────────────────────────────

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
