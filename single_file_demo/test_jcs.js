/**
 * Conformance tests for single_file_demo/jcs.js.
 *
 * Same test vectors as Tests/test_jcs.py — if both pass, the two
 * implementations produce byte-identical canonical output for every
 * input shape SUM actually emits. Run with:
 *     node single_file_demo/test_jcs.js
 *
 * Exit 0 on all-pass. Exit 1 on any failure, with the first failure
 * printed to stderr for debugging.
 *
 * Zero dependencies. Runs on any Node ≥ 14.
 */
'use strict';

const { canonicalize, canonicalizeToStr } = require('./jcs');

let failed = 0;
let passed = 0;

function check(label, actual, expected) {
  const ok = actual === expected;
  if (ok) {
    passed++;
  } else {
    failed++;
    console.error(`FAIL ${label}`);
    console.error(`  expected: ${JSON.stringify(expected)}`);
    console.error(`  actual:   ${JSON.stringify(actual)}`);
  }
}

function checkThrows(label, fn, expectedMatch) {
  try {
    fn();
    failed++;
    console.error(`FAIL ${label} — expected throw`);
  } catch (e) {
    if (!expectedMatch || e.message.includes(expectedMatch)) {
      passed++;
    } else {
      failed++;
      console.error(`FAIL ${label} — threw ${JSON.stringify(e.message)} (expected to contain ${JSON.stringify(expectedMatch)})`);
    }
  }
}

// ─── Primitives ───────────────────────────────────────────────────

check('null',                canonicalizeToStr(null),   'null');
check('true',                canonicalizeToStr(true),   'true');
check('false',               canonicalizeToStr(false),  'false');
check('integer 42',          canonicalizeToStr(42),     '42');
check('integer -7',          canonicalizeToStr(-7),     '-7');
check('integer 0',           canonicalizeToStr(0),      '0');
check('large int (bigint)',  canonicalizeToStr(2n ** 63n), '9223372036854775808');
checkThrows('float rejected', () => canonicalize(1.5), 'floating-point');
checkThrows('NaN rejected',   () => canonicalize(NaN), 'non-finite');
checkThrows('Infinity rejected', () => canonicalize(Infinity), 'non-finite');

// ─── Strings ──────────────────────────────────────────────────────

check('empty string',            canonicalizeToStr(''),        '""');
check('ascii',                   canonicalizeToStr('hello'),   '"hello"');
check('quote escape',            canonicalizeToStr('a"b'),     '"a\\"b"');
check('backslash escape',        canonicalizeToStr('a\\b'),    '"a\\\\b"');
check('solidus not escaped',     canonicalizeToStr('a/b'),     '"a/b"');
check('short control escapes',   canonicalizeToStr('\b\t\n\f\r'), '"\\b\\t\\n\\f\\r"');
check('\\u0001 lowercase hex',   canonicalizeToStr('\u0001'),  '"\\u0001"');
check('unicode cafe passthrough',canonicalizeToStr('café'),    '"café"');

// ─── Objects ──────────────────────────────────────────────────────

check('empty object',            canonicalizeToStr({}),        '{}');
check('single pair',             canonicalizeToStr({a: 1}),    '{"a":1}');

// Insertion {b:1, a:2} must emit a-first.
check(
  'keys sorted',
  canonicalizeToStr({b: 1, a: 2}),
  '{"a":2,"b":1}'
);
check(
  'nested',
  canonicalizeToStr({outer: {b: 1, a: 2}}),
  '{"outer":{"a":2,"b":1}}'
);

// Non-string keys don't exist in JS plain objects the same way, but
// symbol keys are not iterated by Object.keys, so this is implicitly fine.

// Supplementary-character sort: U+1F600 (😀) encodes as surrogate pair
// 0xD83D 0xDE00. 0xD83D < 0xE000 (BMP private-use area), so the
// supplementary character sorts FIRST in UTF-16 code-unit order.
// Default JS string sort produces this ordering without a compareFn.
{
  const bmp = '\uE000';
  const supp = '\uD83D\uDE00'; // "😀" as explicit surrogate pair
  const out = canonicalizeToStr({[bmp]: 1, [supp]: 2});
  check(
    'utf16 supplementary sort (supp first)',
    out.startsWith('{"' + supp + '"'),
    true
  );
}

// ─── Arrays ───────────────────────────────────────────────────────

check('empty array',             canonicalizeToStr([]),         '[]');
check('array ints',              canonicalizeToStr([1,2,3]),    '[1,2,3]');
check('array mixed',             canonicalizeToStr([1, 'a', true, null]), '[1,"a",true,null]');
check(
  'array of objects preserves position',
  canonicalizeToStr([{b:1}, {a:2}]),
  '[{"b":1},{"a":2}]'
);

// ─── Integration ──────────────────────────────────────────────────

{
  const obj = {z: [1, 2], a: {b: true, a: null}};
  const once = canonicalizeToStr(obj);
  const twice = canonicalizeToStr(JSON.parse(once));
  check('idempotent after JSON.parse roundtrip', once, twice);
}

checkThrows('unsupported type (function)', () => canonicalize(() => {}), 'unsupported type');

{
  const obj = {msg: 'café'};
  const raw = canonicalize(obj);
  // Uint8Array -> string via TextDecoder
  const decoded = new TextDecoder('utf-8').decode(raw);
  check('utf8 bytes output shape', decoded, '{"msg":"café"}');
}

// ─── Summary ──────────────────────────────────────────────────────

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
