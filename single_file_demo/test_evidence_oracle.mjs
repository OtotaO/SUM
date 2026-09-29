// The independent oracle (evidence_oracle.mjs) run over the built-in
// examples, an adversarial corpus and a seeded random mutation fuzz: every
// supported sentence form is checked against the raw texts in these cases.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as E from './change_evidence.js';
import { compareTexts } from './review_packet.js';
import { check, ADVERSARIAL, CORPUS_BASES, mutate, mulberry32, charsMatch, countWords } from './evidence_oracle.mjs';

test('oracle: the built-in examples print only true statements', () => {
  for (const ex of Object.values(E.EXAMPLES)) {
    for (const method of ['literal-spans-v1', 'literal-spans-v2']) assert.deepEqual(check(ex.source, ex.output, method), [], method);
  }
});

test('oracle: every adversarial pair prints only true statements', () => {
  for (const [a, b] of ADVERSARIAL) {
    assert.deepEqual(check(a, b), [], `${JSON.stringify(a)} -> ${JSON.stringify(b)}`);
    assert.deepEqual(check(b, a), [], `${JSON.stringify(b)} -> ${JSON.stringify(a)}`);
  }
});

test('oracle: a seeded mutation fuzz finds no false statement', () => {
  const rnd = mulberry32(20260927);
  const failures = [];
  let cases = 0;
  for (let n = 0; n < 3000; n++) {
    const base = CORPUS_BASES[Math.floor(rnd() * CORPUS_BASES.length)];
    const a = rnd() < 0.3 ? mutate(base, rnd) : base;
    const b = mutate(base, rnd);
    cases++;
    const bad = check(a, b);
    if (bad.length) failures.push({ a, b, bad: bad.slice(0, 3) });
    if (failures.length >= 5) break;
  }
  assert.deepEqual(failures, [], `${failures.length} of ${cases} fuzz cases printed a false statement`);
});

test('oracle: planted false counts, scopes, locations, quotes and headings are rejected', () => {
  const badCount = check('Do not smoke, not ever, not here.', 'Do not smoke, not here.', undefined, e => {
    const n = e.passages.flatMap(p => p.notes).find(n => n.scope === 'count');
    n.counts = [999, 888];
  });
  assert.ok(badCount.some(s => s.startsWith('COUNT:')));
  const badScope = check('Bob signs. Alice pays.', 'Alice pays Bob.', undefined, e => {
    const n = e.passages.flatMap(p => p.notes).find(n => n.state === 'other');
    assert.ok(n, 'raw fixture has an elsewhere match');
    n.state = n.a ? 'a-only' : 'b-only'; n.scope = 'nowhere';
  });
  assert.ok(badScope.some(s => s.startsWith('FALSE-NOWHERE:')));
  const badLocation = check('Bob signs. Alice pays.', 'Alice pays Bob.', undefined, e => {
    const n = e.passages.flatMap(p => p.notes).find(n => n.state === 'other');
    n.where.passage = 's999';
  });
  assert.ok(badLocation.some(s => s.startsWith('OTHER-THERE:')));
  const badQuote = check('Do not sign.', 'Do NOT sign.', undefined, e => {
    e.passages[0].inBoth[0].a.text = 'invented';
  });
  assert.ok(badQuote.some(s => s.startsWith('IN-BOTH-FORMS:')));
  const badHeading = check('A b.', 'A  b.', undefined, e => { e.summary.identicalTexts = true; });
  assert.ok(badHeading.some(s => s.startsWith('HEADING-IDENTICAL:')));
  assert.ok(!charsMatch('a space', '  '));
  assert.ok(charsMatch('code point U+200B', '\u200b'));
  assert.equal(countWords("Don't DON’T cannot", "don't"), 2);
});

test('oracle: summary counts, edit bounds and invented quotations are independently rejected', () => {
  const a = 'Alice may cancel.', b = 'Alice can cancel.';
  for (const field of ['passages', 'alsoMarked']) {
    const bad = check(a, b, undefined, e => { e.summary[field] = 999; });
    assert.ok(bad.some(s => s.startsWith('SUMMARY-' + field.toUpperCase() + ':')), field);
  }
  const badBound = check(a, b, undefined, e => {
    e.passages[0].over = { limit: 999999 };
    e.summary.over = [{ number: e.passages[0].number, limit: 999999 }];
  });
  assert.ok(badBound.some(s => s.startsWith('OVER:')));
  const badQuote = check('Alice must pay $30 today.', 'Alice must pay today.', undefined, e => {
    e.passages.flatMap(p => p.notes).find(n => n.scope === 'nowhere').a.text = 'invented quotation';
  });
  assert.ok(badQuote.some(s => s.startsWith('OWN-TEXT:')));
});

test('diff: Myers and the table agree on the length of the common subsequence', () => {
  const rnd = mulberry32(7);
  const vocab = ['a', 'b', 'c', 'd', 'e'];
  for (let n = 0; n < 400; n++) {
    const mk = () => Array.from({ length: Math.floor(rnd() * 60) }, () => ({ k: vocab[Math.floor(rnd() * vocab.length)] }));
    const a = mk(), b = mk();
    const table = E.diffTokens(a, b).ops;
    const eq = ops => ops.filter(o => o.op === 'eq').length;
    // Force the Myers path on the same input by padding far past the table limit, then compare the core.
    const pad = Array.from({ length: 600 }, (_, i) => ({ k: `p${i}` }));
    const big = E.diffTokens([...pad, ...a], [...pad, ...b]);
    assert.ok(big.ops, 'within the budget');
    assert.equal(eq(big.ops) - pad.length, eq(table), 'same LCS length');
    // Ops are valid: every token appears once, in order, and eq pairs have equal keys.
    const as = big.ops.filter(o => o.a !== undefined).map(o => o.a), bs = big.ops.filter(o => o.b !== undefined).map(o => o.b);
    assert.deepEqual(as, [...as.keys()]);
    assert.deepEqual(bs, [...bs.keys()]);
    for (const o of big.ops) if (o.op === 'eq') assert.equal([...pad, ...a][o.a].k, [...pad, ...b][o.b].k);
  }
});

test('diff: a pair past the budget is shown whole, with no notes and a plain message', () => {
  const words = Array.from({ length: 6000 }, (_, i) => `w${i}`).join(' ');
  const reversed = words.split(' ').reverse().join(' ');
  const { passages, summary } = E.buildEvidence(words, reversed, compareTexts(words, reversed));
  assert.ok(passages[0].over);
  assert.equal(passages[0].notes.length, 0);
  assert.match(E.passageMessage(passages[0]), /more than [\d,]+ word insertions and deletions/);
  assert.match(E.summaryFacts(summary).join(' '), /Passage 1: more than/);
  assert.equal(E.headingText(summary), '1 passage compared; the texts are not identical.');
  assert.deepEqual(check(words, reversed), []);
});
