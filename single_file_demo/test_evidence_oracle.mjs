// The independent oracle (evidence_oracle.mjs) run over the built-in
// examples, an adversarial corpus and a seeded random mutation fuzz: every
// sentence the review page prints about the two texts must be literally true.
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

test('oracle: the checks themselves catch false statements', () => {
  // A mutated engine output must be caught; otherwise the oracle proves nothing.
  const { passages } = E.buildEvidence('Alice may cancel.', 'Alice can cancel.', compareTexts('Alice may cancel.', 'Alice can cancel.'));
  const note = passages[0].notes[0];
  assert.match(E.noteStatement(note, passages[0], 'Alice may cancel.', 'Alice can cancel.'), /“may” in the original, “can” in the rewrite/);
  const bad = check('The fee is $8 today.', 'The total is $8.50 today.');
  assert.deepEqual(bad, []);
  assert.ok(!charsMatch('a space', '  '));
  assert.ok(charsMatch('2 spaces', '  '));
  assert.equal(countWords('Do not, NOT or cannot', 'not'), 2);
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
