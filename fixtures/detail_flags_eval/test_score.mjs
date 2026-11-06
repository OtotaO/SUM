// Run: node --test fixtures/detail_flags_eval/test_score.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { score } from './score.mjs';

const corpus = [
  { id: 'a1', label: 'altered', original: 'Refunds within 14 days.', rewrite: 'Refunds within 30 days.', change: { category: 'number', type: 'changed', source_detail: '14 days', rewrite_detail: '30 days' } },
  { id: 'a2', label: 'altered', original: 'You may not smoke unless outside.', rewrite: 'You may not smoke.', change: { category: 'exception', type: 'dropped', source_detail: 'unless', rewrite_detail: null } },
  { id: 'a3', label: 'altered', original: 'Call us.', rewrite: 'Call us by Friday.', change: { category: 'date', type: 'added', source_detail: null, rewrite_detail: 'Friday' } },
  { id: 'p1', label: 'preserving', original: 'Pay $5.', rewrite: 'You pay $5.', change: null },
  { id: 'p2', label: 'preserving', original: 'Pay $6.', rewrite: 'You pay $6.', change: null },
];
const good = {
  a1: [{ category: 'number', kind: 'changed', source: { start: 15, end: 22, text: '14 days' }, output: null }],
  a2: [{ category: 'condition', kind: 'missing', source: { start: 0, end: 33 }, output: null }],
  a3: [{ category: 'date', kind: 'added', source: null, output: { start: 0, end: 5 } }],
  p1: [{ category: 'number', kind: 'changed', source: { start: 4, end: 6 }, output: null }],
  p2: [],
};

test('credit needs an overlapping span; category is secondary; any flag on a preserving pair is an alarm', () => {
  const r = score(corpus, good);
  assert.equal(r.recall.hit, 2);
  assert.equal(r.category_matched_recall.hit, 1);
  assert.equal(r.by_type.added.hit, 0);
  assert.equal(r.false_alarm.flagged, 1);
  assert.equal(r.decision, 'ship-flags');
});

test('a span longer than 80 code units earns no credit', () => {
  const original = 'x'.repeat(90) + ' 14 days.';
  const c = [{ id: 'b', label: 'altered', original, rewrite: 'y', change: { category: 'number', type: 'dropped', source_detail: '14 days', rewrite_detail: null } }, corpus[3]];
  const r = score(c, { b: [{ category: 'number', kind: 'missing', source: { start: 0, end: original.length }, output: null }], p1: [] });
  assert.equal(r.recall.hit, 0);
  assert.equal(r.decision, 'neutral-detail-table');
});

test('malformed input is refused, never scored as a miss', () => {
  const span = (start, end) => ({ category: 'number', kind: 'changed', source: { start, end }, output: null });
  const bad = [
    { ...good, a1: [{ category: 'number', kind: 'changed', source: null, rewrite: { start: 15, end: 22 } }] },
    { ...good, zz: [] },
    { a1: good.a1, a3: good.a3, p1: [], p2: [] },
    { ...good, a1: [{ category: 'amount', kind: 'changed', source: { start: 1, end: 2 }, output: null }] },
    { ...good, a1: [{ category: 'number', kind: 'changed', source: null, output: null }] },
    { ...good, a1: [span(3, 1)] },
    { ...good, a1: [span(16, 16)] },
    { ...good, a1: [span(100, 101)] },
  ];
  for (const flags of bad) assert.throws(() => score(corpus, flags));
  const dup = [{ id: 'c', label: 'altered', original: 'a a', rewrite: 'b', change: { category: 'name', type: 'dropped', source_detail: 'a', rewrite_detail: null } }, corpus[3]];
  assert.throws(() => score(dup, { c: [], p1: [] }), /exactly once/);
});
