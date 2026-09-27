// Change evidence engine: the deterministic rules behind the review page's
// notes. Ported from the design prototype's check (lease and refund examples,
// splitter cases), plus the cases that keep every note statement literally true.
import test from 'node:test';
import assert from 'node:assert/strict';
import { EXAMPLES, buildEvidence, evidenceSummary, blacklineSegments, noteStatement, noteStrings, noteSpans,
  kindLabel, KIND_LABEL, STATE_LABEL, groupByState } from './change_evidence.js';
import { compareTexts, sourceSpans, sourceSpansV2 } from './review_packet.js';

const evidence = (a, b, method) => buildEvidence(a, b, compareTexts(a, b, method)).passages;
const noteStringsOf = p => p.notes.map(n => noteStrings(n).aText || noteStrings(n).bText).join(' | ');

// One line per passage and note, in the shape of the design spec's Appendix A.
function lines(passages, source, output) {
  const out = [];
  for (const p of passages) {
    out.push(`${p.number} ${p.kind} A ${p.a ? `${p.a.id} ${p.a.start}-${p.a.end}` : '-'} | B ${p.b ? `${p.b.id} ${p.b.start}-${p.b.end}` : '-'}`);
    for (const n of p.notes) {
      const { aText, bText } = noteStrings(n);
      const spans = noteSpans(n, source, output).map(s => `${s.side.toUpperCase()} ${s.s}-${s.e}`).join(' ');
      out.push(`  ${n.letter} ${kindLabel(n)} | ${STATE_LABEL[n.state]} | ${[aText, bText].filter(Boolean).join(' -> ')} | ${spans}`);
    }
    for (const ib of p.inBoth) out.push(`  = ${KIND_LABEL[ib.kind]} ${(ib.a || ib.b).text}`);
    if (p.alsoMarked.length) out.push(`  also: ${p.alsoMarked.map(m => m.tok.t).join(', ')}`);
  }
  return out;
}

const LEASE = [
  '1 changed-candidate A s1 0-47 | B s1 0-27',
  '  a Modal verb | Differs | may -> can | A 6-9 B 6-9',
  '  b Duration | Original only | 30 days | A 32-39',
  '  c Wording | Original only | notice | A 40-46',
  '  = Name Alice',
  '  also: with',
  '2 changed-candidate A s2 48-97 | B s2 28-54',
  '  a Exception | Original only | unless rent is overdue | A 74-96',
];

const REFUND = [
  '1 changed-candidate A s1 0-81 | B s1 0-63',
  '  a Wording | Original only | Customers | A 0-9',
  '  b Modal verb | Differs | may -> can | A 10-13 B 4-7',
  '  c Duration | Differs | 30 days -> 60 days | A 43-50 B 37-44',
  '  d Wording | Original only | delivery | A 54-62',
  '  also: You, of',
  '2 changed-candidate A s2 82-141 | B s2 64-115',
  '  a Negation | Original only | not | A 99-102',
  '  b Condition or exception | Differs | unless they arrive damaged -> if they arrive damaged | A 114-140 B 92-114',
  '3 changed-candidate A s3 142-216 | B s3 116-180',
  '  a Wording | Differs | issued -> usually go back | A 154-160 B 124-139',
  '  b Wording | Differs | original payment method -> card | A 168-191 B 148-152',
  '  c Duration | Differs | 10 business days -> a few business days | A 199-215 B 160-179',
  '  also: are, the, your',
  '4 changed-candidate A s4 217-302 | B s4 181-212',
  '  a Number | Original only | $8 | A 234-236',
  '  b Exception | Original only | except where the return is caused by our error | A 255-301',
  '  = Negation not',
  '  also: of',
  '5 changed-candidate A s5 303-365 | B s5 213-267',
  '  a Date | Differs | March 1, 2026 -> March 2026 | A 324-337 B 234-244',
  '  b Wording | Differs | previous -> old | A 349-357 B 256-259',
  '5.1 output-unmatched A - | B s6 268-311',
  '  a Name | Other passage | Northwind | B 301-310 A 366-375',
  '6 source-unmatched A s6 366-421 | B -',
  '  a Name | Original only | Northwind Outfitters | A 366-386',
  '  b Modal verb | Original only | must | A 387-391',
  '  c Number | Original only | over $200 | A 411-420',
];

test('lease example: the worked evidence, identical under both splitters', () => {
  const { source, output } = EXAMPLES.lease;
  for (const method of ['literal-spans-v1', 'literal-spans-v2']) {
    const passages = evidence(source, output, method);
    assert.deepEqual(lines(passages, source, output), LEASE, method);
    const summary = evidenceSummary(passages);
    assert.equal(summary.passages, 2);
    assert.equal(summary.notes, 4);
  }
  const [p1, p2] = evidence(source, output);
  assert.equal(p1.notes.map(n => n.letter).join(''), 'abc');
  assert.equal(noteStatement(p1.notes[0], p1), '“may” in the original, “can” in the rewrite.');
  assert.equal(noteStatement(p1.notes[1], p1), 'Appears in the original, nowhere in the rewrite.');
  assert.equal(noteStatement(p2.notes[0], p2), 'The marker “unless” and the words after it appear in the original, nowhere in the rewrite.');
  for (const p of [p1, p2]) for (const n of p.notes) {
    for (const s of noteSpans(n, source, output)) {
      assert.equal((s.side === 'a' ? source : output).slice(s.s, s.e), s.text, 'every span points at its exact characters');
    }
  }
});

test('refund example: the worked evidence, identical under both splitters', () => {
  const { source, output } = EXAMPLES.refund;
  for (const method of ['literal-spans-v1', 'literal-spans-v2']) {
    assert.deepEqual(lines(evidence(source, output, method), source, output), REFUND, method);
  }
  const passages = evidence(source, output);
  const summary = evidenceSummary(passages);
  assert.equal(summary.passages, 7);
  assert.equal(summary.notes, 17);
  const groups = groupByState(passages);
  assert.deepEqual(Object.fromEntries(Object.entries(groups).map(([k, v]) => [k, v.length])),
    { 'a-only': 8, differs: 8, 'b-only': 0, moved: 1, both: 1 });
  const p2 = passages.find(p => p.number === '2'), p51 = passages.find(p => p.number === '5.1'), p6 = passages.find(p => p.number === '6');
  assert.equal(noteStatement(p2.notes[0], p2), 'In the original passage, not in the paired rewrite passage.');
  assert.equal(noteStatement(p51.notes[0], p51), 'This passage has no partner in the original; the string appears in original passage 6.');
  assert.equal(noteStatement(p6.notes[1], p6), 'In an original passage that has no partner in the rewrite.');
  // Display order puts the rewrite-only passage after passage 5; review.rows keeps compareTexts order.
  assert.deepEqual(passages.map(p => p.number), ['1', '2', '3', '4', '5', '5.1', '6']);
});

test('splitters: v2 fixes the measured v1 defects and agrees elsewhere', () => {
  const cases = [
    ['Shipping costs $7.95 per order. Refunds follow.', 1, 2],
    ['Call Dr. Patel if a rash appears. Stop the tablets.', 3, 2],
    ['Bring ID, e.g. a passport. Then sign.', 1, 2],
    ['Take 3.5 mg daily. Stop if dizzy.', 1, 2],
    ['Made in the U.S. by Acme.', 1, 1],
    ['First line\nSecond line', 2, 2],
  ];
  for (const [text, v1, v2] of cases) {
    assert.equal(sourceSpans(text).length, v1, `v1: ${text}`);
    assert.equal(sourceSpansV2(text).length, v2, `v2: ${text}`);
    for (const span of sourceSpansV2(text)) assert.equal(text.slice(span.start, span.end), span.text);
  }
  for (const ex of Object.values(EXAMPLES)) {
    for (const text of [ex.source, ex.output]) assert.deepEqual(sourceSpans(text), sourceSpansV2(text), 'examples split identically');
  }
});

test('a reorder within a passage is listed as In both, never as missing', () => {
  const a = 'Customers may return items within 30 days.', b = 'Within 30 days, customers may return items.';
  const [p] = evidence(a, b);
  assert.equal(p.notes.length, 0);
  assert.deepEqual(p.inBoth.map(ib => (ib.a || ib.b).text.toLowerCase()).sort(), ['30 days', 'may', 'within']);
  const within = p.inBoth.find(ib => ib.kind === 'wording');
  assert.ok(within.a && within.b, 'both occurrences sit on one In both line');
});

test('a string that is in the paired passage fewer times is stated with both counts', () => {
  const [p] = evidence('Do not smoke, not ever, not here.', 'Do not smoke, not here.');
  const negation = p.notes.find(n => n.kind === 'negation');
  assert.equal(noteStatement(negation, p), 'Appears 3 times in the original passage and twice in the paired rewrite passage.');
  const [q] = evidence('Notice notice given.', 'Notice given.');
  assert.equal(noteStatement(q.notes[0], q), 'Appears twice in the original passage and once in the paired rewrite passage.');
});

test('"no literal differences" is claimed only when every word matches in order', () => {
  assert.equal(evidenceSummary(evidence('Alice may cancel.', 'Alice may cancel.')).noDifferences, true);
  assert.equal(evidenceSummary(evidence('The fee is DUE.', 'the fee is due')).noDifferences, true, 'case and punctuation are not checked');
  const commonOnly = evidenceSummary(evidence('Pay the fee.', 'Pay a fee.'));
  assert.equal(commonOnly.notes, 0);
  assert.equal(commonOnly.noDifferences, false, 'a changed common word is still a difference');
  const dropped = evidence('Rent is due monthly. The tenant pays for water.', 'Rent is due monthly.');
  const summary = evidenceSummary(dropped);
  assert.equal(summary.noDifferences, false);
  assert.ok(summary.notes >= 1, 'an unpartnered passage always carries a note');
  assert.equal(noteStringsOf(dropped[1]), 'tenant pays for water');
});

test('the blackline never marks a space and never runs two runs together', () => {
  for (const ex of Object.values(EXAMPLES)) {
    for (const p of evidence(ex.source, ex.output)) {
      for (const seg of blacklineSegments(p, ex.source, ex.output)) {
        for (const part of seg.t === 'run' ? seg.parts : []) {
          if (part.t !== 'mark') continue;
          assert.equal(part.text, part.text.trim(), `mark "${part.text}" has no leading or trailing space`);
          assert.doesNotMatch(part.text, /^\s|\s$/);
        }
      }
    }
  }
  // An insertion straight after a deletion at the passage start gets one space ("mayYou" never happens).
  const [a, b] = ['May we go now.', 'You can go now.'];
  const pair = evidence(a, b)[0];
  assert.equal(pair.kind, 'changed-candidate');
  const segs = blacklineSegments(pair, a, b);
  const insertion = segs.find(s => s.t === 'run' && s.side === 'i');
  assert.deepEqual(insertion.parts[0], { t: 'gap', text: ' ' });
  // Groups of at most 30 characters are kept on one line.
  const lease = EXAMPLES.lease;
  const marks = blacklineSegments(evidence(lease.source, lease.output)[0], lease.source, lease.output)
    .flatMap(s => (s.t === 'run' ? s.parts : [])).filter(m => m.t === 'mark').map(m => m.text);
  assert.ok(marks.includes('30 days'));
});

test('verbatim passages render as the plain passage text', () => {
  const text = '<img src=x onerror="alert(1)"> Alice may cancel.';
  const [p] = evidence(text, text);
  assert.equal(p.kind, 'verbatim');
  assert.deepEqual(blacklineSegments(p, text, text), [{ t: 'plain', text }]);
});

test('evidence comes from the review rows and is deterministic', () => {
  const { source, output } = EXAMPLES.refund;
  const review = compareTexts(source, output);
  const before = JSON.stringify(review);
  const first = lines(buildEvidence(source, output, review).passages, source, output);
  const second = lines(buildEvidence(source, output, structuredClone(review)).passages, source, output);
  assert.deepEqual(first, second);
  assert.equal(JSON.stringify(review), before, 'the review rows are not modified');
});
