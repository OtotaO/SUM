// Change evidence engine: the deterministic rules behind the review page's
// notes. The worked examples (lease and refund), the splitter cases, and the
// cases that keep every statement literally true. test_evidence_oracle.mjs
// checks the same statements independently over a fuzzed corpus.
import test from 'node:test';
import assert from 'node:assert/strict';
import { EXAMPLES, buildEvidence, blacklineSegments, viewText, noteStatement, noteStrings, noteSpans, kindLabel,
  inBothText, inBothKindLabel, headingText, summaryFacts, passageMessage, STATE_LABEL, groupByState, tokenize } from './change_evidence.js';
import { compareTexts, sourceSpans, sourceSpansV2 } from './review_packet.js';

const evidence = (a, b, method) => buildEvidence(a, b, compareTexts(a, b, method));
const statements = (a, b) => evidence(a, b).passages.flatMap(p => p.notes.map(n => noteStatement(n, p, a, b)));

// One line per passage and note, in the shape of the design spec's Appendix A.
function lines(passages, source, output) {
  const out = [];
  for (const p of passages) {
    out.push(`${p.number} ${p.kind} A ${p.a ? `${p.a.id} ${p.a.start}-${p.a.end}` : '-'} | B ${p.b ? `${p.b.id} ${p.b.start}-${p.b.end}` : '-'}`);
    for (const n of p.notes) {
      const { aText, bText } = noteStrings(n, source, output);
      const spans = noteSpans(n, source, output).map(s => `${s.side.toUpperCase()} ${s.s}-${s.e}`).join(' ');
      out.push(`  ${n.letter} ${kindLabel(n)} | ${STATE_LABEL[n.state]} | ${[aText, bText].filter(Boolean).join(' -> ')} | ${spans}`);
    }
    for (const ib of p.inBoth) out.push(`  = ${inBothKindLabel(ib)} ${inBothText(ib)}`);
    if (p.alsoMarked.length) out.push(`  also: ${p.alsoMarked.map(m => m.tok.t).join(', ')}`);
  }
  return out;
}

// Appendix A of the design spec, with the state names that are true by
// position (Removed, Added) and the honest kind label (Capitalized word).
const LEASE = [
  '1 changed-candidate A s1 0-47 | B s1 0-27',
  '  a Modal-list word | Differs | may -> can | A 6-9 B 6-9',
  '  b Duration | Removed | 30 days | A 32-39',
  '  c Wording | Removed | notice | A 40-46',
  '  = Capitalized word “Alice”',
  '  also: with',
  '2 changed-candidate A s2 48-97 | B s2 28-54',
  '  a Exception-marker phrase | Removed | unless rent is overdue | A 74-96',
];

const REFUND = [
  '1 changed-candidate A s1 0-81 | B s1 0-63',
  '  a Wording | Removed | Customers | A 0-9',
  '  b Modal-list word | Differs | may -> can | A 10-13 B 4-7',
  '  c Duration | Differs | 30 days -> 60 days | A 43-50 B 37-44',
  '  d Wording | Removed | delivery | A 54-62',
  '  also: You, of',
  '2 changed-candidate A s2 82-141 | B s2 64-115',
  '  a Negation-list word | Removed | not | A 99-102',
  '  b Marker phrase | Differs | unless they arrive damaged -> if they arrive damaged | A 114-140 B 92-114',
  '3 changed-candidate A s3 142-216 | B s3 116-180',
  '  a Wording | Differs | issued -> usually go back | A 154-160 B 124-139',
  '  b Wording | Differs | original payment method -> card | A 168-191 B 148-152',
  '  c Duration | Differs | 10 business days -> a few business days | A 199-215 B 160-179',
  '  also: are, the, your',
  '4 changed-candidate A s4 217-302 | B s4 181-212',
  '  a Number | Removed | $8 | A 234-236',
  '  b Exception-marker phrase | Removed | except where the return is caused by our error | A 255-301',
  '  = Negation-list word “not”',
  '  also: of',
  '5 changed-candidate A s5 303-365 | B s5 213-267',
  '  a Date | Differs | March 1, 2026 -> March 2026 | A 324-337 B 234-244',
  '  b Wording | Differs | previous -> old | A 349-357 B 256-259',
  '5.1 output-unmatched A - | B s6 268-311',
  '  a Capitalized word | Other passage | Northwind | B 301-310 A 366-375',
  '6 source-unmatched A s6 366-421 | B -',
  '  a Capitalized words | Removed | Northwind Outfitters | A 366-386',
  '  b Modal-list word | Removed | must | A 387-391',
  '  c Number | Removed | over $200 | A 411-420',
];

test('lease example: the worked evidence, identical under both splitters', () => {
  const { source, output } = EXAMPLES.lease;
  for (const method of ['literal-spans-v1', 'literal-spans-v2']) {
    const { passages, summary } = evidence(source, output, method);
    assert.deepEqual(lines(passages, source, output), LEASE, method);
    assert.equal(summary.passages, 2);
    assert.equal(summary.notes, 4);
  }
  const { passages: [p1, p2], summary } = evidence(source, output);
  assert.equal(headingText(summary), '2 passages compared, 4 differences noted.');
  assert.deepEqual(summaryFacts(summary), ['1 common word is also marked, without a note.']);
  assert.equal(noteStatement(p1.notes[0], p1, source, output), '“may” in the original, “can” in the rewrite.');
  assert.equal(noteStatement(p1.notes[1], p1, source, output), '“30 days” has no normalized word match in the entire rewrite text.');
  assert.equal(noteStatement(p2.notes[0], p2, source, output), '“unless rent is overdue” has no normalized word match in the entire rewrite text.');
});

test('refund example: the worked evidence, identical under both splitters', () => {
  const { source, output } = EXAMPLES.refund;
  for (const method of ['literal-spans-v1', 'literal-spans-v2']) {
    assert.deepEqual(lines(evidence(source, output, method).passages, source, output), REFUND, method);
  }
  const { passages, summary } = evidence(source, output);
  assert.equal(summary.notes, 17);
  const groups = groupByState(passages);
  assert.deepEqual(Object.fromEntries(Object.entries(groups).map(([k, v]) => [k, v.length])),
    { 'a-only': 8, differs: 8, 'b-only': 0, other: 1, moved: 0, both: 1 });
  const p2 = passages.find(p => p.number === '2'), p51 = passages.find(p => p.number === '5.1'), p6 = passages.find(p => p.number === '6');
  assert.equal(noteStatement(p2.notes[0], p2, source, output), '“not” has no normalized word match in the paired rewrite passage.');
  assert.equal(noteStatement(p51.notes[0], p51, source, output), '“Northwind” has a normalized word match in the original, starting in passage 6: “Northwind”.');
  assert.equal(noteStatement(p6.notes[1], p6, source, output), '“must” is in an original passage that has no partner in the rewrite.');
  assert.deepEqual(passages.map(p => p.number), ['1', '2', '3', '4', '5', '5.1', '6']);
});

test('splitters: v2 fixes the measured v1 defects, splits Chinese and Japanese sentences, and agrees elsewhere', () => {
  const cases = [
    ['Shipping costs $7.95 per order. Refunds follow.', 1, 2],
    ['Call Dr. Patel if a rash appears. Stop the tablets.', 3, 2],
    ['Bring ID, e.g. a passport. Then sign.', 1, 2],
    ['Take 3.5 mg daily. Stop if dizzy.', 1, 2],
    ['Made in the U.S. by Acme.', 1, 1],
    ['First line\nSecond line', 2, 2],
    ['租金必须在30天内支付。押金可以退还。', 1, 2],
    ['患者は1日3回服用してください！医師の指示がない限り、超えないでください。', 1, 2],
    ['他说：“好的。”然后离开。', 1, 2],
  ];
  for (const [text, v1, v2] of cases) {
    assert.equal(sourceSpans(text).length, v1, `v1: ${text}`);
    assert.equal(sourceSpansV2(text).length, v2, `v2: ${text}`);
    for (const span of sourceSpansV2(text)) assert.equal(text.slice(span.start, span.end), span.text);
  }
  for (const ex of Object.values(EXAMPLES)) {
    for (const text of [ex.source, ex.output]) assert.deepEqual(sourceSpans(text), sourceSpansV2(text), 'examples split identically');
  }
  // Chinese passages pair by characters, so a one-character change still pairs.
  assert.deepEqual(compareTexts('租金必须在30天内支付。', '租金必须在60天内支付。').rows.map(r => r.kind), ['changed-candidate']);
  assert.deepEqual(tokenize('租金30天').map(t => t.t), ['租', '金', '30', '天']);
});

test('Read as Original and Read as Rewrite show each text exactly, character for character', () => {
  const pairs = [
    ['Keep the dose < 5 mg, taken daily by Alice.', 'Keep the dose > 5 mg taken daily by Bob.'],
    ['On Tuesday, March 3, 2026, Mayor Okafor spoke.', 'Mayor Okafor spoke.'],
    ['Alice may cancel the lease with 30 days notice.', 'Alice can cancel the lease.'],
    ['A b c.', 'A  b\tc.'],
  ];
  for (const [a, b] of pairs) {
    for (const p of evidence(a, b).passages) {
      const segs = blacklineSegments(p, a, b);
      assert.equal(viewText(segs, 'a'), p.a.text, `original of ${JSON.stringify(a)}`);
      assert.equal(viewText(segs, 'b'), p.b.text, `rewrite of ${JSON.stringify(b)}`);
    }
  }
});

test('only character-identical texts are called identical; other character differences are noted', () => {
  assert.equal(headingText(evidence('Alice may cancel.', 'Alice may cancel.').summary), '1 passage compared, no literal differences.');
  for (const [a, b, said] of [
    ['Call if glucose is < 70 mg/dL.', 'Call if glucose is > 70 mg/dL.', '“<” in the original, “>” in the rewrite.'],
    ['Store at -20°C.', 'Store at 20°C.', 'Marked original characters: “-”.'],
    ['Price: 50 € per month.', 'Price: 50 $ per month.', '“€” in the original, “$” in the rewrite.'],
    ['Great job 👍 today.', 'Great job 👎 today.', '“👍” in the original, “👎” in the rewrite.'],
    ['Do not sign.', 'Do NOT sign.', '“not” in the original, “NOT” in the rewrite.'],
    ["Let's eat, Grandma.", "Let's eat Grandma.", 'Marked original characters: “,”.'],
    ['A b.', 'A  b.', 'Marked rewrite characters: a space.'],
    ['A b.', 'A\tb.', 'a space in the original, a tab in the rewrite.'],
  ]) {
    const { passages, summary } = evidence(a, b);
    assert.equal(summary.identicalTexts, false);
    assert.match(headingText(summary), /differences? noted/);
    assert.ok(statements(a, b).includes(said), `${a} -> ${b}: ${statements(a, b).join(' | ')}`);
    assert.equal(passageMessage(passages[0]), 'The normalized words match in order; the characters noted here differ.');
  }
});

test('reordered passages and spacing between passages are reported as such', () => {
  const reordered = evidence('Pay within 30 days. Late fees apply.', 'Late fees apply. Pay within 30 days.').summary;
  assert.equal(headingText(reordered), '2 passages compared; the texts are not identical.');
  assert.deepEqual(summaryFacts(reordered), ['The rewrite has these passages in a different order.']);
  const spaced = evidence('Pay within 30 days.\nLate fees apply.', 'Pay within 30 days.  Late fees apply.').summary;
  assert.deepEqual(summaryFacts(spaced), ['Every passage is identical and in the same order; the texts differ only in the spacing or line breaks outside the passages (before, between or after them).']);
});

test('unaligned equal words are noted as matches without inferring movement', () => {
  for (const [a, b, word] of [
    ['You must not leave, and you may stay.', 'You must leave, and you may not stay.', 'not'],
    ['Tenants may not smoke, and may vape.', 'Tenants may smoke, and may not vape.', 'not'],
    ['Alice pays Bob.', 'Bob pays Alice.', 'Alice'],
  ]) {
    const { passages, summary } = evidence(a, b);
    const moved = passages[0].notes.filter(n => n.state === 'moved');
    assert.ok(moved.some(n => noteStrings(n, a, b).aText === word), `${a}: ${moved.length} moved`);
    assert.ok(statements(a, b).includes(`“${word}” is matched in both passages by normalized words.`));
    assert.match(headingText(summary), /differences? noted/);
  }
});

test('absence claims concern normalized words, not substrings or meaning', () => {
  for (const [a, b, word] of [
    ['Alice may cancel the lease.', "Alice's lease can be cancelled.", 'Alice'],
    ['The fee is $8 today. Nothing else.', 'The fee is waived. Total $8.50 today.', '$8'],
    ['Take 1 tablet every 4 to 6 hours.', 'Take one tablet every 4-6 hours.', '4'],
  ]) assert.ok(statements(a, b).includes(`“${word}” has no normalized word match in the entire rewrite text.`), a);
  assert.ok(statements('You may not leave.', 'You cannot leave.').includes('“not” in the original, “cannot” in the rewrite.'));
});

test('a string that is in the paired passage fewer times is stated with both counts', () => {
  assert.ok(statements('Do not smoke, not ever, not here.', 'Do not smoke, not here.')
    .includes('“not” has non-overlapping normalized word matches 3 times in the original passage and twice in the paired rewrite passage.'));
});

test('in-both lines print what each passage literally has', () => {
  const a = 'Do not sign.', b = 'Do NOT sign.';
  const lines = evidence(a, b).passages[0].inBoth.map(inBothText);
  assert.ok(lines.includes('“not” in the original, “NOT” in the rewrite: matching normalized words; exact characters differ.'));
  const g = evidence('Call if glucose is < 70 mg/dL.', 'Call if glucose is > 70 mg/dL.').passages[0].inBoth.map(inBothText);
  assert.ok(g.includes('“if glucose is < 70 mg/dL” in the original, “if glucose is > 70 mg/dL” in the rewrite: matching normalized words; exact characters differ.'));
});

test('the blackline never marks a space as a word, and each run holds one side only', () => {
  for (const ex of Object.values(EXAMPLES)) {
    for (const p of evidence(ex.source, ex.output).passages) {
      for (const seg of blacklineSegments(p, ex.source, ex.output)) {
        for (const part of seg.t === 'run' ? seg.parts : []) {
          if (part.t === 'mark' && !part.ws) assert.equal(part.text, part.text.trim(), `mark "${part.text}"`);
        }
      }
    }
  }
  // Consecutive words of one note stay one mark.
  const lease = EXAMPLES.lease;
  const marks = blacklineSegments(evidence(lease.source, lease.output).passages[0], lease.source, lease.output)
    .flatMap(s => (s.t === 'run' ? s.parts : [])).filter(m => m.t === 'mark').map(m => m.text);
  assert.ok(marks.includes('30 days'));
});

test('note letters run a to z, then aa, ab', () => {
  const a = Array.from({ length: 30 }, (_, i) => `Item${i} costs $${i}`).join(', ') + '.';
  const b = Array.from({ length: 30 }, (_, i) => `Item${i} costs $${i + 100}`).join(', ') + '.';
  const letters = evidence(a, b).passages[0].notes.map(n => n.letter);
  assert.deepEqual(letters.slice(24, 29), ['y', 'z', 'aa', 'ab', 'ac']);
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

test('long and hostile inputs stay fast (timing guard)', () => {
  const guard = (label, a, b, limitMs) => {
    const started = performance.now();
    const { passages } = evidence(a, b);
    for (const p of passages) blacklineSegments(p, a, b);
    const took = performance.now() - started;
    assert.ok(took < limitMs, `${label}: ${took.toFixed(0)} ms (limit ${limitMs} ms)`);
  };
  const t0 = performance.now();
  sourceSpansV2('a. '.repeat(33000));
  sourceSpansV2('go. '.repeat(25000));
  assert.ok(performance.now() - t0 < 1000, 'the v2 splitter is linear');
  guard("'if ' repeated to 99,000 characters", 'if '.repeat(33000), 'x', 4000);
  // Review reproduction: formerly lowercased 86,769 characters for each of
  // 1,375 absent structural words (7.3 seconds in this Node environment).
  const structuralA = [], structuralB = [];
  for (let i = 0; i < 11000; i++) {
    const word = 'wwwww' + (i % 90);
    structuralA.push(word); structuralB.push(word);
    if (i % 8 === 0) structuralA.push('not');
  }
  guard('1,375 missing words in a 92k-character passage', structuralA.join(' '), structuralB.join(' '), 2000);
  const nums = Array.from({ length: 17000 }, (_, i) => String(i % 997)).join(' ').slice(0, 85000);
  guard('85,000 characters of numbers', nums, 'x', 4000);
  const words = Array.from({ length: 16000 }, (_, i) => ['alpha', 'beta', 'gamma', 'delta', 'tenant', 'pays'][i % 6]).join(' ').slice(0, 99000);
  guard('one 99,000-character passage against its reverse', words, words.split(' ').reverse().join(' '), 4000);
  const long = EXAMPLES.refund.source.replace(/\. /g, ', ').repeat(230).slice(0, 99000);
  guard('one 99,000-character passage with light edits', long, long.replace(/30 days/g, '60 days'), 4000);
});

// Fixed raw-text reproductions from the independent review. These expected
// phrases/counts were specified from the raw strings, not from engine output.
test('normalization is disclosed for hard wraps, apostrophes and extra gaps', () => {
  for (const [a, b] of [
    ['Late fees apply if rent is late. unless waived.', 'Late fees apply if rent is late. Unless waived.'],
    ['Refunds apply unless rent is overdue.', 'Refunds apply unless rent is\noverdue.'],
    ["Don't don't don't enter.", 'Don’t don’t enter.'],
    ['Pay within 30 days.', 'Pay within 30\tdays.'],
  ]) {
    const printed = statements(a, b).join('\n');
    assert.doesNotMatch(printed, /nowhere|only within other words|ignoring capitalization|at different places/);
  }
  assert.ok(statements("Don't don't don't enter.", 'Don’t don’t enter.').some(s => /3 times.*twice/.test(s)), 'apostrophe-normalized count is explicit');
  const cross = statements('Late fees apply if rent is late. unless waived.', 'Late fees apply if rent is late. Unless waived.');
  assert.ok(cross.some(s => /normalized word match.*starting in passage/.test(s)), 'a match spanning passages is found');
});

test('one-sided character notes claim only their own marked text', () => {
  const printed = statements('Pay now.', 'Pay\u200b now.');
  assert.ok(printed.includes('Marked rewrite characters: code point U+200B.'));
  assert.doesNotMatch(printed.join(' '), /original does not|nowhere/);
});

test('lexical labels do not infer grammar or mistake Unicode symbols for capitalization', () => {
  const a = 'Put it in the can.', b = 'Put it in the box.';
  assert.ok(evidence(a, b).passages.flatMap(p => p.notes).some(n => kindLabel(n) === 'Modal-list word'));
  const only = evidence('The only exit is here.', 'The exit is here.').passages.flatMap(p => p.notes);
  assert.ok(only.some(n => kindLabel(n) === 'Marker word'));
  for (const [x, y] of [['K', 'K'], ['Ω', 'Ω']]) {
    const notes = evidence(`Cool it to 5 ${x} now.`, `Cool it to 5 ${y} now.`).passages.flatMap(p => p.notes);
    assert.ok(notes.some(n => kindLabel(n) === 'Characters'));
    assert.ok(notes.every(n => kindLabel(n) !== 'Capitalization'));
  }
  assert.deepEqual(tokenize('⚠️ hot.').map(t => t.t), ['hot'], 'variation selector belongs to the symbol gap');
  assert.deepEqual(tokenize('e\u0301cho').map(t => t.t), ['e\u0301cho'], 'a letter retains its combining mark');
});

test('matched-word notes never overlap aligned in-both words', () => {
  for (const [a, b] of [
    ['Refunds are given if the item is damaged.', 'Refunds are given if if the ITEM is damaged.'],
    ['may? except no, Dr.... ', 'may? except. except no, Dr.... '],
    ['pay €, Alice! 👍,', 'pay €, Pay Alice! 👍,'],
  ]) for (const p of evidence(a, b).passages) {
    for (const n of p.notes.filter(n => n.state === 'moved')) {
      for (const side of ['a', 'b']) {
        const ns = noteSpans(n, a, b).filter(s => s.side === side);
        for (const ib of p.inBoth) assert.ok(ns.every(s => s.e <= ib[side].s || s.s >= ib[side].e), 'matched and aligned notes are disjoint');
      }
    }
  }
});

test('leading and trailing whitespace is described as outside passages', () => {
  for (const [a, b] of [['Pay rent.', 'Pay rent.\n'], ['x', 'x '], [' x', 'x']]) {
    assert.deepEqual(summaryFacts(evidence(a, b).summary), [
      'Every passage is identical and in the same order; the texts differ only in the spacing or line breaks outside the passages (before, between or after them).',
    ]);
  }
});
