import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { detailFlags, DETAIL_FLAGS_ID } from './detail_flags.js';

// Hand-made examples only. The frozen evaluation corpus (fixtures/detail_flags_eval) is scored by
// fixtures/detail_flags_eval/score.mjs, never used here.
const flagsOf = (s, o) => detailFlags(s, o).flags.map(f => [f.category, f.kind, f.source?.text ?? null, f.output?.text ?? null]);

test('each kind of detail change is flagged at the changed words', () => {
  const cases = [
    ['Cancel at least 48 hours before check-in for a full refund.', 'Cancel at least 24 hours before check-in for a full refund.', ['number', 'changed', '48 hours', '24 hours']],
    ['Send the form to the Benefits Office by 30 June 2027.', 'Send the form to the Benefits Office by 13 June 2027.', ['date', 'changed', '30 June 2027', '13 June 2027']],
    ['The warranty does not cover accidental damage.', 'The warranty covers accidental damage.', ['negation', 'missing', 'not', null]],
    ['You must return the device by Friday.', 'You should return the device by Friday.', ['modal', 'changed', 'must', 'should']],
    ['You can return items within 30 days.', 'You can return items within 30 days if you have the receipt.', ['condition', 'added', null, 'if']],
    ['Visitors can park in Lot B unless an event is scheduled.', 'Visitors can park in Lot B.', ['exception', 'missing', 'unless', null]],
    ['Questions go to Priya Shah in Payroll.', 'Questions go to Priya Patel in Payroll.', ['name', 'changed', 'Priya Shah', 'Priya Patel']],
    ['The monthly fee is $24.99 and rises by 10% in June.', 'The fee is about $25 each month and goes up by ten percent in June.', ['number', 'changed', '$24.99', '$25']],
  ];
  for (const [source, output, expected] of cases) assert.deepEqual(flagsOf(source, output), [expected], source);
});

test('details kept in other words are not flagged', () => {
  const cases = [
    ['Dr. Maria Alvarez will see you on March 3, 2027 at 5 p.m. You must bring two forms of ID. If you are late by more than 15 minutes, you may need to rebook.',
      'Your appointment with Dr. Alvarez is on 3 March 2027 at 17:00. You have to bring 2 forms of ID. If you arrive more than fifteen minutes late, you might need to book again.'],
    ['Smoking is not permitted anywhere in the building. Tenants must give 30 days notice before moving out, unless the lease says otherwise. Northgate Housing Association will return the deposit within 14 days.',
      'You cannot smoke anywhere in the building. You need to tell us 30 days before you move out, except where your lease says something different. Northgate Housing will give back your deposit within two weeks.'],
    ['Your policy covers water damage from burst pipes. It does not cover damage caused by flooding from outside the home. Claims must be filed within 60 days of the incident.',
      'Water damage from burst pipes is covered. Flood water coming in from outside is not covered. File any claim no more than 60 days after it happens.'],
    ['Students may leave the school grounds at lunch provided that a parent has signed the permission form. Year 7 students are not allowed to leave under any circumstances.',
      'Students can go off school grounds at lunch as long as a parent signed the permission form. Year 7 students can never leave.'],
  ];
  for (const [source, output] of cases) assert.deepEqual(flagsOf(source, output), [], source);
});

test('the membership example flags the shortened refund window and the dropped notice period', () => {
  const flags = flagsOf('You may cancel this membership at any time with 30 days written notice. Fees already paid are refundable only if you cancel within 14 days of joining. We may change these terms after giving you 60 days notice. Disputes are handled by the courts of your home state.',
    'You can cancel your membership at any time. Fees already paid are refundable only if you cancel within 30 days of joining. We may change these terms after giving you 60 days notice. You can contact support with any questions.');
  assert.deepEqual(flags.filter(f => f[0] === 'number'), [['number', 'missing', '30 days', null], ['number', 'changed', '14 days', '30 days']]);
});

const bills = JSON.parse(await readFile(new URL('../fixtures/meaning_receipts_billsum/corpus_billsum_test_first64.json', import.meta.url))).pairs;
const texts = ['', ' \n ', 'No.', 'May 5 or may not.', ...bills.slice(0, 24).flatMap(p => [p.source, p.rendering || ''])].filter(t => typeof t === 'string');

test('a text compared with itself has no flags, and every span is copied from its text', () => {
  for (const text of texts) {
    let result;
    try { result = detailFlags(text, text); } catch (e) { assert.match(e.message, /up to 300/); continue; }
    assert.equal(result.method, DETAIL_FLAGS_ID);
    assert.deepEqual(result.flags, [], text.slice(0, 80));
  }
  for (let i = 0; i + 1 < texts.length; i++) {
    let result;
    try { result = detailFlags(texts[i], texts[i + 1]); } catch (e) { assert.match(e.message, /up to 300/); continue; }
    for (const f of result.flags) {
      if (f.source) { assert.equal(f.source.text, texts[i].slice(f.source.start, f.source.end)); assert.ok(f.source.end - f.source.start <= 80); }
      if (f.output) { assert.equal(f.output.text, texts[i + 1].slice(f.output.start, f.output.end)); assert.ok(f.output.end - f.output.start <= 80); }
      assert.ok(f.source || f.output);
    }
    assert.deepEqual(detailFlags(texts[i], texts[i + 1]), result);
  }
});

test('the checks make no network request and need no model', async () => {
  const code = await readFile(new URL('./detail_flags.js', import.meta.url), 'utf8');
  assert.doesNotMatch(code, /\bfetch\(|XMLHttpRequest|WebSocket|import\(/);
});
