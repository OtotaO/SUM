// Run: node --test fixtures/depth_gate/test_depth_gate.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, existsSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { distill } from '../../single_file_demo/distiller.js';
import { arms, covered, locateMark, locateMarks, scoreText, split, pooled, verdict, blockOpeners } from './depth_gate.mjs';

const HARNESS = join(dirname(fileURLToPath(import.meta.url)), 'depth_gate.mjs');
const WORDS = ['apples', 'bricks', 'candles', 'doors', 'engines', 'feathers', 'gloves', 'hammers', 'islands', 'jackets', 'kettles', 'ladders', 'mirrors', 'needles', 'oranges', 'pencils', 'quilts', 'ribbons', 'saddles', 'tables'];
const filler = i => `Line ${String.fromCharCode(97 + i)} mentions ${WORDS[i]} only once.`;

// Each arm wins on a text built for it.
const LEAD_MARKS = ['Our opening claim names the zebra.', 'The second claim names the yak.', 'A third claim names the walrus.'];
// The marks fill the first paragraph; every later paragraph opens with a
// sentence the outline arm takes instead of marks 2 and 3.
const LEAD_TEXT = [LEAD_MARKS.join(' '), ...Array.from({ length: 10 }, (_, i) => `Boats ${i} wait. The harbor boat crossed the river harbor near the river boat dock ${['north', 'south', 'east', 'west'][i % 4]} side.`)].join('\n\n');
const DIST_MARKS = ['The beacon tower signal reached the valley.', 'The valley farmers answered the beacon signal.', 'Every tower keeper watched the farmers answer.'];
const DIST_TEXT = [...Array.from({ length: 20 }, (_, i) => filler(i)), ...DIST_MARKS].join(' ');
const OUT_MARKS = ['Rain fell on the eastern hills.', 'Sheep moved to the lower meadow.', 'Prices rose at the autumn market.'];
const section = (h, m, k) => `## ${h}\n\n${m} ${Array.from({ length: 6 }, (_, i) => `Detail ${k}${i} repeats the common phrase about weather reports and weather maps.`).join(' ')}`;
const OUT_TEXT = [section('Weather', OUT_MARKS[0], 'a'), section('Flocks', OUT_MARKS[1], 'b'), section('Trade', OUT_MARKS[2], 'c')].join('\n\n');

test('lead wins when the marks open the text and the body repeats other words', () => {
  const b = scoreText(LEAD_TEXT, LEAD_MARKS).depths.brief;
  assert.equal(b.lead.kept_marks, 3);
  assert.ok(b.distill.kept_marks < b.lead.kept_marks && b.outline.kept_marks < b.lead.kept_marks);
});

test('the depth reader wins when the marks close the text and carry its repeated words', () => {
  const b = scoreText(DIST_TEXT, DIST_MARKS).depths.brief;
  assert.equal(b.lead.kept_marks, 0);
  assert.ok(b.distill.kept_marks > b.lead.kept_marks && b.distill.kept_marks > b.outline.kept_marks);
});

test('the outline arm wins when each mark opens a section', () => {
  const b = scoreText(OUT_TEXT, OUT_MARKS).depths.brief;
  assert.equal(b.outline.kept_marks, 3);
  assert.ok(b.outline.kept_marks > b.lead.kept_marks && b.outline.kept_marks > b.distill.kept_marks);
});

test('every arm stays within the cap unless one passage alone exceeds it, and that is flagged', () => {
  for (const [text, marks] of [[LEAD_TEXT, LEAD_MARKS], [DIST_TEXT, DIST_MARKS], [OUT_TEXT, OUT_MARKS]]) {
    for (const row of Object.values(scoreText(text, marks).depths))
      for (const arm of ['distill', 'lead', 'outline']) {
        assert.ok(row[arm].chars <= row.cap || row[arm].passages === 1, `${arm} ${row[arm].chars} > ${row.cap}`);
        assert.equal(row[arm].over_cap, row[arm].chars > row.cap);
      }
  }
  // One opening passage longer than a tenth of the text: lead takes it whole and is flagged.
  const long = `${'The opening sentence runs on and on with many words about nothing in particular '.repeat(3).trim()}. ${Array.from({ length: 8 }, (_, i) => filler(i)).join(' ')}`;
  const b = scoreText(long, [filler(0), filler(1), filler(2)]).depths.brief;
  assert.equal(b.lead.passages, 1);
  assert.equal(b.lead.over_cap, true);
});

test('the brief set equals the depth reader brief, or the headline when brief was dropped', () => {
  const d = distill(LEAD_TEXT), brief = d.depths.find(x => x.label === 'brief');
  assert.deepEqual(arms(LEAD_TEXT).arms.brief.distill, brief.passages.map(p => ({ start: p.start, end: p.end })));
  // One long passage that ranks first fills the headline and the 10% cap alike.
  const text = `${'The river and the harbor and the boat and the dock and the river harbor boat dock again. '.repeat(1).trim()} ${Array.from({ length: 6 }, (_, i) => filler(i)).join(' ')} The river harbor boat dock.`;
  const dd = distill(text);
  assert.equal(dd.depths.some(x => x.label === 'brief'), false);
  assert.deepEqual(arms(text).arms.brief.distill, dd.depths[0].passages.map(p => ({ start: p.start, end: p.end })));
});

test('a mark must occur exactly once; whitespace runs match any whitespace', () => {
  assert.throws(() => locateMark(LEAD_TEXT, 'A sentence that is not there.'), /not found/);
  assert.throws(() => locateMark('Same. Same.', 'Same.'), /occurs 2 times/);
  assert.throws(() => locateMark('Go on. Go on. Go on. End here.', 'Go on. Go on.'), /occurs 2 times/);
  // A mark that opens with a character outside the BMP (two UTF-16 units) is found once, not looped on.
  assert.deepEqual(locateMark('Start. \u{1F600} smiles here. End.', '\u{1F600} smiles here.'), { start: 7, end: 22 });
  assert.throws(() => locateMark('\u{1D400}b. \u{1D400}b.', '\u{1D400}b.'), /occurs 2 times/);
  assert.deepEqual(locateMark('One line\n  wrapped here.', 'line wrapped here.'), { start: 4, end: 24 });
  assert.throws(() => locateMark('Curly “quote”.', 'Curly "quote".'), /not found/);
  assert.throws(() => locateMarks(LEAD_TEXT, LEAD_MARKS.slice(0, 2)), /need 3 to 5/);
  assert.throws(() => locateMarks(LEAD_TEXT, [...LEAD_MARKS, 'The second claim', 'claim names the yak.']), e => e.message === 'marks 2 and 4 overlap');
  assert.throws(() => locateMarks(LEAD_TEXT, [LEAD_MARKS[0], 'Secret wording absent.', LEAD_MARKS[2]]), e => e.message === 'mark 2: mark not found in the text');
});

test('a mark counts only when every non-space character of it is kept, across passages or a cut', () => {
  const text = 'First half ends here. Second half follows.';
  const { spans } = arms(text);
  const mark = locateMark(text, 'ends here. Second half');
  assert.equal(covered(text, [spans[0]], mark), false);
  assert.equal(covered(text, [spans[0], spans[1]], mark), true);
  assert.equal(covered(text, [spans[0], { start: spans[1].start, end: spans[1].start + 6 }], mark), false);
  assert.equal(covered(text, [spans[0], { start: spans[1].start, end: spans[1].start + 11 }], mark), true);
});

test('block openers: headings, the sentence after one, and the sentence after a blank line', () => {
  const text = '# Title\nFirst sentence. Second sentence.\n\nThird after a gap. Fourth.';
  const { spans } = arms(text);
  assert.deepEqual(blockOpeners(text, spans), [true, true, false, true, false]);
});

test('split: the lower half of SHA-256 values is dev', () => {
  const items = ['f1', '0a', '9c', '3b'].map(h => ({ sha256: h }));
  const { dev, heldout } = split(items);
  assert.deepEqual(dev.map(x => x.sha256), ['0a', '3b']);
  assert.deepEqual(heldout.map(x => x.sha256), ['9c', 'f1']);
});

test('verdict: a tie with the lead arm fails, and so does recall under the minimum', () => {
  const row = (d, l) => ({ depths: { brief: { distill: { kept_marks: d }, lead: { kept_marks: l }, outline: { kept_marks: 0 } }, summary: { distill: { kept_marks: 0 }, lead: { kept_marks: 0 }, outline: { kept_marks: 0 } }, half: { distill: { kept_marks: 0 }, lead: { kept_marks: 0 }, outline: { kept_marks: 0 } } }, marks: 4 });
  assert.equal(verdict(pooled([row(3, 3)]), 0.5).pass, false);
  assert.equal(verdict(pooled([row(1, 0)]), 0.5).pass, false);
  assert.equal(verdict(pooled([row(2, 1)]), 0.5).pass, true);
  assert.deepEqual(verdict(pooled([row(2, 1)]), 0.5).texts_over_cap, { distill: 0, lead: 0 });
});

test('command line: check scores nothing, the split freezes the files, heldout writes once', () => {
  const dir = mkdtempSync(join(tmpdir(), 'depth-gate-'));
  const texts = [[LEAD_TEXT, LEAD_MARKS], [DIST_TEXT, DIST_MARKS], [OUT_TEXT, OUT_MARKS]];
  for (let i = 0; i < 8; i++) {
    const [t, m] = texts[i % 3];
    writeFileSync(join(dir, `t${i}.txt`), `${t}\n\nCopy ${i}.`);
    writeFileSync(join(dir, `t${i}.marks.txt`), m.join('\n') + '\n');
  }
  const run = (...a) => spawnSync(process.execPath, ['--no-warnings', HARNESS, ...a], { encoding: 'utf8' });
  const splitFile = join(dir, 'split.json'), out = join(dir, 'result.json');
  const check = run('check', dir, '--write-split', splitFile);
  assert.equal(check.status, 0, check.stderr);
  assert.match(check.stdout, /Nothing was scored/);
  assert.doesNotMatch(check.stdout, /kept_marks/);
  assert.equal(run('check', dir, '--write-split', splitFile).status, 3);
  for (const f of ['--write-split', '--split', '--out', '--min-recall']) {
    const r = run('check', dir, f);
    assert.equal(r.status, 2); assert.equal(r.stdout, ''); assert.equal(r.stderr, `${f} needs a value\n`);
  }
  assert.equal(run('check', dir, '--write-split', '--split').status, 2);
  const frozen = JSON.parse(readFileSync(splitFile, 'utf8'));
  assert.equal(frozen.dev.length, 4);
  assert.equal(frozen.heldout.length, 4);
  assert.equal(run('dev', dir).status, 2); // no --split
  assert.equal(run('heldout', dir, '--out', out, '--min-recall', '0.5').status, 2); // no --split
  // Anything that is not a split file is refused by name only: a text (whose
  // opening a JSON parse error would quote), JSON of another shape, null.
  writeFileSync(join(dir, 'other.json'), '{"a": 1}');
  writeFileSync(join(dir, 'null.json'), 'null');
  for (const name of ['t0.txt', 'other.json', 'null.json']) {
    const r = run('dev', dir, '--split', join(dir, name));
    assert.equal(r.status, 2, name);
    assert.equal(r.stderr, `${name} is not a split file written by check --write-split\n`);
    assert.equal(r.stdout, '');
  }
  assert.ok(!existsSync(out));
  const dev = run('dev', dir, '--split', splitFile);
  assert.equal(dev.status, 0, dev.stderr);
  assert.equal(JSON.parse(dev.stdout).texts.length, 4);
  assert.equal(run('heldout', dir, '--split', splitFile, '--out', out).status, 2); // no --min-recall
  const first = run('heldout', dir, '--split', splitFile, '--out', out, '--min-recall', '0.5');
  assert.equal(first.status, 0, first.stderr);
  const rec = JSON.parse(readFileSync(out, 'utf8'));
  assert.equal(rec.texts.length, 4);
  assert.deepEqual(rec.texts.map(t => t.sha256), frozen.heldout.map(t => t.sha256));
  assert.ok(!JSON.stringify(rec).includes('zebra'), 'no text wording in the output');
  assert.match(rec.distiller_js_sha256, /^[0-9a-f]{64}$/);
  assert.match(rec.split_sha256, /^[0-9a-f]{64}$/);
  assert.equal(run('heldout', dir, '--split', splitFile, '--out', out, '--min-recall', '0.5').status, 3);
  assert.ok(existsSync(out));
  // A marks file changed after the split is refused, and errors never echo mark wording.
  writeFileSync(join(dir, 't0.marks.txt'), m0());
  assert.equal(run('dev', dir, '--split', splitFile).status, 4);
  writeFileSync(join(dir, 't0.marks.txt'), 'Secret wording absent.\nNor this.\nNor that.\n');
  const bad = run('check', dir);
  assert.equal(bad.status, 1);
  assert.match(bad.stderr, /t0\.txt: mark 1: mark not found/);
  assert.doesNotMatch(bad.stderr, /Secret/);
});

function m0() { return LEAD_MARKS.slice().reverse().join('\n') + '\n'; }
