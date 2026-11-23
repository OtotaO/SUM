// Depth reader gate: does the depth reader's brief view keep the sentences a
// reader marked as must-read more often than the opening of the text within
// the same character budget? See README.md for the marking format and the rule.
//
//   node fixtures/depth_gate/depth_gate.mjs check   DIR [--write-split SPLIT]
//   node fixtures/depth_gate/depth_gate.mjs dev     DIR --split SPLIT
//   node fixtures/depth_gate/depth_gate.mjs heldout DIR --split SPLIT --out FILE --min-recall X
//
// check validates every text and mark and prints the split; it scores nothing.
// With --write-split it freezes the hashes of every text and marks file; dev
// and heldout refuse a folder that differs from that file. dev scores the dev
// half only. heldout scores the held-out half and refuses to run when FILE
// already exists. The JSON record holds hashes and counts, never text or mark
// wording; error messages name files and a mark's position among the
// non-blank lines of its marks file, never wording.
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join, basename, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { distill, passages, DISTILLER_ID } from '../../single_file_demo/distiller.js';

export const GATE_ID = 'sum.depth_gate.v1';
export const TEXT_COUNT = 8;
export const MARKS_MIN = 3, MARKS_MAX = 5;
// The share of the text's passage characters each arm may use. The gate reads
// 0.1 (the depth reader's "brief"); the others are reported, never gated.
export const SHARES = [['brief', 0.1], ['summary', 0.25], ['half', 0.5]];
const DEPTH_ORDER = ['headline', 'brief', 'summary', 'half', 'full text'];

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

// A mark is located as exactly one occurrence in the text, where any run of
// whitespace in the mark matches any run of whitespace in the text. No other
// normalisation: quotes, dashes and case must match.
export function locateMark(text, mark) {
  const m = mark.trim();
  if (!m) throw new Error('empty mark');
  const re = new RegExp(m.split(/\s+/u).map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('\\s+'), 'gu');
  // Search again from one character past each hit, so a copy that overlaps
  // an earlier one is counted too.
  const hits = [];
  for (let m; (m = re.exec(text)); re.lastIndex = m.index + 1) hits.push(m);
  // Errors name the mark by its line, never its wording.
  if (hits.length === 0) throw new Error('mark not found in the text');
  if (hits.length > 1) throw new Error(`mark occurs ${hits.length} times in the text, it must occur once`);
  return { start: hits[0].index, end: hits[0].index + hits[0][0].length };
}

export function locateMarks(text, marks) {
  if (marks.length < MARKS_MIN || marks.length > MARKS_MAX) throw new Error(`${marks.length} marks, need ${MARKS_MIN} to ${MARKS_MAX}`);
  const spans = marks.map((m, i) => { try { return locateMark(text, m); } catch (e) { throw new Error(`mark ${i + 1}: ${e.message}`); } });
  const order = spans.map((s, i) => ({ s, i })).sort((a, b) => a.s.start - b.s.start);
  for (let k = 1; k < order.length; k++) if (order[k].s.start < order[k - 1].s.end) throw new Error(`marks ${Math.min(order[k - 1].i, order[k].i) + 1} and ${Math.max(order[k - 1].i, order[k].i) + 1} overlap`);
  return spans;
}

// A passage opens a block when it is the first passage, a heading line, the
// first passage after a heading, or the first after a blank line.
const HEADING = /^[ \t]*#{1,6}\s/;
export function blockOpeners(text, spans) {
  const isHeading = s => HEADING.test(text.slice(text.lastIndexOf('\n', s.start - 1) + 1, s.end));
  return spans.map((s, i) => i === 0 || isHeading(s) || isHeading(spans[i - 1]) || /\n[ \t]*\r?\n/.test(text.slice(spans[i - 1].end, s.start)));
}

// Each arm is a list of character ranges of the text. Every arm uses the depth
// reader's own budget rule (distiller.js depthCounts): the longest prefix of
// its ordering whose passages fit share x the text's passage characters, and
// always at least one passage. Only the ordering differs:
//   distill: the reader's ranking (its own view at that depth);
//   lead:    the passages in text order;
//   outline: block openers (see blockOpeners) in text order, then the other
//            passages in text order.
// The characters each arm uses differ by text and are recorded; a row whose
// arm goes over the budget (one passage longer than it) is flagged.
const rangesOf = (idx, spans) => [...idx].sort((x, y) => x - y).map(i => ({ start: spans[i].start, end: spans[i].end }));

function prefixWithin(order, spans, cap) {
  const out = []; let used = 0;
  for (const i of order) {
    const len = spans[i].end - spans[i].start;
    if (out.length && used + len > cap) break;
    out.push(i); used += len;
  }
  return rangesOf(out, spans);
}

// The depth reader's set at a share is the depth with that label; a depth the
// reader dropped because it equalled the one before it falls back to that one.
function readerSet(d, label) {
  let last = d.depths[0];
  for (const name of DEPTH_ORDER) {
    const hit = d.depths.find(x => x.label === name);
    if (hit) last = hit;
    if (name === label) break;
  }
  return last.passages.map(p => ({ start: p.start, end: p.end }));
}

export function arms(text) {
  const spans = passages(text), d = distill(text);
  const total = spans.reduce((a, s) => a + s.end - s.start, 0);
  const sourceOrder = spans.map((_, i) => i);
  const open = blockOpeners(text, spans);
  const outlineOrder = [...sourceOrder.filter(i => open[i]), ...sourceOrder.filter(i => !open[i])];
  const out = {};
  for (const [label, share] of SHARES) {
    const cap = share * total;
    out[label] = {
      cap: Math.floor(cap),
      distill: readerSet(d, label),
      lead: prefixWithin(spans.map((_, i) => i), spans, cap),
      outline: prefixWithin(outlineOrder, spans, cap),
    };
  }
  return { spans, total, method: d.method, arms: out };
}

// A mark counts as kept only when every non-whitespace character of it lies
// inside a range the arm kept.
export function covered(text, ranges, mark) {
  const inKept = new Uint8Array(mark.end - mark.start);
  for (const r of ranges) {
    const s = Math.max(r.start, mark.start), e = Math.min(r.end, mark.end);
    if (s < e) inKept.fill(1, s - mark.start, e - mark.start);
  }
  for (let k = mark.start; k < mark.end; k++) if (!inKept[k - mark.start] && /\S/u.test(text[k])) return false;
  return true;
}

export function scoreText(text, marks) {
  const located = locateMarks(text, marks);
  const { spans, total, arms: a } = arms(text);
  const depths = {};
  for (const [label] of SHARES) {
    const row = { cap: a[label].cap };
    for (const arm of ['distill', 'lead', 'outline']) {
      const kept = a[label][arm];
      row[arm] = {
        chars: kept.reduce((n, r) => n + r.end - r.start, 0),
        passages: kept.length,
        kept_marks: located.filter(m => covered(text, kept, m)).length,
      };
      row[arm].over_cap = row[arm].chars > row.cap;
    }
    depths[label] = row;
  }
  return { chars: text.length, passage_chars: total, passages: spans.length, marks: located.length, depths };
}

// Texts are sorted by the SHA-256 of their file bytes; the lower half is dev,
// the upper half is held out.
export function split(items) {
  const sorted = [...items].sort((x, y) => (x.sha256 < y.sha256 ? -1 : x.sha256 > y.sha256 ? 1 : 0));
  const half = sorted.length / 2;
  return { dev: sorted.slice(0, half), heldout: sorted.slice(half) };
}

export function pooled(results) {
  const out = {};
  for (const [label] of SHARES) {
    const marks = results.reduce((n, r) => n + r.marks, 0), row = { marks };
    for (const arm of ['distill', 'lead', 'outline']) {
      const kept = results.reduce((n, r) => n + r.depths[label][arm].kept_marks, 0);
      const over = results.filter(r => r.depths[label][arm].over_cap).length;
      row[arm] = { kept_marks: kept, recall: marks ? kept / marks : 0, texts_over_cap: over };
    }
    out[label] = row;
  }
  return out;
}

// The rule: the reader passes on the held-out texts when its brief keeps at
// least minRecall of the pooled marks AND strictly more marks than the lead
// arm. A tie is not a pass. Texts where an arm went over the budget are
// counted beside the verdict, not removed from it.
export function verdict(pool, minRecall) {
  const b = pool.brief;
  const pass = b.distill.recall >= minRecall && b.distill.kept_marks > b.lead.kept_marks;
  return { pass, brief_recall: b.distill.recall, min_recall: minRecall, distill_kept: b.distill.kept_marks, lead_kept: b.lead.kept_marks, marks: b.marks,
    texts_over_cap: { distill: b.distill.texts_over_cap, lead: b.lead.texts_over_cap } };
}

export function loadDir(dir) {
  const names = readdirSync(dir).filter(n => /\.(txt|md)$/.test(n) && !/\.marks\.txt$/.test(n)).sort();
  return names.map(n => {
    const stem = n.replace(/\.(txt|md)$/, ''), bytes = readFileSync(join(dir, n));
    let marksBytes;
    try { marksBytes = readFileSync(join(dir, `${stem}.marks.txt`)); } catch { throw new Error(`${n}: no ${stem}.marks.txt next to it`); }
    const marksRaw = marksBytes.toString('utf8');
    const text = bytes.toString('utf8');
    const marks = marksRaw.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    try { locateMarks(text, marks); passages(text); } catch (e) { throw new Error(`${n}: ${e.message} (marks count non-blank lines of ${stem}.marks.txt)`); }
    return { name: n, sha256: sha256(bytes), marks_sha256: sha256(marksBytes), text, marks };
  });
}

function main(argv) {
  const [mode, dir, ...rest] = argv;
  const flag = k => { const i = rest.indexOf(k); return i >= 0 ? rest[i + 1] : undefined; };
  if (!['check', 'dev', 'heldout'].includes(mode) || !dir) { console.error('usage: depth_gate.mjs check DIR [--write-split FILE] | dev DIR --split FILE | heldout DIR --split FILE --out FILE --min-recall X'); return 2; }
  const items = loadDir(dir);
  if (items.length !== TEXT_COUNT) { console.error(`found ${items.length} texts, need ${TEXT_COUNT}`); return 2; }
  if (new Set(items.map(x => x.sha256)).size !== items.length) { console.error('two texts have the same bytes'); return 2; }
  const { dev, heldout } = split(items);
  const label = x => `${x.sha256.slice(0, 12)}  ${x.name}  (${x.marks.length} marks, ${x.text.length} chars)`;
  const frozen = { gate: GATE_ID, dev: dev.map(x => ({ sha256: x.sha256, marks_sha256: x.marks_sha256 })), heldout: heldout.map(x => ({ sha256: x.sha256, marks_sha256: x.marks_sha256 })) };
  if (mode === 'check') {
    console.log('All texts and marks are valid. Nothing was scored.');
    console.log('dev:'); dev.forEach(x => console.log('  ' + label(x)));
    console.log('held out:'); heldout.forEach(x => console.log('  ' + label(x)));
    const w = flag('--write-split');
    if (w) {
      try { writeFileSync(w, JSON.stringify(frozen, null, 2) + '\n', { flag: 'wx' }); }
      catch (e) { if (e.code === 'EEXIST') { console.error(`${w} exists: the split is frozen already`); return 3; } throw e; }
      console.log(`split written to ${basename(w)}`);
    }
    return 0;
  }
  // A run is bound to the split file it is given: a text or marks file that
  // differs from it by a byte is refused. That the file given is the one
  // committed before the dev run is shown by the history and split_sha256.
  const splitFile = flag('--split');
  if (!splitFile) { console.error(`${mode} needs --split FILE (written by check --write-split)`); return 2; }
  let want = null;
  try { want = JSON.parse(readFileSync(splitFile, 'utf8')); } catch { /* reported below */ }
  const isSplit = w => w && typeof w === 'object' && w.gate === GATE_ID && Array.isArray(w.dev) && Array.isArray(w.heldout);
  if (!isSplit(want)) { console.error(`${basename(splitFile)} is not a split file written by check --write-split`); return 2; }
  if (JSON.stringify({ dev: want.dev, heldout: want.heldout }) !== JSON.stringify({ dev: frozen.dev, heldout: frozen.heldout })) {
    console.error('the texts or marks differ from the frozen split'); return 4;
  }
  const here = dirname(fileURLToPath(import.meta.url));
  const record = (set, list) => {
    const per = list.map(x => ({ sha256: x.sha256, marks_sha256: x.marks_sha256, ...scoreText(x.text, x.marks) }));
    return {
      gate: GATE_ID, set, distiller: DISTILLER_ID,
      distiller_js_sha256: sha256(readFileSync(join(here, '../../single_file_demo/distiller.js'))),
      harness_sha256: sha256(readFileSync(fileURLToPath(import.meta.url))),
      split_sha256: sha256(readFileSync(splitFile)),
      texts: per, pooled: pooled(per),
    };
  };
  if (mode === 'dev') {
    const r = record('dev', dev);
    console.log(JSON.stringify(r, null, 2));
    return 0;
  }
  const out = flag('--out'), min = Number(flag('--min-recall'));
  if (!out || !(min > 0 && min <= 1)) { console.error('heldout needs --out FILE and --min-recall X (0 < X <= 1)'); return 2; }
  if (existsSync(out)) { console.error(`${out} exists: the held-out set has been scored already`); return 3; }
  const r = record('heldout', heldout);
  r.verdict = verdict(r.pooled, min);
  try { writeFileSync(out, JSON.stringify(r, null, 2) + '\n', { flag: 'wx' }); }
  catch (e) { if (e.code === 'EEXIST') { console.error(`${out} exists: the held-out set has been scored already`); return 3; } throw e; }
  console.log(`${r.verdict.pass ? 'PASS' : 'FAIL'}: brief kept ${r.verdict.distill_kept} of ${r.verdict.marks} marks (lead ${r.verdict.lead_kept}); written to ${basename(out)}`);
  return 0;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  try { process.exitCode = main(process.argv.slice(2)); } catch (e) { console.error(e.message); process.exitCode = 1; }
}
