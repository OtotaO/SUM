import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { distill, passages, rankPassages, DISTILLER_ID } from './distiller.js';

const bills = JSON.parse(await readFile(new URL('../fixtures/meaning_receipts_billsum/corpus_billsum_test_first64.json', import.meta.url))).pairs;
const docs = [...bills.map(p => p.source), await readFile(new URL('../docs/NORTH_STAR.md', import.meta.url), 'utf8'),
  'Alice may cancel the lease with 30 days notice. The deposit is refundable unless rent is overdue.'];

// Seeded generator so the property cases are the same on every run.
function lcg(seed) { let x = seed >>> 0; return () => (x = (Math.imul(x, 1664525) + 1013904223) >>> 0) / 2 ** 32; }
function randomText(seed) {
  const r = lcg(seed), words = ['deposit', 'tenant', 'notice', 'refund', 'the', 'shall', 'not', '30', 'U.S.C.', 'Dr.', 'e.g.', 'café', '\u{1F600}', 'rent'];
  const seps = [' ', ' ', ' ', '. ', '! ', '? ', '\n', '\n\n', '\n# ', '\n- ', '\n(a) ', '  \t', '...'];
  let t = '';
  for (let i = 0, n = Math.floor(r() * 120); i < n; i++) t += words[Math.floor(r() * words.length)] + seps[Math.floor(r() * seps.length)];
  return t;
}
const cases = [...docs, ...Array.from({ length: 300 }, (_, i) => randomText(i + 1))];

test('every passage and tag is copied verbatim from the original at its offsets', () => {
  for (const text of cases) {
    const d = distill(text);
    assert.equal(d.method, DISTILLER_ID);
    for (const depth of d.depths) for (const p of [...depth.passages, ...depth.left_out]) assert.equal(p.text, text.slice(p.start, p.end));
    for (const t of d.tags) assert.equal(t.text, text.slice(t.start, t.end));
  }
});

test('passages cover every non-space character exactly once, in order, never empty', () => {
  for (const text of cases) {
    const ps = passages(text), owner = new Int32Array(text.length).fill(-1);
    ps.forEach((p, i) => {
      assert.ok(p.end > p.start && p.start >= 0 && p.end <= text.length);
      assert.equal(text.slice(p.start, p.end), text.slice(p.start, p.end).trim());
      if (i) assert.ok(p.start >= ps[i - 1].end);
      for (let c = p.start; c < p.end; c++) owner[c] = i;
    });
    for (let c = 0; c < text.length; c++) if (/\S/u.test(text[c])) assert.notEqual(owner[c], -1, `uncovered character ${c} in ${JSON.stringify(text.slice(0, 80))}`);
  }
});

test('depths are nested, kept and left out partition the passages, and the deepest is the full text', () => {
  for (const text of cases) {
    const d = distill(text), n = passages(text).length;
    assert.equal(d.passage_count, n);
    if (!n) { assert.deepEqual(d.depths, []); continue; }
    assert.equal(d.depths.at(-1).count, n);
    assert.equal(d.depths.at(-1).label, 'full text');
    let prev = new Set();
    for (const depth of d.depths) {
      const kept = depth.passages.map(p => p.index), left = depth.left_out.map(p => p.index);
      assert.equal(kept.length, depth.count);
      assert.deepEqual([...kept, ...left].sort((a, b) => a - b), Array.from({ length: n }, (_, i) => i));
      assert.deepEqual(kept, [...kept].sort((a, b) => a - b));
      for (const i of prev) assert.ok(kept.includes(i), 'a shallower passage is missing from a deeper depth');
      assert.ok(kept.length > prev.size || depth === d.depths[0]);
      prev = new Set(kept);
    }
  }
});

test('the same text gives the same depths on every run', () => {
  for (const text of cases.slice(0, 80)) assert.equal(JSON.stringify(distill(text)), JSON.stringify(distill(text)));
});

test('lazy ranking equals plain greedy maximum coverage with ties to the earlier passage', () => {
  function plain(spanKeys, weight) {
    const covered = new Set(), order = [], left = new Set(spanKeys.keys());
    for (;;) {
      let best = -1, bestGain = 0;
      for (const i of left) {
        let s = 0; for (const k of spanKeys[i]) if (!covered.has(k)) s += weight.get(k);
        const g = s / Math.sqrt(1 + spanKeys[i].length);
        if (g > bestGain || (g === bestGain && g > 0 && i < best)) { best = i; bestGain = g; }
      }
      if (best < 0) break;
      order.push(best); left.delete(best); for (const k of spanKeys[best]) covered.add(k);
    }
    return [...order, ...[...left].sort((a, b) => a - b)];
  }
  for (let seed = 1; seed <= 400; seed++) {
    const r = lcg(seed), n = 1 + Math.floor(r() * 40), vocab = 1 + Math.floor(r() * 25);
    const spanKeys = Array.from({ length: n }, () => [...new Set(Array.from({ length: Math.floor(r() * 8) }, () => `w${Math.floor(r() * vocab)}`))].sort());
    const df = new Map(); spanKeys.forEach(ks => ks.forEach(k => df.set(k, (df.get(k) || 0) + 1)));
    const weight = new Map([...df].map(([k, c]) => [k, c - 1]));
    assert.deepEqual(rankPassages(spanKeys, weight), plain(spanKeys, weight), `seed ${seed}`);
  }
});

test('with no repeated words the ranking falls back to source order', () => {
  const d = distill('Alpha bravo charlie. Delta echo foxtrot. Golf hotel india. Juliet kilo lima.');
  assert.deepEqual(d.depths[0].passages.map(p => p.index), [0]);
  assert.deepEqual(d.tags, []);
});

test('segmentation joins wrapped lines and splits at sentences, headings, list items and clauses', () => {
  const text = 'The tenant may cancel\nwith 30 days notice. Dr. Smith agrees with\n20 U.S.C. 1099b. Done!\n# Terms\n- first item\n- second item\n(a) A clause\n\nLast line';
  assert.deepEqual(passages(text).map(p => text.slice(p.start, p.end)), [
    'The tenant may cancel\nwith 30 days notice.', 'Dr. Smith agrees with\n20 U.S.C. 1099b.', 'Done!', '# Terms', '- first item', '- second item', '(a) A clause', 'Last line']);
});

test('empty, blank, single-passage and over-limit inputs', () => {
  assert.equal(distill('').passage_count, 0);
  assert.equal(distill(' \n\t ').passage_count, 0);
  const one = distill('no terminal punctuation here');
  assert.deepEqual(one.depths.map(d => [d.label, d.count]), [['full text', 1]]);
  assert.throws(() => distill('a'.repeat(100001)), /at most 100,000/);
  assert.throws(() => distill(null), /at most 100,000/);
});

test('a real bill reads at three or more depths', () => {
  for (const p of bills) {
    const d = distill(p.source);
    assert.ok(d.depths.length >= 3, `${p.id} has ${d.depths.length} depths`);
    const share = { headline: 0, brief: 0.1, summary: 0.25, half: 0.5, 'full text': 1 };
    for (const depth of d.depths) assert.ok(depth.chars <= share[depth.label] * d.total_chars || depth.count === 1, `${p.id} ${depth.label}`);
  }
});
