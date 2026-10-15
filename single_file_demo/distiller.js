import { MAX_REVIEW_CHARS } from './review_packet.js';

// Deterministic extractive depths over the user's own text. No model, no
// network, no randomness: every passage returned is source.slice(start, end).
// The identifier is distinct from every LLM render path so an extractive view
// cannot be mistaken for a generated rewrite.
export const DISTILLER_ID = 'sum.distill:extractive_coverage_v1';
export const DISTILL_SCOPE = 'Every passage shown is copied from your text at the stated position; nothing is written or paraphrased by a model. Which passages appear at each depth is a word-coverage heuristic. It does not measure importance, meaning or accuracy, and a passage left out may matter more than one kept.';

const STOP = new Set(('a about above after again against all also am an and any are as at be because been before being below between both but by can could did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself just me more most must my myself no nor not now of off on once only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up upon very was we were what when where which while who whom why will with would you your yours yourself yourselves ' +
  'may might shall per via etc one two its').split(' '));

// A line break inside a paragraph is treated as wrapping, so hard-wrapped
// prose yields whole sentences. A blank line, or a next line that opens a
// heading, list item, quote or numbered clause, ends the passage. A heading
// line is always a passage of its own.
const HEADING = /^[ \t]*#{1,6}\s/;
const STRUCTURAL = /^[ \t]*(?:#{1,6}\s|[-*+>]\s|\d{1,3}[.)]\s|\(\w{1,5}\)\s|`{2}|\|)/;
const ABBREV = /(?:(?:^|[\s(.])[A-Z]|(?:^|[\s(])(?:Mr|Mrs|Ms|Dr|Prof|St|No|Nos|Sec|Secs|Art|Fig|vs|etc|Inc|Ltd|Co|Corp|Jr|Sr|e\.g|i\.e|U\.S|cf|al))\.$/;

export function passages(text) {
  if (typeof text !== 'string' || text.length > MAX_REVIEW_CHARS) throw new Error('Text must be at most 100,000 characters.');
  const units = [];
  let unitStart = -1, lineStart = 0;
  const close = end => { if (unitStart >= 0) units.push([unitStart, end]); unitStart = -1; };
  while (lineStart <= text.length) {
    const nl = text.indexOf('\n', lineStart), lineEnd = nl < 0 ? text.length : nl;
    const line = text.slice(lineStart, lineEnd);
    if (!line.trim()) close(lineStart);
    else if (HEADING.test(line)) { close(lineStart); units.push([lineStart, lineEnd]); }
    else { if (STRUCTURAL.test(line)) close(lineStart); if (unitStart < 0) unitStart = lineStart; }
    if (nl < 0) break;
    lineStart = nl + 1;
  }
  close(text.length);
  const out = [];
  for (const [us, ue] of units) {
    let from = us;
    for (const m of text.slice(us, ue).matchAll(/[.!?]+["'”’)\]]*(?=\s)/gu)) {
      const cut = us + m.index + m[0].length;
      if (ABBREV.test(text.slice(Math.max(from, cut - 8), cut))) continue;
      out.push([from, cut]); from = cut;
    }
    out.push([from, ue]);
  }
  return out.map(([s, e]) => {
    const raw = text.slice(s, e), lead = raw.length - raw.trimStart().length;
    return { start: s + lead, end: s + lead + raw.trim().length };
  }).filter(p => p.end > p.start);
}

function terms(text, start, end) {
  const found = [];
  for (const m of text.slice(start, end).matchAll(/[\p{L}\p{N}]+(?:['’]\p{L}+)?/gu)) {
    let key = m[0].toLocaleLowerCase('en').replace(/['’]s$/u, '');
    if (STOP.has(key) || (key.length < 3 && !/\p{N}/u.test(key))) continue;
    found.push({ key, start: start + m.index, end: start + m.index + m[0].length });
  }
  return found;
}

// Ordering used by both the heap and the reference greedy: larger gain wins,
// then the earlier passage.
const better = (ga, ia, gb, ib) => ga > gb || (ga === gb && ia < ib);

function gainOf(keys, weight, covered) {
  let sum = 0;
  for (const k of keys) if (!covered.has(k)) sum += weight.get(k);
  return sum / Math.sqrt(1 + keys.length);
}

// Lazy greedy maximum coverage of repeated content words. Gains only fall as
// words become covered, so a stale heap entry is an upper bound and the
// selection order equals the plain greedy order (ties go to the earlier
// passage). Once no passage adds an uncovered repeated word, the remainder
// follows source order.
export function rankPassages(spanKeys, weight) {
  const n = spanKeys.length, covered = new Set(), order = [], heap = [];
  const push = e => { heap.push(e); let i = heap.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (!better(heap[i].g, heap[i].i, heap[p].g, heap[p].i)) break; [heap[i], heap[p]] = [heap[p], heap[i]]; i = p; } };
  const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < heap.length && better(heap[l].g, heap[l].i, heap[m].g, heap[m].i)) m = l; if (r < heap.length && better(heap[r].g, heap[r].i, heap[m].g, heap[m].i)) m = r; if (m === i) break; [heap[i], heap[m]] = [heap[m], heap[i]]; i = m; } } return top; };
  for (let i = 0; i < n; i++) push({ g: gainOf(spanKeys[i], weight, covered), i });
  const taken = new Uint8Array(n);
  while (heap.length) {
    const top = pop(), g = gainOf(spanKeys[top.i], weight, covered);
    if (g <= 0) continue; // a passage that adds nothing never gains again
    if (heap.length && !better(g, top.i, heap[0].g, heap[0].i)) { push({ g, i: top.i }); continue; }
    order.push(top.i); taken[top.i] = 1;
    for (const k of spanKeys[top.i]) covered.add(k);
  }
  for (let i = 0; i < n; i++) if (!taken[i]) order.push(i);
  return order;
}

// Each depth takes the longest prefix of the ranking that fits a share of the
// text's characters (at least one passage), so every depth contains the one
// above it. Depths that would repeat a smaller one are dropped.
const BUDGETS = [[0, 'headline'], [0.1, 'brief'], [0.25, 'summary'], [0.5, 'half'], [1, 'full text']];
function depthCounts(lengths) {
  const total = lengths.reduce((a, b) => a + b, 0), out = [];
  if (!lengths.length) return out;
  for (const [share, label] of BUDGETS) {
    let count = 0, used = 0;
    while (count < lengths.length && (count === 0 || used + lengths[count] <= share * total)) used += lengths[count++];
    if (share === 1) count = lengths.length;
    if (out.length && out[out.length - 1].count === count) { if (share === 1) out[out.length - 1].label = label; continue; }
    out.push({ count, label });
  }
  return out;
}

export function distill(text) {
  const spans = passages(text);
  const spanTerms = spans.map(s => terms(text, s.start, s.end));
  const spanKeys = spanTerms.map(list => [...new Set(list.map(t => t.key))].sort());
  // A word's weight is the number of further passages that repeat it, so a
  // word used once adds nothing and the most repeated words count most.
  const df = new Map(), first = new Map();
  spanKeys.forEach(keys => keys.forEach(k => df.set(k, (df.get(k) || 0) + 1)));
  spanTerms.forEach(list => list.forEach(t => { if (!first.has(t.key)) first.set(t.key, t); }));
  const weight = new Map([...df].map(([k, c]) => [k, c - 1]));
  const order = rankPassages(spanKeys, weight);
  const rank = new Int32Array(spans.length); order.forEach((idx, r) => { rank[idx] = r; });
  const passage = (s, index) => ({ index, start: s.start, end: s.end, text: text.slice(s.start, s.end) });
  const depths = depthCounts(order.map(i => spans[i].end - spans[i].start)).map(({ count, label }) => {
    const kept = [], left = [];
    spans.forEach((s, i) => (rank[i] < count ? kept : left).push(passage(s, i)));
    return { label, count, passages: kept, left_out: left, chars: kept.reduce((a, p) => a + p.end - p.start, 0) };
  });
  const tags = [...df].filter(([, c]) => c >= 2)
    .sort((a, b) => b[1] - a[1] || first.get(a[0]).start - first.get(b[0]).start)
    .slice(0, 8)
    .map(([k, c]) => { const t = first.get(k); return { text: text.slice(t.start, t.end), start: t.start, end: t.end, passages: c }; });
  return { method: DISTILLER_ID, offset_unit: 'utf16-code-unit', scope: DISTILL_SCOPE, passage_count: spans.length,
    total_chars: spans.reduce((a, s) => a + s.end - s.start, 0), tags, depths };
}
