// Scores critical-detail flags against the frozen corpus, under the rules in PREREG.md.
// Usage: node fixtures/detail_flags_eval/score.mjs <corpus.json> <flags.json>
// flags.json maps each pair id to an array of flags:
//   { category, kind, source: {start, end} | null, output: {start, end} | null }
// Offsets count UTF-16 code units, as String.prototype.indexOf does.
import { readFileSync } from 'node:fs';

export const MAX_FLAG_SPAN = 80;
export const CATEGORIES = ['number', 'date', 'negation', 'modal', 'condition', 'exception', 'name'];

function onlySpan(text, detail, where) {
  const start = text.indexOf(detail);
  if (start < 0 || text.indexOf(detail, start + 1) >= 0) throw new Error(`${where}: detail must occur exactly once: ${JSON.stringify(detail)}`);
  return { start, end: start + detail.length };
}

const overlaps = (span, target) => Boolean(span) && span.end - span.start <= MAX_FLAG_SPAN && span.start < target.end && target.start < span.end;

function credited(flag, pair, src, out) {
  const t = pair.change.type;
  return ((t === 'changed' || t === 'dropped') && overlaps(flag.source, src)) || ((t === 'changed' || t === 'added') && overlaps(flag.output, out));
}

function checkSpan(span, where) {
  if (span === null) return;
  const extra = Object.keys(span || {}).filter(k => !['start', 'end', 'text'].includes(k));
  if (!span || typeof span !== 'object' || !Number.isInteger(span.start) || !Number.isInteger(span.end) || span.start < 0 || span.end < span.start || extra.length) throw new Error(`${where}: a span is {start, end} with optional text, or null`);
}

// Refuses malformed input instead of scoring it as a miss: under the run-once rule a field-name slip
// would otherwise be committed as a result.
export function checkFlags(corpus, flags) {
  const ids = new Set(corpus.map(p => p.id));
  if (!flags || typeof flags !== 'object' || Array.isArray(flags)) throw new Error('flags must be an object keyed by pair id');
  for (const [id, list] of Object.entries(flags)) {
    if (!ids.has(id)) throw new Error(`unknown pair id in flags: ${id}`);
    if (!Array.isArray(list)) throw new Error(`${id}: flags must be an array`);
    list.forEach((f, i) => {
      const where = `${id} flag ${i}`;
      const extra = Object.keys(f || {}).filter(k => !['category', 'kind', 'source', 'output'].includes(k));
      if (!f || typeof f !== 'object' || extra.length || !('source' in f) || !('output' in f)) throw new Error(`${where}: fields are exactly category, kind, source, output`);
      if (!CATEGORIES.includes(f.category)) throw new Error(`${where}: unknown category ${f.category}`);
      if (typeof f.kind !== 'string') throw new Error(`${where}: kind must be a string`);
      checkSpan(f.source, where); checkSpan(f.output, where);
      if (!f.source && !f.output) throw new Error(`${where}: at least one span is required`);
    });
  }
  for (const id of ids) if (!(id in flags)) throw new Error(`missing flags entry for pair ${id} (use [] for none)`);
}

export function score(corpus, flags) {
  checkFlags(corpus, flags);
  const altered = corpus.filter(p => p.label === 'altered');
  const preserving = corpus.filter(p => p.label === 'preserving');
  const tally = () => ({ n: 0, hit: 0 });
  const byCategory = Object.fromEntries(CATEGORIES.map(c => [c, tally()]));
  const byType = { changed: tally(), dropped: tally(), added: tally() };
  let hits = 0, categoryHits = 0, alteredFlags = 0, alarms = 0, preservingFlags = 0;
  for (const pair of altered) {
    const list = flags[pair.id] || [];
    const { type, category, source_detail, rewrite_detail } = pair.change;
    const src = type === 'added' ? null : onlySpan(pair.original, source_detail, `${pair.id} original`);
    const out = type === 'dropped' ? null : onlySpan(pair.rewrite, rewrite_detail, `${pair.id} rewrite`);
    const good = list.filter(f => credited(f, pair, src, out));
    byCategory[category].n++; byType[type].n++;
    if (good.length) { hits++; byCategory[category].hit++; byType[type].hit++; }
    if (good.some(f => f.category === category)) categoryHits++;
    alteredFlags += list.length;
  }
  for (const pair of preserving) {
    const list = flags[pair.id] || [];
    if (list.length) alarms++;
    preservingFlags += list.length;
  }
  const recall = hits / altered.length, falseAlarm = alarms / preserving.length;
  return {
    altered: altered.length, preserving: preserving.length,
    recall: { hit: hits, n: altered.length, rate: recall },
    false_alarm: { flagged: alarms, n: preserving.length, rate: falseAlarm },
    category_matched_recall: { hit: categoryHits, n: altered.length },
    by_category: byCategory, by_type: byType,
    mean_flags_per_altered_pair: alteredFlags / altered.length,
    mean_flags_per_preserving_pair: preservingFlags / preserving.length,
    decision: recall >= 0.5 && falseAlarm <= 0.5 ? 'ship-flags' : 'neutral-detail-table',
  };
}

if (process.argv[1] && import.meta.url.endsWith(process.argv[1].split('/').pop())) {
  const [corpusPath, flagsPath] = process.argv.slice(2);
  if (!corpusPath || !flagsPath) { console.error('usage: node score.mjs <corpus.json> <flags.json>'); process.exit(2); }
  const corpus = JSON.parse(readFileSync(corpusPath, 'utf8'));
  const flags = JSON.parse(readFileSync(flagsPath, 'utf8'));
  console.log(JSON.stringify(score(corpus.pairs || corpus, flags), null, 2));
}
