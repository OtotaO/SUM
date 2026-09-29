// Literal change evidence for the review page. Pure functions, no DOM.
//
// Given the two texts and a review (compareTexts output, or a checked packet's
// review), this module works out, per passage pair, which characters differ
// and what scoped word comparisons establish about them. Every sentence it
// produces is meant to be literally true of the exact characters pasted:
// strings are quoted exactly, word comparisons use explicitly normalized
// token sequences, and anything a check cannot
// establish is left unsaid. Nothing reads for meaning, weights, ranks or
// scores. Passages come from review.rows only; this module never re-splits
// a text, so the evidence always matches the packet's own passages.
//
// Offsets are UTF-16 code units, end-exclusive and absolute in the full text,
// the same unit as textarea.setSelectionRange and the review packet.
// Normalized words lower case with toLocaleLowerCase('en'), replace curly
// apostrophes with straight ones, and ignore gaps between tokens. This is
// neither Unicode full case folding nor a claim about grammatical meaning.

// ---------------------------------------------------------------- built-in examples
// Both examples avoid decimals and abbreviations, so the frozen
// literal-spans-v1 splitter and literal-spans-v2 give identical passages.
export const EXAMPLES = {
  lease: {
    label: 'Lease clause',
    name: 'lease clause',
    blurb: 'A lease clause and an AI rewrite of it, loaded in both boxes. Replace either text with your own.',
    source: 'Alice may cancel the lease with 30 days notice. The deposit is refundable unless rent is overdue.',
    output: 'Alice can cancel the lease. The deposit is refundable.',
  },
  refund: {
    label: 'Refund policy',
    name: 'refund policy',
    blurb: 'A store refund policy and an AI "plain language" rewrite of it, loaded in both boxes. Replace either text with your own.',
    source: 'Customers may return unopened items within 30 days of delivery for a full refund. Opened items are not refundable unless they arrive damaged. Refunds are issued to the original payment method within 10 business days. Shipping fees of $8 are not refunded, except where the return is caused by our error. Orders placed before March 1, 2026 follow the previous policy. Northwind Outfitters must approve any return over $200.',
    output: 'You can return unopened items within 60 days for a full refund. Opened items are refundable if they arrive damaged. Refunds usually go back to your card within a few business days. Shipping fees are not refunded. Orders placed before March 2026 follow the old policy. Large returns need approval from Northwind.',
  },
};

// ---------------------------------------------------------------- tokens
// A word: letters, marks and digits, joined by one of . , : / ' ’ - (so 1,000
// 7.95 $200 50% 2026-03-01 10:30 don't e-mail stay whole). A sentence-final
// "." is never part of a word. Chinese and Japanese characters (Han, Hiragana,
// Katakana) are one word each, because those scripts do not put spaces
// between words.
const CJK = '\\p{Script=Han}\\p{Script=Hiragana}\\p{Script=Katakana}';
const WORD_CHAR = `(?:(?![${CJK}])[\\p{L}\\p{N}][\\p{M}]*)`;
export const TOKEN_RE = new RegExp(`[${CJK}]|[$€£¥]?${WORD_CHAR}+(?:[.,:\\/'’\\-]${WORD_CHAR}+)*%?`, 'gu');
export const keyOf = t => t.toLocaleLowerCase('en').replace(/’/g, "'");
export function tokenize(text, base = 0) {
  return Array.from(text.matchAll(TOKEN_RE), m => ({ t: m[0], k: keyOf(m[0]), s: base + m.index, e: base + m.index + m[0].length }));
}
// Pointwise lowercase helper retained for callers; word matching uses keyOf.
export const lowerChars = text => Array.from(text, c => c.toLocaleLowerCase('en')).join('');


// ---------------------------------------------------------------- word lists
export const STOP = new Set(['a', 'an', 'the', 'this', 'that', 'these', 'those', 'it', 'its', 'i', 'you', 'your', 'yours', 'he', 'him', 'his',
  'she', 'her', 'hers', 'we', 'us', 'our', 'ours', 'they', 'them', 'their', 'theirs', 'who', 'whom', 'whose', 'which', 'what',
  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am', 'has', 'have', 'had', 'do', 'does', 'did',
  'of', 'to', 'in', 'on', 'at', 'for', 'with', 'as', 'into', 'onto', 'than', 'then', 'there', 'here', 'so', 'such', 'very', 'just', 'also', 'too']);
export const MODALS = new Set(['may', 'might', 'can', 'could', 'must', 'shall', 'should', 'will', 'would', 'ought']);
export const NEGATIONS = new Set(['not', 'no', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'neither', 'nor', 'without', 'cannot']);
const isNegation = k => NEGATIONS.has(k) || k.endsWith("n't");
export const CONDITION_MARKERS = [ // longest first; each opens a clause
  'if and only if', 'in the event that', 'on condition that', 'unless and until', 'in the event of',
  'provided that', 'providing that', 'subject to', 'as long as', 'so long as', 'except that', 'except for', 'except where',
  'except when', 'except if', 'only if', 'only when', 'only after', 'only before', 'other than',
  'notwithstanding', 'unless', 'except', 'until', 'if'];
export const SINGLE_TOKEN_MARKERS = new Set(['only']); // marked alone, no clause
export const EXCEPTION_MARKERS = new Set(['unless', 'unless and until', 'except', 'except that', 'except for', 'except where',
  'except when', 'except if', 'other than', 'notwithstanding']);
export const CLAUSE_MAX_WORDS = 40; // a clause stops at , ; : ( ) or the passage end, and after at most 40 words
const MONTHS = new Map(Object.entries({ january: 1, jan: 1, february: 2, feb: 2, march: 3, mar: 3, april: 4, apr: 4, may: 5, june: 6, jun: 6,
  july: 7, jul: 7, august: 8, aug: 8, september: 9, sep: 9, sept: 9, october: 10, oct: 10, november: 11, nov: 11, december: 12, dec: 12 }));
const WEEKDAYS = new Set(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']);
export const NUMBER_WORDS = new Set(['zero', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
  'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty',
  'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million', 'billion', 'dozen', 'half', 'once', 'twice', 'thrice', 'double', 'triple']);
export const TIME_UNITS = new Set(['second', 'seconds', 'minute', 'minutes', 'hour', 'hours', 'day', 'days', 'week', 'weeks',
  'fortnight', 'fortnights', 'month', 'months', 'year', 'years']);
export const DURATION_QUALIFIERS = new Set(['business', 'calendar', 'working', 'full', 'consecutive', 'additional', 'more', 'further']);
export const QUANTITY_UNITS = new Set(['mg', 'mcg', 'g', 'kg', 'ml', 'l', 'oz', 'lb', 'lbs', 'km', 'm', 'cm', 'mm', 'mi', 'mile', 'miles',
  'percent', '%', 'dollar', 'dollars', 'euro', 'euros', 'pound', 'pounds', 'cent', 'cents', 'time', 'times', 'tablet', 'tablets',
  'capsule', 'capsules', 'pill', 'pills', 'dose', 'doses', 'drop', 'drops', 'puff', 'puffs', 'unit', 'units', 'item', 'items',
  'people', 'persons', 'copies', 'pages', 'words', 'degrees', 'calories']);
export const COMPARATORS = ['no more than', 'no less than', 'not more than', 'not less than', 'at least', 'at most', 'up to',
  'more than', 'less than', 'fewer than', 'over', 'under', 'above', 'below', 'exceeding', 'approximately', 'about', 'around',
  'exactly', 'maximum', 'minimum'];
const FREQUENCY_WORDS = new Set(['once', 'twice', 'thrice', 'times', 'per', 'every']);
export const SENTENCE_STARTERS = new Set(['all', 'any', 'each', 'every', 'some', 'most', 'many', 'much', 'more', 'less', 'other', 'another',
  'both', 'either', 'same', 'new', 'old', 'large', 'small', 'big', 'little', 'long', 'short', 'high', 'low', 'full', 'first', 'last',
  'next', 'please', 'note', 'see', 'take', 'give', 'keep', 'make', 'call', 'use', 'stop', 'start', 'go', 'get', 'let', 'ask', 'tell',
  'read', 'send', 'return', 'pay', 'check', 'contact', 'visit', 'avoid', 'allow', 'add', 'apply', 'remove', 'follow', 'include',
  'provide', 'store', 'wash', 'and', 'or', 'but', 'yet', 'because', 'since', 'although', 'though', 'while', 'when', 'where', 'how',
  'why', 'after', 'before', 'during', 'within', 'upon', 'between', 'among', 'through', 'however', 'therefore', 'thus', 'again',
  'today', 'tomorrow', 'yesterday', 'now', 'yes', 'children', 'adults', 'everyone', 'anyone', 'someone', 'people', 'there', 'here',
  'from', 'by', 'about', 'under', 'over', 'above', 'below', 'one', 'once', 'online', 'free', 'late', 'early']);
const NAME_CONNECTORS = new Set(['of', 'de', 'van', 'von', 'der', 'la', 'du', '&']);
const commonWord = k => STOP.has(k) || MODALS.has(k) || isNegation(k) || NUMBER_WORDS.has(k) || SENTENCE_STARTERS.has(k) ||
  SINGLE_TOKEN_MARKERS.has(k) || CONDITION_MARKERS.some(m => m.split(' ')[0] === k);
const isCapitalized = t => /^\p{Lu}/u.test(t.t) && !/^\p{Lu}$/u.test(t.t);
const byFirstWord = list => {
  const map = new Map();
  for (const phrase of list) { const words = phrase.split(' '); if (!map.has(words[0])) map.set(words[0], []); map.get(words[0]).push(words); }
  return map;
};
const MARKERS_BY_FIRST = byFirstWord(CONDITION_MARKERS);
const COMPARATORS_BY_FIRST = byFirstWord(COMPARATORS);

// ---------------------------------------------------------------- item detection
// ctx: { text: the full text of this side,
//        capsElsewhere: token texts capitalized at a non-first position in either text,
//        lowerAnywhere: keys of tokens written in lowercase anywhere in either text }
// Linear in the passage length: clause ends are precomputed and clauses are capped.
export function detectItems(passageText, passageStart, ctx, toks = tokenize(passageText, passageStart)) {
  const passageEnd = passageStart + passageText.length;
  const gap = i => ctx.text.slice(toks[i].e, i + 1 < toks.length ? toks[i + 1].s : passageEnd);
  const singleSpaces = i => /^ +$/.test(gap(i));
  const phraseAt = (i, map) => {
    const options = map.get(toks[i].k);
    if (!options) return null;
    for (const words of options) {
      if (words.every((w, n) => toks[i + n] && toks[i + n].k === w && (n === 0 || singleSpaces(i + n - 1)))) return words;
    }
    return null;
  };
  const seq = (i, words) => words.every((w, n) => toks[i + n] && toks[i + n].k === w && (n === 0 || singleSpaces(i + n - 1)));
  const items = [];
  const consumed = new Uint8Array(toks.length);
  const make = (kind, i, j, extra = {}) => {
    const s = toks[i].s, e = toks[j - 1].e;
    return { kind, ti: i, tj: j, s, e, text: ctx.text.slice(s, e), key: toks.slice(i, j).map(t => t.k).join(' '), ...extra };
  };
  const breaksAfter = toks.map((_, i) => /[,;:()]/.test(gap(i)));

  // Pass 1: conditions and exceptions. Items may overlap later items; only the markers are consumed.
  for (let i = 0; i < toks.length; i++) {
    const words = phraseAt(i, MARKERS_BY_FIRST);
    if (!words) continue;
    const n = words.length;
    let j = i + n; // the clause runs to the last token before , ; : ( ) or the passage end, capped
    while (j < toks.length && !breaksAfter[j - 1] && j - (i + n) < CLAUSE_MAX_WORDS) j++;
    for (let x = i; x < i + n; x++) consumed[x] = 1;
    items.push(make('condition', i, j, { marker: words.join(' ') }));
    i += n - 1;
  }

  const isDay = t => t && /^\d{1,2}(st|nd|rd|th)?$/.test(t.k);
  const isYear = t => t && /^(1[5-9]|2[01])\d{2}$/.test(t.k);
  const isMonth = t => t && MONTHS.has(t.k) && /^\p{Lu}/u.test(t.t);
  const dateAt = i => {
    const t = toks[i];
    if (/^\d{4}-\d{2}-\d{2}$/.test(t.k) || /^\d{1,2}\/\d{1,2}\/\d{2,4}$/.test(t.k)) return i + 1;
    if (isMonth(t)) {
      if (isDay(toks[i + 1]) && /^ $/.test(gap(i))) {
        if (isYear(toks[i + 2]) && /^,? $/.test(gap(i + 1))) return i + 3;
        return i + 2;
      }
      if (isYear(toks[i + 1]) && /^,? $/.test(gap(i))) return i + 2;
      if (t.k !== 'may' && i > 0) return i + 1; // a capitalized month name alone, never "May"
    }
    if (isDay(t) && isMonth(toks[i + 1]) && /^ $/.test(gap(i))) return isYear(toks[i + 2]) && /^,? $/.test(gap(i + 1)) ? i + 3 : i + 2;
    if (WEEKDAYS.has(t.k) && /^\p{Lu}/u.test(t.t)) return i + 1;
    return 0;
  };
  const numericTok = t => t && /\p{N}/u.test(t.k);
  const quantityAt = i => { // index after the quantity words, or 0
    const t = toks[i];
    if (!t) return 0;
    if (numericTok(t) || NUMBER_WORDS.has(t.k) || t.k === 'several') return i + 1;
    if (t.k === 'one') return i + 1;
    if (seq(i, ['a', 'couple', 'of'])) return i + 3;
    if (seq(i, ['a', 'few'])) return i + 2;
    if ((t.k === 'a' || t.k === 'an') && !(i > 0 && FREQUENCY_WORDS.has(toks[i - 1].k))) return i + 1;
    return 0;
  };
  const durationAt = i => {
    const t = toks[i];
    if (/^\d+(\.\d+)?-(second|minute|hour|day|week|month|year)s?$/.test(t.k)) return i + 1;
    const q = quantityAt(i);
    if (!q) return 0;
    let j = q;
    for (let n = 0; n < 2 && toks[j] && DURATION_QUALIFIERS.has(toks[j].k); n++) j++;
    return toks[j] && TIME_UNITS.has(toks[j].k) ? j + 1 : 0;
  };
  const numberAt = i => {
    const t = toks[i];
    if (numericTok(t) || NUMBER_WORDS.has(t.k)) {
      return toks[i + 1] && QUANTITY_UNITS.has(toks[i + 1].k) && /^ ?$/.test(gap(i)) ? i + 2 : i + 1;
    }
    if (t.k === 'one' && toks[i + 1] && QUANTITY_UNITS.has(toks[i + 1].k)) return i + 2;
    return 0;
  };

  // Pass 2: exclusive kinds, in priority order date > duration > number > negation > modal > only > capitalized.
  for (let i = 0; i < toks.length; i++) {
    if (consumed[i]) continue;
    const t = toks[i];
    let j;
    if ((j = dateAt(i))) { items.push(make('date', i, j)); consumed.fill(1, i, j); i = j - 1; continue; }
    const comp = phraseAt(i, COMPARATORS_BY_FIRST);
    const c0 = comp ? i + comp.length : i;
    if (toks[c0] && !consumed[c0]) {
      if ((j = durationAt(c0))) { items.push(make('duration', i, j)); consumed.fill(1, i, j); i = j - 1; continue; }
      if ((j = numberAt(c0))) { items.push(make('number', i, j)); consumed.fill(1, i, j); i = j - 1; continue; }
    }
    if (isNegation(t.k)) { items.push(make('negation', i, i + 1)); consumed[i] = 1; continue; }
    // A capitalized "May" inside a sentence is more likely a month or a name than a verb.
    if (MODALS.has(t.k) && !(i > 0 && isCapitalized(t))) { items.push(make('modal', i, i + 1)); consumed[i] = 1; continue; }
    if (SINGLE_TOKEN_MARKERS.has(t.k)) { items.push(make('condition', i, i + 1, { marker: t.k })); consumed[i] = 1; continue; }
    if (isCapitalized(t)) {
      // A run of capitalized tokens (single spaces; one connector between two capitalized tokens).
      let end = i + 1;
      while (end < toks.length && !consumed[end] && /^ $/.test(gap(end - 1))) {
        if (isCapitalized(toks[end])) { end++; continue; }
        if (NAME_CONNECTORS.has(toks[end].k) && toks[end + 1] && isCapitalized(toks[end + 1]) && /^ $/.test(gap(end))) { end += 2; continue; }
        break;
      }
      let isName = i > 0;
      if (i === 0) {
        isName = end - i >= 2 || ctx.capsElsewhere.has(t.t) ||
          (!commonWord(t.k) && !/(s|ed|ing|ly)$/.test(t.k) && !ctx.lowerAnywhere.has(t.k));
        if (!isName) end = i + 1;
      }
      if (isName) { items.push(make('name', i, end)); consumed.fill(1, i, end); i = end - 1; continue; }
    }
  }
  items.sort((a, b) => a.s - b.s || b.e - a.e);
  return { toks, items };
}

// ---------------------------------------------------------------- word diff
// Small pairs use an LCS table; larger pairs use Myers' O((N+M)D) algorithm,
// which gives up (returns null) once D, the number of word insertions plus
// deletions, exceeds a work-bounded limit. Within a changed stretch,
// deletions come before insertions. Hunk ids increase in reading order.
export const TABLE_MAX_CELLS = 250000;
export const MYERS_WORK = 40000000;
export const diffLimit = (n, m) => Math.max(100, Math.min(2000, Math.floor(MYERS_WORK / Math.max(1, n + m))));

function numberHunks(raw) {
  // raw: [{op:'eq',a,b}|{op:'del',a}|{op:'ins',b}] in order; reorder each changed
  // stretch to deletions first, then number the stretches.
  const ops = [];
  let hunk = 0;
  for (let k = 0; k < raw.length;) {
    if (raw[k].op === 'eq') { ops.push(raw[k++]); continue; }
    hunk++;
    const dels = [], ins = [];
    while (k < raw.length && raw[k].op !== 'eq') (raw[k].op === 'del' ? dels : ins).push(raw[k++]);
    for (const o of dels) ops.push({ op: 'del', a: o.a, hunk });
    for (const o of ins) ops.push({ op: 'ins', b: o.b, hunk });
  }
  return ops;
}

function tableDiff(a, b) {
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    dp[i][j] = a[i].k === b[j].k ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const raw = [];
  let i = 0, j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && a[i].k === b[j].k) { raw.push({ op: 'eq', a: i++, b: j++ }); continue; }
    if (j >= m || (i < n && dp[i + 1][j] >= dp[i][j + 1])) raw.push({ op: 'del', a: i++ });
    else raw.push({ op: 'ins', b: j++ });
  }
  return numberHunks(raw);
}

function myersDiff(a, b, limit) {
  const n = a.length, m = b.length, off = limit + 1;
  const v = new Int32Array(2 * limit + 3);
  const trace = [];
  for (let d = 0; d <= limit; d++) {
    trace.push(v.slice(off - d - 1, off + d + 2));
    for (let k = -d; k <= d; k += 2) {
      let x = (k === -d || (k !== d && v[off + k - 1] < v[off + k + 1])) ? v[off + k + 1] : v[off + k - 1] + 1;
      let y = x - k;
      while (x < n && y < m && a[x].k === b[y].k) { x++; y++; }
      v[off + k] = x;
      if (x === n && y === m) {
        // Walk back through the saved frontiers.
        const raw = [];
        let cx = n, cy = m;
        for (let dd = d; dd > 0; dd--) {
          const pv = trace[dd]; const po = dd + 1; // pv[po + k] = frontier before step dd
          const kk = cx - cy;
          const down = kk === -dd || (kk !== dd && pv[po + kk - 1] < pv[po + kk + 1]);
          const pk = down ? kk + 1 : kk - 1;
          const px = pv[po + pk], py = px - pk;
          while (cx > (down ? px : px + 1) && cy > (down ? py + 1 : py)) raw.push({ op: 'eq', a: --cx, b: --cy });
          if (down) raw.push({ op: 'ins', b: --cy }); else raw.push({ op: 'del', a: --cx });
        }
        while (cx > 0 && cy > 0) raw.push({ op: 'eq', a: --cx, b: --cy });
        raw.reverse();
        return numberHunks(raw);
      }
    }
  }
  return null;
}

export function diffTokens(a, b) {
  if ((a.length + 1) * (b.length + 1) <= TABLE_MAX_CELLS) return { ops: tableDiff(a, b), limit: null };
  const limit = diffLimit(a.length, b.length);
  return { ops: myersDiff(a, b, limit), limit };
}

// ---------------------------------------------------------------- word indexes
// Non-overlapping occurrences of a word sequence, found through an index of
// first words and memoized by the sequence, so repeated questions stay cheap.
function wordIndex(toks) {
  const first = new Map();
  toks.forEach((t, i) => { if (!first.has(t.k)) first.set(t.k, []); first.get(t.k).push(i); });
  return { toks, first, memo: new Map() };
}
function occurrences(index, keys) {
  const id = keys.join(' ');
  if (index.memo.has(id)) return index.memo.get(id);
  const found = [];
  let lastEnd = -1;
  for (const i of index.first.get(keys[0]) || []) {
    if (i < lastEnd || i + keys.length > index.toks.length) continue;
    let ok = true;
    for (let n = 1; n < keys.length; n++) if (index.toks[i + n].k !== keys[n]) { ok = false; break; }
    if (ok) { found.push(i); lastEnd = i + keys.length; }
  }
  index.memo.set(id, found);
  return found;
}
const letterOf = i => { let s = ''; for (let n = i + 1; n > 0; n = Math.floor((n - 1) / 26)) s = String.fromCharCode(97 + ((n - 1) % 26)) + s; return s; };
const passageNumber = spanId => String(spanId).slice(1);
const commonPrefix = (x, y) => { let i = 0; while (i < x.length && i < y.length && x[i] === y[i]) i++; return i; };
const commonSuffix = (x, y, max) => { let i = 0; while (i < max && x[x.length - 1 - i] === y[y.length - 1 - i]) i++; return i; };
// Grapheme boundaries of a short string, so a difference never splits a
// character that is drawn as one (a surrogate pair, a letter and its accent,
// an emoji sequence).
const SEGMENTER = typeof Intl !== 'undefined' && Intl.Segmenter ? new Intl.Segmenter('en', { granularity: 'grapheme' }) : null;
function boundaries(str) {
  const set = new Set([0, str.length]);
  if (SEGMENTER) { for (const g of SEGMENTER.segment(str)) set.add(g.index); return set; }
  for (let i = 1; i < str.length; i++) {
    const c = str.charCodeAt(i);
    if (!(c >= 0xdc00 && c <= 0xdfff) && !/\p{M}/u.test(str[i])) set.add(i);
  }
  return set;
}
// The shared start and end of two strings, cut only at character boundaries of both.
export function sharedEnds(x, y) {
  const bx = boundaries(x), by = boundaries(y);
  let p = commonPrefix(x, y);
  while (p > 0 && !(bx.has(p) && by.has(p))) p--;
  let q = commonSuffix(x, y, Math.min(x.length, y.length) - p);
  while (q > 0 && !(bx.has(x.length - q) && by.has(y.length - q))) q--;
  return { p, q };
}

// ---------------------------------------------------------------- evidence per displayed passage
// review: { rows: [{id, kind, source, output, decision}] } as returned by
// compareTexts or carried by a checked packet. Returns { passages, summary }
// in display order. Each passage: { row, kind, number, a, b, notes, inBoth,
// alsoMarked, ops, aToks, bToks, sameWords, over }.
//
// Note states retain the display schema: 'a-only', 'b-only', 'differs',
// 'other', and 'moved'. The last means matched normalized words outside the
// alignment, not proof that a writer moved them. `inBoth` is aligned words.
// Word scopes are explicit: 'nowhere' means no normalized token sequence in
// the entire other text (not substring absence); 'other' is a found sequence
// that may cross passages; 'count' is a paired-passage count; 'passage' is
// paired-passage absence; 'unpaired' makes no other-text claim. 'here' only
// identifies marked characters. These are lexical checks, not grammar.
export function buildEvidence(source, output, review) {
  const rows = Array.isArray(review?.rows) ? review.rows : [];
  const byStart = (x, y) => x.start - y.start;
  // Every original passage appears in exactly one row, and so does every
  // rewrite passage, so the rows carry both complete passage lists.
  const originals = rows.filter(r => r.source).map(r => r.source).sort(byStart);
  const rewrites = rows.filter(r => r.output).map(r => r.output).sort(byStart);
  const passTokens = new Map();
  const tokensOf = span => { if (!passTokens.has(span)) passTokens.set(span, tokenize(span.text, span.start)); return passTokens.get(span); };
  const aPass = originals.map(span => ({ span, toks: tokensOf(span) }));
  const bPass = rewrites.map(span => ({ span, toks: tokensOf(span) }));
  const allA = aPass.flatMap(p => p.toks), allB = bPass.flatMap(p => p.toks);
  const aFirst = new Set(originals.map(s => s.start)), bFirst = new Set(rewrites.map(s => s.start));
  const capsElsewhere = new Set([...allA.filter(t => !aFirst.has(t.s)), ...allB.filter(t => !bFirst.has(t.s))].filter(isCapitalized).map(t => t.t));
  const lowerAnywhere = new Set([...allA, ...allB].filter(t => t.t === t.t.toLocaleLowerCase('en')).map(t => t.k));
  const ctxA = { text: source, capsElsewhere, lowerAnywhere }, ctxB = { text: output, capsElsewhere, lowerAnywhere };

  // Whole-text normalized word indexes. Matches may cross passage boundaries.
  const textIndex = (passes, all) => {
    const index = wordIndex(all), passageOf = new Int32Array(all.length);
    let at = 0;
    passes.forEach((p, pi) => { for (let x = 0; x < p.toks.length; x++) passageOf[at++] = pi; });
    return { index, passageOf, passes };
  };
  const sideA = { text: source, all: allA, words: textIndex(aPass, allA) };
  const sideB = { text: output, all: allB, words: textIndex(bPass, allB) };
  // First whole-word occurrence of `keys` in the other text outside passage `exceptId`.
  const wordElsewhere = (side, keys, exceptId) => {
    const w = side.words;
    for (const i of occurrences(w.index, keys)) {
      const p = w.passageOf[i];
      const last = w.passageOf[i + keys.length - 1];
      if (p === last && w.passes[p].span.id === exceptId) continue;
      return { passage: w.passes[p].span.id, lastPassage: w.passes[last].span.id, s: side.all[i].s, e: side.all[i + keys.length - 1].e, text: side.text.slice(side.all[i].s, side.all[i + keys.length - 1].e) };
    }
    return null;
  };

  const out = [];
  for (const row of rows) {
    const entry = { row, kind: row.kind, a: row.source, b: row.output, notes: [], inBoth: [], alsoMarked: [], ops: null,
      aToks: row.source ? tokensOf(row.source) : [], bToks: row.output ? tokensOf(row.output) : [], sameWords: false, over: null };
    out.push(entry);
    if (row.kind === 'verbatim') continue;
    const A = row.source ? detectItems(row.source.text, row.source.start, ctxA, entry.aToks) : { toks: [], items: [] };
    const B = row.output ? detectItems(row.output.text, row.output.start, ctxB, entry.bToks) : { toks: [], items: [] };
    const paired = Boolean(row.source && row.output);
    let ops = null;
    if (paired) {
      const diff = diffTokens(A.toks, B.toks);
      if (!diff.ops) { entry.over = { limit: diff.limit }; continue; } // too many differences: shown whole, no notes
      ops = diff.ops;
    }
    entry.ops = ops;
    entry.sameWords = Boolean(ops) && ops.every(o => o.op === 'eq');
    const aIdx = wordIndex(A.toks), bIdx = wordIndex(B.toks);

    const aHunk = new Map(), bHunk = new Map(), aPos = new Map(), bPos = new Map();
    const bOfA = new Int32Array(A.toks.length).fill(-1), aOfB = new Int32Array(B.toks.length).fill(-1);
    if (ops) ops.forEach((o, idx) => {
      if (o.a !== undefined) { aPos.set(o.a, idx); if (o.op === 'del') aHunk.set(o.a, o.hunk); }
      if (o.b !== undefined) { bPos.set(o.b, idx); if (o.op === 'ins') bHunk.set(o.b, o.hunk); }
      if (o.op === 'eq') { bOfA[o.a] = o.b; aOfB[o.b] = o.a; }
    });
    else { A.toks.forEach((_, i) => aPos.set(i, i)); B.toks.forEach((_, i) => bPos.set(i, i)); }
    const range = it => Array.from({ length: it.tj - it.ti }, (_, n) => it.ti + n);
    const hunksOf = (it, map) => new Set(range(it).map(i => map.get(i)).filter(h => h !== undefined));
    const posOf = (it, map) => Math.min(...range(it).map(i => map.get(i)));
    const footprint = (it, map) => { const v = range(it).map(i => map.get(i)); return [Math.min(...v), Math.max(...v)]; };
    const aligned = (it, map) => {
      const first = map[it.ti];
      if (first < 0) return null;
      for (let x = it.ti + 1; x < it.tj; x++) if (map[x] !== first + (x - it.ti)) return null;
      return [first, first + (it.tj - it.ti)];
    };
    const spanOf = (toks, text, i, j) => ({ ti: i, tj: j, s: toks[i].s, e: toks[j - 1].e, text: text.slice(toks[i].s, toks[j - 1].e), key: toks.slice(i, j).map(t => t.k).join(' ') });

    // 1. In both: the item's words are aligned, in order, with the same words in the paired passage.
    if (ops) {
      const bByRange = new Map(B.items.map(y => [`${y.ti}:${y.tj}`, y]));
      const bCovered = new Uint8Array(B.toks.length);
      for (const x of A.items) {
        const r = aligned(x, bOfA);
        if (!r) continue;
        x.state = 'both';
        const twin = bByRange.get(`${r[0]}:${r[1]}`);
        if (twin && twin.kind === x.kind) twin.state = 'both';
        for (let j = r[0]; j < r[1]; j++) bCovered[j] = 1;
        entry.inBoth.push({ kind: x.kind, a: x, b: twin && twin.kind === x.kind ? twin : spanOf(B.toks, output, r[0], r[1]) });
      }
      for (const y of B.items) {
        if (y.state) continue;
        const r = aligned(y, aOfB);
        if (!r) continue;
        y.state = 'both';
        if (range(y).every(j => bCovered[j])) continue; // already on an original item's line
        entry.inBoth.push({ kind: y.kind, a: spanOf(A.toks, source, r[0], r[1]), b: y });
      }
    }
    // Words already shown as one end of a Moved note are not noted again.
    const claimed = { a: new Uint8Array(A.toks.length), b: new Uint8Array(B.toks.length) };
    const claim = (side, r) => claimed[side].fill(1, r.ti, r.tj);
    const allClaimed = (side, from, to) => { for (let i = from; i < to; i++) if (!claimed[side][i]) return false; return true; };
    const noneClaimed = (side, from, to) => { for (let i = from; i < to; i++) if (claimed[side][i]) return false; return true; };
    // 2. Moved: the same words (same kind) in both passages, not aligned.
    const ua = A.items.filter(x => !x.state), ub = B.items.filter(y => !y.state);
    if (paired) {
      const queue = new Map();
      for (const y of ub) { const id = y.kind + '|' + y.key; if (!queue.has(id)) queue.set(id, []); queue.get(id).push(y); }
      for (const x of ua) {
        const q = queue.get(x.kind + '|' + x.key);
        const y = q && q.find(c => !c.state && range(x).every(i => bOfA[i] < 0) && range(c).every(i => aOfB[i] < 0));
        if (y) { x.state = y.state = 'moved'; x.pair = y; y.pair = x; claim('a', x); claim('b', y); }
      }
    }
    // 3. Differs: same condition marker (passage level), else the same kind
    //    sharing a diff hunk or overlapping op-index footprints.
    if (paired) {
      const left = ub.filter(y => !y.state);
      const byMarker = new Map(), byHunk = new Map();
      left.forEach((y, order) => {
        y.order = order;
        if (y.kind === 'condition') { if (!byMarker.has(y.marker)) byMarker.set(y.marker, []); byMarker.get(y.marker).push(y); }
        for (const h of hunksOf(y, bHunk)) { if (!byHunk.has(h)) byHunk.set(h, []); byHunk.get(h).push(y); }
        y.fp = footprint(y, bPos);
      });
      const byStartFp = [...left].sort((p, q) => p.fp[0] - q.fp[0]);
      const maxSpan = Math.max(0, ...left.map(y => y.fp[1] - y.fp[0]));
      for (const x of ua) {
        if (x.state) continue;
        let y = null;
        if (x.kind === 'condition') y = (byMarker.get(x.marker) || []).find(b => !b.state) || null;
        if (!y) {
          const fx = footprint(x, aPos);
          const candidates = new Set();
          for (const h of hunksOf(x, aHunk)) for (const b of byHunk.get(h) || []) candidates.add(b);
          let lo = 0, hi = byStartFp.length;
          while (lo < hi) { const mid = (lo + hi) >> 1; if (byStartFp[mid].fp[0] < fx[0] - maxSpan) lo = mid + 1; else hi = mid; }
          for (let i = lo; i < byStartFp.length && byStartFp[i].fp[0] <= fx[1]; i++) if (byStartFp[i].fp[1] >= fx[0]) candidates.add(byStartFp[i]);
          for (const b of candidates) if (!b.state && b.kind === x.kind && (!y || b.order < y.order)) y = b;
        }
        if (y) { x.pair = y; y.pair = x; x.state = y.state = 'differs'; }
      }
    }
    // 4. One-sided items: say only what the texts establish.
    const structural = it => it.kind === 'negation' || it.kind === 'modal' || (it.kind === 'condition' && it.tj - it.ti === 1);
    const settle = (it, own, other, otherIdx, otherSpan, state) => {
      const keys = it.key.split(' ');
      if (otherSpan) {
        const found = occurrences(otherIdx, keys), mine = occurrences(own, keys).length;
        const ownSide = state === 'a-only' ? 'a' : 'b', otherSide = ownSide === 'a' ? 'b' : 'a';
        if (found.length >= mine && found.length) {
          // The paired passage has these words at least as often: in both, at another place.
          // Words already one end of another Moved note are covered by that note.
          if (allClaimed(ownSide, it.ti, it.tj)) { it.state = 'moved'; it.covered = true; return; }
          const map = state === 'a-only' ? bOfA : aOfB;
          const reverseMap = state === 'a-only' ? aOfB : bOfA;
          const free = j => noneClaimed(otherSide, j, j + keys.length) && keys.every((_, k) => reverseMap[j + k] < 0);
          const at = noneClaimed(ownSide, it.ti, it.tj) && range(it).every(i => map[i] < 0) ? (found.find(j => map[it.ti] !== j && free(j)) ?? found.find(free)) : undefined;
          if (at !== undefined) {
            it.state = 'moved'; it.pair = spanOf(other, state === 'a-only' ? output : source, at, at + keys.length);
            claim(ownSide, it); claim(otherSide, it.pair);
            return;
          }
        }
        // Fewer occurrences there, or each one already the end of another Moved note: state the counts.
        if (found.length) { it.state = state; it.scope = 'count'; it.counts = [mine, found.length]; return; }
        if (structural(it)) {
          it.state = state; it.scope = 'passage';
          return;
        }
      } else if (structural(it)) { it.state = state; it.scope = 'unpaired'; return; }
      const side = state === 'a-only' ? sideB : sideA;
      const hit = wordElsewhere(side, keys, otherSpan?.id);
      if (hit) { it.state = 'other'; it.scope = 'other'; it.where = hit; return; }
      it.state = state;
      it.scope = 'nowhere';
    };
    for (const x of A.items) if (!x.state) settle(x, aIdx, B.toks, bIdx, row.output, 'a-only');
    for (const y of B.items) if (!y.state) settle(y, bIdx, A.toks, aIdx, row.source, 'b-only');
    // 5. Fold items nested inside a one-sided condition clause with the same state.
    const fold = items => {
      const sorted = [...items].sort((p, q) => p.ti - q.ti);
      for (const c of sorted) {
        if (c.foldedInto || c.kind !== 'condition' || c.tj - c.ti <= 1 || !['a-only', 'b-only', 'other'].includes(c.state)) continue;
        let lo = 0, hi = sorted.length;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (sorted[mid].ti < c.ti) lo = mid + 1; else hi = mid; }
        for (let i = lo; i < sorted.length && sorted[i].ti < c.tj; i++) {
          const it = sorted[i];
          if (it !== c && !it.foldedInto && it.tj <= c.tj && it.state === c.state) { it.foldedInto = c; (c.includes ||= []).push(it); }
        }
      }
    };
    fold(A.items); fold(B.items);

    // Notes from items.
    const covered = { a: new Uint8Array(A.toks.length), b: new Uint8Array(B.toks.length) };
    A.items.forEach(it => covered.a.fill(1, it.ti, it.tj)); B.items.forEach(it => covered.b.fill(1, it.ti, it.tj));
    const oneSidedNote = (it, side) => ({ kind: it.kind, state: it.state, a: side === 'a' ? it : null, b: side === 'b' ? it : null,
      where: it.where, scope: it.scope, counts: it.counts,
      pos: posOf(it, side === 'a' ? aPos : bPos) });
    const done = new Set();
    for (const x of A.items) {
      if (x.foldedInto || x.state === 'both' || x.covered) continue;
      if (x.state === 'differs' || x.state === 'moved') {
        done.add(x.pair);
        entry.notes.push({ kind: x.kind, state: x.state, a: x, b: x.pair, pos: Math.min(posOf(x, aPos), posOf(x.pair, bPos)) });
      } else entry.notes.push(oneSidedNote(x, 'a'));
    }
    for (const y of B.items) {
      if (y.foldedInto || y.state === 'both' || y.covered || done.has(y)) continue;
      if (y.state === 'moved') entry.notes.push({ kind: y.kind, state: 'moved', a: y.pair, b: y, pos: Math.min(posOf(y.pair, aPos), posOf(y, bPos)) });
      else if (y.state !== 'differs') entry.notes.push(oneSidedNote(y, 'b'));
    }

    // Wording: per hunk, the changed words not covered by an item, split into
    // runs and trimmed of common words at each end.
    const toRun = (g, toks, text) => ({ ...spanOf(toks, text, g[0], g[g.length - 1] + 1) });
    const groupsOf = idxs => {
      const groups = [];
      for (const i of idxs) { const g = groups[groups.length - 1]; if (g && g[g.length - 1] === i - 1) g.push(i); else groups.push([i]); }
      return groups;
    };
    const trimmed = (idxs, toks, text, side, keepAlso) => {
      const kept = [];
      for (const g of groupsOf(idxs)) {
        let s = 0, e = g.length;
        while (s < e && STOP.has(toks[g[s]].k)) { if (keepAlso) entry.alsoMarked.push({ side, i: g[s], tok: toks[g[s]] }); s++; }
        const tail = [];
        while (e > s && STOP.has(toks[g[e - 1]].k)) { e--; if (keepAlso) tail.unshift({ side, i: g[e], tok: toks[g[e]] }); }
        if (s < e) kept.push(toRun(g.slice(s, e), toks, text));
        if (keepAlso) entry.alsoMarked.push(...tail);
      }
      return kept;
    };
    const runPos = (side, r) => (side === 'a' ? aPos : bPos).get(r.ti);
    const settleRun = (r, side) => {
      // One-sided wording: the same questions as for items, for a run of words.
      if (allClaimed(side, r.ti, r.tj)) return null;
      const keys = r.key.split(' ');
      const own = side === 'a' ? aIdx : bIdx, otherIdx = side === 'a' ? bIdx : aIdx;
      const otherSpan = side === 'a' ? row.output : row.source;
      const state = side === 'a' ? 'a-only' : 'b-only';
      const note = { kind: 'wording', aRuns: side === 'a' ? [r] : [], bRuns: side === 'b' ? [r] : [], pos: runPos(side, r) };
      if (otherSpan) {
        const found = occurrences(otherIdx, keys), mine = occurrences(own, keys).length;
        const otherSide = side === 'a' ? 'b' : 'a';
        // Moved only when both ends are free: a word is one end of at most one Moved note.
        const ownMap = side === 'a' ? bOfA : aOfB, otherMap = side === 'a' ? aOfB : bOfA;
        const at = found.length >= mine && noneClaimed(side, r.ti, r.tj) && range(r).every(i => ownMap[i] < 0)
          ? found.find(j => noneClaimed(otherSide, j, j + keys.length) && keys.every((_, k) => otherMap[j + k] < 0)) : undefined;
        if (at !== undefined) {
          const otherToks = side === 'a' ? B.toks : A.toks;
          const twin = spanOf(otherToks, side === 'a' ? output : source, at, at + keys.length);
          claim(side, r); claim(otherSide, twin);
          return { kind: 'wording', state: 'moved', aRuns: side === 'a' ? [r] : [twin], bRuns: side === 'b' ? [r] : [twin], pos: note.pos };
        }
        if (found.length) return { ...note, state, scope: 'count', counts: [mine, found.length] };
      }
      const other = side === 'a' ? sideB : sideA;
      const hit = wordElsewhere(other, keys, otherSpan?.id);
      if (hit) return { ...note, state: 'other', scope: 'other', where: hit };
      return { ...note, state, scope: 'nowhere' };
    };
    if (ops) {
      const hunks = new Map();
      for (const o of ops) {
        if (o.op === 'eq') continue;
        const h = hunks.get(o.hunk) || { a: [], b: [] };
        if (o.op === 'del' && !covered.a[o.a]) h.a.push(o.a);
        if (o.op === 'ins' && !covered.b[o.b]) h.b.push(o.b);
        hunks.set(o.hunk, h);
      }
      const runs = [...hunks].map(([id, h]) => ({ id, a: trimmed(h.a, A.toks, source, 'a', true), b: trimmed(h.b, B.toks, output, 'b', true) }));
      // Moved wording: the same run of words deleted in one place and inserted in another.
      const inserted = new Map();
      for (const h of runs) for (const r of h.b) { if (!inserted.has(r.key)) inserted.set(r.key, []); inserted.get(r.key).push({ h, r }); }
      for (const h of runs) {
        h.a = h.a.filter(r => {
          const twin = (inserted.get(r.key) || []).find(c => !c.used && c.h !== h);
          if (!twin) return true;
          twin.used = true;
          claim('a', r); claim('b', twin.r);
          entry.notes.push({ kind: 'wording', state: 'moved', aRuns: [r], bRuns: [twin.r], pos: Math.min(runPos('a', r), runPos('b', twin.r)) });
          return false;
        });
      }
      for (const h of runs) h.b = h.b.filter(r => !(inserted.get(r.key) || []).some(c => c.used && c.r === r));
      const oneSided = [];
      for (const h of runs) {
        if (!h.a.length && !h.b.length) continue;
        if (h.a.length && h.b.length) {
          entry.notes.push({ kind: 'wording', state: 'differs', aRuns: h.a, bRuns: h.b, pos: Math.min(...h.a.map(r => runPos('a', r)), ...h.b.map(r => runPos('b', r))) });
          continue;
        }
        const side = h.a.length ? 'a' : 'b';
        for (const r of h.a.length ? h.a : h.b) oneSided.push({ r, side });
      }
      // Longest runs first, so a moved phrase is noted once, not word by word.
      oneSided.sort((x, y) => (y.r.tj - y.r.ti) - (x.r.tj - x.r.ti) || runPos(x.side, x.r) - runPos(y.side, y.r));
      for (const { r, side } of oneSided) { const n = settleRun(r, side); if (n) entry.notes.push(n); }
      entry.alsoMarked.sort((x, y) => (x.side === 'a' ? aPos.get(x.i) : bPos.get(x.i)) - (y.side === 'a' ? aPos.get(y.i) : bPos.get(y.i)));
    } else if (!paired && !entry.notes.length) {
      // An unpartnered passage always gets a note: its uncovered words, or its characters.
      const side = row.source ? 'a' : 'b', toks = side === 'a' ? A.toks : B.toks, text = side === 'a' ? source : output;
      const idxs = toks.map((_, i) => i).filter(i => !covered[side][i]);
      let rs = trimmed(idxs, toks, text, side, false);
      if (!rs.length && idxs.length) rs = groupsOf(idxs).map(g => toRun(g, toks, text));
      for (const r of rs) { const n = settleRun(r, side); if (n) entry.notes.push(n); }
      if (!toks.length) {
        const span = side === 'a' ? row.source : row.output;
        entry.notes.push({ kind: 'characters', state: side === 'a' ? 'a-only' : 'b-only', scope: 'here', pos: 0,
          aChars: side === 'a' ? { s: span.start, e: span.end } : null, bChars: side === 'b' ? { s: span.start, e: span.end } : null });
      }
    }

    // Characters: between aligned words, and inside aligned words, the exact characters that differ.
    if (ops) {
      let curA = row.source.start, curB = row.output.start, prevEq = true;
      const charNote = (sa, ea, sb, eb, pos, kind) => {
        const ga = source.slice(sa, ea), gb = output.slice(sb, eb);
        if (ga === gb) return;
        const { p, q } = sharedEnds(ga, gb);
        const aChars = { s: sa + p, e: ea - q }, bChars = { s: sb + p, e: eb - q };
        const hasA = aChars.e > aChars.s, hasB = bChars.e > bChars.s;
        entry.notes.push({ kind: kind || 'characters', state: hasA && hasB ? 'differs' : hasA ? 'a-only' : 'b-only', scope: 'here', pos,
          aChars: hasA ? aChars : null, bChars: hasB ? bChars : null, atA: sa + p, atB: sb + p });
      };
      ops.forEach((o, idx) => {
        if (o.op === 'eq') {
          const ta = A.toks[o.a], tb = B.toks[o.b];
          if (prevEq) charNote(curA, ta.s, curB, tb.s, idx - 0.5);
          if (ta.t !== tb.t) {
            const kind = 'characters';
            entry.notes.push({ kind, state: 'differs', scope: 'here', pos: idx, aChars: { s: ta.s, e: ta.e }, bChars: { s: tb.s, e: tb.e } });
          }
          curA = ta.e; curB = tb.e; prevEq = true;
        } else {
          if (o.op === 'del') curA = A.toks[o.a].e; else curB = B.toks[o.b].e;
          prevEq = false;
        }
      });
      if (prevEq) charNote(curA, row.source.end, curB, row.output.end, ops.length);
    }
    entry.notes.sort((x, y) => x.pos - y.pos);
    entry.notes.forEach((n, i) => { n.letter = letterOf(i); });
  }

  // Display order: a rewrite-only passage sits after the passage holding the
  // rewrite passage before it, numbered k.1, k.2 (0.1 if it comes first).
  // Display only: review.rows keeps its order.
  const display = out.filter(e => e.row.kind !== 'output-unmatched');
  display.forEach(e => { e.number = passageNumber(e.a.id); });
  for (const e of out.filter(e => e.row.kind === 'output-unmatched')) {
    const q = Number(passageNumber(e.b.id));
    let at = -1;
    for (let k = q - 1; k >= 1 && at < 0; k--) at = display.findIndex(d => d.b && d.b.id === `s${k}`);
    const base = at >= 0 ? display[at].number.split('.')[0] : '0';
    let n = 1;
    while (display.some(d => d.number === `${base}.${n}`)) n++;
    e.number = `${base}.${n}`;
    display.splice(at + 1 + (n - 1), 0, e);
  }
  return { passages: display, summary: evidenceSummary(display, source, output) };
}

// ---------------------------------------------------------------- labels
// One map for the state names, so they read the same everywhere on the page.
export const STATE_ORDER = ['a-only', 'differs', 'b-only', 'other', 'moved', 'both'];
export const STATE_LABEL = { 'a-only': 'Removed', differs: 'Differs', 'b-only': 'Added', other: 'Other passage', moved: 'Matched words', both: 'In both' };
export const KIND_LABEL = { number: 'Number', date: 'Date', duration: 'Duration', negation: 'Negation-list word', modal: 'Modal-list word',
  condition: 'Marker phrase', name: 'Capitalized word', wording: 'Wording', characters: 'Characters', case: 'Characters' };
const conditionLabel = it => it.tj - it.ti === 1 ? 'Marker word' : (EXCEPTION_MARKERS.has(it.marker) ? 'Exception-marker phrase' : 'Condition-marker phrase');
export function kindLabel(note) {
  if (note.kind === 'name') return /\s/.test((note.a || note.b).text) ? 'Capitalized words' : 'Capitalized word';
  if (note.kind !== 'condition') return KIND_LABEL[note.kind];
  const a = note.a && conditionLabel(note.a), b = note.b && conditionLabel(note.b);
  return a && b && a !== b ? 'Marker phrase' : (a || b);
}
export const inBothKindLabel = ib => (ib.kind === 'name' ? (/\s/.test(ib.a.text) ? 'Capitalized words' : 'Capitalized word') : KIND_LABEL[ib.kind]);
const q = s => `“${s}”`;
const times = n => (n === 1 ? 'once' : n === 2 ? 'twice' : `${n} times`);

// How a run of characters reads: whitespace is named, anything else is quoted.
export function charPhrase(s) {
  if (/^\s+$/.test(s)) {
    const n = s.length, kinds = new Set(s);
    if (kinds.size === 1) {
      const c = s[0];
      const name = c === ' ' ? 'space' : c === '\t' ? 'tab' : c === ' ' ? 'non-breaking space' : c === '\n' ? 'line break' : 'whitespace character';
      return n === 1 ? `a ${name}` : `${n} ${name === 'whitespace character' ? 'whitespace characters' : name + 's'}`;
    }
    return `${n} whitespace characters`;
  }
  if (/[\p{Cf}\p{Cc}]/u.test(s) || /^\p{M}+$/u.test(s)) {
    const points = Array.from(s, c => `U+${c.codePointAt(0).toString(16).toUpperCase().padStart(4, '0')}`);
    return `${points.length === 1 ? 'code point' : 'code points'} ${points.join(' ')}`;
  }
  return q(s);
}

export function noteStrings(note, source = '', output = '') {
  const slice = (text, c) => (c ? text.slice(c.s, c.e) : '');
  if (note.kind === 'characters' || note.kind === 'case') return { aText: slice(source, note.aChars), bText: slice(output, note.bChars) };
  return {
    aText: note.a ? note.a.text : (note.aRuns || []).map(r => r.text).join(' … '),
    bText: note.b ? note.b.text : (note.bRuns || []).map(r => r.text).join(' … '),
  };
}

// How two strings holding the same words compare, or null when they are identical.
function sameWordsPhrase(a, b) {
  return a === b ? null : 'matching normalized words; exact characters differ';
}

// The one sentence each note says. Every statement is a literal fact about the
// exact characters of the two texts; none says whether a difference matters.
export function noteStatement(note, passage, source = '', output = '') {
  const { aText, bText } = noteStrings(note, source, output);
  const own = aText ? 'original' : 'rewrite', other = aText ? 'rewrite' : 'original';
  const str = aText || bText;
  const multi = (note.aRuns || note.bRuns || []).length > 1;
  const quoted = multi ? (note.aRuns?.length ? note.aRuns : note.bRuns).map(r => q(r.text)).join(' and ') : q(str);
  if (note.kind === 'characters' || note.kind === 'case') {
    if (note.state === 'differs') return `${charPhrase(aText)} in the original, ${charPhrase(bText)} in the rewrite.`;
    if (passage && (!passage.a || !passage.b)) return `This ${own} passage has no partner in the ${other}.`;
    return note.state === 'a-only' ? `Marked original characters: ${charPhrase(aText)}.`
      : `Marked rewrite characters: ${charPhrase(bText)}.`;
  }
  if (note.state === 'differs') {
    const quote = (single, runs) => (runs && runs.length > 1 ? runs.map(r => q(r.text)).join(' and ') : q(single));
    return `${quote(aText, note.aRuns)} in the original, ${quote(bText, note.bRuns)} in the rewrite.`;
  }
  if (note.state === 'moved') {
    const phrase = sameWordsPhrase(aText, bText);
    return phrase ? `${q(aText)} in the original and ${q(bText)} in the rewrite, ${phrase}.`
      : `${q(aText)} is matched in both passages by normalized words.`;
  }
  if (note.state === 'other') {
    const k = passageNumber(note.where.passage);
    return `${quoted} has a normalized word match in the ${other}, starting in passage ${k}: ${q(note.where.text)}.`;
  }
  if (note.scope === 'count') {
    const [mine, theirs] = note.counts;
    return `${quoted} has non-overlapping normalized word matches ${times(mine)} in the ${own} passage and ${times(theirs)} in the paired ${other} passage.`;
  }
  if (note.scope === 'unpaired') return `${quoted} is in ${own === 'original' ? 'an' : 'a'} ${own} passage that has no partner in the ${other}.`;
  if (note.scope === 'passage') return `${quoted} has no normalized word match in the paired ${other} passage.`;
  return `${quoted} has no normalized word match in the entire ${other} text.`;
}

// The line for words at the same place in both passages.
export function inBothText(ib) {
  const phrase = sameWordsPhrase(ib.a.text, ib.b.text);
  return phrase ? `${q(ib.a.text)} in the original, ${q(ib.b.text)} in the rewrite: ${phrase}.` : q(ib.a.text);
}

// The exact spans a note points at: [{side: 'a'|'b', s, e, text}].
export function noteSpans(note, source, output) {
  const spans = [];
  const add = (side, s, e) => spans.push({ side, s, e, text: (side === 'a' ? source : output).slice(s, e) });
  if (note.kind === 'characters' || note.kind === 'case') {
    if (note.aChars) add('a', note.aChars.s, note.aChars.e);
    if (note.bChars) add('b', note.bChars.s, note.bChars.e);
    return spans;
  }
  if (note.a) add('a', note.a.s, note.a.e);
  for (const r of note.aRuns || []) add('a', r.s, r.e);
  if (note.b) add('b', note.b.s, note.b.e);
  for (const r of note.bRuns || []) add('b', r.s, r.e);
  const otherSide = noteStrings(note).aText ? 'b' : 'a';
  if (note.state === 'other' && note.where) add(otherSide, note.where.s, note.where.e);
  return spans;
}

// Notes and in-both lines grouped by state, in display order (never ranked).
export function groupByState(passages) {
  const groups = Object.fromEntries(STATE_ORDER.map(s => [s, []]));
  for (const p of passages) {
    for (const n of p.notes) groups[n.state].push({ passage: p, note: n });
    p.inBoth.forEach((ib, k) => groups.both.push({ passage: p, inBoth: ib, index: k }));
  }
  return groups;
}

// ---------------------------------------------------------------- summary
// Facts about the whole pair of texts, each one computed, never inferred.
export function evidenceSummary(passages, source, output) {
  const notes = passages.reduce((s, p) => s + p.notes.length, 0);
  const paired = passages.filter(p => p.a && p.b).sort((x, y) => x.a.start - y.a.start);
  const orderDiffers = paired.some((p, i) => i > 0 && p.b.start < paired[i - 1].b.start);
  const allPaired = passages.every(p => p.a && p.b);
  let betweenDiffers = null; // known only when every passage has a partner and the order is the same
  if (allPaired && !orderDiffers && typeof source === 'string' && typeof output === 'string') {
    const gaps = (text, spans) => spans.map((sp, i) => text.slice(i ? spans[i - 1].end : 0, sp.start)).concat(text.slice(spans.length ? spans[spans.length - 1].end : 0));
    const ga = gaps(source, paired.map(p => p.a)), gb = gaps(output, paired.map(p => p.b));
    betweenDiffers = ga.length !== gb.length || ga.some((g, i) => g !== gb[i]);
  }
  return {
    passages: passages.length,
    notes,
    identicalTexts: source === output,
    alsoMarked: passages.reduce((s, p) => s + p.alsoMarked.length, 0),
    over: passages.filter(p => p.over).map(p => ({ number: p.number, limit: p.over.limit })),
    orderDiffers,
    betweenDiffers,
    pairs: passages.filter(p => p.kind === 'changed-candidate').length,
    identical: passages.filter(p => p.kind === 'verbatim').length,
    originalOnly: passages.filter(p => p.kind === 'source-unmatched').length,
    rewriteOnly: passages.filter(p => p.kind === 'output-unmatched').length,
  };
}

const plural = (n, one, many = one + 's') => `${n.toLocaleString('en-US')} ${n === 1 ? one : many}`;
export const pluralOf = plural;

// The sheet heading.
export function headingText(s) {
  const P = plural(s.passages, 'passage');
  if (s.identicalTexts) return `${P} compared, no literal differences.`;
  if (s.notes) return `${P} compared, ${plural(s.notes, 'difference')} noted.`;
  return `${P} compared; the texts are not identical.`;
}

// Sentences about what differs outside the notes, each computed.
export function summaryFacts(s) {
  const facts = [];
  if (s.identicalTexts) return ['The two texts are identical, character for character.'];
  if (s.alsoMarked) facts.push(`${plural(s.alsoMarked, 'common word')} ${s.alsoMarked === 1 ? 'is' : 'are'} also marked, without a note.`);
  if (s.orderDiffers) facts.push('The rewrite has these passages in a different order.');
  if (s.betweenDiffers) facts.push(s.notes || s.alsoMarked || s.over.length ? 'The spacing or line breaks outside the passages also differ.'
    : 'Every passage is identical and in the same order; the texts differ only in the spacing or line breaks outside the passages (before, between or after them).');
  for (const o of s.over) facts.push(`Passage ${o.number}: more than ${plural(o.limit, 'word insertion or deletion', 'word insertions and deletions')} apart, so its differences are not marked one by one.`);
  return facts;
}

// The message a passage shows above its notes, or null.
export function passageMessage(p) {
  if (p.kind === 'verbatim') return 'Identical text in both.';
  if (p.over) return `Turning the original passage into the rewrite passage takes more than ${plural(p.over.limit, 'word insertion or deletion', 'word insertions and deletions')}, so the differences are not marked one by one. Both texts are shown in full.`;
  if (p.sameWords) return 'The normalized words match in order; the characters noted here differ.';
  if (!p.notes.length && p.alsoMarked.length) return 'Only common words, and any characters marked next to them, differ here. They are marked in the passage without a note.';
  return null;
}

// ---------------------------------------------------------------- blackline
// The marked passage as plain data for the renderer. Every character of both
// passages appears exactly once, in order, on the side it belongs to:
//   {t: 'text', text}                        characters of both texts
//   {t: 'eq', text, n}                       a word of both texts (n: in-both id, or null)
//   {t: 'run', side: 'd'|'i', sub, parts}    characters of one text only
//       parts: {t: 'gap', text} (whitespace)
//            | {t: 'mark', text, n, mv, ws}  a word or other characters (mv: moved; ws: spacing that differs)
//            | {t: 'ref', n, letter}
//   {t: 'ref', n, letter}                    a note letter after its last word
//   {t: 'block', side: 'a'|'b', text}        a whole passage, unmarked
// Joining text, eq and the 'd' runs gives the original passage exactly;
// joining text, eq and the 'i' runs gives the rewrite passage exactly.
export function blacklineSegments(passage, source, output) {
  if (passage.kind === 'verbatim') return [{ t: 'text', text: passage.a.text }];
  if (passage.over) return [{ t: 'block', side: 'a', text: passage.a.text }, { t: 'block', side: 'b', text: passage.b.text }];
  const pno = passage.number;
  const A = passage.aToks, B = passage.bToks;
  const opOfA = new Int32Array(A.length), opOfB = new Int32Array(B.length);
  (passage.ops || []).forEach((o, idx) => { if (o.a !== undefined) opOfA[o.a] = idx; if (o.b !== undefined) opOfB[o.b] = idx; });
  const aNote = new Map(), bNote = new Map(), moved = { a: new Set(), b: new Set() };
  const refAt = new Map(); // "a:tokenIndex" or "b:tokenIndex" or "ca:offset"/"cb:offset" -> notes ending there
  const addRef = (key, n) => refAt.set(key, [...(refAt.get(key) || []), n]);
  for (const n of passage.notes) {
    const id = `${pno}${n.letter}`;
    if (n.kind === 'characters' || n.kind === 'case') {
      // A character note's letter follows its last character on the rewrite side, else the original side.
      if (n.bChars) addRef(`cb:${n.bChars.e}`, n); else addRef(`ca:${n.aChars.e}`, n);
      if (n.aChars) aNote.set(`c:${n.aChars.s}`, id);
      if (n.bChars) bNote.set(`c:${n.bChars.s}`, id);
      continue;
    }
    const ranges = [];
    if (n.a) ranges.push(['a', n.a.ti, n.a.tj]);
    if (n.b) ranges.push(['b', n.b.ti, n.b.tj]);
    for (const r of n.aRuns || []) ranges.push(['a', r.ti, r.tj]);
    for (const r of n.bRuns || []) ranges.push(['b', r.ti, r.tj]);
    for (const it of (n.a || n.b)?.includes || []) ranges.push([n.a ? 'a' : 'b', it.ti, it.tj]);
    let last = null, lastPos = -Infinity;
    const posOf = (side, i) => (passage.ops ? (side === 'a' ? opOfA : opOfB)[i] : i);
    for (const [side, i0, i1] of ranges) {
      for (let i = i0; i < i1; i++) { (side === 'a' ? aNote : bNote).set(i, id); if (n.state === 'moved') moved[side].add(i); }
      const p = posOf(side, i1 - 1);
      if (p > lastPos) { lastPos = p; last = `${side}:${i1 - 1}`; }
    }
    addRef(last, n);
  }
  passage.inBoth.forEach((ib, k) => {
    const id = `${pno}-both-${k}`;
    for (let i = ib.a.ti; i < ib.a.tj; i++) if (!aNote.has(i)) aNote.set(i, id);
    for (let i = ib.b.ti; i < ib.b.tj; i++) if (!bNote.has(i)) bNote.set(i, id);
  });

  const segs = [];
  let drun = null, irun = null;
  const refsFor = key => (refAt.get(key) || []).map(n => ({ t: 'ref', n: `${pno}${n.letter}`, letter: n.letter }));
  const openD = () => {
    if (!drun) {
      drun = { t: 'run', side: 'd', sub: false, parts: [] };
      const at = irun ? segs.indexOf(irun) : segs.length; // an original-side run always precedes its rewrite-side run
      segs.splice(at, 0, drun);
    }
    return drun;
  };
  const openI = () => { if (!irun) { irun = { t: 'run', side: 'i', sub: false, parts: [] }; segs.push(irun); } return irun; };
  const closeRuns = () => { if (drun) drun.sub = Boolean(irun); drun = irun = null; };
  const put = (run, text, n = null, mv = false) => {
    // Whitespace stays unmarked; any other characters are marked.
    for (const piece of text.match(/\s+|\S+/g) || []) {
      if (/^\s+$/.test(piece)) run.parts.push({ t: 'gap', text: piece });
      else run.parts.push({ t: 'mark', text: piece, n, mv });
    }
  };
  const charsInto = (run, from, to, text, side) => {
    // Characters of one side between offsets, with any character-note refs and ids.
    if (to <= from) return;
    const id = (side === 'a' ? aNote : bNote).get(`c:${from}`) || null;
    const chars = text.slice(from, to);
    // Spacing that differs is drawn as a visible mark; other spacing stays plain.
    if (id && /^\s+$/.test(chars)) run.parts.push({ t: 'mark', text: chars, n: id, mv: false, ws: true });
    else put(run, chars, id);
    run.parts.push(...refsFor(`${side === 'a' ? 'ca' : 'cb'}:${to}`));
  };
  const both = text => { if (text) segs.push({ t: 'text', text }); };

  if (!passage.ops) {
    // Unpartnered: the whole passage belongs to one side.
    const side = passage.a ? 'a' : 'b', toks = side === 'a' ? A : B, span = side === 'a' ? passage.a : passage.b, text = side === 'a' ? source : output;
    const run = side === 'a' ? openD() : openI();
    const map = side === 'a' ? aNote : bNote;
    let cur = span.start;
    for (let i = 0; i < toks.length; i++) {
      put(run, text.slice(cur, toks[i].s));
      let j = i, word = toks[i].t;
      const id = map.get(i) || null;
      while (toks[j + 1] && (map.get(j + 1) || null) === id && text.slice(toks[j].e, toks[j + 1].s) === ' ' && !refAt.has(`${side}:${j}`)) { j++; word += ' ' + toks[j].t; }
      run.parts.push({ t: 'mark', text: word, n: id, mv: false });
      run.parts.push(...refsFor(`${side}:${j}`));
      cur = toks[j].e; i = j;
    }
    charsInto(run, cur, span.end, text, side);
    if (!toks.length) run.parts.push(...refsFor(`${side === 'a' ? 'ca' : 'cb'}:${span.end}`));
    closeRuns();
    return segs;
  }

  const ops = passage.ops;
  const sharedTail = (x, y) => { const b1 = boundaries(x), b2 = boundaries(y); let t = commonSuffix(x, y, Math.min(x.length, y.length)); while (t > 0 && !(b1.has(x.length - t) && b2.has(y.length - t))) t--; return t; };
  let curA = passage.a.start, curB = passage.b.start;
  // Characters between two positions that are aligned on both sides.
  const alignedGap = (ea, eb) => {
    const ga = source.slice(curA, ea), gb = output.slice(curB, eb);
    if (drun || irun) {
      // After a one-sided run: the shared tail stays shared, the rest joins the run.
      const tail = sharedTail(ga, gb);
      if (ga.length > tail) charsInto(openD(), curA, ea - tail, source, 'a');
      if (gb.length > tail) charsInto(openI(), curB, eb - tail, output, 'b');
      closeRuns();
      both(ga.slice(ga.length - tail));
      return;
    }
    if (ga === gb) { both(ga); return; }
    const { p, q: tail } = sharedEnds(ga, gb);
    both(ga.slice(0, p));
    if (ga.length - p - tail > 0) charsInto(openD(), curA + p, ea - tail, source, 'a');
    if (gb.length - p - tail > 0) charsInto(openI(), curB + p, eb - tail, output, 'b');
    closeRuns();
    both(ga.slice(ga.length - tail));
  };
  for (let k = 0; k < ops.length; k++) {
    const o = ops[k];
    if (o.op === 'eq') {
      const ta = A[o.a], tb = B[o.b];
      alignedGap(ta.s, tb.s);
      if (ta.t === tb.t) {
        segs.push({ t: 'eq', text: tb.t, n: bNote.get(o.b) || aNote.get(o.a) || null });
      } else {
        charsInto(openD(), ta.s, ta.e, source, 'a');
        charsInto(openI(), tb.s, tb.e, output, 'b');
        closeRuns();
      }
      segs.push(...refsFor(`a:${o.a}`), ...refsFor(`b:${o.b}`));
      curA = ta.e; curB = tb.e;
      continue;
    }
    const del = o.op === 'del';
    const toks = del ? A : B, text = del ? source : output, map = del ? aNote : bNote, side = del ? 'a' : 'b';
    const run = del ? openD() : openI();
    const idx = del ? o.a : o.b;
    put(run, text.slice(del ? curA : curB, toks[idx].s));
    const id = map.get(idx) || null;
    let j = k, word = toks[idx].t, last = idx;
    // Consecutive words of the same note, one space apart, form one mark.
    while (ops[j + 1] && ops[j + 1].op === o.op && id && map.get(del ? ops[j + 1].a : ops[j + 1].b) === id &&
      text.slice(toks[last].e, toks[del ? ops[j + 1].a : ops[j + 1].b].s) === ' ' && !refAt.has(`${side}:${last}`)) {
      j++; last = del ? ops[j].a : ops[j].b; word += ' ' + toks[last].t;
    }
    run.parts.push({ t: 'mark', text: word, n: id, mv: moved[side].has(idx) });
    run.parts.push(...refsFor(`${side}:${last}`));
    if (del) curA = toks[last].e; else curB = toks[last].e;
    k = j;
  }
  alignedGap(passage.a.end, passage.b.end);
  closeRuns();
  return segs;
}

// Join a passage as one view shows it: 'a' the original, 'b' the rewrite.
export function viewText(segs, side) {
  return segs.map(s => {
    if (s.t === 'text' || s.t === 'eq') return s.text;
    if (s.t === 'block') return s.side === side ? s.text : '';
    if (s.t === 'run') return (s.side === 'd') === (side === 'a') ? s.parts.filter(p => p.t !== 'ref').map(p => p.text).join('') : '';
    return '';
  }).join('');
}
