// Literal change evidence for the review page. Pure functions, no DOM.
//
// Given the two texts and a review (compareTexts output, or a checked packet's
// review), this module lists which exact strings appear in one text and not
// the other, or in both, per passage pair. Everything here is literal string
// processing: nothing reads for meaning, weights, ranks or scores. Passages
// come from review.rows only; this module never re-splits a text, so the
// evidence always matches the packet's own passages.
//
// Offsets are UTF-16 code units, end-exclusive and absolute in the full text,
// the same unit as textarea.setSelectionRange and the review packet.

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
// Keeps 1,000 / 7.95 / $200 / 50% / 2026-03-01 / 10:30 / don't / e-mail as
// single tokens. A sentence-final "." is never part of a token.
export const TOKEN_RE = /[$€£¥]?[\p{L}\p{M}\p{N}]+(?:[.,:\/'’\-][\p{L}\p{M}\p{N}]+)*%?/gu;
export const keyOf = t => t.toLocaleLowerCase('en').replace(/’/g, "'");
export function tokenize(text, base = 0) {
  return Array.from(text.matchAll(TOKEN_RE), m => ({ t: m[0], k: keyOf(m[0]), s: base + m.index, e: base + m.index + m[0].length }));
}

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

// ---------------------------------------------------------------- item detection
// ctx: { text: the full text of this side,
//        capsElsewhere: token texts capitalized at a non-first position in either text,
//        lowerAnywhere: keys of tokens written in lowercase anywhere in either text }
export function detectItems(passageText, passageStart, ctx) {
  const toks = tokenize(passageText, passageStart);
  const gap = i => (i + 1 < toks.length ? ctx.text.slice(toks[i].e, toks[i + 1].s) : ctx.text.slice(toks[i].e, passageStart + passageText.length));
  const seq = (i, words) => words.every((w, n) => toks[i + n] && toks[i + n].k === w && (n === 0 || /^ +$/.test(gap(i + n - 1))));
  const items = [];
  const consumed = new Array(toks.length).fill(false);
  const make = (kind, i, j, extra = {}) => {
    const s = toks[i].s, e = toks[j - 1].e;
    return { kind, ti: i, tj: j, s, e, text: ctx.text.slice(s, e), key: toks.slice(i, j).map(t => t.k).join(' '), ...extra };
  };

  // Pass 1: conditions and exceptions. Items may overlap later items; only the markers are consumed.
  for (let i = 0; i < toks.length; i++) {
    const marker = CONDITION_MARKERS.find(m => seq(i, m.split(' ')));
    if (!marker) continue;
    const n = marker.split(' ').length;
    let j = i + n; // the clause runs to the last token before , ; : ( ) or to the passage end
    while (j < toks.length && !/[,;:()]/.test(gap(j - 1))) j++;
    if (j === i + n && j < toks.length && !/[,;:()]/.test(gap(j - 1))) j++;
    for (let x = i; x < i + n; x++) consumed[x] = true;
    items.push(make('condition', i, Math.max(j, i + n), { marker }));
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

  // Pass 2: exclusive kinds, in priority order date > duration > number > negation > modal > only > name.
  for (let i = 0; i < toks.length; i++) {
    if (consumed[i]) continue;
    const t = toks[i];
    let j;
    if ((j = dateAt(i))) { items.push(make('date', i, j)); consumed.fill(true, i, j); i = j - 1; continue; }
    const comp = COMPARATORS.find(c => seq(i, c.split(' ')));
    const c0 = comp ? i + comp.split(' ').length : i;
    if (toks[c0] && !consumed[c0]) {
      if ((j = durationAt(c0))) { items.push(make('duration', i, j)); consumed.fill(true, i, j); i = j - 1; continue; }
      if ((j = numberAt(c0))) { items.push(make('number', i, j)); consumed.fill(true, i, j); i = j - 1; continue; }
    }
    if (isNegation(t.k)) { items.push(make('negation', i, i + 1)); consumed[i] = true; continue; }
    if (MODALS.has(t.k)) { items.push(make('modal', i, i + 1)); consumed[i] = true; continue; }
    if (SINGLE_TOKEN_MARKERS.has(t.k)) { items.push(make('condition', i, i + 1, { marker: t.k })); consumed[i] = true; continue; }
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
      if (isName) { items.push(make('name', i, end)); consumed.fill(true, i, end); i = end - 1; continue; }
    }
  }
  items.sort((a, b) => a.s - b.s || b.e - a.e);
  return { toks, items };
}

// ---------------------------------------------------------------- word diff (LCS on token keys)
export const MAX_DIFF_TOKENS = 400;
export function diffTokens(a, b) {
  const n = a.length, m = b.length;
  if (n > MAX_DIFF_TOKENS || m > MAX_DIFF_TOKENS) return null; // the page says the passage is too long for word marks
  const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    dp[i][j] = a[i].k === b[j].k ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const ops = [];
  let i = 0, j = 0, hunk = 0, inChange = false;
  while (i < n || j < m) {
    if (i < n && j < m && a[i].k === b[j].k) { ops.push({ op: 'eq', a: i++, b: j++ }); inChange = false; continue; }
    if (!inChange) { hunk++; inChange = true; }
    // Tie-break: within a changed stretch, deletions come before insertions.
    if (j >= m || (i < n && dp[i + 1][j] >= dp[i][j + 1])) ops.push({ op: 'del', a: i++, hunk });
    else ops.push({ op: 'ins', b: j++, hunk });
  }
  return ops;
}

// ---------------------------------------------------------------- helpers
const countSeq = (toks, keys) => {
  let c = 0;
  for (let i = 0; i + keys.length <= toks.length; i++) if (keys.every((k, n) => toks[i + n].k === k)) { c++; i += keys.length - 1; }
  return c;
};
const findSeq = (toks, keys) => {
  for (let i = 0; i + keys.length <= toks.length; i++) if (keys.every((k, n) => toks[i + n].k === k)) return { s: toks[i].s, e: toks[i + keys.length - 1].e };
  return null;
};
const passageNumber = spanId => String(spanId).slice(1);

// ---------------------------------------------------------------- evidence per displayed passage
// review: { rows: [{id, kind, source, output, decision}] } as returned by
// compareTexts or carried by a checked packet. Returns { passages } in display
// order. Each passage: { row, kind, number, a, b, notes, inBoth, alsoMarked,
// ops, aToks, bToks, sameWords, tooLong }.
//
// Note states: 'a-only' (Original only), 'differs', 'b-only' (Rewrite only),
// 'moved' (Other passage). Note scopes for one-sided notes: 'text' (searched
// in the whole other text), 'passage' (structural word, compared within the
// passage pair), 'count' (the string is in the paired passage, fewer times).
export function buildEvidence(source, output, review) {
  const rows = Array.isArray(review?.rows) ? review.rows : [];
  const byStart = (x, y) => x.start - y.start;
  // Every original passage appears in exactly one row, and so does every
  // rewrite passage, so the rows carry both complete passage lists.
  const originals = rows.filter(r => r.source).map(r => r.source).sort(byStart);
  const rewrites = rows.filter(r => r.output).map(r => r.output).sort(byStart);
  const allA = tokenize(source), allB = tokenize(output);
  const aFirst = new Set(originals.map(s => s.start)), bFirst = new Set(rewrites.map(s => s.start));
  const capsElsewhere = new Set([...allA.filter(t => !aFirst.has(t.s)), ...allB.filter(t => !bFirst.has(t.s))].filter(isCapitalized).map(t => t.t));
  const lowerAnywhere = new Set([...allA, ...allB].filter(t => t.t === t.t.toLocaleLowerCase('en')).map(t => t.k));
  const ctxA = { text: source, capsElsewhere, lowerAnywhere }, ctxB = { text: output, capsElsewhere, lowerAnywhere };
  const passTok = spans => spans.map(s => ({ span: s, toks: tokenize(s.text, s.start) }));
  const aPass = passTok(originals), bPass = passTok(rewrites);
  const elsewhere = (passes, exceptId, keys) => {
    for (const p of passes) {
      if (p.span.id === exceptId) continue;
      const hit = findSeq(p.toks, keys);
      if (hit) return { passage: p.span.id, ...hit };
    }
    return null;
  };

  const out = [];
  for (const row of rows) {
    const entry = { row, kind: row.kind, a: row.source, b: row.output, notes: [], inBoth: [], alsoMarked: [], ops: null,
      aToks: [], bToks: [], sameWords: false, tooLong: false };
    const A = row.source ? detectItems(row.source.text, row.source.start, ctxA) : { toks: [], items: [] };
    const B = row.output ? detectItems(row.output.text, row.output.start, ctxB) : { toks: [], items: [] };
    entry.aToks = A.toks; entry.bToks = B.toks;
    if (row.kind === 'verbatim') { out.push(entry); continue; }
    const paired = Boolean(row.source && row.output);

    const ops = paired ? diffTokens(A.toks, B.toks) : null;
    entry.ops = ops;
    entry.tooLong = paired && !ops;
    entry.sameWords = Boolean(ops) && ops.every(o => o.op === 'eq');
    const aHunk = new Map(), bHunk = new Map(), aPos = new Map(), bPos = new Map();
    if (ops) ops.forEach((o, idx) => {
      if (o.a !== undefined) { aPos.set(o.a, idx); if (o.op === 'del') aHunk.set(o.a, o.hunk); }
      if (o.b !== undefined) { bPos.set(o.b, idx); if (o.op === 'ins') bHunk.set(o.b, o.hunk); }
    });
    else { A.toks.forEach((_, i) => aPos.set(i, i)); B.toks.forEach((_, i) => bPos.set(i, i)); }
    const range = it => Array.from({ length: it.tj - it.ti }, (_, n) => it.ti + n);
    const hunksOf = (it, map) => new Set(range(it).map(i => map.get(i)).filter(h => h !== undefined));
    const posOf = (it, map) => Math.min(...range(it).map(i => map.get(i)));
    const footprint = (it, map) => { const v = range(it).map(i => map.get(i)); return [Math.min(...v), Math.max(...v)]; };

    // 1. In both: the item's token-key sequence occurs in the paired passage,
    //    multiset-counted (a third "not" matches only if the other passage has three).
    const seen = new Map();
    const inBoth = (it, otherToks, side) => {
      const id = side + '|' + it.kind + '|' + it.key;
      const used = seen.get(id) || 0;
      if (used < countSeq(otherToks, it.key.split(' '))) { seen.set(id, used + 1); return true; }
      return false;
    };
    const aLeft = [], bLeft = [];
    for (const it of A.items) (row.output && inBoth(it, B.toks, 'a') ? (it.state = 'both') : aLeft.push(it));
    const bothKeys = new Map();
    A.items.filter(it => it.state === 'both').forEach(it => bothKeys.set(it.kind + '|' + it.key, (bothKeys.get(it.kind + '|' + it.key) || 0) + 1));
    for (const it of B.items) {
      if (row.source && inBoth(it, A.toks, 'b')) {
        it.state = 'both';
        const k = it.kind + '|' + it.key;
        if (bothKeys.get(k)) { bothKeys.set(k, bothKeys.get(k) - 1); it.dupOfA = true; }
      } else bLeft.push(it);
    }
    // 2. Differs: same condition marker (passage level), else the same kind
    //    sharing a diff hunk or overlapping op-index footprints.
    for (const x of aLeft) {
      if (x.pair) continue;
      let y = null;
      if (x.kind === 'condition') y = bLeft.find(b => !b.pair && b.kind === 'condition' && b.marker === x.marker);
      if (!y && ops) {
        const hx = hunksOf(x, aHunk), fx = footprint(x, aPos);
        y = bLeft.find(b => {
          if (b.pair || b.kind !== x.kind) return false;
          const fb = footprint(b, bPos);
          return [...hunksOf(b, bHunk)].some(h => hx.has(h)) || (fx[0] <= fb[1] && fb[0] <= fx[1]);
        });
      }
      if (y) { x.pair = y; y.pair = x; x.state = y.state = 'differs'; }
    }
    // 3. One-sided items. A string that is in the paired passage, only fewer
    //    times, is stated with both counts. Structural single words (negation,
    //    modal verb, "only") recur everywhere, so they are compared within the
    //    passage pair only; every other string is searched in the whole other text.
    const structural = it => it.kind === 'negation' || it.kind === 'modal' || (it.kind === 'condition' && it.tj - it.ti === 1);
    const settle = (it, ownToks, otherToks, hasOther, otherPasses, otherId, state) => {
      const keys = it.key.split(' ');
      const inPair = hasOther ? countSeq(otherToks, keys) : 0;
      if (inPair > 0) { it.state = state; it.scope = 'count'; it.counts = [countSeq(ownToks, keys), inPair]; it.where = null; return; }
      const hit = structural(it) ? null : elsewhere(otherPasses, otherId, keys);
      it.state = hit ? 'moved' : state; it.where = hit; it.scope = structural(it) ? 'passage' : 'text';
    };
    for (const x of aLeft.filter(x => !x.pair)) settle(x, A.toks, B.toks, Boolean(row.output), bPass, row.output?.id, 'a-only');
    for (const y of bLeft.filter(y => !y.pair)) settle(y, B.toks, A.toks, Boolean(row.source), aPass, row.source?.id, 'b-only');
    // 4. Fold items nested inside a one-sided condition clause with the same state.
    const fold = items => {
      for (const c of items.filter(c => c.kind === 'condition' && c.tj - c.ti > 1 && (c.state === 'a-only' || c.state === 'b-only' || c.state === 'moved'))) {
        for (const it of items) if (it !== c && !it.foldedInto && it.ti >= c.ti && it.tj <= c.tj && it.state === c.state) { it.foldedInto = c; (c.includes ||= []).push(it); }
      }
    };
    fold(A.items); fold(B.items);

    // Notes from items.
    const covered = { a: new Set(), b: new Set() };
    const cover = (side, it) => { for (let x = it.ti; x < it.tj; x++) covered[side].add(x); };
    A.items.forEach(it => cover('a', it)); B.items.forEach(it => cover('b', it));
    for (const x of A.items) {
      if (x.foldedInto) continue;
      if (x.state === 'both') entry.inBoth.push({ kind: x.kind, a: x, b: null });
      else if (x.state === 'differs') entry.notes.push({ kind: x.kind, state: 'differs', a: x, b: x.pair, pos: Math.min(posOf(x, aPos), posOf(x.pair, bPos)) });
      else entry.notes.push({ kind: x.kind, state: x.state, a: x, b: null, where: x.where, scope: x.scope, counts: x.counts, pos: posOf(x, aPos) });
    }
    for (const y of B.items) {
      if (y.foldedInto || y.state === 'differs') continue;
      if (y.state === 'both') {
        // The same string counted from the original side: link this occurrence
        // to that line instead of listing it twice.
        const twin = y.dupOfA && entry.inBoth.find(ib => ib.a && !ib.b && ib.kind === y.kind && ib.a.key === y.key);
        if (twin) twin.b = y;
        else if (!y.dupOfA) entry.inBoth.push({ kind: y.kind, a: null, b: y });
      } else entry.notes.push({ kind: y.kind, state: y.state, a: null, b: y, where: y.where, scope: y.scope, counts: y.counts, pos: posOf(y, bPos) });
    }

    // Wording notes: per hunk, the changed tokens not covered by an item,
    // split into contiguous runs and trimmed of common words at each end.
    const toRun = (g, toks, text) => ({ ti: g[0], tj: g[g.length - 1] + 1, s: toks[g[0]].s, e: toks[g[g.length - 1]].e,
      text: text.slice(toks[g[0]].s, toks[g[g.length - 1]].e), key: g.map(i => toks[i].k).join(' ') });
    const groupsOf = idxs => {
      const groups = [];
      for (const i of idxs) { const g = groups[groups.length - 1]; if (g && g[g.length - 1] === i - 1) g.push(i); else groups.push([i]); }
      return groups;
    };
    const wordSeen = new Map();
    // A one-sided run is split three ways: in the paired passage at least as
    // often (moved within the passage: listed as In both), in the paired
    // passage fewer times (a count statement), or absent from the pair.
    const oneSided = (runs, side, pos) => {
      const own = side === 'a' ? A.toks : B.toks, other = side === 'a' ? B.toks : A.toks;
      const hasOther = side === 'a' ? Boolean(row.output) : Boolean(row.source);
      const free = [];
      for (const r of runs) {
        const keys = r.key.split(' ');
        const inPair = hasOther ? countSeq(other, keys) : 0;
        const mine = countSeq(own, keys);
        const id = side + '|' + r.key;
        const used = wordSeen.get(id) || 0;
        if (inPair > 0 && used < inPair && inPair >= mine) {
          wordSeen.set(id, used + 1);
          // A reorder shows up as a deletion in one hunk and an insertion in
          // another: pair the two runs on one In both line.
          const twin = entry.inBoth.find(ib => ib.kind === 'wording' && ib.pending === (side === 'a' ? 'a' : 'b') && ib[side === 'a' ? 'b' : 'a'].key === r.key);
          if (twin) { twin[side] = r; delete twin.pending; delete twin[side === 'a' ? 'aSpan' : 'bSpan']; continue; }
          const span = findSeq(other, keys);
          entry.inBoth.push(side === 'a' ? { kind: 'wording', a: r, b: null, bSpan: span, pending: 'b' } : { kind: 'wording', a: null, b: r, aSpan: span, pending: 'a' });
        } else if (inPair > 0) {
          const note = { kind: 'wording', state: side === 'a' ? 'a-only' : 'b-only', aRuns: side === 'a' ? [r] : [], bRuns: side === 'b' ? [r] : [],
            pos: pos(side, r), scope: 'count', counts: [mine, inPair] };
          entry.notes.push(note);
        } else free.push(r);
      }
      if (!free.length) return;
      const otherPasses = side === 'a' ? bPass : aPass, otherId = side === 'a' ? row.output?.id : row.source?.id;
      const hits = free.map(r => elsewhere(otherPasses, otherId, r.key.split(' ')));
      const absent = free.filter((_, i) => !hits[i]);
      free.forEach((r, i) => {
        if (!hits[i]) return;
        entry.notes.push({ kind: 'wording', state: 'moved', aRuns: side === 'a' ? [r] : [], bRuns: side === 'b' ? [r] : [],
          pos: pos(side, r), scope: 'text', where: hits[i] });
      });
      if (absent.length) entry.notes.push({ kind: 'wording', state: side === 'a' ? 'a-only' : 'b-only', aRuns: side === 'a' ? absent : [],
        bRuns: side === 'b' ? absent : [], pos: Math.min(...absent.map(r => pos(side, r))), scope: 'text' });
    };
    const posRun = (side, r) => (side === 'a' ? aPos : bPos).get(r.ti);
    const trimmed = (idxs, toks, text, side, keepAlso) => {
      const kept = [];
      for (const g of groupsOf(idxs)) {
        let s = 0, e = g.length;
        while (s < e && STOP.has(toks[g[s]].k)) { if (keepAlso) entry.alsoMarked.push({ side, tok: toks[g[s]] }); s++; }
        const tail = [];
        while (e > s && STOP.has(toks[g[e - 1]].k)) { e--; if (keepAlso) tail.unshift({ side, tok: toks[g[e]] }); }
        if (s < e) kept.push(toRun(g.slice(s, e), toks, text));
        if (keepAlso) entry.alsoMarked.push(...tail);
      }
      return kept;
    };
    if (ops) {
      const hunks = new Map();
      for (const o of ops) {
        if (o.op === 'eq') continue;
        const h = hunks.get(o.hunk) || { a: [], b: [] };
        if (o.op === 'del' && !covered.a.has(o.a)) h.a.push(o.a);
        if (o.op === 'ins' && !covered.b.has(o.b)) h.b.push(o.b);
        hunks.set(o.hunk, h);
      }
      for (const [, h] of hunks) {
        const aRuns = trimmed(h.a, A.toks, source, 'a', true), bRuns = trimmed(h.b, B.toks, output, 'b', true);
        if (!aRuns.length && !bRuns.length) continue;
        if (aRuns.length && bRuns.length) {
          const pos = Math.min(...aRuns.map(r => aPos.get(r.ti)), ...bRuns.map(r => bPos.get(r.ti)));
          entry.notes.push({ kind: 'wording', state: 'differs', aRuns, bRuns, pos, scope: 'text' });
        } else if (aRuns.length) oneSided(aRuns, 'a', posRun);
        else oneSided(bRuns, 'b', posRun);
      }
      entry.alsoMarked.sort((x, y) => (x.side === 'a' ? aPos.get(A.toks.indexOf(x.tok)) : bPos.get(B.toks.indexOf(x.tok))) -
        (y.side === 'a' ? aPos.get(A.toks.indexOf(y.tok)) : bPos.get(B.toks.indexOf(y.tok))));
    } else if (!paired && !entry.notes.length) {
      // An unpartnered passage with no noted item still differs as a whole:
      // note its uncovered words so no passage is ever left without a note.
      const side = row.source ? 'a' : 'b', toks = side === 'a' ? A.toks : B.toks, text = side === 'a' ? source : output;
      const idxs = toks.map((_, i) => i).filter(i => !covered[side].has(i));
      let runs = trimmed(idxs, toks, text, side, false);
      if (!runs.length && idxs.length) runs = groupsOf(idxs).map(g => toRun(g, toks, text));
      if (runs.length) oneSided(runs, side, posRun);
    }
    entry.notes.sort((x, y) => x.pos - y.pos);
    entry.notes.forEach((n, i) => { n.letter = String.fromCharCode(97 + (i % 26)).repeat(1 + Math.floor(i / 26)); });
    // In-both lines get the matching span in the other passage.
    for (const ib of entry.inBoth) {
      delete ib.pending;
      if (ib.a && !ib.b && !ib.bSpan) ib.bSpan = findSeq(B.toks, ib.a.key.split(' '));
      if (ib.b && !ib.a && !ib.aSpan) ib.aSpan = findSeq(A.toks, ib.b.key.split(' '));
    }
    out.push(entry);
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
  return { passages: display };
}

// ---------------------------------------------------------------- labels and statements
export const KIND_LABEL = { number: 'Number', date: 'Date', duration: 'Duration', negation: 'Negation', modal: 'Modal verb',
  condition: 'Condition or exception', name: 'Name', wording: 'Wording' };
export const STATE_ORDER = ['a-only', 'differs', 'b-only', 'moved', 'both'];
export const STATE_LABEL = { 'a-only': 'Original only', differs: 'Differs', 'b-only': 'Rewrite only', moved: 'Other passage', both: 'In both' };
const conditionLabel = it => (EXCEPTION_MARKERS.has(it.marker) ? 'Exception' : 'Condition');
export function kindLabel(note) {
  if (note.kind !== 'condition') return KIND_LABEL[note.kind];
  const a = note.a && conditionLabel(note.a), b = note.b && conditionLabel(note.b);
  return a && b && a !== b ? 'Condition or exception' : (a || b);
}
export const noteId = (passage, note) => `${passage.number}${note.letter}`;
export const noteStrings = note => ({
  aText: note.a ? note.a.text : (note.aRuns || []).map(r => r.text).join(' … '),
  bText: note.b ? note.b.text : (note.bRuns || []).map(r => r.text).join(' … '),
});
const q = s => `“${s}”`;
const times = n => (n === 1 ? 'once' : n === 2 ? 'twice' : `${n} times`);

// The one sentence each note says. Every statement is a literal fact about
// where a string occurs; none says whether a difference matters.
export function noteStatement(note, passage) {
  const { aText, bText } = noteStrings(note);
  const unpaired = !passage.a || !passage.b;
  const it = note.a || note.b;
  if (note.state === 'differs') return `${q(aText)} in the original, ${q(bText)} in the rewrite.`;
  if (note.state === 'moved') {
    const other = aText ? 'rewrite' : 'original';
    const k = passageNumber(note.where.passage);
    return unpaired ? `This passage has no partner in the ${other}; the string appears in ${other} passage ${k}.`
      : `Not in the paired ${other} passage; it appears in ${other} passage ${k}.`;
  }
  if (note.scope === 'count') {
    const [own, other] = note.counts;
    return aText ? `Appears ${times(own)} in the original passage and ${times(other)} in the paired rewrite passage.`
      : `Appears ${times(own)} in the rewrite passage and ${times(other)} in the paired original passage.`;
  }
  let say;
  if (note.kind === 'condition' && it && it.tj - it.ti > 1) {
    say = aText ? `The marker ${q(it.marker)} and the words after it appear in the original, nowhere in the rewrite.`
      : `The marker ${q(it.marker)} and the words after it appear in the rewrite, nowhere in the original.`;
    if (it.includes?.length) say += ` Includes ${it.includes.map(x => `the ${KIND_LABEL[x.kind].toLowerCase()} ${q(x.text)}`).join(', ')}.`;
    return say;
  }
  if (note.scope === 'passage') {
    return aText ? (unpaired ? 'In an original passage that has no partner in the rewrite.' : 'In the original passage, not in the paired rewrite passage.')
      : (unpaired ? 'In a rewrite passage that has no partner in the original.' : 'In the rewrite passage, not in the paired original passage.');
  }
  const plural = note.kind === 'wording' && (/\s/.test(aText || bText) || (note.aRuns || note.bRuns || []).length > 1);
  return aText ? `${plural ? 'These words appear' : 'Appears'} in the original, nowhere in the rewrite.`
    : `${plural ? 'These words appear' : 'Appears'} in the rewrite, nowhere in the original.`;
}

// The exact spans a note points at: [{side: 'a'|'b', s, e, text}].
export function noteSpans(note, source, output) {
  const spans = [];
  const add = (side, s, e) => spans.push({ side, s, e, text: (side === 'a' ? source : output).slice(s, e) });
  if (note.a) add('a', note.a.s, note.a.e);
  for (const r of note.aRuns || []) add('a', r.s, r.e);
  if (note.b) add('b', note.b.s, note.b.e);
  for (const r of note.bRuns || []) add('b', r.s, r.e);
  if (note.state === 'moved' && note.where) add(noteStrings(note).aText ? 'b' : 'a', note.where.s, note.where.e);
  return spans;
}

// Notes and in-both lines grouped by state, in display order (never ranked).
export function groupByState(passages) {
  const groups = { 'a-only': [], differs: [], 'b-only': [], moved: [], both: [] };
  for (const p of passages) {
    for (const n of p.notes) groups[n.state].push({ passage: p, note: n });
    p.inBoth.forEach((ib, k) => groups.both.push({ passage: p, inBoth: ib, index: k }));
  }
  return groups;
}

// Heading counts. N counts notes (In both excluded). "No literal differences"
// is only claimed when every passage is paired and its words match in order.
export function evidenceSummary(passages) {
  const notes = passages.reduce((s, p) => s + p.notes.length, 0);
  const noDifferences = notes === 0 && passages.every(p => p.kind === 'verbatim' || (p.kind === 'changed-candidate' && p.sameWords));
  return {
    passages: passages.length,
    notes,
    noDifferences,
    pairs: passages.filter(p => p.kind === 'changed-candidate').length,
    identical: passages.filter(p => p.kind === 'verbatim').length,
    originalOnly: passages.filter(p => p.kind === 'source-unmatched').length,
    rewriteOnly: passages.filter(p => p.kind === 'output-unmatched').length,
  };
}

// ---------------------------------------------------------------- blackline
// The marked passage as plain data for the renderer. Segments:
//   {t: 'plain', text}                      whole passage, no marks
//   {t: 'gap', text}                        characters between tokens
//   {t: 'eq', text, n}                      a token in both (n: note or in-both id, or null)
//   {t: 'run', side: 'd'|'i', sub, parts}   a deletion or insertion run
//       parts: {t: 'gap', text} | {t: 'mark', text, n}
//   {t: 'ref', n, letter}                   a note letter after its last token
// Gap text is never inside a mark, so no leading or trailing space is struck.
export function blacklineSegments(passage, source, output) {
  if (passage.kind === 'verbatim' || passage.sameWords || passage.tooLong) return [{ t: 'plain', text: (passage.b || passage.a).text }];
  const pno = passage.number;
  const A = passage.aToks, B = passage.bToks;
  const ops = passage.ops || (passage.a ? A.map((_, i) => ({ op: 'del', a: i })) : B.map((_, i) => ({ op: 'ins', b: i })));
  const aNote = new Map(), bNote = new Map(), refAfter = new Map();
  const opIndexA = new Map(), opIndexB = new Map();
  ops.forEach((o, i) => { if (o.a !== undefined) opIndexA.set(o.a, i); if (o.b !== undefined) opIndexB.set(o.b, i); });
  for (const n of passage.notes) {
    const id = `${pno}${n.letter}`;
    const ranges = [];
    if (n.a) ranges.push(['a', n.a.ti, n.a.tj]);
    if (n.b) ranges.push(['b', n.b.ti, n.b.tj]);
    for (const r of n.aRuns || []) ranges.push(['a', r.ti, r.tj]);
    for (const r of n.bRuns || []) ranges.push(['b', r.ti, r.tj]);
    for (const it of (n.a || n.b)?.includes || []) ranges.push([n.a ? 'a' : 'b', it.ti, it.tj]);
    let last = -1;
    for (const [side, i0, i1] of ranges) for (let i = i0; i < i1; i++) {
      (side === 'a' ? aNote : bNote).set(i, id);
      last = Math.max(last, side === 'a' ? opIndexA.get(i) : opIndexB.get(i));
    }
    refAfter.set(last, [...(refAfter.get(last) || []), n]);
  }
  passage.inBoth.forEach((ib, k) => {
    const id = `${pno}-both-${k}`;
    if (ib.a) for (let i = ib.a.ti; i < ib.a.tj; i++) aNote.set(i, aNote.get(i) || id);
    if (ib.b) for (let i = ib.b.ti; i < ib.b.tj; i++) bNote.set(i, bNote.get(i) || id);
  });
  const gapBefore = (toks, i, text, pStart) => text.slice(i === 0 ? pStart : toks[i - 1].e, toks[i].s);
  const keepTogether = text => (text.length <= 30 ? text.replace(/ /g, ' ') : text);
  const segs = [];
  let run = null;
  const refs = k => { for (const n of refAfter.get(k) || []) (run ? run.parts : segs).push({ t: 'ref', n: `${pno}${n.letter}`, letter: n.letter }); };
  for (let k = 0; k < ops.length; k++) {
    const o = ops[k];
    if (o.op === 'eq') {
      run = null;
      segs.push({ t: 'gap', text: gapBefore(B, o.b, output, passage.b.start) });
      segs.push({ t: 'eq', text: B[o.b].t, n: bNote.get(o.b) || aNote.get(o.a) || null });
      refs(k);
      continue;
    }
    const del = o.op === 'del';
    const side = del ? 'd' : 'i';
    const wasDel = run?.side === 'd';
    if (run?.side !== side) {
      let sub = false;
      if (del) { const next = ops.slice(k).find(x => x.op !== 'del'); sub = Boolean(next && next.op === 'ins'); }
      run = { t: 'run', side, sub, parts: [] };
      segs.push(run);
    }
    const toks = del ? A : B, text = del ? source : output, span = del ? passage.a : passage.b, map = del ? aNote : bNote;
    const idx = del ? o.a : o.b;
    const g = gapBefore(toks, idx, text, span.start);
    // An insertion straight after a deletion at the passage start gets one
    // space, so the two runs never read as one word ("mayYou").
    const gapText = !del && g === '' && wasDel ? ' ' : g;
    if (gapText) run.parts.push({ t: 'gap', text: gapText });
    const id = map.get(idx) || null;
    let j = k, word = toks[idx].t;
    // Consecutive tokens of the same note, separated by one space, form one mark;
    // in an unpartnered passage consecutive unnoted tokens are grouped too.
    while (ops[j + 1] && ops[j + 1].op === o.op && map.get(del ? ops[j + 1].a : ops[j + 1].b) === (id ?? undefined) &&
      (id || !passage.ops) && gapBefore(toks, del ? ops[j + 1].a : ops[j + 1].b, text, span.start) === ' ' && !refAfter.has(j)) {
      j++; word += ' ' + toks[del ? ops[j].a : ops[j].b].t;
    }
    run.parts.push({ t: 'mark', text: keepTogether(word), n: id });
    refs(j);
    k = j;
  }
  const [T, text, span] = passage.b ? [B, output, passage.b] : [A, source, passage.a];
  const tail = text.slice(T.length ? T[T.length - 1].e : span.start, span.end);
  if (tail) segs.push({ t: 'gap', text: tail });
  return segs;
}
