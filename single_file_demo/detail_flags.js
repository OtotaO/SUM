// Critical-detail checks: lexical, deterministic, computed in the browser with no network request.
// Pure functions shared by the browser and the offline Node tests. A flag says a detail of one of seven
// kinds (number, date, negation, modal, condition, exception, name) has no literal or equal-valued
// counterpart in the other text. It can miss changes and can flag details that were kept in other words.
import { sourceSpans } from './review_packet.js';

export const DETAIL_FLAGS_ID = 'sum.detail_flags.lexical_v1.1';
export const DETAIL_SCOPE = 'Lexical checks for numbers, dates, negations, modal verbs, conditions, exceptions and names. A flag means no literal or equal-valued match was found nearby; it is not a judgement of meaning, and changes outside these words are not checked.';

const UNITS = [
  [/^(?:business|working)\s+days?\b/i, 'businessday', 1], [/^(?:calendar\s+)?days?\b/i, 'minute', 1440], [/^nights?\b/i, 'night', 1],
  [/^weeks?\b/i, 'minute', 10080], [/^fortnights?\b/i, 'minute', 20160], [/^months?\b/i, 'month', 1], [/^years?\b/i, 'month', 12],
  [/^hours?\b/i, 'minute', 60], [/^minutes?\b/i, 'minute', 1], [/^(?:percent|per\s+cent)\b/i, 'fraction', 0.01], [/^%/, 'fraction', 0.01],
  [/^(?:dollars?|usd)\b/i, 'money$', 1], [/^(?:euros?|eur)\b/i, 'money€', 1], [/^(?:pounds?|gbp)\b/i, 'money£', 1], [/^cents?\b/i, 'money$', 0.01],
  [/^(?:tablets?|pills?|capsules?)\b/i, 'tablet', 1], [/^(?:mg|milligrams?)\b/i, 'mg', 1], [/^(?:ml|millilit(?:er|re)s?)\b/i, 'ml', 1],
  [/^times\b/i, 'times', 1], [/^doses?\b/i, 'dose', 1], [/^(?:miles?)\b/i, 'mile', 1], [/^(?:km|kilomet(?:er|re)s?)\b/i, 'km', 1],
];
const SMALL = { zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19, twenty: 20, thirty: 30, forty: 40, fifty: 50, sixty: 60, seventy: 70, eighty: 80, ninety: 90 };
const SCALE = { hundred: 100, thousand: 1000, million: 1e6, billion: 1e9 };
const MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'];
const MONTH_RE = '(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\\.?';

const round = v => Math.round(v * 1e6) / 1e6;
const monthIndex = m => MONTHS.findIndex(name => name.startsWith(m.toLowerCase().replace('.', '').slice(0, 3)));

// Coverage checks run once per match, so they use a per-text mask instead of scanning earlier ranges.
function maskOf(length, ranges) { const m = new Uint8Array(length + 1); for (const r of ranges) m.fill(1, r.start, r.end); return m; }
function covered(mask, start, end) { for (let k = start; k < end; k++) if (mask[k]) return true; return false; }
function coverer(length) { const mask = new Uint8Array(length + 1); return { mask, add: r => mask.fill(1, r.start, r.end), has: (start, end) => covered(mask, start, end) }; }
const window = (text, i, n = 80) => text.slice(i, i + n);

function findDates(text) {
  const items = [];
  const cover = coverer(text.length);
  const add = (m, key) => { const item = { category: 'date', key, start: m.index, end: m.index + m[0].length, text: m[0] }; items.push(item); cover.add(item); };
  const covered = (_, start, end) => cover.has(start, end);
  const year = '(?:,?\\s+(\\d{4}))?';
  for (const m of text.matchAll(new RegExp(`\\b(${MONTH_RE})\\s+(\\d{1,2})(?:st|nd|rd|th)?\\b${year}`, 'g'))) add(m, `d:${monthIndex(m[1])}-${+m[2]}-${m[3] || ''}`);
  for (const m of text.matchAll(new RegExp(`\\b(\\d{1,2})(?:st|nd|rd|th)?\\s+(?:of\\s+)?(${MONTH_RE})(?![a-z])${year}`, 'g'))) if (!covered(items, m.index, m.index + m[0].length)) add(m, `d:${monthIndex(m[2])}-${+m[1]}-${m[3] || ''}`);
  for (const m of text.matchAll(new RegExp(`\\b(${MONTH_RE})\\s+(\\d{4})\\b`, 'g'))) if (!covered(items, m.index, m.index + m[0].length)) add(m, `d:${monthIndex(m[1])}--${m[2]}`);
  for (const m of text.matchAll(/\b(\d{4})-(\d{2})-(\d{2})\b/g)) if (!covered(items, m.index, m.index + m[0].length)) add(m, `d:${+m[2] - 1}-${+m[3]}-${m[1]}`);
  for (const m of text.matchAll(/\b(\d{1,2})\/(\d{1,2})\/(\d{2,4})\b/g)) if (!covered(items, m.index, m.index + m[0].length)) add(m, `d:${+m[1] - 1}-${+m[2]}-${m[3].length === 2 ? '20' + m[3] : m[3]}`);
  for (const m of text.matchAll(/\b(Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)days?\b/g)) if (!covered(items, m.index, m.index + m[0].length)) add(m, `w:${m[1].toLowerCase()}`);
  // Relative dates: "the end of the month" and "end of this month" are the same detail.
  for (const m of text.matchAll(/\b(?:the\s+)?(end|start|beginning|middle)\s+of\s+(?:the\s+|this\s+|each\s+|every\s+)?(day|week|month|quarter|year)\b/gi)) if (!covered(items, m.index, m.index + m[0].length)) add(m, `r:${m[1].toLowerCase().replace('beginning', 'start')}-${m[2].toLowerCase()}`);
  for (const m of text.matchAll(/\b(today|tomorrow|tonight|yesterday)\b|\bnext\s+(week|month|year)\b/gi)) if (!covered(items, m.index, m.index + m[0].length)) add(m, `r:${(m[1] || 'next-' + m[2]).toLowerCase()}`);
  for (const m of text.matchAll(/\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s?m\b\.?/gi)) {
    if (covered(items, m.index, m.index + m[0].length)) continue;
    const h = (+m[1] % 12) + (m[3].toLowerCase() === 'p' ? 12 : 0);
    add(m, `t:${h * 60 + (+m[2] || 0)}`);
  }
  for (const m of text.matchAll(/\b([01]?\d|2[0-3]):([0-5]\d)\b/g)) if (!covered(items, m.index, m.index + m[0].length)) add(m, `t:${+m[1] * 60 + +m[2]}`);
  for (const m of text.matchAll(/\b(noon|midday|midnight)\b/gi)) if (!covered(items, m.index, m.index + m[0].length)) add(m, `t:${/mid(?:night)/i.test(m[1]) && m[1].toLowerCase() === 'midnight' ? 0 : 720}`);
  // A month name standing alone ("in March") is also a date detail; "May" alone is a modal verb.
  for (const m of text.matchAll(new RegExp(`\\b(${MONTH_RE.replace('May|', '')})(?![a-z])`, 'g'))) if (!covered(items, m.index, m.index + m[0].length)) add(m, `d:${monthIndex(m[1])}--`);
  return items.sort((x, y) => x.start - y.start);
}

function spelledAt(text, i) {
  const m = /^(?:(a|an)\s+(?=hundred|thousand|dozen|half)|)((?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[\s-](?:one|two|three|four|five|six|seven|eight|nine))?(?:\s+(?:hundred|thousand|million))?|hundred|thousand|dozen)\b/i.exec(window(text, i));
  if (!m) return null;
  let value = 0, part = 0;
  for (const w of m[2].toLowerCase().split(/[\s-]+/)) {
    if (w === 'dozen') part = (part || 1) * 12;
    else if (w in SCALE) { part = (part || 1) * SCALE[w]; if (SCALE[w] >= 1000) { value += part; part = 0; } }
    else part += SMALL[w];
  }
  return { length: m[0].length, value: value + part, word: m[2].toLowerCase() };
}

function unitAt(text, i, allow = () => true) {
  const ahead = window(text, i, 40);
  const gap = /^[\s-]*/.exec(ahead)[0];
  const u = UNITS.find(([r, name]) => allow(name) && r.test(ahead.slice(gap.length)));
  return u ? { name: u[1], factor: u[2], length: gap.length + u[0].exec(ahead.slice(gap.length))[0].length } : null;
}

function findNumbers(text, taken) {
  const items = [];
  const takenMask = maskOf(text.length, taken);
  let lastEnd = 0;
  const push = item => { items.push(item); lastEnd = item.end; };
  const re = /([$€£]\s?)?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)|\b(?:an?|half|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|dozen)\b/gi;
  for (const m of text.matchAll(re)) {
    const start = m.index;
    if (start < lastEnd || covered(takenMask, start, start + m[0].length)) continue;
    if (/[\p{L}\d]/u.test(text[start - 1] || '') && !m[1]) continue;
    let value, end, unit = '', word = null;
    if (m[2] !== undefined) {
      value = Number(m[2].replace(/,/g, '')); end = start + m[0].length;
      if (m[1]) unit = `money${m[1].trim()}`;
      const scale = /^\s+(thousand|million|billion)\b/i.exec(window(text, end, 20));
      if (scale) { value *= SCALE[scale[1].toLowerCase()]; end += scale[0].length; }
    } else {
      const lower = m[0].toLowerCase();
      if (lower === 'half') {
        const rest = /^half\s+(?:an?\s+)?/i.exec(window(text, start, 20));
        const u = rest && unitAt(text, start + rest[0].length, name => name !== 'fraction');
        if (u) push({ category: 'number', key: `${u.name}:${round(0.5 * u.factor)}`, start, end: start + rest[0].length + u.length, text: text.slice(start, start + rest[0].length + u.length) });
        else push({ category: 'number', key: 'fraction:0.5', start, end: start + 4, text: text.slice(start, start + 4) });
        continue;
      }
      if (lower === 'a' || lower === 'an') {
        // "a month" is a duration; "$10 a month" and "twice a day" are rates, whose number is counted already.
        if (/(?:[\d%]|\b(?:once|twice|thrice|times|dollars?|euros?|pounds?|cents?))\s*$/i.test(text.slice(Math.max(0, start - 16), start))) continue;
        const u = unitAt(text, start + m[0].length, name => name !== 'fraction' && !name.startsWith('money'));
        if (!u) continue;
        push({ category: 'number', key: `${u.name}:${round(u.factor)}`, start, end: start + m[0].length + u.length, text: text.slice(start, start + m[0].length + u.length) });
        continue;
      }
      const sp = spelledAt(text, start);
      if (!sp) continue;
      value = sp.value; end = start + sp.length; word = sp.word;
    }
    const halfMore = /^\s+and\s+a\s+half\b/i.exec(window(text, end, 20));
    if (halfMore) { value += 0.5; end += halfMore[0].length; }
    let factor = 1, u = null;
    if (!unit) {
      u = unitAt(text, end);
      if (u) { unit = u.name; factor = u.factor; end += u.length; }
    } else {
      const dollars = /^\s*dollars?\b/i.exec(window(text, end, 12));
      if (dollars) end += dollars[0].length;
    }
    if (word === 'one' && !u && !halfMore) continue; // "one of", "no one": a pronoun more often than a count
    push({ category: 'number', key: `${unit}:${round(value * factor)}`, start, end, text: text.slice(start, end) });
  }
  // "once" counts only as a frequency ("once a day"), not as a conjunction ("once you pay").
  for (const m of text.matchAll(/\b(once|twice|thrice)\b(?=\s+(?:a|an|per|every|each|daily|weekly|monthly|yearly|annually)\b)/gi)) {
    if (covered(takenMask, m.index, m.index + m[0].length)) continue;
    items.push({ category: 'number', key: `times:${{ once: 1, twice: 2, thrice: 3 }[m[1].toLowerCase()]}`, start: m.index, end: m.index + m[0].length, text: m[0] });
  }
  return items.sort((x, y) => x.start - y.start);
}

const COMMON_ACRONYMS = new Set(['ID', 'IDs', 'PIN', 'PDF', 'FAQ', 'ATM', 'TV', 'OK', 'AM', 'PM', 'USB', 'URL', 'SMS', 'PS', 'NB', 'I']);
const TITLES = new Set(['Dr', 'Mr', 'Mrs', 'Ms', 'Mx', 'Prof', 'Sir', 'Dame', 'St']);
const GENERIC = new Set(['Association', 'Department', 'Office', 'Team', 'Center', 'Centre', 'Clinic', 'Hospital', 'Bank', 'Company', 'Inc', 'Ltd', 'LLC', 'Services', 'Service', 'Group', 'Council', 'Authority', 'Agency', 'School', 'Academy', 'University', 'College', 'Housing', 'Insurance', 'The', 'Of', 'And', 'Street', 'Road', 'Avenue', 'Plan', 'Program', 'Programme', 'Fund', 'Trust', 'Foundation', 'Club', 'Store', 'Pharmacy', 'Hotel', 'Airlines', 'Air', 'Card', 'Account']);
const SENTENCE_STARTERS = new Set(['The', 'A', 'An', 'You', 'Your', 'We', 'Our', 'If', 'This', 'These', 'That', 'Those', 'Please', 'It', 'Its', 'All', 'Any', 'No', 'In', 'On', 'At', 'For', 'To', 'From', 'After', 'Before', 'When', 'Unless', 'Except', 'Once', 'Each', 'Every', 'There', 'They', 'Their', 'He', 'She', 'I', 'As', 'By', 'With', 'Without', 'Only', 'Also', 'However', 'Note', 'Under', 'During', 'While', 'Where', 'What', 'Do', 'Don', 'Can', 'May', 'Must', 'Should', 'Will', 'Keep', 'Take', 'Call', 'Contact', 'Remember', 'Bring', 'Make', 'Send', 'Pay', 'Check', 'Read', 'Use', 'Tell', 'Ask', 'Let', 'Some', 'Most', 'Both', 'Either', 'Neither', 'Not', 'Never', 'So', 'But', 'And', 'Or', 'Then', 'Here', 'Any', 'Anyone', 'Members', 'Customers', 'Tenants', 'Patients', 'Employees', 'Staff', 'Students', 'Parents', 'Refunds', 'Payments', 'Fees']);

function wordsWithOffsets(text) { return Array.from(text.matchAll(/[\p{L}\p{N}][\p{L}\p{N}'’]*/gu), m => ({ w: m[0], start: m.index, end: m.index + m[0].length })); }

function findNames(text, spans, taken) {
  const items = [];
  const capitalizedSomewhere = new Set();
  const starts = new Set(spans.map(s => s.start));
  const isInitial = start => {
    const before = text.slice(Math.max(0, start - 24), start);
    return starts.has(start) || /[.!?:;"“(\n]\s*$/.test(before) || (start <= 24 && !/\S/.test(before));
  };
  const words = wordsWithOffsets(text);
  words.forEach(t => { if (/^\p{Lu}/u.test(t.w) && !isInitial(t.start)) capitalizedSomewhere.add(t.w); });
  const takenMask = maskOf(text.length, taken);
  const isCap = t => /^\p{Lu}/u.test(t.w) && !COMMON_ACRONYMS.has(t.w) && !covered(takenMask, t.start, t.end);
  for (let i = 0; i < words.length; i++) {
    if (!isCap(words[i])) continue;
    let j = i;
    while (j + 1 < words.length && (isCap(words[j + 1]) || (['of', 'and', 'de', 'la', 'van', 'von'].includes(words[j + 1].w) && words[j + 2] && isCap(words[j + 2]))) && /^[ \t.&'-]*$/.test(text.slice(words[j].end, words[j + 1].start))) j++;
    const group = words.slice(i, j + 1);
    // Long capitalized runs are headings or titles, not names.
    if (group.length > 6 || (group.length > 3 && group.every(t => t.w === t.w.toUpperCase()))) { i = j; continue; }
    const tokens = group.map(t => t.w).filter(w => /^\p{Lu}/u.test(w) && !TITLES.has(w));
    const initial = isInitial(group[0].start);
    if (initial && group.length === 1 && (SENTENCE_STARTERS.has(group[0].w) || !capitalizedSomewhere.has(group[0].w))) { i = j; continue; }
    if (initial && SENTENCE_STARTERS.has(group[0].w)) { tokens.shift(); group.shift(); }
    if (!tokens.length || !group.length) { i = j; continue; }
    const distinctive = tokens.filter(w => !GENERIC.has(w));
    items.push({ category: 'name', key: (distinctive.length ? distinctive : tokens).join(' '), tokens: distinctive.length ? distinctive : tokens, start: group[0].start, end: group[group.length - 1].end, text: text.slice(group[0].start, group[group.length - 1].end) });
    i = j;
  }
  return items;
}

const BOUNDS = /\b(?:no|not)\s+(?:more|later|less|fewer|earlier|sooner)\s+than\b/gi;
const NEGATORS = /\b(?:not|no|never|none|nobody|nothing|nowhere|neither|nor|cannot|without|non-?\w+|un(?:able|available|authori[sz]ed|eligible)|ineligible|prohibited|forbidden|banned|\w+n['’]t)\b/gi;
const MODALS = [
  ['prohibition', /\b(?:(?:can|may|must|could)\s+never|must\s+not|may\s+not|shall\s+not|can\s*not|can['’]t|mustn['’]t|(?:is|are|be)\s+not\s+(?:allowed|permitted)|not\s+(?:allowed|permitted)|prohibited|forbidden)\b/gi],
  ['obligation', /\b(?:must|shall|ha(?:ve|s|d)\s+to|needs?\s+to|(?:is|are|be)\s+required|required\s+to|requires?|obliged|obligated|mandatory|compulsory)\b/gi],
  ['permission', /\b(?:may|might|can|could|(?:is|are|be)\s+allowed|allowed\s+to|(?:is|are|be)\s+permitted|permitted\s+to|entitled\s+to|free\s+to|able\s+to)\b/gi],
  ['recommendation', /\b(?:should|ought\s+to|recommends?|recommended|advised?|encouraged?|best\s+to)\b/gi],
];
// "will" is not compared: plain-language rewrites add and drop it constantly ("we will refund" to "we refund").
const QUALIFIERS = [
  ['exception', /\b(?:unless|except(?:\s+(?:for|when|where|if))?|excepting|other\s+than|apart\s+from|aside\s+from|excluding|with\s+the\s+exception\s+of|but\s+not|save\s+for)\b/gi],
  ['condition', /\b(?:only\s+if|if|provided(?:\s+that)?|providing|as\s+long\s+as|so\s+long\s+as|in\s+case|on\s+condition\s+that|in\s+the\s+event\s+that|assuming)\b/gi],
];

function findPattern(text, category, re, taken = []) {
  const items = [];
  const mask = maskOf(text.length, taken);
  for (const m of text.matchAll(re)) {
    if (covered(mask, m.index, m.index + m[0].length)) continue;
    items.push({ category, start: m.index, end: m.index + m[0].length, text: m[0], key: m[0].toLowerCase().replace(/\s+/g, ' ') });
  }
  return items;
}

const STOP = new Set('the a an and or of to in on at for by with from is are be was were this that these those you your we our us it its they their them will may must can shall should not no if as any all each per than then so do does did has have had been being which who whom whose what when where how also only'.split(' '));
const stem = w => w.toLowerCase().replace(/['’]s$/, '').slice(0, 5);
const content = text => new Set((text.match(/[\p{L}\p{N}]+/gu) || []).filter(w => !STOP.has(w.toLowerCase())).map(stem));
function similarity(left, right) {
  const common = [...left].filter(t => right.has(t)).length;
  return common / Math.max(1, Math.min(left.size, right.size));
}

// Groups source and rewrite sentences that correspond, allowing merges and splits.
function align(sourceSpansList, outputSpansList) {
  const parent = new Map();
  const find = x => { while (parent.get(x) !== x) { parent.set(x, parent.get(parent.get(x))); x = parent.get(x); } return x; };
  const union = (a, b) => parent.set(find(a), find(b));
  sourceSpansList.forEach((_, i) => parent.set(`s${i}`, `s${i}`));
  outputSpansList.forEach((_, j) => parent.set(`o${j}`, `o${j}`));
  const left = sourceSpansList.map(s => content(s.text)), right = outputSpansList.map(o => content(o.text));
  const best = (set, others, prefix) => {
    let pick = -1, score = 0.15;
    others.forEach((o, k) => { const v = similarity(set, o); if (v > score) { score = v; pick = k; } });
    return pick < 0 ? null : `${prefix}${pick}`;
  };
  const exact = new Map(outputSpansList.map((o, j) => [o.text, j]));
  sourceSpansList.forEach((s, i) => { if (exact.has(s.text)) union(`s${i}`, `o${exact.get(s.text)}`); });
  left.forEach((set, i) => { const b = best(set, right, 'o'); if (b) union(`s${i}`, b); });
  right.forEach((set, j) => { const b = best(set, left, 's'); if (b) union(`o${j}`, b); });
  return { source: sourceSpansList.map((_, i) => find(`s${i}`)), output: outputSpansList.map((_, j) => find(`o${j}`)) };
}

function sentenceOf(spans, pos) {
  let lo = 0, hi = spans.length - 1, found = spans.length - 1;
  while (lo <= hi) { const mid = (lo + hi) >> 1; if (spans[mid].end <= pos) lo = mid + 1; else { found = mid; hi = mid - 1; } }
  return found;
}
const span = item => item ? { start: item.start, end: item.end, text: item.text } : null;
const bucket = (items, keyOf) => { const m = new Map(); for (const x of items) { const k = keyOf(x); if (!m.has(k)) m.set(k, []); m.get(k).push(x); } return m; };
const valueOf = key => key.slice(key.lastIndexOf(':') + 1), unitOf = key => key.slice(0, key.lastIndexOf(':'));

// Multiset match by key; unmatched items on both sides pair up as "changed" when they share a group.
// Every item carries its aligned group as item.g.
function compareByKey(category, src, out, looseKeys = null) {
  // First match within aligned groups, then across the text, but never take a counterpart from a group
  // that still has an unmatched detail of its own: that pairing is more likely a change.
  const usedOut = new Set(), usedSrc = new Set();
  const byKeyGroup = bucket(out.map((o, j) => j), j => `${out[j].key}\u0000${out[j].g}`);
  src.forEach((item, i) => {
    const list = byKeyGroup.get(`${item.key}\u0000${item.g}`);
    while (list && list.length) { const j = list.shift(); if (!usedOut.has(j)) { usedOut.add(j); usedSrc.add(i); break; } }
  });
  const openInGroup = new Map();
  src.forEach((item, i) => { if (!usedSrc.has(i)) openInGroup.set(item.g, (openInGroup.get(item.g) || 0) + 1); });
  const byKey = bucket(out.map((o, j) => j).filter(j => !usedOut.has(j)), j => out[j].key);
  src.forEach((item, i) => {
    if (usedSrc.has(i)) return;
    const list = byKey.get(item.key) || [];
    const k = list.findIndex(j => !usedOut.has(j) && (openInGroup.get(out[j].g) || 0) - (out[j].g === item.g ? 1 : 0) === 0);
    if (k >= 0) { usedOut.add(list[k]); list.splice(k, 1); usedSrc.add(i); openInGroup.set(item.g, openInGroup.get(item.g) - 1); }
  });
  const missing = src.filter((_, i) => !usedSrc.has(i));
  let added = out.filter((_, j) => !usedOut.has(j));
  if (looseKeys) {
    // Loose equality: each item offers keys; a missing item matches an added item sharing any loose key.
    const pool = new Map();
    added.forEach((o, j) => { for (const k of looseKeys(o, 'out')) { if (!pool.has(k)) pool.set(k, []); pool.get(k).push(j); } });
    const gone = new Set(), still = [];
    for (const item of missing) {
      let hit = -1;
      for (const k of looseKeys(item, 'src')) { const list = pool.get(k) || []; while (list.length && gone.has(list[0])) list.shift(); if (list.length) { hit = list.shift(); break; } }
      if (hit >= 0) gone.add(hit); else still.push(item);
    }
    added = added.filter((_, j) => !gone.has(j));
    return pairUp(category, still, added);
  }
  return pairUp(category, missing, added);
}

function pairUp(category, missing, added) {
  const flags = [];
  const left = bucket(added, o => o.g);
  const pairedOut = new Set();
  const single = missing.length === 1 && added.length === 1;
  for (const item of missing) {
    const list = left.get(item.g);
    let o = list && list.length ? list.shift() : null;
    if (!o && single) o = added[0];
    if (o) { pairedOut.add(o); flags.push({ category, kind: 'changed', source: span(item), output: span(o) }); }
    else flags.push({ category, kind: 'missing', source: span(item), output: null });
  }
  for (const o of added) if (!pairedOut.has(o)) flags.push({ category, kind: 'added', source: null, output: span(o) });
  return flags;
}

// Count-based comparison within aligned groups, for words too common to match across a whole text.
function compareInGroups(category, src, out, classOf) {
  const a = bucket(src, i => i.g), b = bucket(out, i => i.g);
  const flags = [];
  const name = item => typeof category === 'function' ? category(item) : category;
  for (const g of new Set([...a.keys(), ...b.keys()])) {
    const ac = bucket(a.get(g) || [], classOf), bc = bucket(b.get(g) || [], classOf);
    const missing = [], added = [];
    for (const c of new Set([...ac.keys(), ...bc.keys()])) {
      const x = ac.get(c) || [], y = bc.get(c) || [];
      if (x.length > y.length) missing.push(...x.slice(y.length));
      if (y.length > x.length) added.push(...y.slice(x.length));
    }
    while (missing.length && added.length) { const m = missing.shift(), o = added.shift(); flags.push({ category: name(m), kind: 'changed', source: span(m), output: span(o) }); }
    for (const m of missing) flags.push({ category: name(m), kind: 'missing', source: span(m), output: null });
    for (const o of added) flags.push({ category: name(o), kind: 'added', source: null, output: span(o) });
  }
  return flags;
}

export function extractDetails(text) {
  const spans = sourceSpans(text);
  const dates = findDates(text);
  const numbers = findNumbers(text, dates);
  const prohibitions = findPattern(text, 'prohibition', MODALS[0][1]);
  const modals = [...prohibitions];
  for (const [cls, re] of MODALS.slice(1)) modals.push(...findPattern(text, cls, re, modals));
  const negations = findPattern(text, 'negation', NEGATORS, findPattern(text, 'bound', BOUNDS));
  const qualifiers = [];
  for (const [cls, re] of QUALIFIERS) qualifiers.push(...findPattern(text, cls, re, qualifiers));
  const names = findNames(text, spans, dates);
  return { spans, dates, numbers, modals, negations, qualifiers, names };
}

function clauseAfter(text, item) {
  return window(text, item.end, 120).split(/[,;.!?:]/)[0];
}

export function detailFlags(source, output) {
  const a = extractDetails(source), b = extractDetails(output);
  if (a.spans.length > 300 || b.spans.length > 300) throw new Error('Detail checks support up to 300 sentence/line spans per text. Split this document into sections.');
  const groups = align(a.spans, b.spans);
  for (const list of [a.dates, a.numbers, a.modals, a.negations, a.qualifiers, a.names]) for (const item of list) item.g = groups.source[sentenceOf(a.spans, item.start)];
  for (const list of [b.dates, b.numbers, b.modals, b.negations, b.qualifiers, b.names]) for (const item of list) item.g = groups.output[sentenceOf(b.spans, item.start)];
  const flags = [];
  // A bare number equals the same value with a unit or currency sign ("$1,000" and "1000").
  // Added items offer keys; missing items look them up. Only one side of a loose match may be bare.
  flags.push(...compareByKey('number', a.numbers, b.numbers, (x, side) => {
    const v = valueOf(x.key), bare = !unitOf(x.key);
    if (side === 'out') return bare ? [`bare:${v}`, `any:${v}`] : [`any:${v}`];
    return bare ? [`any:${v}`] : [`bare:${v}`];
  }));
  // A date without a year equals the same day with one.
  flags.push(...compareByKey('date', a.dates, b.dates, (x, side) => {
    if (!x.key.startsWith('d:')) return [];
    const [m, d, y] = x.key.slice(2).split('-');
    if (side === 'out') return y ? [`any:${m}-${d}`] : [`noyear:${m}-${d}`, `any:${m}-${d}`];
    return y ? [`noyear:${m}-${d}`] : [`any:${m}-${d}`];
  }));
  // A name is kept when the other text names the same entity in a shorter or longer form (one name's
  // distinctive words are a subset of the other's), or uses all of its distinctive words.
  const capitalized = text => new Set(wordsWithOffsets(text).map(t => t.w).filter(w => /^\p{Lu}/u.test(w)));
  const inB = capitalized(output), inA = capitalized(source);
  const subset = (x, y) => x.every(t => y.includes(t));
  const kept = (n, others, words) => n.tokens.every(t => words.has(t)) || others.some(m => subset(m.tokens, n.tokens) || subset(n.tokens, m.tokens));
  flags.push(...pairUp('name', a.names.filter(n => !kept(n, b.names, inB)), b.names.filter(n => !kept(n, a.names, inA))));
  flags.push(...compareInGroups('negation', a.negations, b.negations, () => 'negation'));
  const modalFlags = compareInGroups('modal', a.modals, b.modals, item => item.category);
  // An obligation or prohibition turned into an instruction ("You must pay" to "Pay ...", "You must not"
  // to "Do not ...") keeps its force, so a missing modal is not flagged when an aligned rewrite sentence
  // starts with the verb that followed it, or with "Do not", "Don't" or "Never" for a prohibition.
  const modalAt = new Map(a.modals.map(m => [m.start, m]));
  const startsOf = bucket(b.spans.map((sp, j) => ({ text: sp.text, g: groups.output[j] })), x => x.g);
  for (const f of modalFlags) {
    if (f.kind === 'missing') {
      const item = modalAt.get(f.source.start);
      const next = (window(source, item.end, 60).match(/[\p{L}]+/gu) || []).filter(w => !/^(?:be|to|not|always)$/i.test(w)).slice(0, 2).map(w => w.toLowerCase().slice(0, 4));
      const imperative = (startsOf.get(item.g) || []).some(({ text: t }) => {
        const first = (t.match(/[\p{L}]+/u) || [''])[0].toLowerCase();
        if (item.category === 'prohibition') return /^(?:do\s+not|don['’]t|never)\b/i.test(t);
        return item.category === 'obligation' && next.includes(first.slice(0, 4));
      });
      if (imperative) continue;
    }
    flags.push(f);
  }
  // Conditions and exceptions: compared as one pool of qualifiers, so "unless" and "except" can trade.
  const qualifierFlags = compareInGroups(item => item.category, a.qualifiers, b.qualifiers, () => 'qualifier');
  flags.push(...qualifierFlags);
  // A qualifier kept on both sides is flagged when its clause loses at least two content words, and at
  // least half of them, against every counterpart clause in the aligned rewrite sentences.
  const flagged = new Set(qualifierFlags.filter(f => f.source).map(f => f.source.start));
  const peersOf = bucket(b.qualifiers, o => o.g);
  for (const q of a.qualifiers) {
    const peers = peersOf.get(q.g) || [];
    if (!peers.length || flagged.has(q.start)) continue;
    const mine = content(clauseAfter(source, q));
    if (mine.size < 2) continue;
    const lost = peers.map(o => { const theirs = content(clauseAfter(output, o)); return [...mine].filter(t => !theirs.has(t)).length; });
    const best = Math.min(...lost);
    if (best >= 2 && best / mine.size >= 0.5) flags.push({ category: q.category, kind: 'changed', source: span(q), output: span(peers[lost.indexOf(best)]) });
  }
  const order = f => (f.source ? f.source.start : 1e9) + (f.output ? f.output.start / 1e6 : 0);
  return { method: DETAIL_FLAGS_ID, scope: DETAIL_SCOPE, offset_unit: 'utf16-code-unit', flags: flags.sort((x, y) => order(x) - order(y)) };
}
