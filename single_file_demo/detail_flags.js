// Critical-detail checks: lexical, deterministic, computed in the browser with no network request.
// Pure functions shared by the browser and the offline Node tests. A flag says a detail of one of seven
// kinds (number, date, negation, modal, condition, exception, name) has no literal or equal-valued
// counterpart in the other text. It can miss changes and can flag details that were kept in other words.
import { sourceSpans } from './review_packet.js';

export const DETAIL_FLAGS_ID = 'sum.detail_flags.lexical_v1';
export const DETAIL_SCOPE = 'Lexical checks for numbers, dates, negations, modal verbs, conditions, exceptions and names. A flag means no literal or equal-valued match was found nearby; it is not a judgement of meaning, and changes outside these words are not checked.';

const UNITS = [
  [/^(?:business|working)\s+days?\b/i, 'businessday', 1], [/^(?:calendar\s+)?days?\b/i, 'day', 1], [/^nights?\b/i, 'night', 1],
  [/^weeks?\b/i, 'day', 7], [/^fortnights?\b/i, 'day', 14], [/^months?\b/i, 'month', 1], [/^years?\b/i, 'month', 12],
  [/^hours?\b/i, 'minute', 60], [/^minutes?\b/i, 'minute', 1], [/^(?:percent|per\s+cent)\b/i, 'fraction', 0.01], [/^%/, 'fraction', 0.01],
  [/^(?:dollars?|usd)\b/i, 'money:$', 1], [/^(?:euros?|eur)\b/i, 'money:€', 1], [/^(?:pounds?|gbp)\b/i, 'money:£', 1], [/^cents?\b/i, 'money:$', 0.01],
  [/^(?:tablets?|pills?|capsules?)\b/i, 'tablet', 1], [/^(?:mg|milligrams?)\b/i, 'mg', 1], [/^(?:ml|millilit(?:er|re)s?)\b/i, 'ml', 1],
  [/^times\b/i, 'times', 1], [/^doses?\b/i, 'dose', 1], [/^(?:miles?)\b/i, 'mile', 1], [/^(?:km|kilomet(?:er|re)s?)\b/i, 'km', 1],
];
const SMALL = { zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19, twenty: 20, thirty: 30, forty: 40, fifty: 50, sixty: 60, seventy: 70, eighty: 80, ninety: 90 };
const SCALE = { hundred: 100, thousand: 1000, million: 1e6 };
const MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'];
const MONTH_RE = '(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\\.?';
const WEEKDAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'];

const round = v => Math.round(v * 1e6) / 1e6;
const monthIndex = m => MONTHS.findIndex(name => name.startsWith(m.toLowerCase().replace('.', '').slice(0, 3)));

function covered(ranges, start, end) { return ranges.some(r => start < r.end && r.start < end); }

function findDates(text) {
  const items = [];
  const add = (m, key) => items.push({ category: 'date', key, start: m.index, end: m.index + m[0].length, text: m[0] });
  const year = '(?:,?\\s+(\\d{4}))?';
  for (const m of text.matchAll(new RegExp(`\\b(${MONTH_RE})\\s+(\\d{1,2})(?:st|nd|rd|th)?\\b${year}`, 'g'))) add(m, `d:${monthIndex(m[1])}-${+m[2]}-${m[3] || ''}`);
  for (const m of text.matchAll(new RegExp(`\\b(\\d{1,2})(?:st|nd|rd|th)?\\s+(?:of\\s+)?(${MONTH_RE})(?![a-z])${year}`, 'g'))) if (!covered(items, m.index, m.index + m[0].length)) add(m, `d:${monthIndex(m[2])}-${+m[1]}-${m[3] || ''}`);
  for (const m of text.matchAll(new RegExp(`\\b(${MONTH_RE})\\s+(\\d{4})\\b`, 'g'))) if (!covered(items, m.index, m.index + m[0].length)) add(m, `d:${monthIndex(m[1])}--${m[2]}`);
  for (const m of text.matchAll(/\b(\d{4})-(\d{2})-(\d{2})\b/g)) add(m, `d:${+m[2] - 1}-${+m[3]}-${m[1]}`);
  for (const m of text.matchAll(/\b(\d{1,2})\/(\d{1,2})\/(\d{2,4})\b/g)) add(m, `d:${+m[1] - 1}-${+m[2]}-${m[3].length === 2 ? '20' + m[3] : m[3]}`);
  for (const m of text.matchAll(/\b(Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)days?\b/g)) add(m, `w:${m[1].toLowerCase()}`);
  for (const m of text.matchAll(/\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s?m\b\.?/gi)) {
    if (covered(items, m.index, m.index + m[0].length)) continue;
    const h = (+m[1] % 12) + (m[3].toLowerCase() === 'p' ? 12 : 0);
    add(m, `t:${h * 60 + (+m[2] || 0)}`);
  }
  for (const m of text.matchAll(/\b([01]?\d|2[0-3]):([0-5]\d)\b/g)) if (!covered(items, m.index, m.index + m[0].length)) add(m, `t:${+m[1] * 60 + +m[2]}`);
  for (const m of text.matchAll(/\b(noon|midday|midnight)\b/gi)) add(m, `t:${/mid(?:night)/i.test(m[1]) && m[1].toLowerCase() === 'midnight' ? 0 : 720}`);
  // A month name standing alone ("in March") is also a date detail; "May" alone is a modal verb.
  for (const m of text.matchAll(new RegExp(`\\b(${MONTH_RE.replace('May|', '')})(?![a-z])`, 'g'))) if (!covered(items, m.index, m.index + m[0].length)) add(m, `d:${monthIndex(m[1])}--`);
  return items;
}

function spelledAt(text, i) {
  const m = /^(?:(a|an)\s+(?=hundred|thousand|dozen|half)|)((?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[\s-](?:one|two|three|four|five|six|seven|eight|nine))?(?:\s+(?:hundred|thousand|million))?|hundred|thousand|dozen)\b/i.exec(text.slice(i));
  if (!m) return null;
  let value = 0, part = 0;
  for (const w of m[2].toLowerCase().split(/[\s-]+/)) {
    if (w === 'dozen') part = (part || 1) * 12;
    else if (w in SCALE) { part = (part || 1) * SCALE[w]; if (SCALE[w] >= 1000) { value += part; part = 0; } }
    else part += SMALL[w];
  }
  return { length: m[0].length, value: value + part, word: m[2].toLowerCase() };
}

function findNumbers(text, taken) {
  const items = [];
  const re = /([$€£]\s?)?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)|\b(?:an?|half|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|dozen)\b/gi;
  for (const m of text.matchAll(re)) {
    const start = m.index;
    if (covered(taken, start, start + m[0].length) || covered(items, start, start + 1)) continue;
    if (/[\p{L}\d]/u.test(text[start - 1] || '') && !m[1]) continue;
    let value, end, unit = '', word = null;
    if (m[2] !== undefined) {
      value = Number(m[2].replace(/,/g, '')); end = start + m[0].length;
      if (m[1]) unit = `money:${m[1].trim()}`;
    } else {
      const lower = m[0].toLowerCase();
      if (lower === 'half') {
        const rest = /^half\s+(?:an?\s+)?/i.exec(text.slice(start));
        const u = UNITS.find(([r]) => r.test(text.slice(start + rest[0].length)));
        if (u && u[1] !== 'fraction') { const len = u[0].exec(text.slice(start + rest[0].length))[0].length; items.push({ category: 'number', key: `${u[1]}:${round(0.5 * u[2])}`, start, end: start + rest[0].length + len, text: text.slice(start, start + rest[0].length + len) }); }
        else items.push({ category: 'number', key: 'fraction:0.5', start, end: start + 4, text: text.slice(start, start + 4) });
        continue;
      }
      if (lower === 'a' || lower === 'an') {
        const after = text.slice(start + m[0].length).replace(/^\s+/, '');
        const offset = text.length - after.length;
        const u = UNITS.find(([r, unitName]) => unitName !== 'fraction' && !unitName.startsWith('money') && r.test(after));
        if (!u) continue;
        const len = u[0].exec(after)[0].length;
        items.push({ category: 'number', key: `${u[1]}:${round(u[2])}`, start, end: offset + len, text: text.slice(start, offset + len) });
        continue;
      }
      const s = spelledAt(text, start);
      if (!s) continue;
      value = s.value; end = start + s.length; word = s.word;
    }
    const tail = text.slice(end);
    const gap = /^[\s-]*/.exec(tail)[0];
    const u = unit ? null : UNITS.find(([r]) => r.test(tail.slice(gap.length)));
    let factor = 1;
    if (u) { unit = u[1]; factor = u[2]; end += gap.length + u[0].exec(tail.slice(gap.length))[0].length; }
    else if (/^\s*(?:dollars?)\b/i.test(tail) && unit.startsWith('money')) end += /^\s*dollars?/i.exec(tail)[0].length;
    if (word === 'one' && !u && !/^\s+and\s+a\s+half\b/i.test(tail)) continue; // "one of", "no one": a pronoun more often than a count
    items.push({ category: 'number', key: `${unit}:${round(value * factor)}`, start, end, text: text.slice(start, end) });
  }
  for (const item of items) {
    const half = /^\s+and\s+a\s+half\b/i.exec(text.slice(item.end));
    if (half && /^:\d|^:$/.test(item.key.replace(/^[^:]*/, ''))) {
      const tail = text.slice(item.end + half[0].length);
      const gap = /^[\s-]*/.exec(tail)[0];
      const u = UNITS.find(([r]) => r.test(tail.slice(gap.length)));
      const value = Number(item.key.split(':')[1]) + 0.5;
      item.end += half[0].length;
      if (u) { item.end += gap.length + u[0].exec(tail.slice(gap.length))[0].length; item.key = `${u[1]}:${round(value * u[2])}`; }
      else item.key = `:${round(value)}`;
      item.text = text.slice(item.start, item.end);
    }
  }
  for (let k = items.length - 1; k > 0; k--) if (items[k].start < items[k - 1].end) items.splice(k, 1);
  for (const m of text.matchAll(/\b(once|twice|thrice)\b/gi)) {
    if (covered(taken, m.index, m.index + m[0].length)) continue;
    items.push({ category: 'number', key: `times:${{ once: 1, twice: 2, thrice: 3 }[m[1].toLowerCase()]}`, start: m.index, end: m.index + m[0].length, text: m[0] });
  }
  return items;
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
  const isInitial = start => start === 0 || starts.has(start) || /[.!?:;"“(\n]$/.test(text.slice(0, start).trimEnd());
  const words = wordsWithOffsets(text);
  words.forEach(t => { if (/^\p{Lu}/u.test(t.w) && !isInitial(t.start)) capitalizedSomewhere.add(t.w); });
  const isCap = t => /^\p{Lu}/u.test(t.w) && !COMMON_ACRONYMS.has(t.w) && !covered(taken, t.start, t.end);
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
  ['prohibition', /\b(?:(?:can|may|must|could)\s+never|must\s+not|may\s+not|shall\s+not|should\s+not|can\s*not|can['’]t|mustn['’]t|(?:is|are|be)\s+not\s+(?:allowed|permitted)|not\s+(?:allowed|permitted)|prohibited|forbidden)\b/gi],
  ['obligation', /\b(?:must|shall|ha(?:ve|s|d)\s+to|needs?\s+to|(?:is|are|be)\s+required|required\s+to|requires?|obliged|obligated|mandatory|compulsory)\b/gi],
  ['permission', /\b(?:may|might|can|could|(?:is|are|be)\s+allowed|allowed\s+to|(?:is|are|be)\s+permitted|permitted\s+to|entitled\s+to|free\s+to|able\s+to)\b/gi],
  ['recommendation', /\b(?:should|ought\s+to|recommends?|recommended|advised?|encouraged?|best\s+to)\b/gi],
];
// "will" is not compared: plain-language rewrites add and drop it constantly ("we will refund" to "we refund").
const QUALIFIERS = [
  ['exception', /\b(?:unless|except(?:\s+(?:for|when|where|if))?|excepting|other\s+than|apart\s+from|aside\s+from|excluding|with\s+the\s+exception\s+of|but\s+not|save\s+for)\b/gi],
  ['condition', /\b(?:only\s+if|if|when(?:ever)?|provided(?:\s+that)?|providing|as\s+long\s+as|so\s+long\s+as|in\s+case|on\s+condition\s+that|in\s+the\s+event\s+that|assuming)\b/gi],
];

function findPattern(text, category, re, taken = []) {
  const items = [];
  for (const m of text.matchAll(re)) {
    if (covered(taken, m.index, m.index + m[0].length)) continue;
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

const sentenceOf = (spans, pos) => { const k = spans.findIndex(s => pos >= s.start && pos < s.end); return k < 0 ? spans.length - 1 : k; };
const span = item => item ? { start: item.start, end: item.end, text: item.text } : null;

// Multiset match by key; unmatched items on both sides pair up as "changed" when they share a group.
function compareByKey(category, src, out, groupOf, looseKey = null) {
  // First match within aligned groups, then across the text, but never take a counterpart from a group
  // that still has an unmatched detail of its own: that pairing is more likely a change.
  const usedOut = new Set(), usedSrc = new Set();
  src.forEach((item, i) => {
    const k = out.findIndex((o, j) => !usedOut.has(j) && o.key === item.key && groupOf.output(o) === groupOf.source(item));
    if (k >= 0) { usedOut.add(k); usedSrc.add(i); }
  });
  src.forEach((item, i) => {
    if (usedSrc.has(i)) return;
    const k = out.findIndex((o, j) => !usedOut.has(j) && o.key === item.key && !src.some((s2, i2) => !usedSrc.has(i2) && i2 !== i && groupOf.source(s2) === groupOf.output(o)));
    if (k >= 0) { usedOut.add(k); usedSrc.add(i); }
  });
  const missing = src.filter((_, i) => !usedSrc.has(i));
  let added = out.filter((_, i) => !usedOut.has(i));
  if (looseKey) {
    const still = [];
    for (const item of missing) {
      const k = added.findIndex(o => looseKey(item, o));
      if (k >= 0) added.splice(k, 1); else still.push(item);
    }
    missing.splice(0, missing.length, ...still);
  }
  return pairUp(category, missing, added, groupOf);
}

function pairUp(category, missing, added, groupOf) {
  const flags = [];
  const left = [...added];
  for (const item of missing) {
    let k = left.findIndex(o => groupOf.output(o) === groupOf.source(item));
    if (k < 0 && missing.length === 1 && left.length === 1) k = 0;
    if (k >= 0) { const o = left.splice(k, 1)[0]; flags.push({ category, kind: 'changed', source: span(item), output: span(o) }); }
    else flags.push({ category, kind: 'missing', source: span(item), output: null });
  }
  for (const o of left) flags.push({ category, kind: 'added', source: null, output: span(o) });
  return flags;
}

// Count-based comparison within aligned groups, for words too common to match across a whole text.
function compareInGroups(category, src, out, groupOf, classOf = item => item.category, skip = () => false) {
  const groups = new Set([...src.map(groupOf.source), ...out.map(groupOf.output)]);
  const flags = [];
  for (const g of groups) {
    const a = src.filter(i => groupOf.source(i) === g), b = out.filter(i => groupOf.output(i) === g);
    const classes = new Set([...a, ...b].map(classOf));
    const missing = [], added = [];
    for (const c of classes) {
      const ac = a.filter(i => classOf(i) === c), bc = b.filter(i => classOf(i) === c);
      if (ac.length > bc.length) missing.push(...ac.slice(bc.length));
      if (bc.length > ac.length) added.push(...bc.slice(ac.length));
    }
    const pairs = [];
    while (missing.length && added.length) pairs.push([missing.shift(), added.shift()]);
    for (const [m, o] of pairs) flags.push({ category: typeof category === 'function' ? category(m) : category, kind: 'changed', source: span(m), output: span(o) });
    for (const m of missing) if (!skip(m)) flags.push({ category: typeof category === 'function' ? category(m) : category, kind: 'missing', source: span(m), output: null });
    for (const o of added) if (!skip(o)) flags.push({ category: typeof category === 'function' ? category(o) : category, kind: 'added', source: null, output: span(o) });
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
  const rest = text.slice(item.end, item.end + 120);
  return rest.split(/[,;.!?:]/)[0];
}

export function detailFlags(source, output) {
  const a = extractDetails(source), b = extractDetails(output);
  if (a.spans.length > 300 || b.spans.length > 300) throw new Error('Detail checks support up to 300 sentence/line spans per text. Split this document into sections.');
  const groups = align(a.spans, b.spans);
  const groupOf = {
    source: item => groups.source[sentenceOf(a.spans, item.start)],
    output: item => groups.output[sentenceOf(b.spans, item.start)],
  };
  const flags = [];
  flags.push(...compareByKey('number', a.numbers, b.numbers, groupOf, (x, y) => x.key.split(':')[1] === y.key.split(':')[1] && (x.key.startsWith(':') || y.key.startsWith(':'))));
  flags.push(...compareByKey('date', a.dates, b.dates, groupOf, (x, y) => {
    const [p, q] = [x.key, y.key];
    if (!p.startsWith('d:') || !q.startsWith('d:')) return false;
    const [pm, pd, py] = p.slice(2).split('-'), [qm, qd, qy] = q.slice(2).split('-');
    return pm === qm && pd === qd && (!py || !qy);
  }));
  // Names: kept if any distinctive word appears capitalized anywhere in the other text.
  // A name is kept when the other text names the same entity in a shorter or longer form (one name's
  // distinctive words are a subset of the other's), or uses all of its distinctive words.
  const capitalized = text => new Set(wordsWithOffsets(text).map(t => t.w).filter(w => /^\p{Lu}/u.test(w)));
  const inB = capitalized(output), inA = capitalized(source);
  const subset = (x, y) => x.every(t => y.includes(t));
  const kept = (n, others, words) => n.tokens.every(t => words.has(t)) || others.some(m => subset(m.tokens, n.tokens) || subset(n.tokens, m.tokens));
  const lostNames = a.names.filter(n => !kept(n, b.names, inB));
  const newNames = b.names.filter(n => !kept(n, a.names, inA));
  flags.push(...pairUp('name', lostNames, newNames, groupOf));
  // Negation: a count per aligned group; a negator kept next to the same word counts as matched.
  flags.push(...compareInGroups('negation', a.negations, b.negations, groupOf, () => 'negation'));
  // Modals: compared by class within aligned groups; a dropped or added "will" alone is not flagged.
  flags.push(...compareInGroups('modal', a.modals, b.modals, groupOf, item => item.category, item => item.category === 'commitment'));
  // An obligation or prohibition turned into an instruction ("You must pay" to "Pay ...", "You must not"
  // to "Do not ...") keeps its force, so a missing modal is not flagged when an aligned rewrite sentence
  // starts with the verb that followed it, or with "Do not", "Don't" or "Never" for a prohibition.
  for (let k = flags.length - 1; k >= 0; k--) {
    const f = flags[k];
    if (f.category !== 'modal' || f.kind !== 'missing') continue;
    const item = a.modals.find(m => m.start === f.source.start);
    const g = groupOf.source(item);
    const starts = b.spans.filter((_, j) => groups.output[j] === g).map(sp => sp.text);
    const next = (source.slice(item.end).match(/[\p{L}]+/gu) || []).filter(w => !/^(?:be|to|not|always)$/i.test(w)).slice(0, 2).map(w => w.toLowerCase().slice(0, 4));
    const imperative = starts.some(t => {
      const first = (t.match(/[\p{L}]+/u) || [''])[0].toLowerCase();
      if (item.category === 'prohibition') return /^(?:do\s+not|don['’]t|never)\b/i.test(t);
      return item.category === 'obligation' && next.includes(first.slice(0, 4));
    });
    if (imperative) flags.splice(k, 1);
  }
  // Conditions and exceptions: compared as one pool of qualifiers, so "unless" and "except" can trade.
  flags.push(...compareInGroups(item => item.category, a.qualifiers, b.qualifiers, groupOf, () => 'qualifier'));
  // A qualifier kept on both sides whose clause shares no content word with any counterpart is flagged.
  for (const q of a.qualifiers) {
    const g = groupOf.source(q);
    const peers = b.qualifiers.filter(o => groupOf.output(o) === g);
    if (!peers.length || flags.some(f => f.source && f.source.start === q.start)) continue;
    const mine = content(clauseAfter(source, q));
    if (mine.size < 2) continue;
    const kept = peers.some(o => { const theirs = content(clauseAfter(output, o)); return [...mine].some(t => theirs.has(t)); });
    if (!kept) flags.push({ category: q.category, kind: 'changed', source: span(q), output: span(peers[0]) });
  }
  const order = f => (f.source ? f.source.start : 1e9) + (f.output ? f.output.start / 1e6 : 0);
  return { method: DETAIL_FLAGS_ID, scope: DETAIL_SCOPE, offset_unit: 'utf16-code-unit', flags: flags.sort((x, y) => order(x) - order(y)) };
}
