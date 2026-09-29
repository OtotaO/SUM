// Independent checks for the evidence forms exercised by this test corpus.
// They compare printed claims and their quoted spans with the raw strings.
//
// The checks do not reuse the engine's matching. They re-tokenize with their
// own copy of the word pattern, lower case with their own code, rebuild each
// Read-as view from the rendered pieces, and parse each statement back into
// the facts it asserts. Unsupported statement forms and edit bounds that
// exceed the independent check's work budget are reported as failures. These
// tests establish coverage for the committed fixtures, not all possible text.
//
// Used by test_evidence_oracle.mjs (the built-in examples, an adversarial
// corpus and a seeded random mutation fuzz) and by
// scripts/real_browser_check.mjs, which compares what the real page prints
// with what this oracle accepts. Not loaded by the page.
import * as E from './change_evidence.js';
import { compareTexts } from './review_packet.js';

// ---------------------------------------------------------------- independent helpers
const CJK = '\\p{Script=Han}\\p{Script=Hiragana}\\p{Script=Katakana}';
const W = `(?:(?![${CJK}])[\\p{L}\\p{N}][\\p{M}]*)`;
const WORD = new RegExp(`[${CJK}]|[$€£¥]?${W}+(?:[.,:\\/'’\\-]${W}+)*%?`, 'gu');
const words = s => Array.from(s.matchAll(WORD), m => ({ t: m[0], k: m[0].toLocaleLowerCase('en').replace(/’/g, "'"), s: m.index, e: m.index + m[0].length }));
export const countWords = (text, phrase) => {
  const hay = words(text).map(w => w.k), keys = words(phrase).map(w => w.k);
  if (!keys.length) return 0;
  let c = 0;
  for (let i = 0; i + keys.length <= hay.length; i++) if (keys.every((k, n) => hay[i + n] === k)) { c++; i += keys.length - 1; }
  return c;
};
const sameWords = (a, b) => { const x = words(a).map(w => w.k), y = words(b).map(w => w.k); return x.length === y.length && x.every((k, i) => k === y[i]); };

// Independent LCS length via match-position subsequences (Hunt-Szymanski),
// not the engine's table/Myers implementation. Counts insertions + deletions.
// Return null instead of silently accepting a bound we did not establish.
function independentEditDistance(a, b, limit) {
  let x = words(a).map(w => w.k), y = words(b).map(w => w.k);
  if (x.length + y.length <= limit) return x.length + y.length; // enough to refute the claimed lower bound
  let start = 0, endX = x.length, endY = y.length;
  while (start < endX && start < endY && x[start] === y[start]) start++;
  while (endX > start && endY > start && x[endX - 1] === y[endY - 1]) { endX--; endY--; }
  x = x.slice(start, endX); y = y.slice(start, endY);
  const positions = new Map();
  y.forEach((key, i) => { if (!positions.has(key)) positions.set(key, []); positions.get(key).push(i); });
  const counts = new Map();
  for (const key of x) counts.set(key, (counts.get(key) || 0) + 1);
  let common = 0, work = 0;
  for (const [key, n] of counts) { const m = positions.get(key)?.length || 0; common += Math.min(n, m); work += n * m; }
  const lowerBound = x.length + y.length - 2 * common;
  if (lowerBound > limit) return lowerBound;
  if (work > 4000000) return null;
  const tails = [];
  for (const key of x) {
    const matches = positions.get(key) || [];
    for (let j = matches.length - 1; j >= 0; j--) {
      const at = matches[j]; let lo = 0, hi = tails.length;
      while (lo < hi) { const mid = (lo + hi) >>> 1; if (tails[mid] < at) lo = mid + 1; else hi = mid; }
      tails[lo] = at;
    }
  }
  return x.length + y.length - 2 * tails.length;
}
const WHITESPACE_NAMES = { space: ' ', tab: '\t', 'non-breaking space': ' ', 'line break': '\n' };
function readChars(phrase) {
  // “x” -> x ; "a space" -> " " ; "3 spaces" -> "   " ; "N whitespace characters" -> null (checked as whitespace)
  let m;
  if ((m = /^“([\s\S]*)”$/.exec(phrase))) return { text: m[1] };
  if ((m = /^code points? ((?:U\+[0-9A-F]+)(?: U\+[0-9A-F]+)*)$/.exec(phrase))) return { text: m[1].split(' ').map(x => String.fromCodePoint(parseInt(x.slice(2), 16))).join('') };
  if ((m = /^a (space|tab|non-breaking space|line break|whitespace character)$/.exec(phrase))) return m[1] === 'whitespace character' ? { ws: 1 } : { text: WHITESPACE_NAMES[m[1]] };
  if ((m = /^(\d+) (spaces|tabs|non-breaking spaces|line breaks|whitespace characters)$/.exec(phrase))) {
    const n = Number(m[1]);
    return m[2] === 'whitespace characters' ? { ws: n } : { text: WHITESPACE_NAMES[m[2].replace(/s$/, '')].repeat(n) };
  }
  return null;
}
export const charsMatch = (phrase, actual) => {
  const r = readChars(phrase);
  if (!r) return false;
  if (r.ws !== undefined) return actual.length === r.ws && /^\s+$/.test(actual);
  return r.text === actual;
};
const timesOf = w => (w === 'once' ? 1 : w === 'twice' ? 2 : Number(/^(\d+) times$/.exec(w)?.[1]));

// Rebuild a view from the rendered pieces, independently of the engine's viewText.
function view(segs, side) {
  let out = '';
  for (const s of segs) {
    if (s.t === 'text' || s.t === 'eq') out += s.text;
    else if (s.t === 'block' && s.side === side) out += s.text;
    else if (s.t === 'run' && (s.side === 'd') === (side === 'a')) for (const p of s.parts) if (p.t !== 'ref') out += p.text;
  }
  return out;
}

// ---------------------------------------------------------------- the oracle
export function check(source, output, method, mutateEvidence = null) {
  const bad = [];
  const fail = (what, detail) => bad.push(`${what}: ${detail}`);
  let review;
  try { review = compareTexts(source, output, method); } catch { return bad; } // refused texts print nothing about the pair
  const evidence = E.buildEvidence(source, output, review);
  if (mutateEvidence) mutateEvidence(evidence);
  const { passages, summary } = evidence;
  const rewritePassage = new Map(review.rows.filter(r => r.output).map(r => [r.output.id, r.output]));
  const originalPassage = new Map(review.rows.filter(r => r.source).map(r => [r.source.id, r.source]));
  const passageOf = (side, k) => (side === 'b' ? rewritePassage : originalPassage).get(`s${k}`);

  for (const p of passages) {
    const tag = `§${p.number} ${JSON.stringify((p.a || p.b).text.slice(0, 40))}`;
    const A = p.a ? p.a.text : '', B = p.b ? p.b.text : '';
    // Views: every character of both passages, exactly.
    const segs = E.blacklineSegments(p, source, output);
    if (view(segs, 'a') !== A) fail('ORIGINAL-VIEW', `${tag} shows ${JSON.stringify(view(segs, 'a'))}`);
    if (view(segs, 'b') !== B) fail('REWRITE-VIEW', `${tag} shows ${JSON.stringify(view(segs, 'b'))}`);
    // Passage message.
    const msg = E.passageMessage(p);
    if (msg === 'Identical text in both.' && A !== B) fail('IDENTICAL', tag);
    if (msg && msg.startsWith('The normalized words match in order') && (!sameWords(A, B) || A === B || !p.notes.length)) fail('SAME-WORDS', tag);
    if (msg && msg.startsWith('Only common words')) {
      const content = s => words(s).map(w => w.k).filter(k => !E.STOP.has(k));
      if (A === B || JSON.stringify(content(A)) !== JSON.stringify(content(B))) fail('COMMON-ONLY', tag);
    }
    if (p.over) {
      const limit = p.over.limit;
      const distance = Number.isSafeInteger(limit) && limit >= 0 && p.a && p.b ? independentEditDistance(A, B, limit) : -1;
      if (distance === null) fail('OVER-UNPROVED', tag);
      else if (distance <= limit) fail('OVER', `${tag}: independent edit bound ${distance}, reported limit ${limit}`);
    }
    if (!msg && p.kind === 'verbatim') fail('VERBATIM-MESSAGE', tag);

    for (const n of p.notes) {
      // Check stored quotations before noteSpans slices them from the raw text;
      // checking the slices against themselves would be tautological.
      for (const [side, raw, passage] of [['a', source, p.a], ['b', output, p.b]]) {
        for (const item of [n[side], ...(n[side + 'Runs'] || [])].filter(Boolean)) {
          if (!passage || item.s < passage.start || item.e > passage.end || item.s >= item.e || raw.slice(item.s, item.e) !== item.text) fail('OWN-TEXT', `${tag} ${side}`);
        }
        const chars = n[side + 'Chars'];
        if (chars && (!passage || chars.s < passage.start || chars.e > passage.end || chars.s >= chars.e)) fail('CHARS-SPAN', `${tag} ${side}`);
      }
      const spans = E.noteSpans(n, source, output);
      for (const sp of spans) if ((sp.side === 'a' ? source : output).slice(sp.s, sp.e) !== sp.text) fail('SPAN', tag);
      const st = E.noteStatement(n, p, source, output);
      const ntag = `${tag} ${n.letter} [${n.kind}/${n.state}] ${st}`;
      const aSpans = spans.filter(sp => sp.side === 'a'), bSpans = spans.filter(sp => sp.side === 'b');
      const inA = s => p.a && aSpans.some(sp => sp.text === s && sp.s >= p.a.start && sp.e <= p.a.end);
      const inB = s => p.b && bSpans.some(sp => sp.text === s && sp.s >= p.b.start && sp.e <= p.b.end);
      let m;
      const SIDE = '(“[\\s\\S]*”|an? [a-z-]+(?: [a-z-]+)*|\\d+ [a-z-]+(?: [a-z-]+)*|code points? U\\+[0-9A-F]+(?: U\\+[0-9A-F]+)*)';
      if ((m = new RegExp(`^${SIDE} in the original, ${SIDE} in the rewrite\\.$`).exec(st))) {
        const sideMatches = (phrase, ss) => charsMatch(phrase, ss.map(sp => sp.text).join('')) || phrase === ss.map(sp => `“${sp.text}”`).join(' and ');
        if (!sideMatches(m[1], aSpans)) fail('DIFFERS-A', ntag);
        if (!sideMatches(m[2], bSpans)) fail('DIFFERS-B', ntag);
      } else if ((m = /^Marked (original|rewrite) characters: ([\s\S]+)\.$/.exec(st))) {
        const sp = m[1] === 'original' ? aSpans[0] : bSpans[0];
        if (!sp || !charsMatch(m[2], sp.text)) fail('CHARS-HERE', ntag);
      } else if ((m = /^This (original|rewrite) passage has no partner in the (rewrite|original)\.$/.exec(st))) {
        if ((m[1] === 'original' && p.b) || (m[1] === 'rewrite' && p.a)) fail('NO-PARTNER', ntag);
      } else if ((m = /^“([\s\S]*)” is matched in both passages by normalized words\.$/.exec(st))) {
        if (!inA(m[1]) || !inB(m[1])) fail('MOVED', ntag);
      } else if ((m = /^“([\s\S]*)” in the original and “([\s\S]*)” in the rewrite, matching normalized words; exact characters differ\.$/.exec(st))) {
        if (!inA(m[1]) || !inB(m[2]) || !sameWords(m[1], m[2]) || m[1] === m[2]) fail('MOVED-FORMS', ntag);
      } else if ((m = /^“([\s\S]*)” has a normalized word match in the (rewrite|original), starting in passage (\d+): “([\s\S]*)”\.$/.exec(st))) {
        const there = passageOf(m[2] === 'rewrite' ? 'b' : 'a', m[3]);
        const raw = m[2] === 'rewrite' ? output : source;
        if (!there || !raw.includes(m[4]) || !sameWords(m[1], m[4])) fail('OTHER-THERE', ntag);
        const matchSpan = spans.find(sp => sp.side === (m[2] === 'rewrite' ? 'b' : 'a') && sp.text === m[4]);
        if (!matchSpan || !there || matchSpan.s < there.start || matchSpan.s >= there.end) fail('OTHER-START', ntag);
      } else if ((m = /^“([\s\S]*)” has non-overlapping normalized word matches (\S+(?: times)?) in the (original|rewrite) passage and (\S+(?: times)?) in the paired (rewrite|original) passage\.$/.exec(st))) {
        const ownText = m[3] === 'original' ? A : B, otherText = m[3] === 'original' ? B : A;
        if (countWords(ownText, m[1]) !== timesOf(m[2]) || countWords(otherText, m[1]) !== timesOf(m[4])) fail('COUNT', ntag);
      } else if ((m = /^“([\s\S]*)” is in (?:an original|a rewrite) passage that has no partner in the (rewrite|original)\.$/.exec(st))) {
        if ((m[2] === 'rewrite' && p.b) || (m[2] === 'original' && p.a)) fail('UNPAIRED', ntag);
        if (!(m[2] === 'rewrite' ? A : B).includes(m[1])) fail('UNPAIRED-TEXT', ntag);
      } else if ((m = /^“([\s\S]*)” has no normalized word match in the paired (rewrite|original) passage\.$/.exec(st))) {
        if (countWords(m[2] === 'rewrite' ? B : A, m[1])) fail('NOT-IN-PAIRED', ntag);
      } else if ((m = /^“([\s\S]*)” has no normalized word match in the entire (rewrite|original) text\.$/.exec(st))) {
        if (countWords(m[2] === 'rewrite' ? output : source, m[1])) fail('FALSE-NOWHERE', ntag);
      } else {
        fail('UNRECOGNISED', ntag);
      }
    }
    // A word is one end of at most one Moved note.
    for (const side of ['a', 'b']) {
      const seen = new Set();
      for (const n of p.notes.filter(x => x.state === 'moved')) {
        for (const sp of E.noteSpans(n, source, output).filter(x => x.side === side)) {
          for (let i = sp.s; i < sp.e; i++) { if (seen.has(i)) { fail('MOVED-OVERLAP', `${tag} ${side} ${sp.text}`); break; } }
          for (let i = sp.s; i < sp.e; i++) seen.add(i);
        }
      }
    }
    for (const ib of p.inBoth) {
      const line = E.inBothText(ib);
      const aText = source.slice(ib.a.s, ib.a.e), bText = output.slice(ib.b.s, ib.b.e);
      let m;
      if ((m = /^“([\s\S]*)”$/.exec(line))) {
        if (m[1] !== aText || m[1] !== bText) fail('IN-BOTH', `${tag} ${line}`);
      } else if ((m = /^“([\s\S]*)” in the original, “([\s\S]*)” in the rewrite: matching normalized words; exact characters differ\.$/.exec(line))) {
        if (m[1] !== aText || m[2] !== bText || !sameWords(aText, bText)) fail('IN-BOTH-FORMS', `${tag} ${line}`);
        if (aText === bText) fail('IN-BOTH-EXACT-DIFFERS', tag);
      } else fail('UNRECOGNISED-IN-BOTH', `${tag} ${line}`);
      if (!p.a || !p.b || ib.a.s < p.a.start || ib.a.e > p.a.end || ib.b.s < p.b.start || ib.b.e > p.b.end) fail('IN-BOTH-OUTSIDE', tag);
    }
  }

  // Summary projections must agree with the actual review rows and evidence.
  const expectSummary = {
    passages: review.rows.length,
    notes: passages.reduce((n, p) => n + p.notes.length, 0),
    alsoMarked: passages.reduce((n, p) => n + p.alsoMarked.length, 0),
    identicalTexts: source === output,
    pairs: review.rows.filter(r => r.kind === 'changed-candidate').length,
    identical: review.rows.filter(r => r.kind === 'verbatim').length,
    originalOnly: review.rows.filter(r => r.kind === 'source-unmatched').length,
    rewriteOnly: review.rows.filter(r => r.kind === 'output-unmatched').length,
  };
  for (const [key, value] of Object.entries(expectSummary)) if (summary[key] !== value) fail('SUMMARY-' + key.toUpperCase(), `${summary[key]} != ${value}`);
  const expectedOver = passages.filter(p => p.over).map(p => ({ number: p.number, limit: p.over.limit }));
  if (JSON.stringify(summary.over) !== JSON.stringify(expectedOver)) fail('SUMMARY-OVER', 'does not match passage bounds');
  for (const p of passages) for (const item of p.alsoMarked) {
    const raw = item.side === 'a' ? source : output;
    if (raw.slice(item.tok.s, item.tok.e) !== item.tok.t || !E.STOP.has(words(item.tok.t)[0]?.k)) fail('COMMON-WORD', item.tok.t);
  }

  // Heading and facts.
  const heading = E.headingText(summary);
  if (/no literal differences/.test(heading) !== (source === output)) fail('HEADING-IDENTICAL', heading);
  if (/not identical/.test(heading) && source === output) fail('HEADING-NOT-IDENTICAL', heading);
  const noted = /(\d[\d,]*) differences? noted/.exec(heading);
  if (noted && Number(noted[1].replace(/,/g, '')) !== passages.reduce((s, p) => s + p.notes.length, 0)) fail('HEADING-COUNT', heading);
  if (!noted && source !== output && passages.some(p => p.notes.length)) fail('HEADING-MISSES-NOTES', heading);
  const paired = review.rows.filter(r => r.source && r.output).sort((x, y) => x.source.start - y.source.start);
  const reordered = paired.some((r, i) => i && r.output.start < paired[i - 1].output.start);
  for (const fact of E.summaryFacts(summary)) {
    if (/identical, character for character/.test(fact) && source !== output) fail('FACT-IDENTICAL', fact);
    if (/in a different order/.test(fact) && !reordered) fail('FACT-ORDER', fact);
    if (/differ only in the spacing or line breaks outside the passages/.test(fact)) {
      const all = review.rows.every(r => r.kind === 'verbatim');
      const joinA = paired.map(r => r.source.text).join('\u0000'), joinB = paired.map(r => r.output.text).join('\u0000');
      const outside = (text, spans) => { let t = text; for (const sp of [...spans].sort((x, y) => y.start - x.start)) t = t.slice(0, sp.start) + t.slice(sp.end); return t; };
      if (!all || reordered || joinA !== joinB || source === output || /\S/.test(outside(source, paired.map(r => r.source)) + outside(output, paired.map(r => r.output)))) fail('FACT-SPACING', fact);
    }
    if (/also differ\.$/.test(fact) && /spacing or line breaks/.test(fact)) {
      const gaps = (text, spans) => spans.map((sp, i) => text.slice(i ? spans[i - 1].end : 0, sp.start)).concat(text.slice(spans.length ? spans[spans.length - 1].end : 0)).join('\u0000');
      if (reordered || review.rows.some(r => !r.source || !r.output) || gaps(source, paired.map(r => r.source)) === gaps(output, paired.map(r => r.output))) fail('FACT-SPACING-ALSO', fact);
    }
  }
  return bad;
}

// ---------------------------------------------------------------- corpora
export const ADVERSARIAL = [
  ['Payment applies if the “red” box is checked, not otherwise.', 'Payment applies if the “red” box is checked, never otherwise.'],
  ['you can return it if unused. we refund in 10 days.', 'You can return it if unused. We refund in 10 days.'],
  ["Tenants don't smoke and don't vape.", 'Tenants don’t smoke.'],
  ['Pay within 30 days or 30 days after notice.', 'Pay within 30\u00a0days.'],
  ['Pay within 30 days or 30 days after notice.', 'Pay in 30\tdays.'],
  ['Visit οδος today.', 'Visit today. ΟΔΟΣ.'],
  ['STRASSE.', 'Straße.'],
  ['Late fees apply if rent is late. unless waived.', 'Late fees apply if rent is late. Unless waived.'],
  ['Refunds apply unless rent is overdue.', 'Refunds apply unless rent is\noverdue.'],
  ["Don't don't don't enter.", 'Don’t don’t enter.'],
  ['Refunds are given if the item is damaged.', 'Refunds are given if if the ITEM is damaged.'],
  ['may? except no, Dr.... ', 'may? except. except no, Dr.... '],
  ['pay €, Alice! 👍,', 'pay €, Pay Alice! 👍,'],
  ['Pay rent.', 'Pay rent.\n'],
  [' x', 'x'],
  ['x', 'x '],
  ['Cool it to 5 K now.', 'Cool it to 5 K now.'],
  ['Cool it to 5 Ω now.', 'Cool it to 5 Ω now.'],
  ['Put it in the can.', 'Put it in the box.'],
  ['Warning: ⚠️ hot. ⚠️ sharp.', 'Warning: ⚠ hot.'],
  ['Pay now.', 'Pay\u200b now.'],
  ['Pay the deposit within 30 days.', 'Pay the deposit within \u202e30 days.'],
  ['Do not pay. Call us.', 'Do pay. We will not call.'],
  ['You must not leave, and you may stay.', 'You must leave, and you may not stay.'],
  ['Tenants may not smoke, and may vape.', 'Tenants may smoke, and may not vape.'],
  ['Alice pays Bob.', 'Bob pays Alice.'],
  ['Call if glucose is < 70 mg/dL.', 'Call if glucose is > 70 mg/dL.'],
  ['Store at -20°C.', 'Store at 20°C.'],
  ['The change is +1 point.', 'The change is -1 point.'],
  ['Price: 50 € per month.', 'Price: 50 $ per month.'],
  ['Price: 50€ per month.', 'Price: 50$ per month.'],
  ['Great job team 👍 we ship Friday!', 'Great job team 👎 we ship Friday!'],
  ['Pay within 30 days. Late fees apply.', 'Late fees apply. Pay within 30 days.'],
  ["Let's eat, Grandma.", "Let's eat Grandma."],
  ['Do not sign.', 'Do NOT sign!'],
  ['NOTICE is required.', 'notice is required.'],
  ['The fee is $8 today. Nothing else.', 'The fee is waived. Total $8.50 today.'],
  ['Alice may cancel the lease.', "Alice's lease can be cancelled."],
  ['Bob signs. Alice pays.', "Bob signs. Alice's bank pays."],
  ['Take 1 tablet every 4 to 6 hours.', 'Take one tablet every 4-6 hours.'],
  ['You may not leave.', 'You cannot leave.'],
  ['Refunds are given unless the item is damaged. Returns are free.', 'Refunds are given. Returns are free unless you paid by card.'],
  ['Payment is due on 2026-03-01.', 'Payment is due at 2026-03-01T17:00.'],
  ['Pay the fee.', 'Pay a fee.'],
  ['Pay within 30 days.\nLate fees apply.', 'Pay within 30 days.  Late fees apply.'],
  ['Visit Paris in May.', 'Visit Paris.'],
  ['租金必须在30天内支付。押金可以退还。', '租金必须在60天内支付。押金不可退还。'],
  ['患者は1日3回、食後に2錠を服用してください。医師の指示がない限り、1日6錠を超えないでください。', '患者は1日2回、食後に2錠を服用してください。1日8錠を超えないでください。'],
  ['请在30天内退货。运费为$8，不予退还。', '请在60天内退货。运费不予退还。'],
  ['Do not run, do not jump. Stay calm.', 'Do not run, and jump. Stay calm.'],
  ['Items are refund eligible.', 'Items are non-refundable.'],
  ['On Tuesday, March 3, 2026, Mayor Jane Okafor announced the closure.', 'Mayor Jane Okafor announced the closure on March 4.'],
  ['Only adults may enter.', 'Adults may enter.'],
  ['Notwithstanding the foregoing, the Supplier may disclose it.', 'The Supplier may disclose it.'],
  ['The ÜBER tax applies.', 'The über tax applies.'],
  ['The deposit is refundable (unless rent is\n overdue).', 'The deposit is refundable.'],
  ['—', 'Something new.'],
  ['Keep the dose < 5 mg, taken daily by Alice.', 'Keep the dose > 5 mg taken daily by Bob.'],
  ['The dose is -5 mg.', 'The dose is 5 mg.'],
  ['Pay ₹500 now.', 'Pay 500 now.'],
  ['You must pay the fee before you leave.', 'Before you leave, you must pay the fee.'],
  ['Take the pill after you eat.', 'After you take the pill, eat.'],
  ['The Supplier shall not disclose any Confidential Information, except as required by law.', 'The Supplier will not share Confidential Information.'],
  ['Save 30 now.', 'Save 30% now.'],
  ['The fee is 7.95 per order.', 'The fee is $7.95 per order.'],
  ['Alice may cancel the lease with 30 days notice.', 'Alice may cancel the lease with 30 days notice.'],
  ['Call Mr. Smith on Monday.', 'Call Mr. Smith on Tuesday.'],
  ['Line one\n\nLine two.', 'Line one\nLine two changed.'],
  ['éclair costs 5.', 'éclair costs 5.'],
  ['A b c.', 'A  b\tc.'],
  ['<img src=x onerror="alert(1)"> Alice may cancel.', 'javascript:alert(3) Alice can cancel "><svg onload=alert(4)> the lease.'],
  ['', 'Only a rewrite.'],
  ['Only an original.', ''],
  // The QA pass's repro pairs for the first review round (cases.mjs), kept as a regression corpus.
  ["Take 1 tablet by mouth every 4 to 6 hours as needed for pain. Do not exceed 6 tablets in 24 hours. Do not take with alcohol. If symptoms persist for more than 3 days, stop use and call Dr. Patel. Children under 12 years should not use this product unless directed by a doctor.", "Take one tablet every 4-6 hours when you need it for pain. Don't take more than 8 tablets a day. Avoid alcohol. If symptoms last longer than 3 days, call your doctor. Children under 12 can use this product."], // QA pair: medication
  ["Returns are accepted within 30 days of purchase unless the item is marked final sale. Shipping costs $7.95 per order and is non-refundable. Refunds are issued to the original payment method within 5-7 business days. Store credit may be offered instead of a refund.", "You can return anything within 30 days. Shipping is $7.95 and we refund it. Refunds are issued to your card in 5 business days. We will always give a refund, never store credit."], // QA pair: refund
  ["The Supplier shall not disclose any Confidential Information to any third party, except as required by law or with the prior written consent of the Customer. This obligation survives termination of this Agreement for a period of five (5) years. Notwithstanding the foregoing, the Supplier may disclose Confidential Information to its auditors.", "The Supplier will not share Confidential Information with third parties, except when required by law. This obligation lasts for 5 years after the Agreement ends. The Supplier may disclose Confidential Information to its auditors and affiliates."], // QA pair: contract
  ["On Tuesday, March 3, 2026, Mayor Jane Okafor announced that the city of Riverton will close Elm Street Bridge for repairs. The closure will last approximately 14 weeks, according to city engineer Tomás Ruiz. Officials said about 12,000 vehicles cross the bridge daily.", "Mayor Jane Okafor said on March 4 that Riverton will close the Elm Street Bridge for about four months. City engineer Tomas Ruiz said the work was overdue. Roughly 12,000 cars use the bridge each day."], // QA pair: news
  ["Do NOT exceed 4 doses, in 24 hours; call us.", "do not exceed 4 doses in 24 hours: call us!"], // QA pair: punct_case
  ["Alice pays rent. Bob pays the deposit of $500.", "Bob pays the deposit. Alice pays rent of $500."], // QA pair: moved
  ["<img src=x onerror=alert(1)> Alice may cancel </script><script>alert(2)</script> the lease.", "javascript:alert(3) Alice can cancel \"><svg onload=alert(4)> the lease."], // QA pair: xss
  ["James will call Dr. Patel on Monday.", "Jim will call the doctor on Monday."], // QA pair: names
  ["Pay $1,000 by 2026-03-01 at 10:30.", "Pay $1000 by 2026-03-02 at 10:30."], // QA pair: dashes
  ["First line\n\nSecond line.", "First line\nSecond line changed."], // QA pair: empty_line
  ["Call Dr. Patel if a rash appears. Stop the tablets.", "Call your doctor if a rash appears. Stop the tablets."], // QA pair: abbrev
];

export const CORPUS_BASES = [
  ...Object.values(E.EXAMPLES).flatMap(ex => [ex.source, ex.output]),
  'Take 1 tablet by mouth every 4 to 6 hours as needed for pain. Do not exceed 6 tablets in 24 hours. Do not take with alcohol. If symptoms persist for more than 3 days, stop use and call Dr. Patel.',
  'The Supplier shall not disclose any Confidential Information to any third party, except as required by law or with the prior written consent of the Customer. Notwithstanding the foregoing, the Supplier may disclose Confidential Information to its auditors.',
  'On Tuesday, March 3, 2026, Mayor Jane Okafor announced that the city of Riverton will close Elm Street Bridge for repairs. Officials said about 12,000 vehicles cross the bridge daily.',
  '患者は1日3回、食後に2錠を服用してください。医師の指示がない限り、1日6錠を超えないでください。',
];
const VOCAB = ['not', 'no', 'never', 'may', 'must', 'can', 'shall', 'unless', 'if', 'except', 'only', 'within', '30', '60', '$8', '$200', '5%',
  '2026', 'March', 'Monday', 'days', 'Alice', 'Bob', 'Northwind', 'the', 'a', 'of', 'and', 'or', 'refund', 'Refund', 'NOT', "Alice's",
  'cannot', "don't", 'don’t', '4-6', '12,000', '👍', '€', '<', '>', '-', '+', ',', ';', '!', '(', ')', '—', '租金', '天', 'é', 'é', 'ÜBER'];

export function mulberry32(seed) {
  return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
export function mutate(text, rnd) {
  const pick = list => list[Math.floor(rnd() * list.length)];
  let t = text;
  const edits = 1 + Math.floor(rnd() * 4);
  for (let e = 0; e < edits; e++) {
    const parts = t.split(/(\s+)/);
    const wordsAt = parts.map((p, i) => (p && !/^\s+$/.test(p) ? i : -1)).filter(i => i >= 0);
    const i = wordsAt.length ? pick(wordsAt) : 0;
    switch (Math.floor(rnd() * 12)) {
      case 0: parts[i] = ''; break;                                             // delete a word
      case 1: parts[i] = `${parts[i] || ''} ${pick(VOCAB)}`; break;             // insert a word
      case 2: parts[i] = pick(VOCAB); break;                                    // replace a word
      case 3: { const j = pick(wordsAt); if (j !== undefined) [parts[i], parts[j]] = [parts[j], parts[i]]; break; } // swap two words
      case 4: parts[i] = (parts[i] || '').toUpperCase(); break;                 // change case
      case 5: parts[i] = (parts[i] || '') + pick([',', ';', '!', '.', ':']); break; // add punctuation
      case 6: parts[i] = (parts[i] || '').replace(/[,.;:!]$/, ''); break;      // drop punctuation
      case 7: parts[i] = pick(['<', '>', '-', '+', '€', '$', '👍', '(']) + (parts[i] || ''); break; // a symbol
      case 8: { const s = t.split(/(?<=[.!?。])\s*/); if (s.length > 1) { const a = Math.floor(rnd() * s.length), b = Math.floor(rnd() * s.length); [s[a], s[b]] = [s[b], s[a]]; } return s.join(' '); }
      case 9: parts[i] = (parts[i] || '') + pick(['\n', '  ', '\t']); break;    // spacing
      case 10: return t + ' ' + t.slice(0, Math.floor(rnd() * t.length));      // duplicate a stretch
      default: { const j = pick(wordsAt); if (j !== undefined) { const w = parts[j]; parts[j] = ''; parts[i] = `${w} ${parts[i] || ''}`; } } // move a word
    }
    t = parts.join('');
  }
  return t;
}

