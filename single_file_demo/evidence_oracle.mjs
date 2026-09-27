// An independent oracle for the change evidence: every sentence the review
// page prints about the two texts is checked against the raw strings.
//
// The checks do not reuse the engine's matching. They re-tokenize with their
// own copy of the word pattern, lower case with their own code, rebuild each
// Read-as view from the rendered pieces, and parse each statement back into
// the facts it asserts. A statement the oracle cannot parse is itself a
// failure, so a new sentence cannot ship without a check.
//
// Used by test_evidence_oracle.mjs (the built-in examples, an adversarial
// corpus and a seeded random mutation fuzz) and by
// scripts/real_browser_check.mjs, which compares what the real page prints
// with what this oracle accepts. Not loaded by the page.
import * as E from './change_evidence.js';
import { compareTexts } from './review_packet.js';

// ---------------------------------------------------------------- independent helpers
const CJK = '\\p{Script=Han}\\p{Script=Hiragana}\\p{Script=Katakana}';
const W = `(?:(?![${CJK}])[\\p{L}\\p{M}\\p{N}])`;
const WORD = new RegExp(`[${CJK}]|[$€£¥]?${W}+(?:[.,:\\/'’\\-]${W}+)*%?`, 'gu');
const words = s => Array.from(s.matchAll(WORD), m => ({ t: m[0], k: m[0].toLocaleLowerCase('en').replace(/’/g, "'"), s: m.index, e: m.index + m[0].length }));
const ci = s => Array.from(s, c => c.toLocaleLowerCase('en')).join('');
export const countWords = (text, phrase) => {
  const hay = words(text).map(w => w.k), keys = words(phrase).map(w => w.k);
  if (!keys.length) return 0;
  let c = 0;
  for (let i = 0; i + keys.length <= hay.length; i++) if (keys.every((k, n) => hay[i + n] === k)) { c++; i += keys.length - 1; }
  return c;
};
const sameWords = (a, b) => { const x = words(a).map(w => w.k), y = words(b).map(w => w.k); return x.length === y.length && x.every((k, i) => k === y[i]); };
const WHITESPACE_NAMES = { space: ' ', tab: '\t', 'non-breaking space': ' ', 'line break': '\n' };
function readChars(phrase) {
  // “x” -> x ; "a space" -> " " ; "3 spaces" -> "   " ; "N whitespace characters" -> null (checked as whitespace)
  let m;
  if ((m = /^“([\s\S]*)”$/.exec(phrase))) return { text: m[1] };
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
const quotes = str => Array.from(str.matchAll(/“([^”]*)”/g), m => m[1]);
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
export function check(source, output, method) {
  const bad = [];
  const fail = (what, detail) => bad.push(`${what}: ${detail}`);
  let review;
  try { review = compareTexts(source, output, method); } catch { return bad; } // refused texts print nothing about the pair
  const { passages, summary } = E.buildEvidence(source, output, review);
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
    if (msg && msg.startsWith('The same words in the same order') && (!sameWords(A, B) || A === B || !p.notes.length)) fail('SAME-WORDS', tag);
    if (msg && msg.startsWith('Only common words')) {
      const content = s => words(s).map(w => w.k).filter(k => !E.STOP.has(k));
      if (A === B || JSON.stringify(content(A)) !== JSON.stringify(content(B))) fail('COMMON-ONLY', tag);
    }
    if (msg && msg.startsWith('Turning the original passage') && (!p.a || !p.b || A === B)) fail('OVER', tag);
    if (!msg && p.kind === 'verbatim') fail('VERBATIM-MESSAGE', tag);

    for (const n of p.notes) {
      const spans = E.noteSpans(n, source, output);
      for (const sp of spans) if ((sp.side === 'a' ? source : output).slice(sp.s, sp.e) !== sp.text) fail('SPAN', tag);
      const own = spans.filter(sp => sp.side === (n.a || n.aRuns?.length || n.aChars ? 'a' : 'b'));
      const st = E.noteStatement(n, p, source, output);
      const ntag = `${tag} ${n.letter} [${n.kind}/${n.state}] ${st}`;
      const aSpans = spans.filter(sp => sp.side === 'a'), bSpans = spans.filter(sp => sp.side === 'b');
      const inA = s => p.a && aSpans.some(sp => sp.text === s && sp.s >= p.a.start && sp.e <= p.a.end);
      const inB = s => p.b && bSpans.some(sp => sp.text === s && sp.s >= p.b.start && sp.e <= p.b.end);
      let m;
      const SIDE = '(“[^”]*”(?: and “[^”]*”)*|an? [a-z-]+(?: [a-z-]+)*|\\d+ [a-z-]+(?: [a-z-]+)*)';
      if ((m = new RegExp(`^${SIDE} in the original, ${SIDE} in the rewrite\\.$`).exec(st))) {
        // Differs: each quoted side names exactly the characters at the note's spans.
        const qa = m[1].startsWith('“') ? quotes(m[1]) : null, qb = m[2].startsWith('“') ? quotes(m[2]) : null;
        if (qa && qa.join('\u0000') !== aSpans.map(sp => sp.text).join('\u0000') && !charsMatch(m[1], aSpans.map(sp => sp.text).join(''))) fail('DIFFERS-A', ntag);
        if (qb && qb.join('\u0000') !== bSpans.map(sp => sp.text).join('\u0000') && !charsMatch(m[2], bSpans.map(sp => sp.text).join(''))) fail('DIFFERS-B', ntag);
        if (!qa && !charsMatch(m[1], aSpans.map(sp => sp.text).join(''))) fail('DIFFERS-A-CHARS', ntag);
        if (!qb && !charsMatch(m[2], bSpans.map(sp => sp.text).join(''))) fail('DIFFERS-B-CHARS', ntag);
      } else if ((m = /^The (original|rewrite) has (.+) here; the (rewrite|original) does not\.$/.exec(st))) {
        const sp = m[1] === 'original' ? aSpans[0] : bSpans[0];
        if (!sp || !charsMatch(m[2], sp.text)) fail('CHARS-HERE', ntag);
      } else if ((m = /^This (original|rewrite) passage has no partner in the (rewrite|original)\.$/.exec(st))) {
        if ((m[1] === 'original' && p.b) || (m[1] === 'rewrite' && p.a)) fail('NO-PARTNER', ntag);
      } else if ((m = /^“([^”]*)” is in both passages, at different places\.$/.exec(st))) {
        if (!inA(m[1]) || !inB(m[1])) fail('MOVED', ntag);
      } else if ((m = /^“([^”]*)” in the original and “([^”]*)” in the rewrite, at different places: the same words, with (.+)\.$/.exec(st))) {
        if (!inA(m[1]) || !inB(m[2]) || !sameWords(m[1], m[2]) || m[1] === m[2]) fail('MOVED-FORMS', ntag);
        checkWith(m[1], m[2], m[3], ntag);
      } else if ((m = /^(.+) (?:is|are) not a word of the paired (rewrite|original) passage; \2 passage (\d+) has it(?:, as “([^”]*)”)?\.$/.exec(st))) {
        const paired = m[2] === 'rewrite' ? B : A, str = quotes(m[1])[0];
        if (countWords(paired, str) !== 0) fail('OTHER-PAIRED', ntag);
        const there = passageOf(m[2] === 'rewrite' ? 'b' : 'a', m[3]);
        if (!there || countWords(there.text, str) === 0) fail('OTHER-THERE', ntag);
        if (m[4] !== undefined && !there?.text.includes(m[4])) fail('OTHER-AS', ntag);
      } else if ((m = /^This passage has no partner in the (rewrite|original); \1 passage (\d+) has “([^”]*)”(?:, as “([^”]*)”)?\.$/.exec(st))) {
        if (p.a && p.b) fail('OTHER-UNPAIRED-PAIRED', ntag);
        const there = passageOf(m[1] === 'rewrite' ? 'b' : 'a', m[2]);
        if (!there || countWords(there.text, m[3]) === 0) fail('OTHER-UNPAIRED-THERE', ntag);
        if (m[4] !== undefined && !there?.text.includes(m[4])) fail('OTHER-UNPAIRED-AS', ntag);
      } else if ((m = /^(.+) (?:appears|appear) as a word (\S+(?: times)?) in the (original|rewrite) passage and (\S+(?: times)?) in the paired (rewrite|original) passage, ignoring capitalization\.$/.exec(st))) {
        const str = quotes(m[1])[0];
        const ownText = m[3] === 'original' ? A : B, otherText = m[3] === 'original' ? B : A;
        if (countWords(ownText, str) !== timesOf(m[2]) || countWords(otherText, str) !== timesOf(m[4])) fail('COUNT', ntag);
      } else if ((m = /^“([^”]*)” is in an (original|rewrite) passage that has no partner in the (rewrite|original)\.$/.exec(st))) {
        if ((m[2] === 'original' && p.b) || (m[2] === 'rewrite' && p.a)) fail('UNPAIRED', ntag);
        if (!(m[2] === 'original' ? A : B).includes(m[1])) fail('UNPAIRED-TEXT', ntag);
      } else if ((m = /^“([^”]*)” is in the (original|rewrite) passage, not in the paired (rewrite|original) passage\.$/.exec(st))) {
        const paired = m[2] === 'original' ? B : A;
        if (ci(paired).includes(ci(m[1]))) fail('NOT-IN-PAIRED', ntag);
      } else if ((m = /^“([^”]*)” is in the (original|rewrite) passage; the paired (rewrite|original) passage has it only within other words, first in “([^”]*)”\.$/.exec(st))) {
        const paired = m[2] === 'original' ? B : A;
        if (!ci(paired).includes(ci(m[1])) || countWords(paired, m[1]) || !paired.includes(m[4]) || !ci(m[4]).includes(ci(m[1]))) fail('INSIDE-PAIRED', ntag);
      } else if ((m = /^(The clause )?(.+?) (?:appears|appear) in the (original|rewrite)(, nowhere in the (?:rewrite|original)|; the (?:rewrite|original) has (?:it|them) only within other words, first in “([^”]*)”)\.( The (?:rewrite|original) does have the word “([^”]*)”, in (?:rewrite|original) passage (\d+)\.)?( It includes .+\.)?$/.exec(st))) {
        const strs = quotes(m[2]), otherText = m[3] === 'original' ? output : source;
        if (!strs.length || strs.some(x => !(m[3] === 'original' ? source : output).includes(x))) fail('OWN-TEXT', ntag);
        if (m[4].startsWith(', nowhere')) {
          for (const x of strs) if (ci(otherText).includes(ci(x))) fail('FALSE-NOWHERE', ntag);
        } else {
          const w = m[5];
          for (const x of strs) {
            if (!ci(otherText).includes(ci(x)) || countWords(otherText, x) !== 0) fail('FALSE-INSIDE', ntag);
          }
          const first = ci(otherText).indexOf(ci(strs[0]));
          const around = spans.find(sp => sp.side !== own[0]?.side && sp.text === w);
          if (!otherText.includes(w) || !ci(w).includes(ci(strs[0])) || !around || first < around.s || first >= around.e) fail('INSIDE-FIRST', ntag);
        }
        if (m[6]) {
          const there = passageOf(m[3] === 'original' ? 'b' : 'a', m[8]);
          if (!there || countWords(there.text, m[7]) === 0) fail('MARKER-ELSEWHERE', ntag);
        }
        if (m[9]) for (const x of quotes(m[9])) if (!strs[0].includes(x)) fail('INCLUDES', ntag);
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
      if ((m = /^“([^”]*)”$/.exec(line))) {
        if (m[1] !== aText || m[1] !== bText) fail('IN-BOTH', `${tag} ${line}`);
      } else if ((m = /^“([^”]*)” in the original, “([^”]*)” in the rewrite: the same words, with (.+)\.$/.exec(line))) {
        if (m[1] !== aText || m[2] !== bText || !sameWords(aText, bText)) fail('IN-BOTH-FORMS', `${tag} ${line}`);
        checkWith(aText, bText, m[3], `${tag} ${line}`);
      } else fail('UNRECOGNISED-IN-BOTH', `${tag} ${line}`);
      if (!p.a || !p.b || ib.a.s < p.a.start || ib.a.e > p.a.end || ib.b.s < p.b.start || ib.b.e > p.b.end) fail('IN-BOTH-OUTSIDE', tag);
    }
  }
  function checkWith(a, b, parts, where) {
    const wa = words(a), wb = words(b);
    const caseDiff = wa.some((w, i) => w.t !== wb[i]?.t);
    const gapsOf = (s, ws) => ws.map((w, i) => s.slice(i ? ws[i - 1].e : 0, w.s)).concat(s.slice(ws.length ? ws[ws.length - 1].e : 0)).join('\u0000');
    const gapDiff = gapsOf(a, wa) !== gapsOf(b, wb);
    const saysCase = /capitalization/.test(parts), saysGap = /different characters between the words/.test(parts);
    if (saysCase !== caseDiff || saysGap !== gapDiff) fail('WITH', `${where} (case ${caseDiff}, gaps ${gapDiff})`);
    if (/different capitalization$|different capitalization and/.test(parts) && wa.some((w, i) => ci(w.t) !== ci(wb[i].t))) fail('WITH-CASE-ONLY', where);
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
    if (/differ only in the spacing or line breaks between passages/.test(fact)) {
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
  ['You must pay the fee before you leave.', 'Before you leave, you must pay the fee.'],
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

