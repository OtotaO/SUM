import { compareTexts, makeReviewPacket, verifyReviewPacket, checkRenderBinding, publicJwks, hashText,
  sourceSpans, sourceSpansV2, MAX_REVIEW_CHARS, REVIEW_METHOD, REVIEW_METHOD_V1 } from './review_packet.js';
import { EXAMPLES, buildEvidence, evidenceSummary, groupByState, blacklineSegments, noteStatement, noteStrings,
  noteSpans, kindLabel, KIND_LABEL, STATE_LABEL, STATE_ORDER } from './change_evidence.js';

// Every node below is built with createElement and text nodes. Texts, packet
// fields and receipt fields are untrusted and never reach an HTML parser.

const $ = id => document.getElementById(id);
let current = null;        // { source, output, review, render, origin }
let passages = [];         // change evidence for `current`, recomputed, never stored
let generation = 0;
let exportRevision = 0;
let render = null;
let jwks = null;
let receiptChecked = false;
let generated = false;     // box B holds a rewrite generated on this page
let origin = 'user';       // 'example' | 'user' | 'packet'
let spanOrigin = null;     // the span button that selected text, for Escape
let showToken = 0;
const status = message => { $('workbench-status').textContent = message; };
const reducedMotion = () => Boolean(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches);
const plural = (n, one, many = one + 's') => `${n.toLocaleString('en-US')} ${n === 1 ? one : many}`;
const passageNo = spanId => String(spanId).slice(1);
const domId = s => s.replace(/\./g, '-');
const shortHash = hash => hash.slice(0, 15) + '…' + hash.slice(-6);

function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'className') node.className = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : String(value));
  }
  for (const child of children.flat(Infinity)) if (child !== null && child !== undefined && child !== false) node.append(child);
  return node;
}
const joined = (nodes, sep) => nodes.flatMap((n, i) => (i ? [typeof sep === 'function' ? sep() : sep, n] : [n]));

// ---------------------------------------------------------------- page state
function setResultsCurrent(isCurrent) {
  const button = $('compare-btn');
  button.textContent = isCurrent ? 'Compare again' : 'Compare texts';
  button.classList.toggle('ink', !isCurrent);
  $('compare-hint').hidden = !isCurrent;
}

const PLACEHOLDER = {
  empty: ['Nothing compared yet', 'Paste an original into A and its rewrite into B, then Compare texts. Or load an example above.'],
  stale: ['Texts changed', 'These results were for the previous texts, so they are hidden. Compare again to see the change evidence.'],
  error: ['Could not compare', ''],
};
function showPlaceholder(kind, message = '') {
  const [title, body] = PLACEHOLDER[kind];
  $('placeholder-h').textContent = title;
  $('placeholder-body').textContent = message || body;
  $('review-placeholder').hidden = false;
}

function setStrip(kind, key = null) {
  origin = kind;
  const tag = $('strip-tag'), copy = $('strip-copy');
  if (kind === 'example') { tag.hidden = false; tag.textContent = 'Example'; copy.textContent = EXAMPLES[key].blurb; }
  else if (kind === 'packet') { tag.hidden = false; tag.textContent = 'Opened packet'; copy.textContent = 'Texts from a review packet you checked. Its decisions are shown as recorded; they are unsigned.'; }
  else { tag.hidden = true; copy.textContent = 'Your texts. Examples, each fills both boxes:'; }
  for (const button of document.querySelectorAll('.ex-btn')) button.setAttribute('aria-pressed', String(kind === 'example' && button.dataset.example === key));
}

function resetReview() {
  generation++;
  current = null;
  passages = [];
  $('review-panel').hidden = true;
  $('export-review-btn').disabled = true;
  $('compare-btn').disabled = false;
  setResultsCurrent(false);
  showPlaceholder(!$('prose').value && !$('rewrite').value ? 'empty' : 'stale');
  status('Texts changed. Compare again to start a new review.');
}

function resetReceipt() {
  generation++;
  render = null;
  jwks = null;
  receiptChecked = false;
  $('verify-receipt-btn').disabled = true;
  $('verify-receipt-result').textContent = '';
  $('verify-receipt-result').style.display = 'none';
  $('render-trust-status').textContent = 'Receipt not checked:';
  if (current) {
    // The text comparison remains useful, but can no longer export an old
    // receipt after settings change, a failed render, or an input edit.
    current.render = null;
    $('export-review-btn').disabled = false;
    updateReceiptLine();
    status('Render evidence cleared. The current text comparison remains unsigned.');
  }
}

const userEdited = () => { if (origin !== 'user') setStrip('user'); };
$('prose').addEventListener('input', () => { resetReview(); userEdited(); });
$('rewrite').addEventListener('input', () => { window.invalidateRender(); generated = false; resetReview(); userEdited(); });
document.addEventListener('sum:render-reset', resetReceipt);

// ---------------------------------------------------------------- counts and textarea sizing
function countLabel(text) {
  const n = text.length;
  if (!n) return '0 characters';
  const chars = plural(n, 'character');
  if (n > MAX_REVIEW_CHARS) return `${chars} · over the ${MAX_REVIEW_CHARS.toLocaleString('en-US')} limit`;
  const split = current?.review?.method === REVIEW_METHOD_V1 ? sourceSpans : sourceSpansV2;
  try { return `${chars} · ${plural(split(text).length, 'passage')}`; } catch { return chars; }
}
function updateCounts() {
  $('char-count').textContent = countLabel($('prose').value);
  $('rewrite-count').textContent = countLabel($('rewrite').value);
  fitAll();
}
window.__sumUpdateCounts = updateCounts;

// Boxes grow with their text up to 6 lines (5 on a phone). Longer text gets a
// fade and a "Show all N lines" button; it is never clipped without one.
function fit(textarea) {
  const button = document.querySelector(`.show-all[data-for="${textarea.id}"]`);
  const open = textarea.classList.contains('open');
  textarea.style.height = 'auto';
  if (!textarea.scrollHeight) return; // no layout (hidden, or a test DOM)
  textarea.style.height = `${textarea.scrollHeight + 2}px`;
  const clipped = textarea.scrollHeight > textarea.clientHeight + 2;
  textarea.parentElement.classList.toggle('clipped', clipped && !open);
  const lineHeight = parseFloat(window.getComputedStyle(textarea).lineHeight) || 26;
  const lines = Math.max(1, Math.round((textarea.scrollHeight - 20) / lineHeight));
  button.hidden = !clipped && !open;
  button.textContent = open ? 'Show fewer lines' : `Show all ${lines} lines`;
  button.setAttribute('aria-expanded', String(open));
}
const fitAll = () => { fit($('prose')); fit($('rewrite')); };
for (const button of document.querySelectorAll('.show-all')) {
  button.addEventListener('click', () => { $(button.dataset.for).classList.toggle('open'); fit($(button.dataset.for)); });
}
window.addEventListener('resize', () => { fitAll(); syncHeadHeight(); });

// ---------------------------------------------------------------- compare
function fail(message) {
  generation++;
  current = null;
  passages = [];
  $('review-panel').hidden = true;
  $('export-review-btn').disabled = true;
  $('compare-btn').disabled = false;
  setResultsCurrent(false);
  showPlaceholder('error', message);
  status(message);
}

function compare({ example = null } = {}) {
  generation++;
  const version = generation;
  const source = $('prose').value, output = $('rewrite').value;
  // Never compare a silently shortened text: over the limit, say so and stop.
  for (const [box, text] of [['A', source], ['B', output]]) {
    if (text.length > MAX_REVIEW_CHARS) {
      return fail(`Text must be at most 100,000 characters. Box ${box} has ${text.length.toLocaleString('en-US')}. Nothing was compared; split the document into sections.`);
    }
  }
  if (!source.trim()) return fail('Add an original to box A first.');
  if (!output.trim()) return fail('Add the rewrite to box B first. No rewrite yet? Annex 2 can generate one.');
  const run = () => {
    if (version !== generation) return;
    $('compare-btn').disabled = false;
    let review;
    try { review = compareTexts(source, output, REVIEW_METHOD); } catch (e) { return fail(`${e.message} Nothing was compared.`); }
    current = { source, output, review, render, origin };
    showReview();
    const summary = evidenceSummary(passages);
    status(example ? `Loaded the ${EXAMPLES[example].name} example into both boxes and compared them.`
      : `Compared in this browser · ${plural(summary.passages, 'passage')} · ${plural(summary.notes, 'literal difference')}`);
  };
  if (source.length + output.length > 20000) {
    status('Comparing…');
    $('compare-btn').disabled = true;
    setTimeout(run, 0);
  } else run();
}
$('compare-btn').addEventListener('click', () => compare());

function loadExample(key, { firstView = false } = {}) {
  const example = EXAMPLES[key];
  $('prose').value = example.source;
  $('rewrite').value = example.output;
  generated = false;
  window.updateCharCount();
  window.invalidateSource();
  resetReview();
  setStrip('example', key);
  // The first view is not a click, so it reports the comparison itself.
  compare({ example: firstView ? null : key });
}
for (const button of document.querySelectorAll('.ex-btn[data-example]')) {
  button.addEventListener('click', () => loadExample(button.dataset.example));
}

$('clear-both').addEventListener('click', () => {
  $('prose').value = '';
  $('rewrite').value = '';
  generated = false;
  window.updateCharCount();
  window.invalidateSource();
  resetReview();
  setStrip('user');
  showPlaceholder('empty');
  status('Both boxes cleared.');
  $('prose').focus();
});

// ---------------------------------------------------------------- rendering
const GLYPH = { 'a-only': ['g g-del', 'ab'], differs: ['g', 'a→b'], 'b-only': ['g g-ins', 'ab'], moved: ['g', '¶→'], both: ['g', '='] };
const glyph = state => h('span', { className: GLYPH[state][0], 'aria-hidden': 'true' }, GLYPH[state][1]);
const KIND_TEXT = {
  'changed-candidate': 'paired by shared words',
  verbatim: 'identical in both texts',
  'source-unmatched': 'no partner passage in the rewrite',
  'output-unmatched': 'in the rewrite only; no partner passage in the original',
};
const DECISION_LABEL = { unreviewed: 'Not reviewed', accepted: 'Accepted', 'needs-change': 'Needs change' };
const keepTogether = text => (text.length <= 30 ? text.replace(/ /g, ' ') : text);
const noteKey = (p, n) => `${p.number}${n.letter}`;

function spanButton(side, s, e, what) {
  const field = side === 'a' ? 'prose' : 'rewrite';
  return h('button', { type: 'button', className: 'span-link',
    'aria-label': `Select ${what}, characters ${s} to ${e} of the ${side === 'a' ? 'original' : 'rewrite'}`,
    onclick: event => selectSpan(event.currentTarget, field, s, e) }, `${side === 'a' ? 'A' : 'B'} ${s}–${e}`);
}

function selectSpan(button, field, s, e) {
  const textarea = $(field);
  textarea.focus({ preventScroll: true });
  textarea.setSelectionRange(s, e);
  textarea.scrollIntoView({ block: 'center', behavior: reducedMotion() ? 'auto' : 'smooth' });
  spanOrigin = button;
  const text = textarea.value.slice(s, e);
  status(`Selected characters ${s} to ${e} of the ${field === 'prose' ? 'original' : 'rewrite'}: “${text.length > 90 ? text.slice(0, 90) + '…' : text}”. Press Escape to go back.`);
}
for (const id of ['prose', 'rewrite']) {
  $(id).addEventListener('keydown', event => {
    if (event.key !== 'Escape' || !spanOrigin?.isConnected) return;
    event.preventDefault();
    spanOrigin.focus();
    spanOrigin = null;
  });
}

// Hover or focus on a note outlines its marks; nothing is highlighted at rest.
function highlight(id) {
  clearHighlight();
  for (const node of document.querySelectorAll('#review-rows [data-n]')) if (node.dataset.n === id) node.classList.add('hl');
}
function clearHighlight() { for (const node of document.querySelectorAll('#review-rows .hl')) node.classList.remove('hl'); }
function linkHighlight(node, id, focus = true) {
  node.addEventListener('mouseenter', () => highlight(id));
  node.addEventListener('mouseleave', clearHighlight);
  if (focus) { node.addEventListener('focusin', () => highlight(id)); node.addEventListener('focusout', clearHighlight); }
}

function markNode(side, text, n, prefix = true) {
  return h(side === 'a' ? 'del' : 'ins', { 'data-n': n || null },
    prefix ? h('span', { className: 'vh' }, side === 'a' ? 'original only: ' : 'rewrite only: ') : null, text);
}
function refNode(seg) {
  const sup = h('sup', { className: 'ref', 'data-n': seg.n, 'aria-hidden': 'true' },
    h('a', { href: `#n${domId(seg.n)}`, tabindex: '-1' }, seg.letter));
  linkHighlight(sup, seg.n, false);
  return sup;
}

function renderBlackline(p) {
  const para = h('p', { className: 'blackline' });
  for (const seg of blacklineSegments(p, current.source, current.output)) {
    if (seg.t === 'plain' || seg.t === 'gap') { if (seg.text) para.append(seg.text); }
    else if (seg.t === 'eq') para.append(seg.n ? h('span', { 'data-n': seg.n }, seg.text) : seg.text);
    else if (seg.t === 'ref') para.append(refNode(seg));
    else {
      const run = h('span', { className: seg.side === 'd' ? `drun ${seg.sub ? 'sub' : 'gap'}` : 'irun' });
      for (const part of seg.parts) {
        if (part.t === 'gap') run.append(part.text);
        else if (part.t === 'ref') run.append(refNode(part));
        else run.append(markNode(seg.side === 'd' ? 'a' : 'b', part.text, part.n));
      }
      para.append(run);
    }
  }
  return para;
}

function renderNote(p, note) {
  const key = noteKey(p, note);
  const { aText, bText } = noteStrings(note);
  const lit = h('span', { className: 'lit' });
  if (note.state === 'differs') lit.append(markNode('a', aText, null, false), h('span', { className: 'arrow', 'aria-hidden': 'true' }, '→'), h('span', { className: 'vh' }, ' changed to '), markNode('b', bText, null, false));
  else lit.append(aText ? markNode('a', aText, null, false) : markNode('b', bText, null, false));
  const spans = noteSpans(note, current.source, current.output).map(sp => spanButton(sp.side, sp.s, sp.e, `“${sp.text}”`));
  const li = h('li', { className: 'note', id: `n${domId(key)}`, 'data-n': key, 'data-state': note.state },
    h('span', { className: 'nl', 'aria-hidden': 'true' }, note.letter),
    h('span', { className: 'vh' }, `Note ${key}: `),
    h('div', {},
      h('p', { className: 'nh' }, h('span', { className: 'lk' }, lit, ' ', h('span', { className: 'kind' }, kindLabel(note))),
        h('span', { className: 'chip' }, glyph(note.state), STATE_LABEL[note.state])),
      h('p', { className: 'ns' }, noteStatement(note, p), spans)));
  linkHighlight(li, key);
  return li;
}

function renderInBoth(p, ib, k) {
  const key = `${p.number}-both-${k}`;
  const item = ib.a || ib.b;
  const aSpan = ib.a || ib.aSpan, bSpan = ib.b || ib.bSpan;
  const line = h('p', { className: 'inboth', 'data-n': key, 'data-state': 'both' },
    glyph('both'), h('span', {}, 'In both passages:'), ' ', h('span', { className: 'lit' }, item.text), ' ',
    h('span', { className: 'kind' }, KIND_LABEL[ib.kind]),
    aSpan ? spanButton('a', aSpan.s, aSpan.e, `“${current.source.slice(aSpan.s, aSpan.e)}”`) : null,
    bSpan ? spanButton('b', bSpan.s, bSpan.e, `“${current.output.slice(bSpan.s, bSpan.e)}”`) : null);
  linkHighlight(line, key);
  return line;
}

function passageMessage(p) {
  if (p.kind === 'verbatim') return 'Identical text in both. No literal differences.';
  if (p.sameWords) return 'No word-level differences. This passage differs only in capitalization, punctuation or spacing, which are not checked.';
  if (p.tooLong) return 'This passage is longer than 400 words, so words are not marked individually.';
  if (!p.notes.length && (p.kind === 'source-unmatched' || p.kind === 'output-unmatched')) return 'This passage has no words to compare, only punctuation or symbols, which are not checked.';
  if (!p.notes.length && p.alsoMarked.length) return 'Only common words differ here. They are marked in the passage, without a note.';
  return null;
}

function renderPassage(p) {
  const num = p.number, pid = `p${domId(num)}`;
  const changed = p.kind !== 'verbatim' && !p.sameWords;
  const article = h('article', { className: `passage ${changed ? 'changed' : 'same'}`, id: pid, 'aria-labelledby': `${pid}-t ${pid}-k` });
  article.append(h('div', { className: 'gut', 'aria-hidden': 'true' },
    h('span', { className: num.includes('.') ? 'num small' : 'num' }, num),
    h('span', { className: 'refs' }, h('span', {}, p.a ? `A§${passageNo(p.a.id)}` : 'A –'), ' ', h('span', {}, p.b ? `B§${passageNo(p.b.id)}` : 'B –')),
    h('span', { className: 'initial' })));
  // The whole source passage link is the first button in each row.
  article.append(h('h3', { className: 'pmeta', id: `${pid}-h` },
    h('span', { className: 'sc', id: `${pid}-t` }, `Passage ${num}`), h('span', { id: `${pid}-k` }, KIND_TEXT[p.kind]),
    p.a ? spanButton('a', p.a.start, p.a.end, `original passage ${passageNo(p.a.id)}`) : null,
    p.b ? spanButton('b', p.b.start, p.b.end, `rewrite passage ${passageNo(p.b.id)}`) : null));
  article.append(renderBlackline(p));
  if (p.alsoMarked.length) {
    article.append(h('p', { className: 'also' }, 'Also marked, no note (common words): ',
      joined(p.alsoMarked.map(m => markNode(m.side, m.tok.t, null)), ', ')));
  }
  const notes = h('div', { className: 'notes' }, h('h4', { className: 'notes-h sc' }, `Change evidence · §${num}`));
  const message = passageMessage(p);
  if (message) notes.append(h('p', { className: 'ns alone' }, message));
  if (p.notes.length) notes.append(h('ol', {}, p.notes.map(n => renderNote(p, n))));
  p.inBoth.forEach((ib, k) => notes.append(renderInBoth(p, ib, k)));
  article.append(notes);
  const name = `d-${p.row.id}`;
  const fieldset = h('fieldset', { className: 'decision' }, h('legend', { className: 'vh' }, `Your decision on passage ${num}`));
  for (const [value, off, on] of [['accepted', 'Accept', 'Accepted'], ['needs-change', 'Needs change', 'Needs change'], ['unreviewed', 'Not reviewed', 'Not reviewed']]) {
    const input = h('input', { type: 'radio', name, id: `${name}-${value}`, value });
    input.checked = p.row.decision === value;
    input.addEventListener('change', () => { if (input.checked) decide(p, value); });
    fieldset.append(input, h('label', { for: `${name}-${value}` }, h('span', { className: 'off' }, off), h('span', { className: 'on', 'aria-hidden': 'true' }, on)));
  }
  fieldset.append(h('span', { className: 'dmeta' }, p.row.decision === 'unreviewed' ? 'no decision yet' : 'recorded here, unsigned'));
  article.append(fieldset);
  return article;
}

function ledgerItem(entry) {
  const { passage: p, note, inBoth } = entry;
  if (inBoth) {
    const item = inBoth.a || inBoth.b;
    return h('a', { href: `#p${domId(p.number)}` }, h('span', { className: 'it' }, keepTogether(item.text)), ' ', h('span', { className: 'k' }, KIND_LABEL[inBoth.kind].toLowerCase()));
  }
  const { aText, bText } = noteStrings(note);
  const body = note.state === 'differs'
    ? [markNode('a', keepTogether(aText), null, false), h('span', { 'aria-hidden': 'true' }, ' → '), h('span', { className: 'vh' }, ' changed to '), markNode('b', keepTogether(bText), null, false)]
    : [aText ? markNode('a', keepTogether(aText), null, false) : markNode('b', keepTogether(bText), null, false)];
  return h('a', { href: `#n${domId(noteKey(p, note))}` }, h('span', { className: 'it' }, body), ' ', h('span', { className: 'k' }, kindLabel(note).toLowerCase()));
}

function renderLedger(summary) {
  const list = $('ledger-list');
  list.replaceChildren();
  if (summary.noDifferences) {
    list.append(h('li', { className: 'line' }, 'No literal differences. Every word of each original passage appears, in order, in its paired rewrite passage.' +
      (summary.identical ? ` ${plural(summary.identical, 'passage is', 'passages are')} identical.` : '')));
    return;
  }
  if (!summary.notes) {
    const tooLong = passages.some(p => p.tooLong);
    list.append(h('li', { className: 'line' }, 'Nothing noted: the differences are in common words or punctuation, which are marked in the passages below without a note.' +
      (tooLong ? ' Passages longer than 400 words are not marked word by word.' : '')));
  }
  const groups = groupByState(passages);
  const empty = [];
  for (const state of STATE_ORDER) {
    const entries = groups[state];
    if (!entries.length) { empty.push(STATE_LABEL[state]); continue; }
    const button = h('button', { type: 'button', className: 'filter', 'data-state': state, 'aria-pressed': 'false' },
      h('span', { className: 'g-wrap' }, glyph(state)), h('span', { className: 'state' }, STATE_LABEL[state]), h('span', { className: 'n' }, String(entries.length)));
    button.addEventListener('click', () => {
      button.setAttribute('aria-pressed', button.getAttribute('aria-pressed') === 'true' ? 'false' : 'true');
      applyFilters(true);
    });
    // At most six per state, in document order; the cap never selects.
    const shown = entries.slice(0, 6).map(ledgerItem);
    const sep = () => h('span', { className: 'sep', 'aria-hidden': 'true' }, '·');
    const items = h('p', { className: 'items' }, joined(shown, sep),
      entries.length > 6 ? [sep(), h('span', { className: 'more' }, `and ${entries.length - 6} more, listed by passage below`)] : null);
    list.append(h('li', { className: state === 'both' ? 'both' : null }, button, items));
  }
  if (empty.length && summary.notes) list.append(h('li', { className: 'zero' }, `Nothing listed under: ${empty.join(' · ')}`));
}

function applyFilters(announce) {
  const buttons = [...document.querySelectorAll('#ledger-list .filter')];
  const want = buttons.filter(b => b.getAttribute('aria-pressed') === 'true').map(b => b.dataset.state);
  for (const node of document.querySelectorAll('#review-rows .note, #review-rows .inboth')) {
    node.classList.toggle('filtered-out', want.length > 0 && !want.includes(node.dataset.state));
  }
  if (announce) status(want.length ? `Showing notes: ${want.map(s => STATE_LABEL[s]).join(', ')}.` : 'Showing all notes.');
}

function renderHeading(summary) {
  const P = plural(summary.passages, 'passage');
  $('review-heading').textContent = summary.noDifferences ? `${P} compared, no literal differences.`
    : summary.notes ? `${P} compared, ${plural(summary.notes, 'literal difference')} marked.`
      : `${P} compared, no noted differences.`;
  const lonely = [summary.originalOnly && plural(summary.originalOnly, 'original passage'), summary.rewriteOnly && plural(summary.rewriteOnly, 'rewrite passage')].filter(Boolean).join(' and ');
  const whose = current.origin === 'example' ? 'the example texts' : current.origin === 'packet' ? 'the packet’s texts' : 'your texts';
  $('review-summary').textContent = `${plural(summary.pairs, 'pair')} matched by shared words · ${summary.identical.toLocaleString('en-US')} identical · ${lonely || 'none'} without a partner · ${whose}`;
}

function updateReceiptLine() {
  const cell = $('export-receipt');
  cell.replaceChildren();
  if (!current) return;
  const r = current.render;
  if (!r) {
    cell.append(current.origin === 'packet' ? 'none in this packet' : generated ? 'none; the service returned no signed receipt' : 'none; this rewrite was pasted, not generated here');
    return;
  }
  const kid = typeof r.receipt?.kid === 'string' ? r.receipt.kid : 'unnamed';
  if (current.origin === 'packet') cell.append(`signed render receipt, key ${kid}; signature and bindings checked in this browser when the packet was checked`);
  else if (receiptChecked) cell.append(`signed render receipt from this site, key ${kid}; signature and bindings checked in this browser`);
  else cell.append(`signed render receipt from this site, key ${kid}; not checked yet. `, h('a', { href: '#generate-panel' }, 'Check it in annex 2'));
}

function renderSignoff() {
  const token = showToken;
  const describe = (text, hash) => [`${plural(text.length, 'character')} · `, h('span', { className: 'mono' }, hash ? shortHash(hash) : 'hashing…')];
  $('export-original').replaceChildren(...describe(current.source));
  $('export-rewrite').replaceChildren(...describe(current.output));
  $('export-passages').textContent = `${plural(current.review.rows.length, 'row')} with exact character spans`;
  updateReceiptLine();
  const snapshot = current;
  Promise.all([hashText(snapshot.source), hashText(snapshot.output)]).then(([a, b]) => {
    if (token !== showToken || current !== snapshot) return;
    $('export-original').replaceChildren(...describe(snapshot.source, a));
    $('export-rewrite').replaceChildren(...describe(snapshot.output, b));
  }).catch(() => {});
}

function updateDecided() {
  const total = passages.length;
  const decided = passages.filter(p => p.row.decision !== 'unreviewed').length;
  $('decided-minis').replaceChildren(...passages.map(p => h('span', { className: 'mini', 'data-d': p.row.decision })));
  $('decided-count').replaceChildren(`${decided} of ${total} `, h('span', { className: 'w' }, total === 1 ? 'passage ' : 'passages '), 'decided');
  $('review-record-count').textContent = `${decided} of ${total}`;
  $('record-list').replaceChildren(...passages.map(p => h('li', {},
    h('span', { className: 'ref' }, `§${p.number}`), h('span', { className: 'mini', 'data-d': p.row.decision }), h('span', {}, DECISION_LABEL[p.row.decision]))));
  $('export-decisions').textContent = `${total}, as recorded above, unsigned`;
  for (const p of passages) {
    const meta = document.querySelector(`#p${domId(p.number)} .dmeta`);
    if (meta) meta.textContent = p.row.decision === 'unreviewed' ? 'no decision yet' : 'recorded here, unsigned';
  }
}

function decide(p, value) {
  p.row.decision = value;
  exportRevision++;
  $('export-review-btn').disabled = false;
  updateDecided();
}

// The sticky decided-and-export bar spans the margin column; at rest it is as
// tall as the results heading beside it.
function syncHeadHeight() {
  const title = document.querySelector('.head-title');
  if (title?.offsetHeight) $('review-panel').style.setProperty('--headh', `${title.offsetHeight}px`);
}

function showReview() {
  showToken++;
  // Recomputed from the two texts and the review rows; never stored in a packet.
  passages = buildEvidence(current.source, current.output, current.review).passages;
  const summary = evidenceSummary(passages);
  renderHeading(summary);
  renderLedger(summary);
  $('review-rows').replaceChildren(...passages.map(renderPassage));
  renderSignoff();
  updateDecided();
  $('review-placeholder').hidden = true;
  $('review-panel').hidden = false;
  $('export-review-btn').disabled = false;
  setResultsCurrent(true);
  updateCounts();
  syncHeadHeight();
}

// ---------------------------------------------------------------- generated rewrite and receipt
document.addEventListener('sum:render', event => {
  const data = event.detail;
  if (data !== window.__sumLastRender || data.source_text !== $('prose').value) return;
  $('rewrite').value = data.tome;
  render = data.render_receipt ? structuredClone({ receipt: data.render_receipt, triples: data.triples_used, sliders: data.quantized_sliders }) : null;
  generated = true;
  receiptChecked = false;
  $('verify-receipt-btn').disabled = !render;
  $('render-trust-status').textContent = render ? 'Receipt not checked:' : 'No signed receipt:';
  window.updateCharCount();
  setStrip('user');
  compare();
  if (current && !render) status('Generated rewrite is in box B and compared. The service returned no signed receipt; export will be an unsigned review packet.');
});

async function getPublicKeys() {
  if (jwks) return jwks;
  const response = await fetch('/.well-known/jwks.json', { cache: 'no-cache' });
  if (!response.ok) throw new Error(`Public keys unavailable (${response.status}).`);
  return publicJwks(await response.json());
}

$('verify-receipt-btn').addEventListener('click', async () => {
  if (!current?.render) return;
  const snapshot = current, version = generation;
  const result = $('verify-receipt-result');
  $('verify-receipt-btn').disabled = true;
  result.style.display = '';
  result.textContent = 'Checking the signature and the exact content bindings…';
  try {
    const keys = await getPublicKeys();
    const checks = await checkRenderBinding(snapshot.render, snapshot.output, keys);
    if (version !== generation || current !== snapshot) return;
    jwks = keys;
    receiptChecked = true;
    $('render-trust-status').textContent = 'Signature and bindings checked:';
    result.textContent = `Signature matches key ${checks.kid}. The output bytes, selected claims and slider settings match the signed receipt. The original and your decisions are unsigned. Key ownership, revocation and freshness were not checked. Meaning was not measured.`;
    updateReceiptLine();
  } catch (e) {
    if (version !== generation || current !== snapshot) return;
    $('render-trust-status').textContent = 'Check failed:';
    result.textContent = e.message;
  } finally {
    if (version === generation && current === snapshot) $('verify-receipt-btn').disabled = false;
  }
});

// ---------------------------------------------------------------- export
$('export-review-btn').addEventListener('click', async () => {
  if (!current) return;
  const snapshot = structuredClone(current), version = generation;
  const exportVersion = ++exportRevision;
  $('export-review-btn').disabled = true;
  try {
    // Export is not a trust promotion. Preserve the receipt unchanged even
    // if verification has not run, and require the recipient to recheck it.
    const keys = snapshot.render ? await getPublicKeys() : null;
    const packet = await makeReviewPacket({ ...snapshot, jwks: keys });
    if (version !== generation || exportVersion !== exportRevision) return;
    const blob = new Blob([JSON.stringify(packet, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'sum-review-packet.json';
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    status('Review packet exported with both full texts, the passage spans, your decisions and any render receipt. The packet itself is unsigned.');
  } catch (e) {
    if (version === generation && exportVersion === exportRevision) status(`Export failed: ${e.message} The signed receipt has not been silently dropped. Try again when public keys are available.`);
  } finally {
    if (version === generation && exportVersion === exportRevision && current) $('export-review-btn').disabled = false;
  }
});

// ---------------------------------------------------------------- open and check a packet
let packetRevision = 0;
let checkedPacket = null;
$('packet-input').addEventListener('input', () => { packetRevision++; checkedPacket = null; $('open-packet-btn').disabled = true; $('packet-status').textContent = 'Packet changed. Check it again.'; });
$('verify-packet-btn').addEventListener('click', async () => {
  const version = ++packetRevision;
  checkedPacket = null;
  $('open-packet-btn').disabled = true;
  $('packet-status').textContent = 'Checking packet…';
  try {
    const packet = JSON.parse($('packet-input').value);
    const result = await verifyReviewPacket(packet);
    if (version !== packetRevision) return;
    checkedPacket = packet;
    $('open-packet-btn').disabled = false;
    $('packet-status').textContent = 'Checks completed:\n' + JSON.stringify(result, null, 2) + '\nThe texts and human decisions are unsigned. Included keys do not establish who the issuer is. Meaning was not measured.';
  } catch (e) {
    if (version === packetRevision) $('packet-status').textContent = 'Check failed: ' + e.message;
  }
});

$('open-packet-btn').addEventListener('click', () => {
  if (!checkedPacket) return;
  const packet = structuredClone(checkedPacket);
  window.invalidateSource();
  $('prose').value = packet.source.text;
  $('rewrite').value = packet.output.text;
  window.updateCharCount();
  resetReview();
  // Imported evidence can be exported unchanged; the generated-output pane
  // remains hidden because this is a recipient opening a supplied packet.
  render = packet.render;
  jwks = packet.jwks;
  generated = false;
  receiptChecked = false;
  setStrip('packet');
  current = { source: packet.source.text, output: packet.output.text, review: packet.review, render, origin: 'packet' };
  showReview();
  status('Opened the checked packet. Its decisions are shown as recorded and remain unsigned.');
  $('review-panel').scrollIntoView({ block: 'start' });
});

// ---------------------------------------------------------------- anchors into closed <details>
function openTarget(hash) {
  if (!hash || hash.length < 2) return null;
  let target = null;
  try { target = document.getElementById(decodeURIComponent(hash.slice(1))); } catch { return null; }
  const details = target && (target.tagName === 'DETAILS' ? target : target.closest('details'));
  if (details && !details.open) details.open = true;
  return target;
}
document.addEventListener('click', event => {
  const link = event.target.closest?.('a[href^="#"]');
  if (!link) return;
  const target = openTarget(link.getAttribute('href'));
  // A ledger link to a note hidden by a filter shows all notes again.
  if (target?.classList.contains('filtered-out')) {
    for (const button of document.querySelectorAll('#ledger-list .filter')) button.setAttribute('aria-pressed', 'false');
    applyFilters(true);
  }
});
window.addEventListener('hashchange', () => openTarget(window.location.hash));

// ---------------------------------------------------------------- first view
// With both boxes empty, the page opens on the lease example, compared: a
// complete working state with no network request and no decision pre-set.
if (!$('prose').value && !$('rewrite').value) loadExample('lease', { firstView: true });
else { setStrip('user'); showPlaceholder('empty'); updateCounts(); }
if (window.ResizeObserver) new window.ResizeObserver(syncHeadHeight).observe(document.querySelector('.head-title'));
const arrived = openTarget(window.location.hash);
if (arrived && arrived.tagName === 'DETAILS') arrived.scrollIntoView({ block: 'start' });
