import { compareTexts, makeReviewPacket, verifyReviewPacket, checkRenderBinding, publicJwks, hashText,
  sourceSpans, sourceSpansV2, MAX_REVIEW_CHARS, MAX_REVIEW_SPANS, REVIEW_METHOD, REVIEW_METHOD_V1 } from './review_packet.js';
import { EXAMPLES, buildEvidence, groupByState, blacklineSegments, noteStatement, noteStrings, noteSpans, kindLabel,
  inBothText, inBothKindLabel, headingText, summaryFacts, passageMessage, STATE_LABEL, STATE_ORDER, pluralOf } from './change_evidence.js';

// Every node below is built with createElement and text nodes. Texts, packet
// fields and receipt fields are untrusted and never reach an HTML parser.

const $ = id => document.getElementById(id);
let current = null;        // { source, output, review, render, origin }
let evidence = null;       // change evidence for `current`, recomputed, never stored
let generation = 0;
let exportRevision = 0;
let render = null;
let jwks = null;
let receiptChecked = false;
let generated = false;     // box B holds a rewrite generated on this page
let origin = 'user';       // 'example' | 'user' | 'packet'
let exampleKey = null;
let spanOrigin = null;     // the span button that selected text, for Escape
let showToken = 0;
let stash = null;          // the last review hidden by an edit, with its decisions
let editConfirmed = false; // the user chose to edit texts that carry recorded decisions
const NOTE_CAP = 150;      // notes rendered per passage before "Show the other N notes"
const PART_CAP = 1500;     // marks rendered per passage before "Show the rest of the passage"
const status = message => { $('workbench-status').textContent = message; };
const reducedMotion = () => Boolean(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches);
const plural = pluralOf;
const passageNo = spanId => String(spanId).slice(1);
const domId = s => s.replace(/\./g, '-');
const shortHash = hash => hash.slice(0, 15) + '…' + hash.slice(-6);
const decidedCount = review => review.rows.filter(r => r.decision !== 'unreviewed').length;

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
  const decided = kind === 'stale' && stash ? decidedCount(stash.review) : 0;
  $('restore-review').hidden = !(kind === 'stale' && stash);
  $('restore-review').textContent = decided ? `Restore the reviewed texts and ${plural(decided, 'decision')}` : 'Restore the reviewed texts';
  $('review-placeholder').hidden = false;
}

function setStrip(kind, key = null) {
  origin = kind;
  exampleKey = kind === 'example' ? key : null;
  const tag = $('strip-tag'), copy = $('strip-copy');
  if (kind === 'example') { tag.hidden = false; tag.textContent = 'Example'; copy.textContent = EXAMPLES[key].blurb; }
  else if (kind === 'packet') { tag.hidden = false; tag.textContent = 'Opened packet'; copy.textContent = 'Texts from a review packet you checked. Its decisions are shown as recorded; they are unsigned.'; }
  else { tag.hidden = true; copy.textContent = 'Your texts. Examples, each fills both boxes:'; }
  for (const button of document.querySelectorAll('.ex-btn')) button.setAttribute('aria-pressed', String(kind === 'example' && button.dataset.example === key));
}

function resetReview() {
  generation++;
  if (current && (!stash || stash.review !== current.review)) stash = current; // keep the hidden review for Restore and for carrying decisions over
  current = null;
  evidence = null;
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

// ---------------------------------------------------------------- in-page confirmation
// One strip near the boxes asks before anything replaces the user's texts or
// clears recorded decisions. It never uses a browser dialog.
let guardAction = null;
function ask(message, confirmLabel, action, cancelLabel = 'Keep them') {
  $('guard-text').textContent = message;
  $('guard-yes').textContent = confirmLabel;
  $('guard-no').textContent = cancelLabel;
  guardAction = action;
  $('edit-guard').hidden = false;
  status(message);
  $('guard-no').focus();
}
function closeGuard() { $('edit-guard').hidden = true; guardAction = null; }
$('guard-yes').addEventListener('click', () => { const act = guardAction; closeGuard(); act?.(); });
$('guard-no').addEventListener('click', () => { closeGuard(); status('Nothing was changed.'); });

const hasOwnText = () => {
  const a = $('prose').value, b = $('rewrite').value;
  if (!a && !b) return false;
  return !Object.values(EXAMPLES).some(ex => ex.source === a && ex.output === b);
};
const recordedDecisions = () => (current ? decidedCount(current.review) : 0);

// Editing a box whose review holds decisions asks first. A cancelable
// beforeinput covers typing, pasting, dropping and deleting.
for (const id of ['prose', 'rewrite']) {
  $(id).addEventListener('beforeinput', event => {
    // A box that is read-only after a span jump takes no input; its keydown handler says why.
    if ($(id).readOnly) { if (event.cancelable) event.preventDefault(); return; }
    const n = recordedDecisions();
    if (!n || editConfirmed || !event.cancelable) return;
    event.preventDefault();
    ask(`This review has ${plural(n, 'recorded decision')}. Editing a text hides the review; decisions for passages you leave unchanged are kept when you compare again.`,
      'Edit the texts', () => { editConfirmed = true; $(id).focus(); }, 'Keep the review');
  });
}

const userEdited = () => { if (origin !== 'user') setStrip('user'); };
$('prose').addEventListener('input', () => { resetReview(); userEdited(); scheduleCounts(); });
$('rewrite').addEventListener('input', () => { window.invalidateRender(); generated = false; resetReview(); userEdited(); scheduleCounts(); });
document.addEventListener('sum:render-reset', resetReceipt);

// ---------------------------------------------------------------- counts and textarea sizing
function countLabel(text) {
  const n = text.length;
  if (!n) return '0 characters';
  const chars = plural(n, 'character');
  if (n > MAX_REVIEW_CHARS) return `${chars} · over the ${MAX_REVIEW_CHARS.toLocaleString('en-US')} limit`;
  const split = current?.review?.method === REVIEW_METHOD_V1 ? sourceSpans : sourceSpansV2;
  let passages;
  try { passages = split(text).length; } catch { return chars; }
  return passages > MAX_REVIEW_SPANS ? `${chars} · ${plural(passages, 'passage')}, over the ${MAX_REVIEW_SPANS} limit` : `${chars} · ${plural(passages, 'passage')}`;
}
function updateCounts({ refit = true } = {}) {
  clearTimeout(countTimer);
  $('char-count').textContent = countLabel($('prose').value);
  $('rewrite-count').textContent = countLabel($('rewrite').value);
  if (refit) fitAll();
}
// Small texts are counted as you type; long ones a moment after typing stops.
let countTimer = 0;
function scheduleCounts() {
  clearTimeout(countTimer);
  if ($('prose').value.length + $('rewrite').value.length <= 20000) updateCounts();
  else countTimer = setTimeout(updateCounts, 250);
}
window.__sumUpdateCounts = scheduleCounts;

// Boxes grow with their text up to 6 lines (5 on a phone). Longer text gets a
// fade and a "Show all N lines" button; it is never clipped without one.
function fit(textarea) {
  const button = document.querySelector(`.show-all[data-for="${textarea.id}"]`);
  const open = textarea.classList.contains('open');
  if (textarea.value.length > 20000 && !open) {
    // Measuring a very long text forces a full layout; it is certainly taller
    // than the cap, so cap it directly and name its length instead.
    textarea.style.height = window.getComputedStyle(textarea).maxHeight;
    textarea.parentElement.classList.add('clipped');
    button.hidden = false;
    button.textContent = `Show the whole text (${textarea.value.length.toLocaleString('en-US')} characters)`;
    button.setAttribute('aria-expanded', 'false');
    return;
  }
  textarea.style.height = 'auto';
  if (!textarea.scrollHeight) return; // no layout (hidden, or a test DOM)
  textarea.style.height = `${textarea.scrollHeight + 2}px`;
  const clipped = textarea.scrollHeight > textarea.clientHeight + 2;
  textarea.parentElement.classList.toggle('clipped', clipped && !open);
  const lineHeight = parseFloat(window.getComputedStyle(textarea).lineHeight) || 26;
  const lines = Math.max(1, Math.round((textarea.scrollHeight - 20) / lineHeight));
  button.hidden = !clipped && !open;
  button.textContent = open ? 'Show fewer lines' : `Show all ${lines.toLocaleString('en-US')} lines`;
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
  if (current) stash = current;
  current = null;
  evidence = null;
  $('review-panel').hidden = true;
  $('export-review-btn').disabled = true;
  $('compare-btn').disabled = false;
  setResultsCurrent(false);
  showPlaceholder('error', message);
  status(message);
}

// Decisions recorded for a passage pair carry over to the same pair of exact
// passage texts after an edit elsewhere. Returns [kept, dropped].
function carryDecisions(review, from) {
  if (!from) return [0, 0];
  const pool = from.review.rows.filter(r => r.decision !== 'unreviewed');
  let kept = 0;
  for (const row of review.rows) {
    const i = pool.findIndex(r => r.kind === row.kind && r.source?.text === row.source?.text && r.output?.text === row.output?.text);
    if (i < 0) continue;
    row.decision = pool[i].decision;
    pool.splice(i, 1);
    kept++;
  }
  return [kept, pool.length];
}

function compare({ example = null } = {}) {
  closeGuard();
  const source = $('prose').value, output = $('rewrite').value;
  // Nothing changed since the last comparison: keep it, with its decisions.
  if (current && current.source === source && current.output === output) {
    showReview();
    status(`The texts have not changed since the last comparison; its ${plural(current.review.rows.length, 'passage')} and ${plural(decidedCount(current.review), 'recorded decision')} are kept.`);
    return;
  }
  generation++;
  const version = generation;
  // Never compare a silently shortened text: over the limit, say so and stop.
  for (const [box, text] of [['A', source], ['B', output]]) {
    if (text.length > MAX_REVIEW_CHARS) {
      return fail(`Text must be at most 100,000 characters. Box ${box} has ${text.length.toLocaleString('en-US')}. Nothing was compared; split the document into sections.`);
    }
  }
  if (!source.trim()) return fail('Add an original to box A first.');
  if (!output.trim()) return fail('Add the rewrite to box B first. No rewrite yet? Annex 2 can generate one.');
  // An opened packet keeps its own passage rules when its texts are compared again.
  const previous = current || stash;
  const method = previous?.origin === 'packet' && previous.review.method === REVIEW_METHOD_V1 && origin !== 'example' ? REVIEW_METHOD_V1 : REVIEW_METHOD;
  const run = () => {
    if (version !== generation) return;
    $('compare-btn').disabled = false;
    let review;
    try { review = compareTexts(source, output, method); } catch (e) { return fail(`${e.message} Nothing was compared.`); }
    const [kept, dropped] = example ? [0, 0] : carryDecisions(review, previous);
    current = { source, output, review, render, origin };
    stash = null;
    editConfirmed = false;
    showReview();
    const s = evidence.summary;
    const notes = [];
    if (kept) notes.push(`kept ${plural(kept, 'decision')} for passages that did not change`);
    if (dropped) notes.push(`${plural(dropped, 'decision')} for changed passages ${dropped === 1 ? 'was' : 'were'} not carried over`);
    if (method === REVIEW_METHOD_V1) notes.push("passages split with the packet's rules (literal-spans-v1)");
    status(example ? `Loaded the ${EXAMPLES[example].name} example into both boxes and compared them.`
      : `Compared in this browser · ${plural(s.passages, 'passage')} · ${plural(s.notes, 'difference')} noted${notes.length ? ' · ' + notes.join(' · ') : ''}`);
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
  stash = null;
  current = null;
  window.updateCharCount();
  window.invalidateSource();
  resetReview();
  stash = null;
  setStrip('example', key);
  // The first view is not a click, so it reports the comparison itself.
  compare({ example: firstView ? null : key });
}
for (const button of document.querySelectorAll('.ex-btn[data-example]')) {
  button.addEventListener('click', () => {
    const key = button.dataset.example, n = recordedDecisions();
    if (!hasOwnText() && !n) return loadExample(key);
    ask(`Replace ${hasOwnText() ? 'your texts' : 'these texts'}${n ? ` and ${plural(n, 'recorded decision')}` : ''} with the ${EXAMPLES[key].name} example?`,
      'Replace them', () => loadExample(key));
  });
}

function clearBoth() {
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
}
$('clear-both').addEventListener('click', () => {
  const n = recordedDecisions();
  if (!hasOwnText() && !n) return clearBoth();
  ask(`Clear both boxes${n ? ` and ${plural(n, 'recorded decision')}` : ''}? The texts are not saved anywhere else.`, 'Clear them', clearBoth);
});

$('restore-review').addEventListener('click', () => {
  if (!stash) return;
  const back = stash;
  $('prose').value = back.source;
  $('rewrite').value = back.output;
  window.invalidateSource();
  render = back.render;
  current = back;
  stash = null;
  editConfirmed = false;
  setStrip(back.origin === 'packet' ? 'packet' : back.origin === 'example' ? 'user' : 'user');
  window.updateCharCount();
  showReview();
  status(`Restored the reviewed texts and ${plural(decidedCount(back.review), 'recorded decision')}.`);
});

// ---------------------------------------------------------------- rendering
const GLYPH = { 'a-only': ['g g-del', 'ab'], differs: ['g', 'a→b'], 'b-only': ['g g-ins', 'ab'], other: ['g', '¶→'], moved: ['g g-mv', 'ab'], both: ['g', '='] };
const glyph = state => h('span', { className: GLYPH[state][0], 'aria-hidden': 'true' }, GLYPH[state][1]);
const KIND_TEXT = {
  'changed-candidate': 'paired by shared words',
  verbatim: 'identical in both texts',
  'source-unmatched': 'no partner passage in the rewrite',
  'output-unmatched': 'in the rewrite only; no partner passage in the original',
};
const DECISION_LABEL = { unreviewed: 'Not reviewed', accepted: 'Accepted', 'needs-change': 'Needs change' };
const noteKey = (p, n) => `${p.number}${n.letter}`;

function spanButton(side, s, e, what) {
  const field = side === 'a' ? 'prose' : 'rewrite';
  return h('button', { type: 'button', className: 'span-link',
    'aria-label': `Select ${what}, characters ${s} to ${e} of the ${side === 'a' ? 'original' : 'rewrite'}`,
    onclick: event => selectSpan(event.currentTarget, field, s, e) }, `${side === 'a' ? 'A' : 'B'} ${s}–${e}`);
}

// A span jump selects characters for reading, not for replacing: the box is
// read-only until you click into it, press Escape, or move focus away.
function selectSpan(button, field, s, e) {
  const textarea = $(field);
  textarea.readOnly = true;
  textarea.dataset.jumped = 'true';
  textarea.focus({ preventScroll: true });
  textarea.setSelectionRange(s, e);
  scrollSelectionIntoBox(textarea, s);
  textarea.scrollIntoView({ block: 'center', behavior: reducedMotion() ? 'auto' : 'smooth' });
  spanOrigin = button;
  const text = textarea.value.slice(s, e);
  status(`Selected characters ${s} to ${e} of the ${field === 'prose' ? 'original' : 'rewrite'}: “${text.length > 90 ? text.slice(0, 90) + '…' : text}”. The box is read-only until you click in it. Press Escape to go back.`);
}
function releaseJump(textarea, collapse) {
  if (textarea.dataset.jumped !== 'true') return;
  delete textarea.dataset.jumped;
  textarea.readOnly = false;
  if (collapse) { const end = textarea.selectionEnd; textarea.setSelectionRange(end, end); }
}
// Scroll a capped box so the selected line is inside it, measured with a mirror.
function scrollSelectionIntoBox(textarea, offset) {
  if (!textarea.clientHeight || textarea.scrollHeight <= textarea.clientHeight) return;
  const style = window.getComputedStyle(textarea);
  const mirror = document.createElement('div');
  for (const prop of ['boxSizing', 'width', 'paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft', 'borderTopWidth', 'borderRightWidth', 'borderBottomWidth', 'borderLeftWidth', 'fontFamily', 'fontSize', 'fontWeight', 'lineHeight', 'letterSpacing', 'wordSpacing', 'tabSize']) mirror.style[prop] = style[prop];
  Object.assign(mirror.style, { position: 'absolute', visibility: 'hidden', whiteSpace: 'pre-wrap', overflowWrap: 'break-word', top: '0', left: '-9999px' });
  mirror.textContent = textarea.value.slice(0, offset);
  const marker = document.createElement('span');
  marker.textContent = '​';
  mirror.append(marker);
  document.body.append(mirror);
  const top = marker.offsetTop;
  mirror.remove();
  textarea.scrollTop = Math.max(0, top - textarea.clientHeight / 3);
}
for (const id of ['prose', 'rewrite']) {
  const textarea = $(id);
  textarea.addEventListener('keydown', event => {
    if (event.key === 'Escape' && spanOrigin?.isConnected) {
      event.preventDefault();
      releaseJump(textarea, true);
      spanOrigin.focus();
      spanOrigin = null;
      return;
    }
    if (textarea.readOnly && textarea.dataset.jumped === 'true' && event.key.length === 1 && !event.metaKey && !event.ctrlKey) {
      status('This box is read-only after a jump, so a key cannot replace the selection. Click in the box to edit, or press Escape to go back.');
    }
  });
  textarea.addEventListener('pointerdown', () => releaseJump(textarea, false));
  textarea.addEventListener('blur', () => releaseJump(textarea, true));
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

const PREFIX = { a: 'original only: ', b: 'rewrite only: ', am: 'moved from here: ', bm: 'moved to here: ' };
function markNode(side, text, n, { prefix = true, mv = false, ws = false } = {}) {
  const cls = [mv ? 'mv' : '', ws ? 'ws' : '', text.length <= 30 ? 'nw' : ''].filter(Boolean).join(' ') || null;
  return h(side === 'a' ? 'del' : 'ins', { 'data-n': n || null, className: cls },
    prefix ? h('span', { className: 'vh' }, PREFIX[side + (mv ? 'm' : '')]) : null, text);
}
function refNode(seg) {
  const sup = h('sup', { className: 'ref', 'data-n': seg.n, 'aria-hidden': 'true' },
    h('a', { href: `#n${domId(seg.n)}`, tabindex: '-1' }, seg.letter));
  linkHighlight(sup, seg.n, false);
  return sup;
}

function renderSegments(segs, para, from = 0, budget = Infinity) {
  // Returns the index of the first segment not rendered, or -1 when all were.
  let used = 0;
  for (let i = from; i < segs.length; i++) {
    const seg = segs[i];
    if (used >= budget) return i;
    if (seg.t === 'text') { if (seg.text) para.append(seg.text); }
    else if (seg.t === 'eq') { para.append(seg.n ? h('span', { 'data-n': seg.n }, seg.text) : seg.text); used++; }
    else if (seg.t === 'ref') para.append(refNode(seg));
    else if (seg.t === 'block') {
      para.append(h('span', { className: `block side-${seg.side}` }, h('span', { className: 'block-label' }, seg.side === 'a' ? 'Original' : 'Rewrite'), seg.text));
    } else {
      const run = h('span', { className: seg.side === 'd' ? `drun ${seg.sub ? 'sub' : 'gap'}` : 'irun' });
      for (const part of seg.parts) {
        // Spacing inside a run is an element too, so a view can hide the whole run.
        if (part.t === 'gap') run.append(h('span', { className: 'gp' }, part.text));
        else if (part.t === 'ref') run.append(refNode(part));
        else { run.append(markNode(seg.side === 'd' ? 'a' : 'b', part.text, part.n, { mv: part.mv, ws: part.ws })); used++; }
      }
      // Marked view only: a thin space so two runs never read as one word.
      const next = segs[i + 1];
      if (seg.side === 'd' && next?.t === 'run' && next.side === 'i' && !/\s$/.test(runText(seg)) && !/^\s/.test(runText(next))) {
        para.append(run, h('span', { className: 'sep', 'aria-hidden': 'true' }, ' '));
        continue;
      }
      para.append(run);
    }
  }
  return -1;
}
const runText = seg => seg.parts.filter(p => p.t !== 'ref').map(p => p.text).join('');

function renderBlackline(p) {
  const para = h('p', { className: p.over ? 'blackline whole' : 'blackline' });
  const segs = blacklineSegments(p, current.source, current.output);
  const marks = segs.reduce((n, s) => n + (s.t === 'eq' ? 1 : s.t === 'run' ? s.parts.length : 0), 0);
  const stop = renderSegments(segs, para, 0, marks > PART_CAP ? PART_CAP : Infinity);
  if (stop < 0) return para;
  // A very long marked passage is drawn in part, with a button that says so.
  const more = h('button', { type: 'button', className: 'text-btn more-btn' }, `The marked passage is shown in part. Show the rest (${plural(segs.length - stop, 'more piece')})`);
  more.addEventListener('click', () => { more.remove(); renderSegments(segs, para, stop); });
  return h('div', { className: 'blackline-wrap' }, para, more);
}

function renderNote(p, note) {
  const key = noteKey(p, note);
  const { aText, bText } = noteStrings(note, current.source, current.output);
  const lit = h('span', { className: 'lit' });
  const mv = note.state === 'moved';
  const ws = s => /^\s+$/.test(s);
  if (note.state === 'differs' || mv) lit.append(markNode('a', aText, null, { prefix: false, mv, ws: ws(aText) }), h('span', { className: 'arrow', 'aria-hidden': 'true' }, mv ? '↷' : '→'), h('span', { className: 'vh' }, mv ? ' and in the rewrite ' : ' changed to '), markNode('b', bText, null, { prefix: false, mv, ws: ws(bText) }));
  else lit.append(aText ? markNode('a', aText, null, { prefix: false, ws: ws(aText) }) : markNode('b', bText, null, { prefix: false, ws: ws(bText) }));
  const spans = noteSpans(note, current.source, current.output).map(sp => spanButton(sp.side, sp.s, sp.e, `“${sp.text}”`));
  const li = h('li', { className: 'note', id: `n${domId(key)}`, 'data-n': key, 'data-state': note.state },
    h('span', { className: 'nl', 'aria-hidden': 'true' }, note.letter),
    h('span', { className: 'vh' }, `Note ${key}: `),
    h('div', {},
      h('p', { className: 'nh' }, h('span', { className: 'lk' }, lit, ' ', h('span', { className: 'kind' }, kindLabel(note))),
        h('span', { className: 'chip' }, glyph(note.state), STATE_LABEL[note.state])),
      h('p', { className: 'ns' }, noteStatement(note, p, current.source, current.output), spans)));
  linkHighlight(li, key);
  return li;
}

function renderInBoth(p, ib, k) {
  const key = `${p.number}-both-${k}`;
  const line = h('p', { className: 'inboth', 'data-n': key, 'data-state': 'both' },
    glyph('both'), h('span', {}, 'In both passages:'), ' ', h('span', { className: 'lit' }, inBothText(ib)), ' ',
    h('span', { className: 'kind' }, inBothKindLabel(ib)),
    spanButton('a', ib.a.s, ib.a.e, `“${current.source.slice(ib.a.s, ib.a.e)}”`),
    spanButton('b', ib.b.s, ib.b.e, `“${current.output.slice(ib.b.s, ib.b.e)}”`));
  linkHighlight(line, key);
  return line;
}

function renderNotes(p, list) {
  // Notes are drawn in order; beyond NOTE_CAP, a button says how many remain.
  const ol = h('ol', {});
  const addFrom = start => { for (const n of p.notes.slice(start, start + (start ? p.notes.length : NOTE_CAP))) ol.append(renderNote(p, n)); };
  addFrom(0);
  list.append(ol);
  if (p.notes.length > NOTE_CAP) {
    const more = h('button', { type: 'button', className: 'text-btn more-btn' }, `Show the other ${plural(p.notes.length - NOTE_CAP, 'note')} for this passage`);
    more.addEventListener('click', () => { more.remove(); addFrom(NOTE_CAP); applyFilters(false); });
    list.append(more);
  }
}

function renderPassage(p) {
  const num = p.number, pid = `p${domId(num)}`;
  const changed = p.kind !== 'verbatim';
  const article = h('article', { className: `passage ${changed ? 'changed' : 'same'}`, id: pid, 'aria-labelledby': `${pid}-h` });
  article.append(h('div', { className: 'gut', 'aria-hidden': 'true' },
    h('span', { className: num.includes('.') ? 'num small' : 'num' }, num),
    h('span', { className: 'refs' }, h('span', {}, p.a ? `A§${passageNo(p.a.id)}` : 'A –'), ' ', h('span', {}, p.b ? `B§${passageNo(p.b.id)}` : 'B –')),
    h('span', { className: 'initial' })));
  // The heading names the passage; the span links sit beside it, outside the heading.
  // The whole source passage link is the first button in each row.
  article.append(h('div', { className: 'pmeta' },
    h('h3', { className: 'pmeta-h', id: `${pid}-h` }, h('span', { className: 'sc' }, `Passage ${num}`), ' ', h('span', { className: 'pkind' }, KIND_TEXT[p.kind])),
    p.a ? spanButton('a', p.a.start, p.a.end, `original passage ${passageNo(p.a.id)}`) : null,
    p.b ? spanButton('b', p.b.start, p.b.end, `rewrite passage ${passageNo(p.b.id)}`) : null));
  article.append(renderBlackline(p));
  if (p.alsoMarked.length) {
    article.append(h('p', { className: 'also' }, 'Also marked, no note (common words): ',
      joined(p.alsoMarked.map(m => markNode(m.side, m.tok.t, null)), ', ')));
  }
  // The decision comes before the notes, in reading order and in tab order.
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
  const notes = h('div', { className: 'notes' }, h('h4', { className: 'notes-h sc' }, `Change evidence · §${num}`));
  const message = passageMessage(p);
  if (message) notes.append(h('p', { className: 'ns alone' }, message));
  if (p.notes.length) renderNotes(p, notes);
  p.inBoth.forEach((ib, k) => notes.append(renderInBoth(p, ib, k)));
  article.append(notes);
  return article;
}

function ledgerItem(entry) {
  const { passage: p, note, inBoth } = entry;
  if (inBoth) {
    // Both forms when they differ, so the ledger never shows a string one passage lacks.
    const forms = inBoth.a.text === inBoth.b.text ? inBoth.a.text : `${inBoth.a.text} / ${inBoth.b.text}`;
    return h('a', { href: `#p${domId(p.number)}` }, h('span', { className: 'it' }, forms), ' ', h('span', { className: 'k' }, inBothKindLabel(inBoth).toLowerCase()));
  }
  const { aText, bText } = noteStrings(note, current.source, current.output);
  const mv = note.state === 'moved';
  const ws = s => /^\s+$/.test(s);
  const body = note.state === 'differs' || mv
    ? [markNode('a', aText, null, { prefix: false, mv, ws: ws(aText) }), h('span', { 'aria-hidden': 'true' }, mv ? ' ↷ ' : ' → '), h('span', { className: 'vh' }, mv ? ' and in the rewrite ' : ' changed to '), markNode('b', bText, null, { prefix: false, mv, ws: ws(bText) })]
    : [aText ? markNode('a', aText, null, { prefix: false, ws: ws(aText) }) : markNode('b', bText, null, { prefix: false, ws: ws(bText) })];
  return h('a', { href: `#n${domId(noteKey(p, note))}` }, h('span', { className: 'it' }, body), ' ', h('span', { className: 'k' }, kindLabel(note).toLowerCase()));
}

function renderLedger(summary) {
  const list = $('ledger-list');
  list.replaceChildren();
  const facts = summaryFacts(summary);
  if (summary.identicalTexts) { list.append(h('li', { className: 'line' }, facts.join(' '))); return; }
  const groups = groupByState(evidence.passages);
  const empty = [];
  for (const state of STATE_ORDER) {
    const entries = groups[state];
    if (!entries.length) { empty.push(STATE_LABEL[state]); continue; }
    const button = h('button', { type: 'button', className: 'filter', 'data-state': state, 'aria-pressed': 'false' },
      h('span', { className: 'g-wrap' }, glyph(state)), h('span', { className: 'state' }, STATE_LABEL[state]), h('span', { className: 'n' }, entries.length.toLocaleString('en-US')));
    button.addEventListener('click', () => {
      button.setAttribute('aria-pressed', button.getAttribute('aria-pressed') === 'true' ? 'false' : 'true');
      applyFilters(true);
    });
    // At most six per state, in document order; the cap never selects.
    const shown = entries.slice(0, 6).map(ledgerItem);
    const sep = () => h('span', { className: 'sep', 'aria-hidden': 'true' }, '·');
    const items = h('p', { className: 'items' }, joined(shown, sep),
      entries.length > 6 ? [sep(), h('span', { className: 'more' }, `and ${(entries.length - 6).toLocaleString('en-US')} more, listed by passage below`)] : null);
    list.append(h('li', { className: state === 'both' ? 'both' : null }, button, items));
  }
  // What differs outside the notes goes on one line, after the empty states.
  const line = [empty.length && summary.notes ? `Nothing listed under: ${empty.join(' · ')}.` : '', ...facts].filter(Boolean).join(' ');
  if (line) list.append(h('li', { className: summary.notes ? 'zero' : 'line' }, line));
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
  $('review-heading').textContent = headingText(summary);
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
    cell.append(current.origin === 'packet' ? 'none in this packet' : current.origin === 'example' ? 'none; this is a built-in example'
      : generated ? 'none; the service returned no signed receipt' : 'none; this rewrite was pasted, not generated here');
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
  const passages = evidence.passages;
  const total = passages.length;
  const decided = passages.filter(p => p.row.decision !== 'unreviewed').length;
  $('decided-minis').replaceChildren(...passages.slice(0, 64).map(p => h('span', { className: 'mini', 'data-d': p.row.decision })));
  $('decided-count').replaceChildren(`${decided} of ${total} `, h('span', { className: 'w' }, total === 1 ? 'passage ' : 'passages '), 'decided');
  $('review-record-count').textContent = `${decided} of ${total}`;
  $('review-record-noun').textContent = total === 1 ? 'passage' : 'passages';
  $('record-list').replaceChildren(...passages.map(p => h('li', {},
    h('span', { className: 'ref' }, `§${p.number}`), h('span', { className: 'mini', 'data-d': p.row.decision }), h('span', {}, DECISION_LABEL[p.row.decision]))));
  $('export-decisions').textContent = `${total.toLocaleString('en-US')}, as recorded above, unsigned`;
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
  evidence = buildEvidence(current.source, current.output, current.review);
  renderHeading(evidence.summary);
  renderLedger(evidence.summary);
  $('review-rows').replaceChildren(...evidence.passages.map(renderPassage));
  renderSignoff();
  updateDecided();
  $('review-placeholder').hidden = true;
  $('review-panel').hidden = false;
  $('export-review-btn').disabled = false;
  setResultsCurrent(true);
  updateCounts({ refit: false }); // the boxes did not change; re-measuring them would force a layout
  window.requestAnimationFrame?.(syncHeadHeight);
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
  if (current) { stash = current; current = null; }
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

function openPacket() {
  const packet = structuredClone(checkedPacket);
  window.invalidateSource();
  $('prose').value = packet.source.text;
  $('rewrite').value = packet.output.text;
  window.updateCharCount();
  resetReview();
  stash = null;
  // Imported evidence can be exported unchanged; the generated-output pane
  // remains hidden because this is a recipient opening a supplied packet.
  render = packet.render;
  jwks = packet.jwks;
  generated = false;
  receiptChecked = false;
  setStrip('packet');
  current = { source: packet.source.text, output: packet.output.text, review: packet.review, render, origin: 'packet' };
  editConfirmed = false;
  showReview();
  status('Opened the checked packet. Its decisions are shown as recorded and remain unsigned.');
  $('review-panel').scrollIntoView({ block: 'start' });
}
$('open-packet-btn').addEventListener('click', () => {
  if (!checkedPacket) return;
  const n = recordedDecisions();
  if (!hasOwnText() && !n) return openPacket();
  ask(`Replace ${hasOwnText() ? 'your texts' : 'these texts'}${n ? ` and ${plural(n, 'recorded decision')}` : ''} with the packet's texts and decisions?`, 'Open the packet', openPacket);
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
