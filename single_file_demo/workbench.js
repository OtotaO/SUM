import { distill } from './distiller.js';
import { DETAIL_SCOPE } from './detail_flags.js';
import { hashText, compareTexts, makeReviewPacket, verifyReviewPacket, checkRenderBinding, publicJwks, pinPacketKey, keyWarning, quoteKid, receiptFailure } from './review_packet.js';

const $ = id => document.getElementById(id);
let current = null;
let generation = 0;
let exportRevision = 0;
let render = null;
let jwks = null; // keys carried by an opened packet (unsigned), or null
let siteKeys = null; // this site's /.well-known/jwks.json; a packet never sets it
const status = message => { $('workbench-status').textContent = message; };
// The depth reader sits above the comparison, so it reports beside its own button. A cached page from
// before that line existed falls back to the shared status line instead of failing.
const distillLine = () => $('distill-status') || $('workbench-status');
const distillStatus = message => { distillLine().textContent = message; };

function resetReview() {
  generation++;
  stopDetailChecks();
  current = null;
  $('review-panel').hidden = true;
  $('export-review-btn').disabled = true;
  status('Texts changed. Compare again to start a new review.');
}

function resetReceipt() {
  generation++;
  render = null;
  jwks = null;
  siteKeys = null;
  $('verify-receipt-btn').disabled = true;
  $('verify-receipt-result').textContent = '';
  $('verify-receipt-result').style.display = 'none';
  $('render-trust-status').textContent = 'Receipt not checked:';
  if (current) {
    // The text comparison remains useful, but can no longer export an old
    // receipt after settings change, a failed render, or an input edit.
    current.render = null;
    $('export-review-btn').disabled = false;
    status('Render evidence cleared. The current text comparison remains unsigned.');
  }
}

$('prose').addEventListener('input', resetReview);
$('rewrite').addEventListener('input', () => { window.invalidateRender(); resetReview(); });
document.addEventListener('sum:render-reset', resetReceipt);

// A find action highlights the passage in its box. The box is read-only until the user clicks into it or
// leaves it, so a stray keystroke cannot replace the highlighted text and discard the recorded review.
const FOUND_PASSAGE_WARNING = 'That box is showing a found passage. Click in it to edit; editing the texts starts a new review.';
function selectIn(field, start, end) {
  const box = $(field);
  box.readOnly = true;
  box.focus();
  box.setSelectionRange(start, end);
  scrollToPassage(box, start);
  box.scrollIntoView({ block: 'center' });
}
// Chrome selects a passage without scrolling the box's own text to it, so after a long paste, which leaves
// the box scrolled to its end, a passage near the top stayed out of sight. A hidden copy of the box, with
// its font, padding, border, box sizing, wrapping and text width, measures how far down the passage
// starts, and the box scrolls to put that line near its middle.
const MIRRORED = ['box-sizing', 'padding-top', 'padding-right', 'padding-bottom', 'padding-left',
  'border-top-width', 'border-right-width', 'border-bottom-width', 'border-left-width',
  'border-top-style', 'border-right-style', 'border-bottom-style', 'border-left-style',
  'font-family', 'font-size', 'font-style', 'font-variant', 'font-weight', 'font-stretch', 'line-height',
  'letter-spacing', 'word-spacing', 'tab-size', 'text-indent', 'text-transform'];
function scrollToPassage(box, start) {
  const style = window.getComputedStyle(box), mirror = document.createElement('div'), marker = document.createElement('span');
  for (const name of MIRRORED) mirror.style.setProperty(name, style.getPropertyValue(name));
  const px = name => parseFloat(style.getPropertyValue(name)) || 0;
  // The client width leaves out the border and any scrollbar, so the copy's text is exactly as wide as the box's.
  const width = style.getPropertyValue('box-sizing') === 'border-box'
    ? box.clientWidth + px('border-left-width') + px('border-right-width')
    : box.clientWidth - px('padding-left') - px('padding-right');
  Object.assign(mirror.style, { position: 'absolute', top: '0', left: '-10000px', visibility: 'hidden', overflow: 'hidden',
    whiteSpace: 'pre-wrap', overflowWrap: 'break-word', width: `${width}px` });
  mirror.setAttribute('aria-hidden', 'true');
  marker.textContent = box.value.slice(start) || '.';
  mirror.append(box.value.slice(0, start), marker);
  document.body.append(mirror);
  const top = marker.offsetTop;
  mirror.remove();
  box.scrollTop = Math.max(0, top - box.clientHeight / 2);
}
for (const field of ['prose', 'rewrite']) {
  const unlock = () => {
    $(field).readOnly = false;
    if (field === 'prose' && distillLine().textContent === FOUND_PASSAGE_WARNING) distillStatus('');
  };
  $(field).addEventListener('pointerdown', unlock);
  // Switching tab or app also blurs the box; it stays read-only until focus really moves elsewhere on the page.
  $(field).addEventListener('blur', () => setTimeout(() => { if (document.activeElement !== $(field) && document.hasFocus()) unlock(); }, 0));
  $(field).addEventListener('keydown', event => {
    if (!$(field).readOnly || event.key.length !== 1 || event.ctrlKey || event.metaKey) return;
    // Report next to the box: the original's nearest status line is the one under its depth button.
    (field === 'prose' ? distillStatus : status)(FOUND_PASSAGE_WARNING);
  });
}
// A box's value has each CR LF as LF (HTML line-break normalisation). An opened packet's texts can keep
// CR LF, so an offset into them moves back one for each CR LF before it.
const boxOffset = (text, i) => i - (text.slice(0, i).match(/\r\n/g)?.length ?? 0);

function spanCell(span, field) {
  const cell = document.createElement('div');
  if (!span) { cell.textContent = field === 'prose' ? 'No matched source passage.' : 'No literal match in the rewrite.'; return cell; }
  const link = document.createElement('button');
  link.className = 'link-btn';
  link.textContent = `${field === 'prose' ? 'Source' : 'Rewrite'} passage ${span.id.slice(1)}`;
  const compared = field === 'prose' ? current.source : current.output;
  link.addEventListener('click', () => selectIn(field, boxOffset(compared, span.start), boxOffset(compared, span.end)));
  const text = document.createElement('p');
  text.textContent = span.text;
  cell.append(link, text);
  return cell;
}

const DETAIL_SHOWN = 200;
const DETAIL_DEADLINE_MS = 3000;
const DETAIL_NAMES = { number: 'Number', date: 'Date or time', negation: 'Negation', modal: 'Modal verb', condition: 'Condition', exception: 'Exception', name: 'Name' };

function quoteButton(span, field) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'link-btn';
  const q = document.createElement('q');
  q.textContent = span.text;
  button.append(q);
  button.title = `Select it in the ${field === 'prose' ? 'original' : 'rewrite'} above`;
  const compared = field === 'prose' ? current.source : current.output;
  button.addEventListener('click', () => selectIn(field, boxOffset(compared, span.start), boxOffset(compared, span.end)));
  return button;
}

// Recomputed from the two texts every time, so an opened packet shows the same checks; nothing is stored.
// detail_flags.js is frozen at the version its registered evaluation measured (fixtures/detail_flags_eval).
// Some of its comparisons can cost the product of the two texts' counts on crafted input, so it runs in a
// worker that the page stops at a deadline: no input can hold the page.
let detailWorker = null, detailTimer = null, detailRun = 0;
function stopDetailChecks() {
  detailRun++;
  clearTimeout(detailTimer);
  if (detailWorker) { detailWorker.terminate(); detailWorker = null; }
}

function showDetailChecks() {
  stopDetailChecks();
  const run = detailRun;
  $('detail-list').replaceChildren();
  $('detail-scope').textContent = DETAIL_SCOPE;
  const summary = text => { $('detail-summary').textContent = text; };
  let worker;
  try { worker = new Worker(new URL('./detail_worker.js', import.meta.url), { type: 'module' }); }
  catch { summary('Detail checks are not available in this browser. Read the passages below.'); return; }
  detailWorker = worker;
  summary('Checking details...');
  const finish = () => { clearTimeout(timer); if (run !== detailRun) return false; worker.terminate(); detailWorker = null; return true; };
  const timer = detailTimer = setTimeout(() => {
    if (finish()) summary(`Detail checks took longer than ${DETAIL_DEADLINE_MS / 1000} seconds on these texts and were stopped. Read the passages below.`);
  }, DETAIL_DEADLINE_MS);
  worker.onmessage = ({ data }) => {
    if (!finish()) return;
    if (data.ok) showDetailFlags(data.result);
    else summary(data.message || 'Detail checks could not run on this text. Read the passages below.');
  };
  worker.onerror = event => {
    if (event.preventDefault) event.preventDefault();
    if (finish()) summary('Detail checks could not run on this text. Read the passages below.');
  };
  worker.postMessage({ id: run, source: current.source, output: current.output });
}

function showDetailFlags(result) {
  const list = $('detail-list');
  const kinds = { changed: 'changed', missing: 'not found in the rewrite', added: 'only in the rewrite' };
  // A list of thousands of items would stall the page; the count still reports them all.
  for (const f of result.flags.slice(0, DETAIL_SHOWN)) {
    const item = document.createElement('li');
    item.append(`${DETAIL_NAMES[f.category]} ${kinds[f.kind]}: `);
    if (f.source) item.append(quoteButton(f.source, 'prose'), ' in the original');
    if (f.source && f.output) item.append('; ');
    if (f.output) item.append(quoteButton(f.output, 'rewrite'), ' in the rewrite');
    item.append('.');
    list.append(item);
  }
  const more = result.flags.length > DETAIL_SHOWN ? ` The first ${DETAIL_SHOWN} are listed.` : '';
  $('detail-summary').textContent = result.flags.length
    ? `${result.flags.length} detail${result.flags.length === 1 ? '' : 's'} to check. Select a quote to find it above. Some may be kept in other words.${more}`
    : 'No changed numbers, dates, negations, modal verbs, conditions, exceptions or names were flagged. These checks can miss changes and do not look at anything else; read the passages below.';
}

function showReview() {
  showDetailChecks();
  const rows = $('review-rows');
  rows.replaceChildren();
  const labels = { verbatim: 'Verbatim match', 'changed-candidate': 'Possible changed passage: check the relationship', 'source-unmatched': 'Source passage without a literal match', 'output-unmatched': 'Rewrite passage without a literal match' };
  for (const row of current.review.rows) {
    const item = document.createElement('article');
    item.className = 'review-row';
    const heading = document.createElement('h3');
    heading.textContent = labels[row.kind];
    const pair = document.createElement('div');
    pair.className = 'review-pair';
    pair.append(spanCell(row.source, 'prose'), spanCell(row.output, 'rewrite'));
    const label = document.createElement('label');
    // In an opened packet a row stays labelled as the sender's until this reader changes it.
    const decisionNames = { unreviewed: 'Not reviewed', accepted: 'Accept as reviewed', 'needs-change': 'Needs a change' };
    const senders = current.fromPacket?.senderDecisions.get(row.id);
    const labelText = () => !current.fromPacket ? 'Your decision: '
      : row.decision === senders ? "Sender's decision in the packet (unsigned), editable: "
      : `Your decision (the sender's in the packet was "${decisionNames[senders] || 'not given'}"): `;
    const labelNode = document.createTextNode(labelText());
    label.append(labelNode);
    const select = document.createElement('select');
    select.setAttribute('aria-label', `Decision for ${row.id}`);
    for (const [value, text] of [['unreviewed', 'Not reviewed'], ['accepted', 'Accept as reviewed'], ['needs-change', 'Needs a change']]) {
      const option = document.createElement('option');
      option.value = value; option.textContent = text; select.append(option);
    }
    select.value = row.decision;
    select.addEventListener('change', () => {
      row.decision = select.value;
      labelNode.textContent = labelText();
      exportRevision++;
      $('export-review-btn').disabled = false;
      updateSummary();
    });
    label.append(select);
    item.append(heading, pair, label);
    rows.append(item);
  }
  updateSummary();
  $('review-panel').hidden = false;
  $('export-review-btn').disabled = false;
}

const changedHere = () => current.review.rows.filter(r => r.decision !== current.fromPacket.senderDecisions.get(r.id)).length;

function updateSummary() {
  if (!current) return;
  const rows = current.review.rows;
  $('review-summary').textContent = `${rows.filter(r => r.kind === 'verbatim').length} verbatim matches; ${rows.filter(r => r.kind !== 'verbatim').length} passages to inspect. ${rows.filter(r => r.decision !== 'unreviewed').length} of ${rows.length} decisions recorded${current.fromPacket ? ` (from a packet; ${current.fromPacket.createdAt ? `its unsigned creation time is ${current.fromPacket.createdAt}` : 'it gives no creation time in the expected format'}; the decisions are unsigned${changedHere() ? `; ${changedHere()} changed here` : ''})` : ''}. No meaning score is computed.`;
}

function compare() {
  generation++;
  try {
    const source = $('prose').value, output = $('rewrite').value;
    if (!source.trim()) throw new Error('Add an original source first.');
    current = { source, output, review: compareTexts(source, output), render };
    showReview();
    status('Comparison ready. Inspect the passages and record your decisions. Export includes the complete source and rewrite.');
  } catch (e) { resetReview(); status(e.message); }
}
$('compare-btn').addEventListener('click', compare);

// Made-up examples with planted changes. Loading one never replaces text the user typed, and sets the page
// address to #example=<name>, a shareable link that opens the same example. The fragment is never sent to a
// server.
const EXAMPLES = {
  "patient": {
    "original": "Take one tablet twice a day with food for 10 days. Do not drink alcohol while taking this medicine. If you develop a rash, stop taking the tablets and call the Quillmere Clinic at once. You should finish the full course even if you feel better.",
    "rewrite": "Take one tablet two times a day with food for 7 days. Drink alcohol only in moderation while you take this medicine. If you get a rash, stop the tablets and call your doctor at once. You should finish all the tablets even if you start to feel better."
  },
  "lease": {
    "original": "Tenants must give at least 60 days written notice before moving out. Harbor Lane Property Management will return the deposit of $1,800 within 21 days, unless there is damage beyond normal wear and tear. Pets are not allowed in the building.",
    "rewrite": "You must give at least 60 days written notice before you move out. Harbor Lane will return your $1,800 deposit within 21 days. Pets are allowed in the building."
  },
  "policy": {
    "original": "Employees may work from home up to 2 days per week, provided that their manager approves the schedule in advance. Requests must be submitted by the 25th of the preceding month. Staff in the Customer Support team are not eligible for this policy.",
    "rewrite": "You can work from home up to 3 days per week. Submit your request by the 25th of the month before. Customer Support staff are also eligible for this policy."
  }
};
const isExampleText = (text, field) => !text.trim() || text === $(field).placeholder || Object.values(EXAMPLES).some(e => text === (field === 'prose' ? e.original : e.rewrite));

function loadExample(name, { fromAddress = false } = {}) {
  const example = Object.hasOwn(EXAMPLES, name) ? EXAMPLES[name] : null;
  if (!example) return;
  if (!isExampleText($('prose').value, 'prose') || !isExampleText($('rewrite').value, 'rewrite')) {
    status('Clear both boxes to load an example. Your text was kept.');
    return;
  }
  for (const [field, text] of [['prose', example.original], ['rewrite', example.rewrite]]) {
    $(field).value = text;
    $(field).dispatchEvent(new window.Event('input', { bubbles: true }));
  }
  if (window.location.hash !== `#example=${name}`) window.history.replaceState(null, '', `#example=${name}`);
  compare();
  if (fromAddress) $('review-panel').scrollIntoView({ block: 'start' });
}
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => loadExample(button.dataset.example)));
const exampleFromAddress = () => { const m = /^#example=([a-z]+)$/.exec(window.location.hash); if (m) loadExample(m[1], { fromAddress: true }); };
window.addEventListener('hashchange', exampleFromAddress);
exampleFromAddress();

// Extractive depths of the original: computed here, nothing leaves the page.
let distilled = null, distilledSource = null;
// The reader's own keep and leave-out choices, each tied to the depth it was made at, keyed
// "depth:passage". They sit beside the rule's choice and never change it: the counts stay the
// rule's and the copy lists the reader's changes apart. `picked` is the passage last picked in
// the whole-text view.
let readerChoice = new Map(), picked = -1, lastPicked = -1, depthIndex = 0, fingerprint = null, announced = '';
// Counts view rebuilds, so a copy still waiting on the clipboard does not report into a later view.
let viewBuilds = 0;
function hideDistill() {
  distilled = null; distilledSource = null; readerChoice = new Map(); picked = -1; fingerprint = null; announced = ''; viewBuilds++;
  $('distill-panel').hidden = true;
  if ($('distill-status')) distillStatus('');
  if ($('distill-copy-status')) $('distill-copy-status').textContent = '';
  if ($('distill-out')) $('distill-out').textContent = '';
}
$('prose').addEventListener('input', hideDistill);
// Offsets are only meaningful against the exact text they were computed from;
// any replacement of the original (typed or programmatic) retires the panel.
function selectOriginal(start, end) {
  if ($('prose').value !== distilledSource) {
    hideDistill();
    distillStatus('The original changed since the depths were made. Read the original at depths again.');
    return;
  }
  selectIn('prose', start, end);
}

const choiceKey = i => `${depthIndex}:${i}`;
function passageState(i, ruleKept) {
  const choice = readerChoice.get(choiceKey(i)), rule = ruleKept.has(i);
  if (choice === 'drop' && rule) return 'dropped';
  if (choice === 'add' && !rule) return 'mine';
  return rule ? 'kept' : 'out';
}
const isKept = state => state === 'kept' || state === 'mine';
const STATE_WORDS = { kept: 'kept at this depth', out: 'left out at this depth', mine: 'left out at this depth, kept by you', dropped: 'kept at this depth, left out by you' };

// In the kept-only view and the left-out list a passage finds itself in the original, as before.
// In the whole-text view it is already in place, so selecting it picks it instead.
function passageButton(p, state, whole = false) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'distill-passage' + (whole ? ` ${state}` : '') + (whole && p.index === picked ? ' picked' : '');
  button.dataset.i = String(p.index);
  button.textContent = p.text;
  if (whole) {
    button.title = `Characters ${p.start} to ${p.end} of the original: ${STATE_WORDS[state]}. Select for actions.`;
    button.setAttribute('aria-expanded', String(p.index === picked));
    // Selecting the picked passage again closes its actions; focus stays on that passage.
    button.addEventListener('click', () => { lastPicked = p.index; picked = p.index === picked ? -1 : p.index; showDepth(depthIndex, picked >= 0 ? 'act' : 'passage'); });
  } else {
    button.title = `Characters ${p.start} to ${p.end} of the original. Select to find it above.`;
    button.addEventListener('click', () => selectOriginal(p.start, p.end));
  }
  return button;
}

// The actions for the picked passage. showDepth puts them on the line below it, as the next Tab stop.
function pickActions(p, state) {
  const box = document.createElement('span');
  box.className = 'distill-pick';
  box.append(`Passage ${p.index + 1} of ${distilled.passage_count}, characters ${p.start} to ${p.end}: ${STATE_WORDS[state]}. `);
  const act = (label, run) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'link-btn'; b.textContent = label;
    b.addEventListener('click', run);
    box.append(b, ' ');
    return b;
  };
  const choose = change => () => {
    change();
    showDepth(depthIndex, 'act');
    const now = depthView().state.get(p.index);
    announced = `Passage ${p.index + 1}: ${STATE_WORDS[now]}.`;
    distillStatus(announced);
  };
  const main = state === 'kept' ? act('Leave it out', choose(() => readerChoice.set(choiceKey(p.index), 'drop')))
    : state === 'out' ? act('Keep it', choose(() => readerChoice.set(choiceKey(p.index), 'add')))
    : act('Undo my change', choose(() => readerChoice.delete(choiceKey(p.index))));
  main.dataset.act = 'main';
  act('Find it in the original', () => selectOriginal(p.start, p.end));
  return box;
}

// The passages kept at the current depth, the rule's choice changed by the reader's, in text order.
function depthView() {
  const d = distilled.depths[depthIndex];
  const ruleKept = new Set(d.passages.map(p => p.index));
  const all = [...d.passages, ...d.left_out].sort((a, b) => a.start - b.start);
  const state = new Map(all.map(p => [p.index, passageState(p.index, ruleKept)]));
  return { d, all, state, kept: all.filter(p => isKept(state.get(p.index))) };
}

// Plain text: the rule's count, the kept passages exactly as written with their positions, then
// the text's fingerprint, the method and the reader's own changes listed apart.
function copyText(hash) {
  const { d, all, state, kept } = depthView();
  const n = distilled.passage_count, total = distilled.total_chars.toLocaleString('en');
  const chars = kept.reduce((a, p) => a + p.end - p.start, 0);
  const pos = i => { const p = all.find(x => x.index === i); return `[${p.start}-${p.end}]`; };
  const changed = kind => [...state].filter(([, s]) => s === kind).map(([i]) => i).sort((a, b) => a - b).map(pos);
  const added = changed('mine'), removed = changed('dropped');
  const lines = [`At depth "${d.label}" the method keeps ${d.count} of ${n} passages, ${d.chars.toLocaleString('en')} of ${total} characters.`];
  if (added.length || removed.length) lines.push(`With the reader's changes: ${kept.length} of ${n} passages, ${chars.toLocaleString('en')} of ${total} characters.`);
  lines.push('');
  let prev = -1;
  for (const p of kept) {
    if (p.index !== prev + 1) lines.push('[…]');
    lines.push(`[${p.start}-${p.end}] ${p.text}`);
    prev = p.index;
  }
  if (prev !== n - 1) lines.push('[…]');
  const changes = [added.length ? `kept ${added.join(', ')}` : '', removed.length ? `left out ${removed.join(', ')}` : ''].filter(Boolean).join('; ');
  lines.push('', `Text fingerprint: ${hash || 'computing'}`, `Passages picked by: ${distilled.method}. ${distilled.scope}`,
    `Changed by the reader: ${changes || 'none'}`, 'Positions count UTF-16 code units in the text as entered.');
  return lines.join('\n');
}
function refreshCopy() { if ($('distill-out') && distilled) $('distill-out').textContent = copyText(fingerprint); }

function gap() {
  const mark = document.createElement('span');
  mark.className = 'distill-gap';
  mark.textContent = '[…]';
  mark.title = 'Passages left out at this depth';
  return mark;
}

// focusAfter 'act' puts focus on the picked passage's main action after the view is rebuilt.
function showDepth(i, focusAfter = null) {
  // A change is announced once; when the reader moves on (another depth, view or selection) it is retired.
  if (announced && distillLine().textContent === announced) distillStatus('');
  announced = '';
  // A copy message was about the text copied before this view; the selection it names is gone.
  if ($('distill-copy-status')) $('distill-copy-status').textContent = '';
  viewBuilds++;
  depthIndex = i;
  const n = distilled.passage_count;
  const { d, all, state, kept } = depthView();
  $('distill-depth-name').textContent = `${d.label} (${i + 1} of ${distilled.depths.length})`;
  const share = distilled.total_chars ? Math.round(100 * d.chars / distilled.total_chars) : 100;
  const mine = [...state.values()].filter(s => s === 'mine').length, dropped = [...state.values()].filter(s => s === 'dropped').length;
  $('distill-stats').textContent = `${d.count} of ${n} passages; ${d.chars.toLocaleString('en')} of ${distilled.total_chars.toLocaleString('en')} characters (${share}%).`
    + (mine || dropped ? ` Your changes at this depth: ${mine} kept, ${dropped} left out.` : '');
  const view = $('distill-view');
  view.replaceChildren();
  const whole = Boolean($('distill-whole')?.checked);
  view.classList.toggle('whole', whole);
  if (whole) {
    // The whole original, in order, with the text between passages as it stands. The picked
    // passage's actions box is a line of its own. It goes after the text that follows the passage,
    // up to and including that text's first line end; with no line end before the next passage, it
    // goes after all of that text, and the rest of the passage's line starts below the box.
    // No blank line is added or lost, and apart from the box the text is the original.
    let pos = 0, box = null;
    const between = text => {
      if (!box) { if (text) view.append(text); return; }
      const cut = text.includes('\n') ? text.indexOf('\n') + 1 : text.length;
      if (cut) view.append(text.slice(0, cut));
      view.append(box);
      if (cut < text.length) view.append(text.slice(cut));
      box = null;
    };
    for (const p of all) {
      between(distilledSource.slice(pos, p.start));
      view.append(passageButton(p, state.get(p.index), true));
      if (p.index === picked) box = pickActions(p, state.get(p.index));
      pos = p.end;
    }
    between(distilledSource.slice(pos));
  } else {
    // The kept-only view is the rule's choice, as before; the reader's changes show in the whole view and the copy.
    d.passages.forEach((p, j) => {
      const prev = j ? d.passages[j - 1].index : -1;
      if (p.index !== prev + 1) view.append(gap());
      else if (j) view.append(' ');
      view.append(passageButton(p, state.get(p.index)));
    });
    if (d.passages.length && d.passages[d.passages.length - 1].index !== n - 1) view.append(gap());
  }
  $('distill-left-summary').textContent = d.left_out.length ? `${d.left_out.length} passages left out at this depth` : 'Nothing left out at this depth';
  const list = $('distill-left-list');
  list.replaceChildren(...d.left_out.map(p => { const row = document.createElement('p'); row.append(passageButton(p, state.get(p.index))); return row; }));
  if (focusAfter === 'act') view.querySelector('.distill-pick [data-act="main"]')?.focus();
  else if (focusAfter === 'passage' && whole) view.querySelector(`.distill-passage[data-i="${lastPicked}"]`)?.focus();
  refreshCopy();
}

const READY_KEPT = 'Depths ready. Move the depth slider; select any passage to find it in the original.';
const READY_WHOLE = 'Depths ready. Move the depth slider; select any passage for its actions.';
const readyLine = () => ($('distill-whole')?.checked ? READY_WHOLE : READY_KEPT);

function distillOriginal() {
  try {
    const source = $('prose').value;
    if (!source.trim()) throw new Error('Add an original source first.');
    distilled = distill(source);
    distilledSource = source;
    readerChoice = new Map(); picked = -1; fingerprint = null; announced = '';
    if ($('distill-copy-status')) $('distill-copy-status').textContent = '';
    hashText(source).then(h => { if (distilledSource === source) { fingerprint = h; refreshCopy(); } });
    $('distill-scope').textContent = distilled.scope;
    $('distill-method').textContent = `Method ${distilled.method}: computed in this browser, with no model and no network request. Positions count UTF-16 code units.`;
    const tags = $('distill-tags');
    tags.replaceChildren();
    if (distilled.tags.length) {
      const label = document.createElement('span');
      label.textContent = 'Most repeated words: ';
      tags.append(label, ...distilled.tags.map(t => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'link-btn';
        button.textContent = t.text;
        button.title = `In ${t.passages} passages. Select to find its first use above.`;
        button.addEventListener('click', () => selectOriginal(t.start, t.end));
        return button;
      }));
    }
    const slider = $('distill-depth');
    slider.max = String(distilled.depths.length - 1);
    slider.value = '0';
    showDepth(0);
    $('distill-panel').hidden = false;
    // Bring the depth reader's status line into view; the panel sits directly below it.
    ($('distill-status') || $('distill-panel')).scrollIntoView({ block: 'start' });
    distillStatus(readyLine());
  } catch (e) { hideDistill(); distillStatus(e.message); }
}
$('distill-btn').addEventListener('click', distillOriginal);
$('distill-depth').addEventListener('input', () => { if (distilled) showDepth(Number($('distill-depth').value)); });
// Elements added with the whole-text view. A cached page from before them has none and keeps the
// depth reader it had; they start hidden, so a new page with a cached older script shows none of them.
for (const id of ['distill-whole-row', 'distill-copy-row', 'distill-out-box']) if ($(id)) $(id).hidden = false;
$('distill-whole')?.addEventListener('change', () => {
  if (!distilled) return;
  // The ready line says what selecting does, which differs between the two views.
  const wasReady = [READY_KEPT, READY_WHOLE].includes(distillLine().textContent);
  showDepth(depthIndex);
  if (wasReady) distillStatus(readyLine());
});
$('distill-copy')?.addEventListener('click', async () => {
  if (!distilled) return;
  const source = distilledSource;
  const hash = fingerprint || await hashText(source);
  if (distilledSource !== source) return;
  const text = copyText(hash);
  const said = $('distill-copy-status');
  const built = viewBuilds;
  try {
    await navigator.clipboard.writeText(text);
    if (viewBuilds === built) said.textContent = 'Copied.';
  } catch {
    // The view changed while the clipboard was asked: this text and its message are about the earlier view.
    if (viewBuilds !== built) return;
    // No clipboard access: show the text and select it for the reader to copy.
    $('distill-out').textContent = text;
    $('distill-out-box').open = true;
    const range = document.createRange(); range.selectNodeContents($('distill-out'));
    const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
    said.textContent = 'Could not copy here. The text below is selected: press Ctrl+C or Cmd+C.';
  }
});

document.addEventListener('sum:render', event => {
  const data = event.detail;
  if (data !== window.__sumLastRender || data.source_text !== $('prose').value) return;
  $('rewrite').value = data.tome;
  render = data.render_receipt ? structuredClone({ receipt: data.render_receipt, triples: data.triples_used, sliders: data.quantized_sliders }) : null;
  $('verify-receipt-btn').disabled = !render;
  $('render-trust-status').textContent = render ? 'Receipt not checked:' : 'No signed receipt:';
  compare();
  if (!render) status('Generated output is ready to review. The service returned no signed receipt; export will be an unsigned review packet.');
});

async function getSiteKeys() {
  if (siteKeys) return siteKeys;
  // The packet check must not hang forever if the JWKS fetch never
  // settles: abort after a bounded wait so the status moves on. The wait
  // covers reading the body as well as the response headers.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  const timedOut = () => new Error('Public keys request timed out.');
  try {
    let response;
    try {
      response = await fetch('/.well-known/jwks.json', { cache: 'no-cache', signal: controller.signal });
    } catch (e) {
      if (controller.signal.aborted) throw timedOut();
      throw e;
    }
    if (!response.ok) throw new Error(`Public keys unavailable (${response.status}).`);
    let body = null;
    try { body = await response.json(); } catch { if (controller.signal.aborted) throw timedOut(); /* not JSON: reported as malformed below */ }
    if (!Array.isArray(body?.keys)) throw new Error('Public keys are malformed.');
    return (siteKeys = publicJwks(body));
  } finally {
    clearTimeout(timer);
  }
}
// Export keeps an opened packet's own keys unchanged; a fresh render exports
// this site's keys.
async function getPublicKeys() { return jwks || getSiteKeys(); }

// In status and result messages, text from a packet appears only as a key
// ID shown by quoteKid: quoted, every space and every character outside
// printable ASCII escaped, at most 40 characters. Receipt failures are
// rebuilt by receiptFailure, and an unparsable packet gets a fixed message.
// Every other message (such as a network error) is shown by plainMessage:
// characters outside printable ASCII and every space after the first in a
// run escaped, at most 400 characters. So no packet text containing a space
// appears in these messages and a packet cannot print the site-key
// sentence; a key ID fragment without spaces can still begin a wrapped
// line. After Open, the packet's source, rewrite and review spans are shown
// as the texts under review, and its created_at only if it is an ISO-8601
// UTC timestamp (digits, '-', ':', '.', 'T', 'Z'; see packetTime).
const escapeUnits = c => Array.from({ length: c.length }, (_, i) => '\\u' + c.charCodeAt(i).toString(16).padStart(4, '0')).join('');
const plainMessage = text => {
  const shown = String(text).replace(/[^\x20-\x7e]/g, escapeUnits).replace(/ {2,}/g, run => ' ' + escapeUnits(run.slice(1)));
  return shown.length > 400 ? `${shown.slice(0, 400)} (message shortened)` : shown;
};
// The result fields are this page's own values, except kid.
const resultJson = result => '{\n' + Object.entries(result).map(([field, value]) =>
  `  ${JSON.stringify(field)}: ${field === 'kid' ? quoteKid(value) : plainMessage(JSON.stringify(value))}`).join(',\n') + '\n}';
const asSentence = text => /[.!?]$/.test(text) ? text : `${text}.`;

// Plain wording for the key pin, shared by both verify buttons. Key IDs are
// always quoted and escaped by quoteKid. Site keys are fetched once per
// render or session, so the wording says what the fetched list showed. The
// public render service signs any caller's claims, so a site key does not
// show who asked for the render.
function keyPinText(checks, unpinned) {
  const kid = quoteKid(checks.kid);
  const warning = checks.key_warning === 'public-test-vector-key' ? ` Warning: key ${kid} is a published test key (its private key is public), so anyone can sign with it.` : '';
  const unknown = ' Anyone can create a packet signed with their own key, so the signer is unknown.';
  if (checks.key_pin === 'site-key') return `Signed by key ${kid}, which this site's /.well-known/jwks.json listed when this page fetched it. This site signs renders for anyone, so this does not show who requested the render or what source it came from.${warning}`;
  if (checks.key_pin === 'not-site-key') return `Key ${kid} came from the packet and was not in this site's /.well-known/jwks.json when this page fetched it.${unknown}${warning}`;
  if (checks.key_pin === 'not-checked') return `Key ${kid} was not checked against this site's /.well-known/jwks.json. Reason: ${asSentence(plainMessage(unpinned || 'site keys not loaded'))}${unknown}${warning}`;
  return '';
}

// Only an ISO-8601 UTC timestamp, as this page writes, is shown; any other value is not repeated.
const packetTime = value => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?Z$/.test(value)
  && !Number.isNaN(Date.parse(value)) && new Date(value).toISOString().slice(0, 19) === value.slice(0, 19) ? value : null;

// One plain sentence for the recipient, built only from this page's own result values.
function packetVerdict(result) {
  if (result.signature !== 'verified') return 'In short: nothing in this packet is signed. Its texts and review rows are consistent with each other, but anyone could have written or changed any of it, including the decisions.';
  if (result.key_pin === 'site-key') return 'In short: the rewrite matches a render signed with this site\'s published key. The site signs renders for anyone, so this does not show who sent the packet, what the original was, or whether the decisions are genuine.';
  return 'In short: the signature is valid, but the key came from the packet itself, so the signer is unknown.';
}

$('verify-receipt-btn').addEventListener('click', async () => {
  if (!current?.render) return;
  const snapshot = current, version = generation, packetKeys = jwks;
  const result = $('verify-receipt-result');
  $('verify-receipt-btn').disabled = true;
  result.style.display = '';
  result.textContent = 'Checking signature and exact content bindings...';
  try {
    // A fresh render is checked against this site's keys. An opened packet is
    // checked against the keys it carries, then pinned to this site's keys.
    let siteJwks = null, unpinned = '';
    try { siteJwks = await getSiteKeys(); } catch (e) { if (!packetKeys) throw e; unpinned = e.message; }
    const keys = packetKeys || siteJwks;
    const checks = await checkRenderBinding(snapshot.render, snapshot.output, keys).catch(e => { throw receiptFailure(e, snapshot.render?.receipt); });
    // A fresh render was just verified against this site's keys as published,
    // so its key is a site key. The packet key checks apply to opened packets.
    Object.assign(checks, packetKeys ? pinPacketKey(packetKeys, siteJwks, checks.kid) : { key_pin: 'site-key', ...keyWarning(siteJwks, checks.kid) });
    if (version !== generation || current !== snapshot) return;
    const siteKey = checks.key_pin === 'site-key';
    $('render-trust-status').textContent = siteKey ? 'Signature and render bytes verified:' : 'Signature valid, signer unknown:';
    result.textContent = `${siteKey ? 'Verified the signature, output bytes, selected claims and slider settings.' : 'The signature is valid, and the output bytes, selected claims and slider settings match the signed receipt.'} ${keyPinText(checks, unpinned)} Original source and human decisions are unsigned. Revocation and freshness were not checked. Meaning preservation was not measured.`;
  } catch (e) {
    if (version !== generation || current !== snapshot) return;
    $('render-trust-status').textContent = 'Verification failed:';
    result.textContent = plainMessage(e.message);
  } finally {
    if (version === generation && current === snapshot) $('verify-receipt-btn').disabled = false;
  }
});

$('export-review-btn').addEventListener('click', async () => {
  if (!current) return;
  const snapshot = structuredClone(current), version = generation;
  const exportVersion = ++exportRevision;
  $('export-review-btn').disabled = true;
  try {
    // Export is not a trust promotion. Preserve the receipt unchanged even
    // if verification has not run, and require the recipient to recheck it.
    const keys = snapshot.render ? await getPublicKeys() : null;
    const packet = await makeReviewPacket({ ...snapshot, jwks: keys, openUrl: OPEN_URL });
    if (version !== generation || exportVersion !== exportRevision) return;
    const blob = new Blob([JSON.stringify(packet, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'sum-review-packet.json';
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    const open = snapshot.review.rows.filter(r => r.decision === 'unreviewed').length;
    status(`${open ? `${open} of ${snapshot.review.rows.length} passages are still Not reviewed. ` : ''}Review packet exported with the complete source, exact rewrite, passage decisions and available render receipt/public keys. The packet itself is unsigned. To have someone check it, send them the file; they open ${OPEN_URL}, choose "Received a review packet? Open it here" and pick the file.`);
  } catch (e) {
    if (version === generation && exportVersion === exportRevision) status(`Export failed: ${e.message} The signed receipt has not been silently dropped. Try again when public keys are available.`);
  } finally {
    if (version === generation && exportVersion === exportRevision && current) $('export-review-btn').disabled = false;
  }
});

// The page a recipient opens to check an exported packet: this page when it
// is served over http(s), otherwise (a local file) the public site.
const OPEN_URL = /^https?:$/.test(window.location.protocol) ? window.location.origin + window.location.pathname : 'https://sum-demo.ototao.workers.dev/';
$('open-received-packet').addEventListener('click', event => {
  event.preventDefault();
  $('packet-verifier-panel').open = true;
  $('packet-verifier-panel').scrollIntoView({ block: 'start' });
  $('packet-file').focus();
});
// Choosing a packet file fills the paste box and runs the same check. After a
// successful read the picker keeps showing the chosen file until the box is
// edited by hand; then it is cleared, since it no longer describes the box,
// and choosing the same file again fires change again. A failed read clears it.
let fillingFromFile = false;
function fillPacketInput(text) {
  fillingFromFile = true;
  try { $('packet-input').value = text; $('packet-input').dispatchEvent(new window.Event('input')); } finally { fillingFromFile = false; }
}
$('packet-file').addEventListener('change', async () => {
  const file = $('packet-file').files?.[0];
  if (!file) return;
  let text = '';
  try { text = await file.text(); } catch (e) {
    fillPacketInput(''); // drops any earlier checked packet
    $('packet-file').value = '';
    $('packet-status').textContent = `Could not read ${file.name}: ${e.message}`;
    return;
  }
  fillPacketInput(text);
  $('verify-packet-btn').click();
});

let packetRevision = 0;
let checkedPacket = null;
$('packet-input').addEventListener('input', () => {
  packetRevision++; checkedPacket = null; $('open-packet-btn').disabled = true; $('packet-status').textContent = 'Packet changed. Verify again.';
  if (!fillingFromFile) $('packet-file').value = '';
});
$('verify-packet-btn').addEventListener('click', async () => {
  const version = ++packetRevision;
  checkedPacket = null;
  $('open-packet-btn').disabled = true;
  $('packet-status').textContent = 'Checking packet...';
  try {
    let packet;
    try { packet = JSON.parse($('packet-input').value); } catch { throw new Error('The packet is not valid JSON.'); }
    // Only a signed packet needs this site's keys; the packet's keys never stand in for them.
    let siteJwks = null, unpinned = '';
    if (packet?.render) siteJwks = await getSiteKeys().catch(e => { unpinned = e.message; return null; });
    const result = await verifyReviewPacket(packet, { siteJwks });
    if (version !== packetRevision) return;
    checkedPacket = packet;
    $('open-packet-btn').disabled = false;
    const pin = keyPinText(result, unpinned);
    $('packet-status').textContent = 'Checks completed:\n' + packetVerdict(result) + '\n' + (pin ? pin + '\n' : '') + resultJson(result) + '\nSource and human decisions are unsigned.' + (packet.render ? ' Included keys do not establish real-world issuer identity.' : '') + ' Meaning preservation was not measured.';
    $('packet-status').scrollIntoView({ block: 'nearest' });
  } catch (e) {
    if (version === packetRevision) $('packet-status').textContent = 'Verification failed: ' + plainMessage(e.message);
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
  hideDistill();
  // Imported evidence can be exported unchanged; the generated-output pane
  // remains hidden because this is a recipient opening a supplied packet.
  render = packet.render;
  jwks = packet.jwks;
  current = { source: packet.source.text, output: packet.output.text, review: packet.review, render, fromPacket: { createdAt: packetTime(packet.created_at), senderDecisions: new Map(packet.review.rows.map(r => [r.id, r.decision])) } };
  showReview();
  status('Opened the checked packet. Review decisions are supplied by the packet and remain unsigned.');
  $('review-panel').scrollIntoView({ block: 'start' });
});
