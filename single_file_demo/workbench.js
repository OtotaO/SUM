import { compareTexts, makeReviewPacket, verifyReviewPacket, checkRenderBinding, publicJwks } from './review_packet.js';

const $ = id => document.getElementById(id);
let current = null;
let generation = 0;
let exportRevision = 0;
let render = null;
let jwks = null;
const status = message => { $('workbench-status').textContent = message; };

function resetReview() {
  generation++;
  current = null;
  $('review-panel').hidden = true;
  $('export-review-btn').disabled = true;
  status('Texts changed. Compare again to start a new review.');
}

function resetReceipt() {
  generation++;
  render = null;
  jwks = null;
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

function spanCell(span, field) {
  const cell = document.createElement('div');
  if (!span) { cell.textContent = field === 'prose' ? 'No matched source passage.' : 'No literal match in the rewrite.'; return cell; }
  const link = document.createElement('button');
  link.className = 'link-btn';
  link.textContent = `${field === 'prose' ? 'Source' : 'Rewrite'} passage ${span.id.slice(1)}`;
  link.addEventListener('click', () => {
    $(field).focus();
    $(field).setSelectionRange(span.start, span.end);
    $(field).scrollIntoView({ block: 'center' });
  });
  const text = document.createElement('p');
  text.textContent = span.text;
  cell.append(link, text);
  return cell;
}

function showReview() {
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
    label.textContent = 'Your decision: ';
    const select = document.createElement('select');
    select.setAttribute('aria-label', `Decision for ${row.id}`);
    for (const [value, text] of [['unreviewed', 'Not reviewed'], ['accepted', 'Accept as reviewed'], ['needs-change', 'Needs a change']]) {
      const option = document.createElement('option');
      option.value = value; option.textContent = text; select.append(option);
    }
    select.value = row.decision;
    select.addEventListener('change', () => {
      row.decision = select.value;
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

function updateSummary() {
  if (!current) return;
  const rows = current.review.rows;
  $('review-summary').textContent = `${rows.filter(r => r.kind === 'verbatim').length} verbatim matches; ${rows.filter(r => r.kind !== 'verbatim').length} passages to inspect. ${rows.filter(r => r.decision !== 'unreviewed').length} of ${rows.length} decisions recorded. No meaning score is computed.`;
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
  result.textContent = 'Checking signature and exact content bindings...';
  try {
    const keys = await getPublicKeys();
    const checks = await checkRenderBinding(snapshot.render, snapshot.output, keys);
    if (version !== generation || current !== snapshot) return;
    jwks = keys;
    $('render-trust-status').textContent = 'Signature and render bytes verified:';
    result.textContent = `Verified signature for key ${checks.kid}, output bytes, selected claims and slider settings. Original source and human decisions are unsigned. Key ownership, revocation and freshness were not checked. Meaning preservation was not measured.`;
  } catch (e) {
    if (version !== generation || current !== snapshot) return;
    $('render-trust-status').textContent = 'Verification failed:';
    result.textContent = e.message;
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
    const packet = await makeReviewPacket({ ...snapshot, jwks: keys });
    if (version !== generation || exportVersion !== exportRevision) return;
    const blob = new Blob([JSON.stringify(packet, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'sum-review-packet.json';
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    status('Review packet exported with the complete source, exact rewrite, passage decisions and available render receipt/public keys. The packet itself is unsigned.');
  } catch (e) {
    if (version === generation && exportVersion === exportRevision) status(`Export failed: ${e.message} The signed receipt has not been silently dropped. Try again when public keys are available.`);
  } finally {
    if (version === generation && exportVersion === exportRevision && current) $('export-review-btn').disabled = false;
  }
});

let packetRevision = 0;
let checkedPacket = null;
$('packet-input').addEventListener('input', () => { packetRevision++; checkedPacket = null; $('open-packet-btn').disabled = true; $('packet-status').textContent = 'Packet changed. Verify again.'; });
$('verify-packet-btn').addEventListener('click', async () => {
  const version = ++packetRevision;
  checkedPacket = null;
  $('open-packet-btn').disabled = true;
  $('packet-status').textContent = 'Checking packet...';
  try {
    const packet = JSON.parse($('packet-input').value);
    const result = await verifyReviewPacket(packet);
    if (version !== packetRevision) return;
    checkedPacket = packet;
    $('open-packet-btn').disabled = false;
    $('packet-status').textContent = 'Checks completed:\n' + JSON.stringify(result, null, 2) + '\nSource and human decisions are unsigned. Included keys do not establish real-world issuer identity. Meaning preservation was not measured.';
  } catch (e) {
    if (version === packetRevision) $('packet-status').textContent = 'Verification failed: ' + e.message;
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
  current = { source: packet.source.text, output: packet.output.text, review: packet.review, render };
  showReview();
  status('Opened the checked packet. Review decisions are supplied by the packet and remain unsigned.');
  $('review-panel').scrollIntoView({ block: 'start' });
});
