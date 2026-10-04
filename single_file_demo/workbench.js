import { compareTexts, makeReviewPacket, verifyReviewPacket, checkRenderBinding, publicJwks, pinPacketKey, keyWarning, quoteKid, receiptFailure } from './review_packet.js';

const $ = id => document.getElementById(id);
let current = null;
let generation = 0;
let exportRevision = 0;
let render = null;
let jwks = null; // keys carried by an opened packet (unsigned), or null
let siteKeys = null; // this site's /.well-known/jwks.json; a packet never sets it
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

async function getSiteKeys() {
  if (siteKeys) return siteKeys;
  const response = await fetch('/.well-known/jwks.json', { cache: 'no-cache' });
  if (!response.ok) throw new Error(`Public keys unavailable (${response.status}).`);
  let body = null;
  try { body = await response.json(); } catch { /* not JSON: reported as malformed below */ }
  if (!Array.isArray(body?.keys)) throw new Error('Public keys are malformed.');
  return (siteKeys = publicJwks(body));
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
// as the texts under review.
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
    $('packet-status').textContent = 'Checks completed:\n' + (pin ? pin + '\n' : '') + resultJson(result) + '\nSource and human decisions are unsigned. Included keys do not establish real-world issuer identity. Meaning preservation was not measured.';
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
  // Imported evidence can be exported unchanged; the generated-output pane
  // remains hidden because this is a recipient opening a supplied packet.
  render = packet.render;
  jwks = packet.jwks;
  current = { source: packet.source.text, output: packet.output.text, review: packet.review, render };
  showReview();
  status('Opened the checked packet. Review decisions are supplied by the packet and remain unsigned.');
  $('review-panel').scrollIntoView({ block: 'start' });
});
