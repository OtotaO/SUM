import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { webcrypto } from 'node:crypto';
import { JSDOM, VirtualConsole } from 'jsdom';

const html = await readFile(new URL('./index.html', import.meta.url), 'utf8');
const captured = JSON.parse(await readFile(new URL('../fixtures/render_receipts/source_render.json', import.meta.url)));
const keys = JSON.parse(await readFile(new URL('../fixtures/render_receipts/jwks_at_capture.json', import.meta.url)));
let instance = 0;
async function until(predicate) {
  for (let i = 0; i < 500; i++) { if (predicate()) return; await new Promise(r => setTimeout(r, 2)); }
  assert.fail('Timed out waiting for the workbench action');
}
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };

async function page(handler = async () => new Response('{}', { status: 503 }), { crypto = webcrypto } = {}) {
  const errors = [], requests = [], downloads = [];
  const mockFetch = async (url, init) => {
    if (String(url).includes('altitude_rungs')) return new Response('{}', { status: 404 });
    requests.push({ url, init });
    return handler(url, init);
  };
  const console = new VirtualConsole();
  console.on('jsdomError', e => { if (!e.message.includes('navigation')) errors.push(e); });
  const dom = new JSDOM(html, { url: 'https://workbench.example', runScripts: 'dangerously', virtualConsole: console,
    beforeParse(window) {
      Object.defineProperty(window, 'crypto', { value: crypto });
      window.TextEncoder = TextEncoder; window.structuredClone = structuredClone;
      window.fetch = mockFetch; window.alert = message => { errors.push(new Error(message)); };
      window.HTMLElement.prototype.scrollIntoView = () => {};
    } });
  const window = dom.window;
  globalThis.window = window; globalThis.document = window.document; globalThis.fetch = mockFetch;
  const originalCreate = URL.createObjectURL, originalRevoke = URL.revokeObjectURL;
  URL.createObjectURL = blob => { downloads.push(blob); return 'blob:test'; };
  URL.revokeObjectURL = () => {};
  await import(`./workbench.js?test=${++instance}`);
  const $ = id => window.document.getElementById(id);
  const input = (id, value) => { $(id).value = value; $(id).dispatchEvent(new window.Event('input', { bubbles: true })); };
  return { $, input, window, requests, downloads, errors, close() { dom.window.close(); URL.createObjectURL = originalCreate; URL.revokeObjectURL = originalRevoke; } };
}

test('existing rewrite comparison, source selection, decision, export and offline recipient checks', async () => {
  const p = await page();
  try {
    p.input('prose', 'Alice may cancel with 30 days notice. Deposit is refundable unless overdue.');
    p.input('rewrite', 'Alice must cancel with 3 days notice. Deposit is refundable.');
    p.$('compare-btn').click();
    assert.equal(p.$('review-panel').hidden, false);
    assert.match(p.$('review-rows').textContent, /paired by shared words/);
    const sourceLink = p.$('review-rows').querySelector('button'); sourceLink.click();
    assert.equal(p.$('prose').value.slice(p.$('prose').selectionStart, p.$('prose').selectionEnd), 'Alice may cancel with 30 days notice.');
    const decision = p.$('review-rows').querySelector('input[value="needs-change"]');
    decision.checked = true; decision.dispatchEvent(new p.window.Event('change'));
    p.$('export-review-btn').click();
    await until(() => p.downloads.length === 1);
    const packet = JSON.parse(await p.downloads[0].text());
    assert.equal(packet.review.rows[0].decision, 'needs-change');
    assert.equal(packet.render, null);
    assert.equal(packet.source.text, p.$('prose').value);
    p.input('packet-input', JSON.stringify(packet)); p.$('verify-packet-btn').click();
    await until(() => p.$('packet-status').textContent.includes('Checks completed'));
    assert.match(p.$('packet-status').textContent, /"signature": "absent"/);
    p.$('open-packet-btn').click();
    assert.equal(p.$('review-rows').querySelector('input[value="needs-change"]').checked, true);
    assert.equal(p.$('strip-tag').textContent, 'Opened packet');
    assert.match(p.$('review-summary').textContent, /the packet’s texts$/);
    assert.equal(p.requests.length, 0, 'comparison and recipient verification must make no network requests');
    p.input('rewrite', 'Changed again.');
    assert.equal(p.$('export-review-btn').disabled, true);
    assert.equal(p.$('review-panel').hidden, true);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('review text treats markup as literal content and every textarea has an accessible name', async () => {
  const p = await page();
  try {
    const markup = '<img src=x onerror="alert(1)"> Alice may cancel.';
    p.input('prose', markup); p.input('rewrite', markup); p.$('compare-btn').click();
    assert.equal(p.$('review-rows').querySelector('img'), null);
    assert.ok(p.$('review-rows').textContent.includes(markup));
    for (const textarea of p.window.document.querySelectorAll('textarea')) {
      assert.ok(textarea.hasAttribute('aria-label') || p.window.document.querySelector(`label[for="${textarea.id}"]`));
    }
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('full extraction is retained and each density selection starts at the original baseline', async () => {
  const triples = Array.from({ length: 10 }, (_, i) => [`person${i}`, 'likes', 'cats']);
  const p = await page(async (url, init) => {
    if (url === '/api/complete') return Response.json({ completion: JSON.stringify(triples) });
    const body = JSON.parse(init.body);
    const kept = body.triples.slice(0, Math.floor(body.triples.length * body.slider_position.density));
    return Response.json({ tome: kept.map(t => t.join(' ')).join('. '), triples_used: kept, quantized_sliders: body.slider_position });
  });
  try {
    p.input('prose', 'Ten source claims.'); p.input('density', '0.5');
    p.$('attest').click(); await until(() => !p.$('attest').disabled);
    assert.equal(p.window.__sumLastTriples.length, 10);
    assert.equal(p.$('fact-count').textContent, '10');
    p.$('render-btn').click(); await until(() => !p.$('render-btn').disabled);
    let request = JSON.parse(p.requests.find(r => r.url === '/api/render').init.body);
    assert.equal(request.triples.length, 10); assert.equal(request.slider_position.density, 0.5);
    assert.equal(p.window.__sumLastRender.triples_used.length, 5);
    p.input('density', '0.8'); p.$('render-btn').click(); await until(() => !p.$('render-btn').disabled);
    request = JSON.parse(p.requests.filter(r => r.url === '/api/render')[1].init.body);
    assert.equal(request.triples.length, 10); assert.equal(request.slider_position.density, 0.8);
    assert.equal(p.window.__sumLastRender.triples_used.length, 8);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('source edits invalidate a pending extraction and unavailable extraction never invents claims', async () => {
  const pending = deferred();
  const p = await page(() => pending.promise);
  try {
    p.input('prose', 'Alice may cancel.'); p.$('attest').click();
    p.input('prose', 'Marie Curie won two prizes.');
    pending.resolve(Response.json({ completion: '[["alice","cancel","lease"]]' }));
    await until(() => !p.$('attest').disabled);
    assert.equal(p.window.__sumLastTriples, null);
    assert.equal(p.$('result').style.display, 'none');
    assert.equal(p.window.naiveExtract('Alice may cancel.').length, 0);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('render and verification responses cannot revive stale evidence after source or slider changes', async () => {
  const waitingRender = deferred(), waitingKeys = deferred(); let renderCalls = 0;
  const p = await page(async url => {
    if (url === '/api/complete') return Response.json({ completion: JSON.stringify(captured.triples_used) });
    if (url === '/.well-known/jwks.json') return waitingKeys.promise;
    if (url === '/api/render') return ++renderCalls === 1 ? Response.json(captured) : waitingRender.promise;
    return new Response('{}', { status: 404 });
  });
  try {
    p.input('prose', 'Alice was born in 1990. Alice graduated in 2012.');
    p.$('attest').click(); await until(() => !p.$('attest').disabled);
    p.$('render-btn').click(); await until(() => !p.$('render-btn').disabled);
    assert.equal(p.$('render-trust-status').textContent, 'Receipt not checked:');
    p.$('verify-receipt-btn').click();
    p.input('density', '0.5');
    waitingKeys.resolve(Response.json(keys));
    await new Promise(r => setTimeout(r, 30));
    assert.equal(p.$('verify-receipt-btn').disabled, true);
    assert.equal(p.$('verify-receipt-result').textContent, '');
    p.$('render-btn').click(); p.input('prose', 'A new source.');
    waitingRender.resolve(Response.json(captured));
    await until(() => !p.$('render-btn').disabled);
    assert.equal(p.window.__sumLastRender, null);
    assert.equal(p.$('render-output').style.display, 'none');
    assert.equal(p.$('export-review-btn').disabled, true);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('signed render export contains the displayed output, exact receipt and public keys; later failure clears it', async () => {
  let calls = 0;
  const p = await page(async url => {
    if (url === '/api/complete') return Response.json({ completion: JSON.stringify(captured.triples_used) });
    if (url === '/.well-known/jwks.json') return Response.json(keys);
    return ++calls === 1 ? Response.json(captured) : Response.json({ error: 'fixture failure' }, { status: 502 });
  });
  try {
    p.input('prose', 'Alice was born in 1990. Alice graduated in 2012.');
    p.$('attest').click(); await until(() => !p.$('attest').disabled);
    p.$('render-btn').click(); await until(() => !p.$('render-btn').disabled);
    p.$('verify-receipt-btn').click(); await until(() => p.$('render-trust-status').textContent.startsWith('Signature and'));
    p.$('export-review-btn').click(); await until(() => p.downloads.length === 1);
    const packet = JSON.parse(await p.downloads[0].text());
    assert.equal(packet.output.text, p.$('rewrite').value);
    assert.deepEqual(packet.render.receipt, captured.render_receipt);
    assert.deepEqual(packet.jwks, keys);
    p.$('render-btn').click(); await until(() => !p.$('render-btn').disabled);
    assert.equal(p.window.__sumLastRender, null);
    assert.equal(p.$('verify-receipt-btn').disabled, true);
    assert.match(p.$('render-meta').textContent, /fixture failure/);
    assert.equal(p.$('render-trust-status').textContent, 'Receipt not checked:');
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('decision changes during pending public-key retrieval cancel an outdated export', async () => {
  const pending = deferred(); let keyCalls = 0;
  const p = await page(async url => {
    if (url === '/api/complete') return Response.json({ completion: JSON.stringify(captured.triples_used) });
    if (url === '/.well-known/jwks.json') return ++keyCalls === 1 ? pending.promise : Response.json(keys);
    return Response.json(captured);
  });
  try {
    p.input('prose', 'Alice was born in 1990. Alice graduated in 2012.');
    p.$('attest').click(); await until(() => !p.$('attest').disabled);
    p.$('render-btn').click(); await until(() => !p.$('render-btn').disabled);
    p.$('export-review-btn').click();
    // The generated heading is a rewrite-only passage shown first as 0.1, so pick row 0 by its radio name.
    const decision = p.$('review-rows').querySelector('input[name="d-source-s1"][value="needs-change"]');
    decision.checked = true; decision.dispatchEvent(new p.window.Event('change'));
    assert.equal(p.$('export-review-btn').disabled, false);
    pending.resolve(Response.json(keys));
    await new Promise(r => setTimeout(r, 30));
    assert.equal(p.downloads.length, 0);
    p.$('export-review-btn').click(); await until(() => p.downloads.length === 1);
    assert.equal(JSON.parse(await p.downloads[0].text()).review.rows[0].decision, 'needs-change');
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('slider change during pending export clears receipt and leaves unsigned review exportable', async () => {
  const pending = deferred();
  const p = await page(async url => {
    if (url === '/api/complete') return Response.json({ completion: JSON.stringify(captured.triples_used) });
    if (url === '/.well-known/jwks.json') return pending.promise;
    return Response.json(captured);
  });
  try {
    p.input('prose', 'Alice was born in 1990. Alice graduated in 2012.');
    p.$('attest').click(); await until(() => !p.$('attest').disabled);
    p.$('render-btn').click(); await until(() => !p.$('render-btn').disabled);
    p.$('export-review-btn').click();
    p.input('density', '0.5');
    assert.equal(p.$('export-review-btn').disabled, false);
    pending.resolve(Response.json(keys));
    await new Promise(r => setTimeout(r, 30));
    assert.equal(p.downloads.length, 0);
    p.$('export-review-btn').click(); await until(() => p.downloads.length === 1);
    const packet = JSON.parse(await p.downloads[0].text());
    assert.equal(packet.render, null);
    assert.equal(packet.jwks, null);
    assert.equal(packet.output.text, captured.tome);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

// ---------------------------------------------------------------- the redesigned review sheet
import { EXAMPLES } from './change_evidence.js';
import { compareTexts, makeReviewPacket } from './review_packet.js';
const text = node => node.textContent.replace(/\s+/g, ' ').trim();
// Text of each text node, space-joined: how the pieces read, independent of CSS gaps.
const words = node => {
  const out = [], walker = node.ownerDocument.createTreeWalker(node, 4 /* SHOW_TEXT */);
  while (walker.nextNode()) { const t = walker.currentNode.data.replace(/\s+/g, ' ').trim(); if (t) out.push(t); }
  return out.join(' ');
};

test('first view opens on the lease example, compared, with no request, no decision and nothing highlighted', async () => {
  const p = await page();
  try {
    assert.equal(p.$('prose').value, EXAMPLES.lease.source);
    assert.equal(p.$('rewrite').value, EXAMPLES.lease.output);
    assert.equal(p.$('review-panel').hidden, false);
    assert.equal(p.$('review-placeholder').hidden, true);
    assert.equal(p.$('review-heading').textContent, '2 passages compared, 4 literal differences marked.');
    assert.equal(p.$('review-summary').textContent, '2 pairs matched by shared words · 0 identical · none without a partner · the example texts');
    assert.equal(p.$('try-example').getAttribute('aria-pressed'), 'true');
    assert.equal(p.$('compare-btn').textContent, 'Compare again');
    assert.equal(p.$('compare-btn').classList.contains('ink'), false, 'Compare is secondary while results are current');
    assert.equal(p.$('export-review-btn').disabled, false);
    assert.equal(p.$('char-count').textContent, '97 characters · 2 passages');
    assert.equal(p.$('rewrite-count').textContent, '54 characters · 2 passages');
    assert.equal(p.$('review-record-count').textContent, '0 of 2');
    assert.equal(p.window.document.querySelectorAll('#review-rows input[value="unreviewed"]:checked').length, 2);
    assert.equal(p.window.document.querySelectorAll('.hl').length, 0, 'nothing is highlighted at rest');
    assert.equal(p.requests.length, 0);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('each example button fills both boxes and compares in one click, with no synthetic input events', async () => {
  const p = await page();
  try {
    let inputs = 0;
    for (const id of ['prose', 'rewrite']) p.$(id).addEventListener('input', () => inputs++);
    p.window.document.querySelector('[data-example="refund"]').click();
    assert.equal(p.$('prose').value, EXAMPLES.refund.source);
    assert.equal(p.$('rewrite').value, EXAMPLES.refund.output);
    assert.equal(p.$('review-heading').textContent, '7 passages compared, 17 literal differences marked.');
    assert.equal(p.window.document.querySelector('[data-example="refund"]').getAttribute('aria-pressed'), 'true');
    assert.equal(p.$('try-example').getAttribute('aria-pressed'), 'false');
    assert.equal(p.$('workbench-status').textContent, 'Loaded the refund policy example into both boxes and compared them.');
    assert.equal(p.$('strip-copy').textContent, EXAMPLES.refund.blurb);
    // From empty boxes too: the lease button fills both, not just box A.
    p.$('clear-both').click();
    assert.equal(p.$('prose').value + p.$('rewrite').value, '');
    p.$('try-example').click();
    assert.equal(p.$('prose').value, EXAMPLES.lease.source);
    assert.equal(p.$('rewrite').value, EXAMPLES.lease.output);
    assert.equal(p.$('review-panel').hidden, false);
    assert.equal(inputs, 0);
    assert.equal(p.requests.length, 0);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('the lease evidence on the page matches the worked example', async () => {
  const p = await page();
  try {
    const note = id => p.$(id);
    assert.equal(text(note('n1a').querySelector('.lit')), 'may→ changed to can');
    assert.equal(words(note('n1a').querySelector('.chip')), 'a→b Differs');
    assert.equal(text(note('n1a').querySelector('.kind')), 'Modal verb');
    assert.equal(words(note('n1a').querySelector('.ns')), '“may” in the original, “can” in the rewrite. A 6–9 B 6–9');
    assert.equal(text(note('n1b').querySelector('.lit')), '30 days');
    assert.equal(words(note('n1b').querySelector('.chip')), 'ab Original only');
    assert.equal(words(note('n1b').querySelector('.ns')), 'Appears in the original, nowhere in the rewrite. A 32–39');
    assert.equal(text(note('n1c').querySelector('.lit')), 'notice');
    assert.equal(text(note('n2a').querySelector('.lit')), 'unless rent is overdue');
    assert.equal(text(note('n2a').querySelector('.kind')), 'Exception');
    assert.equal(words(p.$('p1').querySelector('.inboth')), '= In both passages: Alice Name A 0–5 B 0–5');
    assert.equal(text(p.$('p1').querySelector('.also')), 'Also marked, no note (common words): original only: with');
    const rows = [...p.window.document.querySelectorAll('#ledger-list > li')].map(words);
    assert.deepEqual(rows, [
      'ab Original only 3 30 days duration · notice wording · unless rent is overdue exception',
      'a→b Differs 1 may → changed to can modal verb',
      '= In both 1 Alice name',
      'Nothing listed under: Rewrite only · Other passage',
    ]);
    // Marks read as "original only: may", "rewrite only: can"; the letter follows its last token.
    const line = p.$('p1').querySelector('.blackline');
    assert.equal(line.querySelector('del[data-n="1a"]').textContent, 'original only: may');
    assert.equal(line.querySelector('ins[data-n="1a"]').textContent, 'rewrite only: can');
    assert.equal(line.querySelector('del[data-n="1b"]').textContent, 'original only: 30 days');
    assert.equal(line.querySelector('del[data-n="1b"]').nextElementSibling.textContent, 'b');
    assert.equal(p.$('export-passages').textContent, '2 rows with exact character spans');
    assert.equal(p.$('export-receipt').textContent, 'none; this rewrite was pasted, not generated here');
    await until(() => p.$('export-original').textContent.includes('sha256-712c650c…3868fa'));
    assert.equal(p.$('export-original').textContent, '97 characters · sha256-712c650c…3868fa');
    assert.equal(p.$('export-rewrite').textContent, '54 characters · sha256-36e8b003…d67db4');
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('editing either text hides the results and shows the stale state; Clear both shows the empty state', async () => {
  const p = await page();
  try {
    p.input('rewrite', EXAMPLES.lease.output + ' Extra.');
    assert.equal(p.$('review-panel').hidden, true);
    assert.equal(p.$('review-placeholder').hidden, false);
    assert.equal(p.$('placeholder-h').textContent, 'Texts changed');
    assert.equal(p.$('export-review-btn').disabled, true);
    assert.equal(p.$('compare-btn').textContent, 'Compare texts');
    assert.equal(p.$('compare-btn').classList.contains('ink'), true);
    assert.equal(p.$('compare-hint').hidden, true);
    assert.equal(p.$('strip-tag').hidden, true);
    assert.equal(p.$('strip-copy').textContent, 'Your texts. Examples, each fills both boxes:');
    assert.equal(p.$('try-example').getAttribute('aria-pressed'), 'false');
    assert.equal(p.$('rewrite-count').textContent, '61 characters · 3 passages');
    p.$('compare-btn').click();
    assert.equal(p.$('review-panel').hidden, false);
    assert.match(p.$('review-summary').textContent, /1 rewrite passage without a partner · your texts$/);
    p.$('clear-both').click();
    assert.equal(p.$('placeholder-h').textContent, 'Nothing compared yet');
    assert.equal(p.window.document.activeElement, p.$('prose'));
    p.$('compare-btn').click();
    assert.equal(p.$('placeholder-h').textContent, 'Could not compare');
    assert.equal(p.$('placeholder-body').textContent, 'Add an original to box A first.');
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('text over 100,000 characters is refused explicitly and nothing is compared (M18)', async () => {
  const p = await page();
  try {
    // A maxlength attribute would make the browser cut a longer paste silently,
    // so the boxes must not carry one: the page itself refuses the text instead.
    for (const id of ['prose', 'rewrite']) assert.equal(p.$(id).hasAttribute('maxlength'), false, `#${id} must not truncate silently`);
    p.input('prose', 'x'.repeat(100001));
    assert.equal(p.$('char-count').textContent, '100,001 characters · over the 100,000 limit');
    p.$('compare-btn').click();
    const message = p.$('workbench-status').textContent;
    assert.match(message, /at most 100,000 characters/);
    assert.match(message, /Box A has 100,001/);
    assert.match(message, /Nothing was compared/);
    assert.equal(p.$('review-panel').hidden, true);
    assert.equal(p.$('placeholder-h').textContent, 'Could not compare');
    assert.equal(p.$('placeholder-body').textContent, message);
    assert.equal(p.$('export-review-btn').disabled, true);
    // Over the passage cap: also explicit, also nothing compared.
    p.input('prose', 'Go now. '.repeat(301));
    p.input('rewrite', 'Go now.');
    p.$('compare-btn').click();
    assert.match(p.$('workbench-status').textContent, /up to 300 sentence or line passages per text\. .*Nothing was compared\./);
    assert.equal(p.$('review-panel').hidden, true);
    assert.equal(p.$('export-review-btn').disabled, true);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('filters hide other notes, and hover or focus outlines only the inspected marks', async () => {
  const p = await page();
  try {
    const filter = p.window.document.querySelector('#ledger-list .filter[data-state="a-only"]');
    filter.click();
    assert.equal(filter.getAttribute('aria-pressed'), 'true');
    assert.equal(p.$('workbench-status').textContent, 'Showing notes: Original only.');
    assert.equal(p.$('n1a').classList.contains('filtered-out'), true);
    assert.equal(p.$('n1b').classList.contains('filtered-out'), false);
    assert.equal(p.$('p1').querySelector('.inboth').classList.contains('filtered-out'), true);
    filter.click();
    assert.equal(p.$('workbench-status').textContent, 'Showing all notes.');
    assert.equal(p.window.document.querySelectorAll('.filtered-out').length, 0);
    p.$('n1b').dispatchEvent(new p.window.MouseEvent('mouseenter'));
    const lit = [...p.window.document.querySelectorAll('.hl')];
    assert.ok(lit.includes(p.$('n1b')));
    assert.ok(lit.some(n => n.tagName === 'DEL' && n.dataset.n === '1b'));
    assert.ok(lit.every(n => n.dataset.n === '1b'));
    p.$('n1b').dispatchEvent(new p.window.MouseEvent('mouseleave'));
    assert.equal(p.window.document.querySelectorAll('.hl').length, 0);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('a note span button selects the exact characters and Escape returns to it', async () => {
  const p = await page();
  try {
    const button = p.$('n1b').querySelector('.span-link');
    assert.equal(button.getAttribute('aria-label'), 'Select “30 days”, characters 32 to 39 of the original');
    button.click();
    const prose = p.$('prose');
    assert.equal(prose.value.slice(prose.selectionStart, prose.selectionEnd), '30 days');
    assert.equal(p.$('workbench-status').textContent, 'Selected characters 32 to 39 of the original: “30 days”. Press Escape to go back.');
    prose.dispatchEvent(new p.window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    assert.equal(p.window.document.activeElement, button);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('an exported packet verifies, and a shipped-format v1 packet opens with its evidence recomputed', async () => {
  const p = await page();
  try {
    const decision = p.$('p2').querySelector('input[value="accepted"]');
    decision.checked = true; decision.dispatchEvent(new p.window.Event('change'));
    assert.equal(p.$('review-record-count').textContent, '1 of 2');
    p.$('export-review-btn').click();
    await until(() => p.downloads.length === 1);
    const packet = JSON.parse(await p.downloads[0].text());
    assert.equal(packet.review.method, 'literal-spans-v2');
    assert.deepEqual(Object.keys(packet.review.rows[0]).sort(), ['decision', 'id', 'kind', 'output', 'source'], 'no evidence or extra field is stored');
    assert.equal(packet.review.rows[1].decision, 'accepted');
    p.input('packet-input', JSON.stringify(packet)); p.$('verify-packet-btn').click();
    await until(() => p.$('packet-status').textContent.includes('Checks completed'));
    // A literal-spans-v1 packet, as the previous page exported them.
    const source = 'Alice may cancel the lease with 30 days notice. The deposit is refundable unless rent is overdue.';
    const review = compareTexts(source, 'Alice can cancel the lease. The deposit is refundable.', 'literal-spans-v1');
    review.rows[0].decision = 'needs-change';
    const old = await makeReviewPacket({ source, output: 'Alice can cancel the lease. The deposit is refundable.', review });
    p.input('packet-input', JSON.stringify(old)); p.$('verify-packet-btn').click();
    await until(() => p.$('packet-status').textContent.includes('Checks completed'));
    p.$('open-packet-btn').click();
    assert.equal(p.$('review-heading').textContent, '2 passages compared, 4 literal differences marked.');
    assert.equal(p.$('p1').querySelector('input[value="needs-change"]').checked, true);
    assert.equal(p.$('export-receipt').textContent, 'none in this packet');
    assert.equal(p.requests.length, 0);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

// ---------------------------------------------------------------- M3: bundle Ed25519 fails closed
async function signedBundle(p) {
  const bundle = await p.window.makeBundle([['alice', 'likes', 'cats'], ['bob', 'owns', 'dogs']], 'Test');
  const pair = await webcrypto.subtle.generateKey({ name: 'Ed25519' }, true, ['sign', 'verify']);
  const raw = new Uint8Array(await webcrypto.subtle.exportKey('raw', pair.publicKey));
  const payload = new TextEncoder().encode(`${bundle.canonical_tome}|${bundle.state_integer}|${bundle.timestamp}`);
  const signature = new Uint8Array(await webcrypto.subtle.sign('Ed25519', pair.privateKey, payload));
  return { ...bundle, public_key: 'ed25519:' + Buffer.from(raw).toString('base64'), public_signature: 'ed25519:' + Buffer.from(signature).toString('base64'), raw };
}

test('the bundle check fails closed on a malformed Ed25519 key or signature (M3)', async () => {
  const p = await page();
  try {
    const good = await signedBundle(p);
    const verdict = async bundle => p.window.verifyBundle(JSON.parse(JSON.stringify({ ...bundle, raw: undefined })));
    const ok = await verdict(good);
    assert.equal(ok.ok, true);
    assert.equal(ok.signatures.ed25519.status, 'verified');
    const malformed = {
      'a 31-byte key': { public_key: 'ed25519:' + Buffer.from(good.raw.slice(0, 31)).toString('base64') },
      'a 33-byte key': { public_key: 'ed25519:' + Buffer.from([...good.raw, 0]).toString('base64') },
      'undecodable base64': { public_key: 'ed25519:!!!not-base64!!!' },
      'a key without its prefix': { public_key: Buffer.from(good.raw).toString('base64') },
      'a non-string key': { public_key: 12345 },
      'undecodable signature': { public_signature: 'ed25519:%%%' },
    };
    for (const [name, change] of Object.entries(malformed)) {
      const r = await verdict({ ...good, ...change });
      assert.equal(r.ok, false, `${name} must not pass`);
      assert.equal(r.signatures.ed25519.status, 'malformed', name);
      assert.match(r.reason, /MALFORMED/);
    }
    const tampered = await verdict({ ...good, public_signature: 'ed25519:' + Buffer.alloc(64, 1).toString('base64') });
    assert.equal(tampered.ok, false);
    assert.equal(tampered.signatures.ed25519.status, 'invalid');
    // The rendered verdict says so, with the label escaped as text.
    p.input('verify-input', JSON.stringify({ ...good, raw: undefined, public_key: 'ed25519:AAAA' }));
    p.$('verify').click();
    await until(() => p.$('verify-result').textContent.includes('MALFORMED'));
    assert.match(p.$('verify-result').textContent, /^✗/);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('only a browser without Ed25519 in WebCrypto reports the signature as unsupported (M3)', async () => {
  const subtle = {
    digest: (...args) => webcrypto.subtle.digest(...args),
    importKey: async () => { throw new DOMException('Ed25519 is not supported', 'NotSupportedError'); },
    verify: async () => { throw new DOMException('Ed25519 is not supported', 'NotSupportedError'); },
  };
  const noEd25519 = { subtle, getRandomValues: array => webcrypto.getRandomValues(array) };
  const reference = await page();
  let good;
  try { good = await signedBundle(reference); } finally { reference.close(); }
  const p = await page(undefined, { crypto: noEd25519 });
  try {
    const r = await p.window.verifyBundle(JSON.parse(JSON.stringify({ ...good, raw: undefined })));
    assert.equal(r.signatures.ed25519.status, 'unsupported');
    assert.match(r.signatures.ed25519.label, /not checked/);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});
