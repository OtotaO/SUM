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
    // The boxes hold the user's own texts, so opening asks first, in the page.
    assert.equal(p.$('edit-guard').hidden, false);
    assert.match(p.$('guard-text').textContent, /Replace your texts and 1 recorded decision with the packet's texts and decisions\?/);
    p.$('guard-yes').click();
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
    assert.equal(p.$('review-heading').textContent, '2 passages compared, 4 differences noted.');
    assert.equal(p.$('review-summary').textContent, '2 pairs matched by shared words · 0 identical · none without a partner · the example texts');
    assert.equal(p.$('try-example').getAttribute('aria-pressed'), 'true');
    assert.equal(p.$('compare-btn').textContent, 'Compare again');
    assert.equal(p.$('compare-btn').classList.contains('ink'), false, 'Compare is secondary while results are current');
    assert.equal(p.$('export-review-btn').disabled, false);
    assert.equal(p.$('char-count').textContent, '97 UTF-16 code units · 2 passages');
    assert.equal(p.$('rewrite-count').textContent, '54 UTF-16 code units · 2 passages');
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
    assert.equal(p.$('review-heading').textContent, '7 passages compared, 17 differences noted.');
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
    assert.equal(text(note('n1a').querySelector('.kind')), 'Modal-list word');
    assert.equal(words(note('n1a').querySelector('.ns')), '“may” in the original, “can” in the rewrite. A 6–9 B 6–9');
    assert.equal(text(note('n1b').querySelector('.lit')), '30 days');
    assert.equal(words(note('n1b').querySelector('.chip')), 'ab Removed');
    assert.equal(words(note('n1b').querySelector('.ns')), '“30 days” has no normalized word match in the entire rewrite text. A 32–39');
    assert.equal(text(note('n1c').querySelector('.lit')), 'notice');
    assert.equal(text(note('n2a').querySelector('.lit')), 'unless rent is overdue');
    assert.equal(text(note('n2a').querySelector('.kind')), 'Exception-marker phrase');
    assert.equal(words(p.$('p1').querySelector('.inboth')), '= In both passages: “Alice” Capitalized word A 0–5 B 0–5');
    assert.equal(text(p.$('p1').querySelector('.also')), 'Also marked, no note (common words): original only: with');
    const rows = [...p.window.document.querySelectorAll('#ledger-list > li')].map(words);
    assert.deepEqual(rows, [
      'ab Removed 3 30 days duration · notice wording · unless rent is overdue exception-marker phrase',
      'a→b Differs 1 may → changed to can modal-list word',
      '= In both 1 Alice capitalized word',
      'Nothing listed under: Added · Other passage · Matched words. 1 common word is also marked, without a note.',
    ]);
    // Marks read as "original only: may", "rewrite only: can"; the letter follows its last token.
    const line = p.$('p1').querySelector('.blackline');
    assert.equal(line.querySelector('del[data-n="1a"]').textContent, 'original only: may');
    assert.equal(line.querySelector('ins[data-n="1a"]').textContent, 'rewrite only: can');
    assert.equal(line.querySelector('del[data-n="1b"]').textContent, 'original only: 30 days');
    assert.equal(line.querySelector('del[data-n="1b"]').nextElementSibling.textContent, 'b');
    // Read as Original and Read as Rewrite would show each passage exactly.
    const viewOf = (el, hide) => { const c = el.cloneNode(true); for (const n of c.querySelectorAll(hide + ', .sep, .vh, sup.ref')) n.remove(); return c.textContent; };
    assert.equal(viewOf(line, '.irun'), EXAMPLES.lease.source.slice(0, 47));
    assert.equal(viewOf(line, '.drun'), EXAMPLES.lease.output.slice(0, 27));
    assert.equal(p.$('export-passages').textContent, '2 rows with exact UTF-16 code-unit spans');
    assert.equal(p.$('export-receipt').textContent, 'none; this is a built-in example');
    await until(() => p.$('export-original').textContent.includes('sha256-712c650c…3868fa'));
    assert.equal(p.$('export-original').textContent, '97 UTF-16 code units · sha256-712c650c…3868fa');
    assert.equal(p.$('export-rewrite').textContent, '54 UTF-16 code units · sha256-36e8b003…d67db4');
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
    assert.equal(p.$('rewrite-count').textContent, '61 UTF-16 code units · 3 passages');
    p.$('compare-btn').click();
    assert.equal(p.$('review-panel').hidden, false);
    assert.match(p.$('review-summary').textContent, /1 rewrite passage without a partner · your texts$/);
    p.$('clear-both').click();
    assert.equal(p.$('edit-guard').hidden, false, 'clearing your own texts asks first');
    p.$('guard-yes').click();
    assert.equal(p.$('placeholder-h').textContent, 'Nothing compared yet');
    assert.equal(p.window.document.activeElement, p.$('prose'));
    p.$('compare-btn').click();
    assert.equal(p.$('placeholder-h').textContent, 'Could not compare');
    assert.equal(p.$('placeholder-body').textContent, 'Add an original to box A first.');
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('text over 100,000 UTF-16 code units is refused explicitly and nothing is compared (M18)', async () => {
  const p = await page();
  try {
    // A maxlength attribute would make the browser cut a longer paste silently,
    // so the boxes must not carry one: the page itself refuses the text instead.
    for (const id of ['prose', 'rewrite']) assert.equal(p.$(id).hasAttribute('maxlength'), false, `#${id} must not truncate silently`);
    p.input('prose', 'x'.repeat(100001));
    await until(() => p.$('char-count').textContent === '100,001 UTF-16 code units · over the 100,000 limit');
    p.$('compare-btn').click();
    const message = p.$('workbench-status').textContent;
    assert.match(message, /at most 100,000 UTF-16 code units/);
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
    assert.equal(p.$('workbench-status').textContent, 'Showing notes: Removed.');
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

test('a note span button selects the exact UTF-16 code units and Escape returns to it', async () => {
  const p = await page();
  try {
    const button = p.$('n1b').querySelector('.span-link');
    assert.equal(button.getAttribute('aria-label'), 'Select “30 days”, UTF-16 code units 32 to 39 of the original');
    button.click();
    const prose = p.$('prose');
    assert.equal(prose.value.slice(prose.selectionStart, prose.selectionEnd), '30 days');
    assert.equal(p.$('workbench-status').textContent, 'Selected UTF-16 code units 32 to 39 of the original: “30 days”. The box is read-only until you click in it. Press Escape to go back.');
    assert.equal(prose.readOnly, true, 'a jump selects for reading, so one key cannot replace the selection');
    // With decisions recorded, a key on the read-only box is swallowed without the edit question.
    const d = p.$('d-source-s1-accepted'); d.checked = true; d.dispatchEvent(new p.window.Event('change'));
    button.click();
    assert.equal(beforeInput(p, 'prose'), true);
    assert.equal(p.$('edit-guard').hidden, true);
    prose.dispatchEvent(new p.window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    assert.equal(p.window.document.activeElement, button);
    assert.equal(prose.readOnly, false);
    assert.equal(prose.selectionStart, prose.selectionEnd, 'the selection is collapsed when the box is released');
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
    if (!p.$('edit-guard').hidden) p.$('guard-yes').click();
    assert.equal(p.$('review-heading').textContent, '2 passages compared, 4 differences noted.');
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

// ---------------------------------------------------------------- data safety
const beforeInput = (p, id) => {
  const event = new p.window.InputEvent('beforeinput', { bubbles: true, cancelable: true, inputType: 'insertText', data: ' ' });
  p.$(id).dispatchEvent(event);
  return event.defaultPrevented;
};

test('an edit that would hide recorded decisions asks in the page first, and Restore brings them back', async () => {
  const p = await page();
  try {
    const decide = (row, value) => { const r = p.$(`d-${row}-${value}`); r.checked = true; r.dispatchEvent(new p.window.Event('change')); };
    assert.equal(beforeInput(p, 'rewrite'), false, 'no decisions yet: editing is not interrupted');
    decide('source-s1', 'needs-change'); decide('source-s2', 'accepted');
    assert.equal(beforeInput(p, 'prose'), true, 'a keystroke is held back while decisions exist');
    assert.equal(p.$('edit-guard').hidden, false);
    assert.match(p.$('guard-text').textContent, /2 recorded decisions/);
    assert.equal(p.$('prose').value, EXAMPLES.lease.source);
    p.$('guard-no').click();
    assert.equal(p.$('edit-guard').hidden, true);
    assert.equal(p.$('review-panel').hidden, false);
    assert.equal(beforeInput(p, 'prose'), true);
    p.$('guard-yes').click();
    assert.equal(beforeInput(p, 'prose'), false, 'after "Edit the texts", typing goes through');
    p.input('prose', EXAMPLES.lease.source + ' More text.');
    assert.equal(p.$('review-panel').hidden, true);
    assert.equal(p.$('restore-review').hidden, false);
    assert.equal(p.$('restore-review').textContent, 'Restore the reviewed texts and 2 decisions');
    p.$('restore-review').click();
    assert.equal(p.$('prose').value, EXAMPLES.lease.source);
    assert.equal(p.$('d-source-s1-needs-change').checked, true);
    assert.equal(p.$('d-source-s2-accepted').checked, true);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('Compare again keeps decisions: unchanged texts keep all, edited texts keep those of unchanged passages', async () => {
  const p = await page();
  try {
    const decide = (row, value) => { const r = p.$(`d-${row}-${value}`); r.checked = true; r.dispatchEvent(new p.window.Event('change')); };
    decide('source-s1', 'needs-change'); decide('source-s2', 'accepted');
    p.$('compare-btn').click();
    assert.equal(p.$('d-source-s1-needs-change').checked, true);
    assert.equal(p.$('d-source-s2-accepted').checked, true);
    assert.match(p.$('workbench-status').textContent, /have not changed .* 2 recorded decisions are kept/);
    // Edit passage 2 only: passage 1's decision carries over, passage 2's does not.
    beforeInput(p, 'rewrite'); p.$('guard-yes').click();
    p.input('rewrite', 'Alice can cancel the lease. The deposit is fully refundable.');
    p.$('compare-btn').click();
    assert.equal(p.$('d-source-s1-needs-change').checked, true);
    assert.equal(p.$('d-source-s2-unreviewed').checked, true);
    assert.match(p.$('workbench-status').textContent, /kept 1 decision for passages that did not change · 1 decision for changed or ambiguous passages was not carried over/);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('an opened literal-spans-v1 packet is compared again with its own passage rules', async () => {
  const p = await page();
  try {
    const source = 'Shipping costs $7.95 per order. Refunds follow.';
    const output = 'Shipping costs $7.95 per order. Refunds are quick.';
    const review = compareTexts(source, output, 'literal-spans-v1');
    review.rows[0].decision = 'accepted';
    const packet = await makeReviewPacket({ source, output, review });
    p.input('packet-input', JSON.stringify(packet)); p.$('verify-packet-btn').click();
    await until(() => p.$('packet-status').textContent.includes('Checks completed'));
    p.$('open-packet-btn').click();
    if (!p.$('edit-guard').hidden) p.$('guard-yes').click();
    assert.equal(p.$('review-heading').textContent, '1 passage compared, 1 difference noted.', 'v1 reads this as one passage');
    p.$('compare-btn').click();
    assert.equal(p.$('review-heading').textContent, '1 passage compared, 1 difference noted.', 'unchanged: kept as the packet had it');
    assert.equal(p.$('d-source-s1-accepted').checked, true);
    beforeInput(p, 'rewrite'); p.$('guard-yes').click();
    p.input('rewrite', output + ' ');
    p.$('compare-btn').click();
    assert.match(p.$('workbench-status').textContent, /packet's rules \(literal-spans-v1\)/);
    assert.equal(p.$('review-heading').textContent.startsWith('1 passage compared'), true, 'still one v1 passage, not re-split');
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('example buttons ask before replacing your own texts', async () => {
  const p = await page();
  try {
    p.input('prose', 'My own contract text.');
    p.$('try-example').click();
    assert.equal(p.$('edit-guard').hidden, false);
    assert.equal(p.$('prose').value, 'My own contract text.', 'nothing replaced before you answer');
    p.$('guard-no').click();
    assert.equal(p.$('prose').value, 'My own contract text.');
    p.window.document.querySelector('[data-example="refund"]').click();
    p.$('guard-yes').click();
    assert.equal(p.$('prose').value, EXAMPLES.refund.source);
    assert.equal(p.$('rewrite').value, EXAMPLES.refund.output);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('repeated identical passage pairs never inherit another occurrence\'s decision after edits', async () => {
  const p = await page();
  try {
    const source = 'Yes. Pay rent. Yes. Call us. Yes.';
    p.input('prose', source); p.input('rewrite', source); p.$('compare-btn').click();
    const decision = p.$('d-source-s3-needs-change');
    decision.checked = true; decision.dispatchEvent(new p.window.Event('change'));
    beforeInput(p, 'rewrite'); p.$('guard-yes').click();
    p.input('rewrite', 'Yes. Pay rent. Yes. Call them. Yes.'); p.$('compare-btn').click();
    assert.equal(p.$('d-source-s1-unreviewed').checked, true);
    assert.equal(p.$('d-source-s3-unreviewed').checked, true);
    assert.match(p.$('workbench-status').textContent, /1 decision for changed or ambiguous passages was not carried over/);
    // Removing an occurrence is ambiguous too, even if the survivor has the same offset.
    p.input('prose', 'OK. Pay now. OK.'); p.input('rewrite', 'OK. Pay later. OK.'); p.$('compare-btn').click();
    for (const [id, value] of [['s1', 'accepted'], ['s3', 'needs-change']]) {
      const d = p.$(`d-source-${id}-${value}`); d.checked = true; d.dispatchEvent(new p.window.Event('change'));
    }
    beforeInput(p, 'prose'); p.$('guard-yes').click();
    p.input('prose', 'Pay now. OK.'); p.$('compare-btn').click();
    assert.equal(p.window.document.querySelectorAll('#review-rows input[value="accepted"]:checked, #review-rows input[value="needs-change"]:checked').length, 0);
    assert.match(p.$('workbench-status').textContent, /2 decisions for changed or ambiguous passages were not carried over/);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('noncancelable IME input restores the snapshot before invalidating review or extraction', async () => {
  const p = await page();
  try {
    const decision = p.$('d-source-s1-accepted'); decision.checked = true; decision.dispatchEvent(new p.window.Event('change'));
    const original = p.$('prose').value;
    p.window.__sumLastTriples = [['sentinel', 'is', 'retained']];
    for (const confirm of [false, true]) {
      p.$('prose').setSelectionRange(2, 2);
      p.$('prose').dispatchEvent(new p.window.InputEvent('beforeinput', { bubbles: true, cancelable: false, inputType: 'insertCompositionText', data: 'か', isComposing: true }));
      p.$('prose').value = original.slice(0, 2) + 'か' + original.slice(2);
      p.$('prose').dispatchEvent(new p.window.InputEvent('input', { bubbles: true, inputType: 'insertCompositionText', data: 'か', isComposing: true }));
      assert.equal(p.$('prose').value, original);
      assert.equal(p.$('review-panel').hidden, false);
      assert.equal(p.$('d-source-s1-accepted').checked, true);
      assert.deepEqual(p.window.__sumLastTriples, [['sentinel', 'is', 'retained']]);
      assert.equal(p.$('edit-guard').hidden, false);
      p.$(confirm ? 'guard-yes' : 'guard-no').click();
    }
    p.input('prose', original + 'か');
    assert.equal(p.$('prose').value, original + 'か');
    assert.equal(p.$('review-panel').hidden, true);
    p.$('restore-review').click();
    assert.equal(p.$('prose').value, original);
    assert.equal(p.$('d-source-s1-accepted').checked, true);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('replacement prompts count decisions in a hidden restorable review', async () => {
  const p = await page();
  try {
    const d = p.$('d-source-s1-accepted'); d.checked = true; d.dispatchEvent(new p.window.Event('change'));
    beforeInput(p, 'rewrite'); p.$('guard-yes').click(); p.input('rewrite', 'New working text.');
    for (const button of [p.window.document.querySelector('[data-example="refund"]'), p.$('clear-both')]) {
      button.click();
      assert.match(p.$('guard-text').textContent, /1 recorded decision/);
      p.$('guard-no').click();
      assert.match(p.$('restore-review').textContent, /1 decision/);
    }
    const source = 'A packet source.', output = 'A packet output.';
    const packet = await makeReviewPacket({ source, output, review: compareTexts(source, output) });
    p.input('packet-input', JSON.stringify(packet)); p.$('verify-packet-btn').click();
    await until(() => !p.$('open-packet-btn').disabled);
    p.$('open-packet-btn').click();
    assert.match(p.$('guard-text').textContent, /1 recorded decision/);
    p.$('guard-no').click(); p.$('restore-review').click();
    assert.equal(p.$('d-source-s1-accepted').checked, true);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('generated replacement waits for confirmation and reports decision carry without overwriting it', async () => {
  const p = await page();
  try {
    const d = p.$('d-source-s1-accepted'); d.checked = true; d.dispatchEvent(new p.window.Event('change'));
    const output = p.$('rewrite').value;
    const data = { source_text: p.$('prose').value, tome: 'A completely different generated rewrite.', triples_used: [], quantized_sliders: {} };
    const deliver = () => { p.window.__sumLastRender = data; p.window.document.dispatchEvent(new p.window.CustomEvent('sum:render', { detail: data })); };
    deliver();
    assert.equal(p.$('rewrite').value, output);
    assert.match(p.$('guard-text').textContent, /1 recorded decision/);
    p.$('guard-no').click();
    assert.equal(p.$('d-source-s1-accepted').checked, true);
    deliver(); p.$('guard-yes').click();
    assert.equal(p.$('rewrite').value, data.tome);
    assert.match(p.$('workbench-status').textContent, /1 decision for changed or ambiguous passages was not carried over/);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('bidi and invisible characters are labeled and isolated in evidence but exported unchanged', async () => {
  const p = await page();
  try {
    const source = 'Pay the deposit within 30 days. שלום is welcome.';
    const output = 'Pay the deposit within \u202e30 days. שלום\u200b is welcome.';
    p.input('prose', source); p.input('rewrite', output); p.$('compare-btn').click();
    const displayed = [...p.window.document.querySelectorAll('#review-rows .notes, #ledger-list')].map(n => n.textContent).join(' ');
    assert.doesNotMatch(displayed, /[\u202e\u200b]/);
    assert.match(displayed, /U\+202E/);
    assert.match(displayed, /U\+200B/);
    assert.ok(p.window.document.querySelector('#review-rows .notes bdi[dir="auto"]'));
    p.$('export-review-btn').click(); await until(() => p.downloads.length === 1);
    const packet = JSON.parse(await p.downloads[0].text());
    assert.equal(packet.source.text, source); assert.equal(packet.output.text, output);
    p.input('prose', '👍👍🏽 ');
    assert.match(p.$('char-count').textContent, /^7 UTF-16 code units/);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('repeated shared-word evidence is paged and the sticky decision indicator is bounded', async () => {
  const p = await page();
  try {
    p.input('prose', 'if '.repeat(1500)); p.input('rewrite', 'if '.repeat(1499) + 'x'); p.$('compare-btn').click();
    assert.equal(p.window.document.querySelectorAll('#review-rows .inboth').length, 100);
    const more = p.window.document.querySelector('.more-inboth');
    assert.match(more.textContent, /not yet shown/); more.click();
    assert.equal(p.window.document.querySelectorAll('#review-rows .inboth').length, 200);
    assert.ok(p.window.document.querySelectorAll('*').length < 5000);
    p.input('prose', 'A passage. '.repeat(100)); p.input('rewrite', 'A passage. '.repeat(100)); p.$('compare-btn').click();
    assert.equal(p.$('decided-minis').children.length, 8);
    assert.equal(p.$('decided-count').textContent, '0 of 100 passages decided');
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('one huge changed run stays bounded on first draw and announces the hidden remainder', async () => {
  const p = await page();
  try {
    p.input('prose', 'if '.repeat(33000)); p.input('rewrite', 'x'); p.$('compare-btn').click();
    await until(() => !p.$('review-panel').hidden);
    assert.ok(p.window.document.querySelectorAll('*').length < 15000);
    assert.ok(p.window.document.querySelectorAll('.blackline del, .blackline ins').length <= 1600);
    assert.match(p.window.document.querySelector('.blackline-wrap .more-btn').textContent, /shown in part.*Show the rest/);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('a duplicate unpaired output passage is not claimed absent from the original', async () => {
  const p = await page();
  try {
    p.input('prose', 'Pay rent. Leave.'); p.input('rewrite', 'Pay rent. Leave. Pay rent.'); p.$('compare-btn').click();
    const labels = [...p.window.document.querySelectorAll('.pkind')].map(n => n.textContent);
    assert.ok(labels.includes('no partner passage in the original'));
    assert.equal(labels.some(t => t.includes('rewrite only')), false);
  } finally { p.close(); }
});

test('changing a checked packet while its confirmation is open cannot replace or invalidate the review', async () => {
  const p = await page();
  try {
    const d = p.$('d-source-s1-accepted'); d.checked = true; d.dispatchEvent(new p.window.Event('change'));
    const source = 'Incoming source.', output = 'Incoming output.';
    const packet = await makeReviewPacket({ source, output, review: compareTexts(source, output) });
    p.input('packet-input', JSON.stringify(packet)); p.$('verify-packet-btn').click();
    await until(() => !p.$('open-packet-btn').disabled);
    p.$('open-packet-btn').click();
    p.window.__sumLastTriples = [['sentinel', 'is', 'retained']];
    p.input('packet-input', '{}'); p.$('guard-yes').click();
    assert.equal(p.$('prose').value, EXAMPLES.lease.source);
    assert.equal(p.$('d-source-s1-accepted').checked, true);
    assert.deepEqual(p.window.__sumLastTriples, [['sentinel', 'is', 'retained']]);
    assert.match(p.$('workbench-status').textContent, /packet changed.*Check it again/);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('the page prints only true statements for the adversarial pairs', async () => {
  const { ADVERSARIAL } = await import('./evidence_oracle.mjs');
  const p = await page();
  try {
    for (const [a, b] of ADVERSARIAL.slice(0, 12)) {
      p.$('clear-both').click(); if (!p.$('edit-guard').hidden) p.$('guard-yes').click();
      p.input('prose', a); p.input('rewrite', b); p.$('compare-btn').click();
      const heading = p.$('review-heading').textContent;
      if (a === b) assert.match(heading, /no literal differences/); else assert.doesNotMatch(heading, /no literal differences/);
      assert.doesNotMatch(p.$('ledger-list').textContent, /common words or punctuation/);
      // Every In-both item in the ledger shows strings both passages literally have.
      for (const it of p.window.document.querySelectorAll('#ledger-list li.both .it')) {
        const literal = it.textContent.replace(/\[U\+([0-9A-F]{4,6})(?: [^\]]+)?\]/g, (_, cp) => String.fromCodePoint(parseInt(cp, 16)));
        for (const form of literal.split(' / ')) assert.ok(a.includes(form) || b.includes(form), form);
        if (!literal.includes(' / ')) assert.ok(a.includes(literal) && b.includes(literal), literal);
      }
    }
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});
