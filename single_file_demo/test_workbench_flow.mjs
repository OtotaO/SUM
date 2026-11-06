import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { webcrypto } from 'node:crypto';
import { JSDOM, VirtualConsole } from 'jsdom';
import { handle as detailHandle } from './detail_worker.js';

// jsdom has no Worker. This stand-in answers with the real worker module's handler, asynchronously, so the
// page's message protocol and deadline are exercised; 'hang' never answers, 'manual' answers only when the
// test calls deliver() (even after terminate, as a message already queued would), and 'absent' leaves
// Worker undefined, as in a browser without module workers.
function fakeWorker(mode, created) {
  return class {
    constructor(url, options) { this.url = String(url); this.options = options; this.terminated = false; created.push(this); }
    postMessage(data) {
      if (mode === 'manual') { this.deliver = () => this.onmessage({ data: structuredClone(detailHandle(data)) }); return; }
      if (mode === 'hang') return;
      setTimeout(() => { if (!this.terminated && this.onmessage) this.onmessage({ data: structuredClone(detailHandle(data)) }); }, 0);
    }
    terminate() { this.terminated = true; }
  };
}

const html = await readFile(new URL('./index.html', import.meta.url), 'utf8');
const captured = JSON.parse(await readFile(new URL('../fixtures/render_receipts/source_render.json', import.meta.url)));
const keys = JSON.parse(await readFile(new URL('../fixtures/render_receipts/jwks_at_capture.json', import.meta.url)));
let instance = 0;
async function until(predicate) {
  for (let i = 0; i < 500; i++) { if (predicate()) return; await new Promise(r => setTimeout(r, 2)); }
  assert.fail('Timed out waiting for the workbench action');
}
const detailsDone = p => until(() => p.$('detail-summary').textContent && !p.$('detail-summary').textContent.startsWith('Checking'));
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };

async function page(handler = async () => new Response('{}', { status: 503 }), { worker = 'answer', url = 'https://workbench.example' } = {}) {
  const errors = [], requests = [], downloads = [], workers = [];
  const mockFetch = async (url, init) => {
    if (String(url).includes('altitude_rungs')) return new Response('{}', { status: 404 });
    requests.push({ url, init });
    return handler(url, init);
  };
  const console = new VirtualConsole();
  console.on('jsdomError', e => { if (!e.message.includes('navigation')) errors.push(e); });
  const dom = new JSDOM(html, { url, runScripts: 'dangerously', virtualConsole: console,
    beforeParse(window) {
      Object.defineProperty(window, 'crypto', { value: webcrypto });
      window.TextEncoder = TextEncoder; window.structuredClone = structuredClone;
      window.fetch = mockFetch; window.alert = message => { errors.push(new Error(message)); };
      window.HTMLElement.prototype.scrollIntoView = () => {};
      if (worker !== 'absent') window.Worker = fakeWorker(worker, workers);
    } });
  const window = dom.window;
  globalThis.window = window; globalThis.document = window.document; globalThis.fetch = mockFetch;
  // workbench.js runs as a Node module here, so it sees Node's globals, not the jsdom window's.
  globalThis.Worker = window.Worker;
  const originalCreate = URL.createObjectURL, originalRevoke = URL.revokeObjectURL;
  URL.createObjectURL = blob => { downloads.push(blob); return 'blob:test'; };
  URL.revokeObjectURL = () => {};
  await import(`./workbench.js?test=${++instance}`);
  const $ = id => window.document.getElementById(id);
  const input = (id, value) => { $(id).value = value; $(id).dispatchEvent(new window.Event('input', { bubbles: true })); };
  return { $, input, window, requests, downloads, errors, workers, close() { dom.window.close(); URL.createObjectURL = originalCreate; URL.revokeObjectURL = originalRevoke; } };
}

test('existing rewrite comparison, source selection, decision, export and offline recipient checks', async () => {
  const p = await page();
  try {
    p.input('prose', 'Alice may cancel with 30 days notice. Deposit is refundable unless overdue.');
    p.input('rewrite', 'Alice must cancel with 3 days notice. Deposit is refundable.');
    p.$('compare-btn').click();
    assert.equal(p.$('review-panel').hidden, false);
    assert.match(p.$('review-rows').textContent, /Possible changed passage/);
    const sourceLink = p.$('review-rows').querySelector('button'); sourceLink.click();
    assert.equal(p.$('prose').value.slice(p.$('prose').selectionStart, p.$('prose').selectionEnd), 'Alice may cancel with 30 days notice.');
    const decision = p.$('review-rows').querySelector('select');
    decision.value = 'needs-change'; decision.dispatchEvent(new p.window.Event('change'));
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
    assert.equal(p.$('review-rows').querySelector('select').value, 'needs-change');
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
    const decision = p.$('review-rows').querySelector('select');
    decision.value = 'needs-change'; decision.dispatchEvent(new p.window.Event('change'));
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

// ---------------------------------------------------------------- key pin
// Packets are unsigned containers: anyone can sign a render receipt with a
// fresh key and carry that key in the packet. The workbench must pin the
// signing key to this site's /.well-known/jwks.json and say plainly when it
// is not one of them.
// A namespace import, so these names cannot collide with imports elsewhere in this file.
import * as packetLib from './review_packet.js';
import { canonicalize } from './vendor/sum-verify-deps.js';

const packetSource = 'Alice was born in 1990. Alice graduated in 2012.';
const siteKeysOnly = async url => url === '/.well-known/jwks.json' ? Response.json(keys) : new Response('{}', { status: 404 });
const sitePacket = () => packetLib.makeReviewPacket({ source: packetSource, output: captured.tome, review: packetLib.compareTexts(packetSource, captured.tome),
  render: { receipt: captured.render_receipt, triples: captured.triples_used, sliders: captured.quantized_sliders }, jwks: keys });

// Signed the way worker/src/receipt/sign.ts signs: detached JWS, header
// {alg, kid, b64: false, crit: ['b64']}, Ed25519 over header + '.' + JCS(payload).
async function forgedPacket(kid, privateKey = null, x = null, header = {}) {
  if (!privateKey) {
    const pair = await webcrypto.subtle.generateKey({ name: 'Ed25519' }, true, ['sign', 'verify']);
    privateKey = pair.privateKey; x = (await webcrypto.subtle.exportKey('jwk', pair.publicKey)).x;
  }
  const tome = 'Alice was born in 1990. Alice graduated in 2099.';
  const triples = [['alice', 'born_in', '1990'], ['alice', 'graduated_in', '2099']];
  const payload = { ...captured.render_receipt.payload, render_id: 'f0f0f0f0f0f0f0f0', triples_hash: await packetLib.hashText(canonicalize(triples)), tome_hash: await packetLib.hashText(tome) };
  const protectedHeader = Buffer.from(JSON.stringify({ alg: 'EdDSA', kid, b64: false, crit: ['b64'], ...header })).toString('base64url');
  const signature = await webcrypto.subtle.sign('Ed25519', privateKey, new TextEncoder().encode(protectedHeader + '.' + canonicalize(payload)));
  const render = { receipt: { schema: 'sum.render_receipt.v1', kid, payload, jws: `${protectedHeader}..${Buffer.from(signature).toString('base64url')}` }, triples, sliders: payload.sliders_quantized };
  return packetLib.makeReviewPacket({ source: packetSource, output: tome, review: packetLib.compareTexts(packetSource, tome), render, jwks: { keys: [{ kty: 'OKP', crv: 'Ed25519', x, kid, alg: 'EdDSA', use: 'sig' }] } });
}

// The site-key sentence, now and before: a packet must never print it.
const SITE_KEY_WORDING = /Signed by key|listed when this page fetched it|This site publishes key/;

async function checkPacket(p, packet) {
  p.input('packet-input', JSON.stringify(packet)); p.$('verify-packet-btn').click();
  // Wait for any final status, whatever its wording.
  await until(() => !p.$('packet-status').textContent.startsWith('Checking packet'));
  return p.$('packet-status').textContent;
}

test('a packet signed with its own key never shows the site-key wording, checked or opened', async () => {
  const forged = await forgedPacket('sum-render-2026-10-01-1');
  const p = await page(siteKeysOnly);
  try {
    const shown = await checkPacket(p, forged);
    assert.match(shown, /^Checks completed/);
    assert.match(shown, /"signature": "verified"/, 'the forged packet is internally consistent');
    assert.match(shown, /"key_pin": "not-site-key"/);
    assert.match(shown, /\nKey "sum-render-2026-10-01-1" came from the packet and was not in this site's \/\.well-known\/jwks\.json when this page fetched it\. Anyone can create a packet signed with their own key, so the signer is unknown\.\n/);
    assert.doesNotMatch(shown, SITE_KEY_WORDING);
    assert.ok(p.requests.some(r => r.url === '/.well-known/jwks.json'), 'site keys are fetched from this site');
    p.$('open-packet-btn').click();
    // Opening a packet resets the receipt: the receipt button stays disabled
    // and the render pane hidden, so no user can reach the receipt check from
    // here. The handler's packet-key branch is kept as defense in depth and
    // is exercised by enabling the button directly.
    assert.equal(p.$('verify-receipt-btn').disabled, true);
    assert.equal(p.$('render-output').style.display, 'none');
    p.$('verify-receipt-btn').click(); await new Promise(r => setTimeout(r, 20));
    assert.equal(p.$('verify-receipt-result').textContent, '', 'a disabled button does not run the check');
    p.$('verify-receipt-btn').disabled = false; p.$('verify-receipt-btn').click();
    await until(() => p.$('render-trust-status').textContent !== 'Receipt not checked:');
    assert.equal(p.$('render-trust-status').textContent, 'Signature valid, signer unknown:');
    const result = p.$('verify-receipt-result').textContent;
    assert.match(result, /^The signature is valid, and the output bytes, selected claims and slider settings match the signed receipt\. Key "sum-render-2026-10-01-1" came from the packet and was not in this site's/);
    assert.match(result, /signer is unknown/);
    assert.doesNotMatch(result, SITE_KEY_WORDING);
    assert.doesNotMatch(result, /^Verified/);
    // Export re-emits the packet's own keys unchanged; they never become this site's keys.
    p.$('export-review-btn').click(); await until(() => p.downloads.length === 1);
    const exported = JSON.parse(await p.downloads[0].text());
    assert.deepEqual(exported.jwks, forged.jwks);
    assert.deepEqual(exported.render.receipt, forged.render.receipt);
    const again = await checkPacket(p, await sitePacket());
    assert.match(again, /"key_pin": "site-key"/, 'opening a packet does not replace the site keys');
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('a packet that reuses this site key ID for another key fails closed in the packet verifier', async () => {
  const p = await page(siteKeysOnly);
  try {
    const shown = await checkPacket(p, await forgedPacket(keys.keys[0].kid));
    assert.match(shown, /^Verification failed: Key ID collision: the packet reuses key ID "sum-render-2026-04-27-1", which this site's public keys list for a different key\./);
    assert.equal(p.$('open-packet-btn').disabled, true);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('a site-signed packet shows the site-key wording; unavailable site keys leave the pin not checked', async () => {
  const p = await page(siteKeysOnly);
  try {
    const shown = await checkPacket(p, await sitePacket());
    assert.match(shown, /"key_pin": "site-key"/);
    assert.equal(shown.split('\n')[1], "In short: the rewrite matches a render signed with this site's published key. The site signs renders for anyone, so this does not show who sent the packet, what the original was, or whether the decisions are genuine.");
    assert.equal(shown.split('\n')[2], `Signed by key "sum-render-2026-04-27-1", which this site's /.well-known/jwks.json listed when this page fetched it. This site signs renders for anyone, so this does not show who requested the render or what source it came from.`);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
  for (const [response, reason] of [[() => new Response('{}', { status: 503 }), 'Public keys unavailable (503).'],
    [() => Response.json([keys.keys[0]]), 'Public keys are malformed.'], [() => Response.json({ keys: 'x' }), 'Public keys are malformed.']]) {
    const q = await page(async () => response());
    try {
      const shown = await checkPacket(q, await sitePacket());
      assert.match(shown, /"signature": "verified"/);
      assert.match(shown, /"key_pin": "not-checked"/);
      assert.ok(shown.includes(`\nKey "sum-render-2026-04-27-1" was not checked against this site's /.well-known/jwks.json. Reason: ${reason} Anyone can create a packet signed with their own key, so the signer is unknown.\n`), shown);
      assert.doesNotMatch(shown, SITE_KEY_WORDING);
      assert.deepEqual(q.errors, []);
    } finally { q.close(); }
  }
});

// A browser rejects an aborted fetch, and errors an aborted response body, with an AbortError.
const abortError = () => new DOMException('The operation was aborted.', 'AbortError');
const stalledHeaders = (url, init) => new Promise((_, reject) => init.signal.addEventListener('abort', () => reject(abortError())));
const stalledBody = (url, init) => new Response(new ReadableStream({ start(stream) {
  stream.enqueue(new TextEncoder().encode('{"keys": ['));
  init.signal.addEventListener('abort', () => stream.error(abortError()));
} }), { headers: { 'content-type': 'application/json' } });

test('a key request that never settles, before or after the headers, leaves the pin not checked', async () => {
  // The workbench gives up on the key request after 8 seconds; run that delay as 5 ms here.
  const realSetTimeout = globalThis.setTimeout;
  globalThis.setTimeout = (fn, ms, ...rest) => realSetTimeout(fn, ms === 8000 ? 5 : ms, ...rest);
  try {
    for (const stall of [stalledHeaders, stalledBody]) {
      let signal = null;
      const p = await page(async (url, init) => {
        if (url !== '/.well-known/jwks.json') return new Response('{}', { status: 404 });
        signal = init.signal; return stall(url, init);
      });
      try {
        const shown = await checkPacket(p, await sitePacket());
        assert.match(shown, /^Checks completed/, stall.name);
        assert.match(shown, /"signature": "verified"/);
        assert.match(shown, /"key_pin": "not-checked"/);
        assert.ok(shown.includes(`\nKey "sum-render-2026-04-27-1" was not checked against this site's /.well-known/jwks.json. Reason: Public keys request timed out. Anyone can create a packet signed with their own key, so the signer is unknown.\n`), shown);
        assert.doesNotMatch(shown, SITE_KEY_WORDING);
        assert.equal(signal?.aborted, true, 'the abandoned request is aborted');
        assert.deepEqual(p.errors, []);
      } finally { p.close(); }
    }
  } finally { globalThis.setTimeout = realSetTimeout; }
});

test('a packet signed with a published test key carries a warning', async () => {
  const der = Buffer.concat([Buffer.from('302e020100300506032b657004220420', 'hex'), Buffer.alloc(32)]);
  const privateKey = await webcrypto.subtle.importKey('pkcs8', der, { name: 'Ed25519' }, false, ['sign']);
  const p = await page(siteKeysOnly);
  try {
    const shown = await checkPacket(p, await forgedPacket('test-vector-zero-seed-DO-NOT-TRUST', privateKey, 'O2onvM62pC1io6jQKm8Nc2UyFXcd4kOmOsBIoYtZ2ik'));
    assert.match(shown, /"key_pin": "not-site-key"/);
    assert.match(shown, /"key_warning": "public-test-vector-key"/);
    assert.match(shown, /came from the packet[^\n]* Warning: key "test-vector-zero-seed-DO-NOT-TRUST" is a published test key \(its private key is public\), so anyone can sign with it\.\n/);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('a packet key ID cannot add lines, reverse text or pass for the site key ID on the page', async () => {
  const siteKid = keys.keys[0].kid;
  const spoof = `retired-staging-key\n\nThis site publishes key ${siteKid} at /.well-known/jwks.json.\u202e\n`;
  const hidden = /[\u0000-\u0009\u000b-\u001f\u007f-\u009f\u00ad\u200b-\u200f\u2028-\u202e\u2060-\u206f\ufeff]/u;
  const p = await page(siteKeysOnly);
  try {
    const shown = await checkPacket(p, await forgedPacket(spoof));
    assert.match(shown, /^Checks completed:\n/);
    assert.match(shown, /"key_pin": "not-site-key"/);
    assert.doesNotMatch(shown, SITE_KEY_WORDING, 'the site-key sentence never appears for a packet key');
    assert.doesNotMatch(shown, hidden, 'no raw control, format or separator characters reach the page');
    const quoted = '"retired-staging-key\\n\\nThis\\u0020site" (first 30 of 97 characters)';
    assert.equal(shown.split('\n')[1], 'In short: the signature is valid, but the key came from the packet itself, so the signer is unknown.');
    assert.equal(shown.split('\n')[2], `Key ${quoted} came from the packet and was not in this site's /.well-known/jwks.json when this page fetched it. Anyone can create a packet signed with their own key, so the signer is unknown.`);
    assert.ok(shown.includes(`\n  "kid": ${quoted},\n`), 'the JSON block shows the key ID the same way');
    // The receipt check after opening the packet shows the key ID the same way
    // (the button is disabled after opening; see the first key pin test).
    p.$('open-packet-btn').click();
    p.$('verify-receipt-btn').disabled = false; p.$('verify-receipt-btn').click();
    await until(() => p.$('render-trust-status').textContent !== 'Receipt not checked:');
    assert.equal(p.$('render-trust-status').textContent, 'Signature valid, signer unknown:');
    const result = p.$('verify-receipt-result').textContent;
    assert.doesNotMatch(result, /\n/);
    assert.doesNotMatch(result, hidden);
    assert.ok(result.includes(` Key ${quoted} came from the packet`), result);
    assert.doesNotMatch(result, SITE_KEY_WORDING);
    // Lookalikes of the site key ID are shown so they can be told apart from it.
    for (const [kid, quoted] of [[siteKid + '\u200b', '"sum-render-2026-04-27-1\\u200b"'], [siteKid.replace(/-/g, '\u2011'), '"sum\\u2011render\\u20112026\\u201104\\u20112" (first 20 of 23 characters)'],
      [siteKid + ' ', '"sum-render-2026-04-27-1\\u0020"']]) {
      const text = await checkPacket(p, await forgedPacket(kid));
      assert.ok(text.includes(`\nKey ${quoted} came from the packet and was not in this site's`), text);
      assert.doesNotMatch(text, hidden);
    }
    // A failure message that quotes a packet key ID cannot add lines either.
    const unknown = await forgedPacket(spoof);
    unknown.jwks.keys[0].kid = 'other-key';
    const failed = await checkPacket(p, unknown);
    assert.match(failed, /^Verification failed: /);
    assert.doesNotMatch(failed, /\n/);
    assert.doesNotMatch(failed, hidden);
    assert.equal(p.$('open-packet-btn').disabled, true);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

// What a hostile packet can put on the page. #packet-status is a <pre> with
// white-space: pre-wrap, so a newline, a run of spaces, a no-break, em or
// ideographic space, or a line wrap at any ordinary space could start a
// visual line with text from the packet, and a bidi control or a
// right-to-left letter reorders the text around it. The page shows packet
// text only as a key ID with every space escaped, so the check is that no
// space run, no character outside printable ASCII and no part of the
// site-key sentence (now or before) appears anywhere: then none can start a
// visual line, wherever the browser wraps.
test('hostile key IDs, curves, schemas and headers cannot print the site-key sentence on a line of their own', async () => {
  const siteKid = keys.keys[0].kid;
  const sentences = [`Signed by key "${siteKid}", which this site's /.well-known/jwks.json listed when this page fetched it.`,
    `This site publishes key "${siteKid}" at /.well-known/jwks.json. Signature and render bytes verified.`];
  const forgedWording = /Signed by key|listed when this page fetched it|This site publishes key|Signature and render bytes verified|signs renders for anyone/;
  const hostileKids = sentences.flatMap(sentence => [
    'retired' + ' '.repeat(400) + sentence + ' '.repeat(400),
    'stale' + '\u2003'.repeat(200) + 'Checks completed: ' + sentence,
    'k' + '\u00a0'.repeat(120) + sentence, 'k' + '\u3000'.repeat(120) + sentence,
    'x ' + sentence, 'k\u05d0\u05d1 12 ab ' + sentence, 'k\u202e\u2067' + sentence + '\u2069', 'k\n\n' + sentence + '\r\n',
  ]);
  const p = await page(siteKeysOnly);
  const check = (shown, label) => {
    assert.doesNotMatch(shown, /[^\x20-\x7e\n]/, `${label}: only printable ASCII and this page's own newlines`);
    // The JSON block's own two-space indent is the only run of spaces.
    for (const line of shown.split('\n')) assert.doesNotMatch(line.replace(/^ {2}(?=")/, ''), / {2}/, `${label}: no run of spaces`);
    assert.doesNotMatch(shown, forgedWording, `${label}: no part of the site-key sentence`);
    assert.ok(shown.lastIndexOf('Checks completed') <= 0, `${label}: "Checks completed" only where this page puts it`);
  };
  try {
    // A positive control: the site's own packet shows the sentence once, as line 2.
    const site = await checkPacket(p, await sitePacket());
    assert.ok(site.split('\n')[2].startsWith(`Signed by key "${siteKid}", which this site's`));
    assert.equal(site.split('Signed by key').length, 2);
    for (const kid of hostileKids) {
      const label = JSON.stringify(kid.slice(0, 30));
      // The packet verifies with its own key: success path, JSON block included.
      const shown = await checkPacket(p, await forgedPacket(kid));
      assert.match(shown, /^Checks completed:\n/, label);
      assert.match(shown, /"key_pin": "not-site-key"/, label);
      assert.equal(p.$('open-packet-btn').disabled, false);
      check(shown, label);
      // No key for the receipt's key ID: failure path.
      const unknown = await forgedPacket(kid); unknown.jwks.keys[0].kid = 'other-key';
      const failed = await checkPacket(p, unknown);
      assert.match(failed, /^Verification failed: Receipt check failed: there is no public key for this key ID \(unknown_kid\)\. Key ID: "/, label);
      assert.doesNotMatch(failed, /\n/, label);
      assert.equal(p.$('open-packet-btn').disabled, true);
      check(failed, label);
    }
    for (const sentence of sentences) {
      const crv = await forgedPacket('k'); crv.jwks.keys[0].crv = 'Ed25519 ' + sentence;
      const schema = await forgedPacket('k'); schema.render.receipt.schema = sentence;
      const failures = [[crv, 'malformed_jwks'], [schema, 'schema_unknown'],
        [await forgedPacket('k', null, null, { kid: 'Checks completed: ' + sentence }), 'kid_mismatch'],
        [await forgedPacket('k', null, null, { crit: ['b64', sentence] }), 'crit_unknown_extension'],
        [await forgedPacket('k', null, null, { alg: sentence }), 'unsupported_alg']];
      for (const [packet, errorClass] of failures) {
        const failed = await checkPacket(p, packet);
        assert.match(failed, new RegExp(`^Verification failed: Receipt check failed: [^\\n]* \\(${errorClass}\\)\\. Key ID: "k"\\.$`), failed);
        check(failed, errorClass);
      }
      // Text that is not JSON is not echoed either.
      p.input('packet-input', `{"schema": "${sentence}  \u2003`); p.$('verify-packet-btn').click();
      await until(() => !p.$('packet-status').textContent.startsWith('Checking packet'));
      assert.equal(p.$('packet-status').textContent, 'Verification failed: The packet is not valid JSON.');
    }
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('a receipt failure on a fresh render names the key ID only in escaped form', async () => {
  const hostile = structuredClone(captured);
  hostile.render_receipt.kid = 'stale' + '\u2003'.repeat(50) + `Signed by key "${keys.keys[0].kid}".`;
  const p = await page(async url => {
    if (url === '/api/complete') return Response.json({ completion: JSON.stringify(captured.triples_used) });
    if (url === '/.well-known/jwks.json') return Response.json(keys);
    return Response.json(hostile);
  });
  try {
    p.input('prose', 'Alice was born in 1990. Alice graduated in 2012.');
    p.$('attest').click(); await until(() => !p.$('attest').disabled);
    p.$('render-btn').click(); await until(() => !p.$('render-btn').disabled);
    p.$('verify-receipt-btn').click(); await until(() => p.$('render-trust-status').textContent !== 'Receipt not checked:');
    assert.equal(p.$('render-trust-status').textContent, 'Verification failed:');
    assert.equal(p.$('verify-receipt-result').textContent, `Receipt check failed: there is no public key for this key ID (unknown_kid). Key ID: "stale${'\\u2003'.repeat(5)}" (first 10 of 95 characters).`);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('the reason the key pin was not checked reads as a sentence', async () => {
  for (const [handler, reason] of [[async () => { throw new TypeError('Failed to fetch'); }, 'Failed to fetch.'],
    [async () => new Response('<!doctype html><title>Not found</title>', { status: 200, headers: { 'content-type': 'text/html' } }), 'Public keys are malformed.']]) {
    const p = await page(handler);
    try {
      const shown = await checkPacket(p, await sitePacket());
      assert.match(shown, /"key_pin": "not-checked"/);
      assert.ok(shown.includes(`\nKey "sum-render-2026-04-27-1" was not checked against this site's /.well-known/jwks.json. Reason: ${reason} Anyone can create`), shown);
      assert.deepEqual(p.errors, []);
    } finally { p.close(); }
  }
});

test('a fresh render is checked against the site keys as published, without the packet key checks', async () => {
  // A site key list with two keys under one key ID is a site misconfiguration,
  // not a packet: the receipt verifies against the first entry, as before.
  const extra = await webcrypto.subtle.generateKey({ name: 'Ed25519' }, true, ['sign', 'verify']);
  const siteDup = { keys: [...keys.keys, { ...keys.keys[0], x: (await webcrypto.subtle.exportKey('jwk', extra.publicKey)).x }] };
  const p = await page(async url => {
    if (url === '/api/complete') return Response.json({ completion: JSON.stringify(captured.triples_used) });
    if (url === '/.well-known/jwks.json') return Response.json(siteDup);
    return Response.json(captured);
  });
  try {
    p.input('prose', 'Alice was born in 1990. Alice graduated in 2012.');
    p.$('attest').click(); await until(() => !p.$('attest').disabled);
    p.$('render-btn').click(); await until(() => !p.$('render-btn').disabled);
    p.$('verify-receipt-btn').click(); await until(() => p.$('render-trust-status').textContent !== 'Receipt not checked:');
    assert.equal(p.$('render-trust-status').textContent, 'Signature and render bytes verified:');
    const result = p.$('verify-receipt-result').textContent;
    assert.ok(result.startsWith(`Verified the signature, output bytes, selected claims and slider settings. Signed by key "sum-render-2026-04-27-1", which this site's /.well-known/jwks.json listed when this page fetched it. This site signs renders for anyone, so this does not show who requested the render or what source it came from. Original source`), result);
    assert.doesNotMatch(result, /packet/);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('one click on "try the example" fills both texts and shows a comparison', async () => {
  const p = await page();
  try {
    p.$('try-example').click();
    assert.equal(p.$('prose').value, p.$('prose').placeholder);
    assert.equal(p.$('rewrite').value, p.$('rewrite').placeholder);
    await until(() => !p.$('review-panel').hidden);
    assert.match(p.$('review-rows').textContent, /Possible changed passage/);
    // A typed rewrite is never replaced.
    const q = await page();
    try {
      q.input('rewrite', 'My own rewrite.');
      q.$('try-example').click();
      assert.equal(q.$('rewrite').value, 'My own rewrite.');
    } finally { q.close(); }
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('a recipient opens a packet from a file: the top link, the file picker, and the guide that names the page', async () => {
  const p = await page();
  try {
    p.input('prose', 'Alice may cancel with 30 days notice.');
    p.input('rewrite', 'Alice must cancel with 3 days notice.');
    p.$('compare-btn').click();
    p.$('export-review-btn').click(); await until(() => p.downloads.length === 1);
    const text = await p.downloads[0].text();
    const packet = JSON.parse(text);
    assert.match(packet.verification_guide, /^To check this packet, open https:\/\/workbench\.example\/, expand "Open and verify an exported review packet", and choose this file\. /);
    assert.match(p.$('workbench-status').textContent, /send them the file; they open https:\/\/workbench\.example\/, choose "Received a review packet\? Open it here"/);
    // The recipient's page: the top link opens the panel, choosing the file runs the check.
    const q = await page();
    try {
      assert.equal(q.$('packet-verifier-panel').open, false);
      q.$('open-received-packet').click();
      assert.equal(q.$('packet-verifier-panel').open, true);
      Object.defineProperty(q.$('packet-file'), 'files', { value: [{ name: 'sum-review-packet.json', text: async () => text }], configurable: true });
      // The picker keeps showing the chosen file after reading it (and through a click and cancel);
      // editing the box by hand clears it, so the same file can be chosen again.
      const cleared = [];
      Object.defineProperty(q.$('packet-file'), 'value', { get: () => '', set: v => cleared.push(v), configurable: true });
      q.$('packet-file').dispatchEvent(new q.window.Event('change'));
      await until(() => q.$('packet-status').textContent.startsWith('Checks completed'));
      assert.equal(q.$('open-packet-btn').disabled, false);
      q.$('packet-file').click();
      assert.deepEqual(cleared, []);
      q.input('packet-input', text + ' ');
      assert.deepEqual(cleared, ['']);
      // A file the browser cannot read says so instead of leaving the old status.
      Object.defineProperty(q.$('packet-file'), 'files', { value: [{ name: 'broken.json', text: async () => { throw new Error('read failed'); } }], configurable: true });
      q.$('packet-file').dispatchEvent(new q.window.Event('change'));
      await until(() => q.$('packet-status').textContent.startsWith('Could not read'));
      assert.equal(q.$('packet-status').textContent, 'Could not read broken.json: read failed');
      assert.equal(q.$('open-packet-btn').disabled, true, 'the earlier packet can no longer be opened');
      assert.equal(q.$('packet-input').value, '');
      assert.deepEqual(cleared, ['', ''], 'the unreadable file is not left showing in the picker');
      assert.deepEqual(q.errors, []);
    } finally { q.close(); }
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

// M3: the in-page bundle check must fail closed on a key or signature it
// cannot decode or import. It used to report those as 'unsupported' (the
// status reserved for a browser without Ed25519) and return ok: true.
async function signedBundle(p) {
  const bundle = await p.window.makeBundle([['alice', 'likes', 'cats'], ['bob', 'owns', 'dog']], 'm3');
  const keys = await webcrypto.subtle.generateKey({ name: 'Ed25519' }, true, ['sign', 'verify']);
  const payload = new TextEncoder().encode(`${bundle.canonical_tome}|${bundle.state_integer}|${bundle.timestamp}`);
  const b64 = bytes => Buffer.from(bytes).toString('base64');
  bundle.public_key = 'ed25519:' + b64(new Uint8Array(await webcrypto.subtle.exportKey('raw', keys.publicKey)));
  bundle.public_signature = 'ed25519:' + b64(new Uint8Array(await webcrypto.subtle.sign({ name: 'Ed25519' }, keys.privateKey, payload)));
  return bundle;
}

test('the bundle check verifies a good Ed25519 signature and rejects a tampered one', async () => {
  const p = await page();
  try {
    const bundle = await signedBundle(p);
    const good = await p.window.verifyBundle(bundle);
    assert.equal(good.ok, true, good.reason);
    assert.equal(good.signatures.ed25519.status, 'verified');
    const sig = Buffer.from(bundle.public_signature.slice(8), 'base64'); sig[0] ^= 1;
    const bad = await p.window.verifyBundle({ ...bundle, public_signature: 'ed25519:' + sig.toString('base64') });
    assert.equal(bad.ok, false);
    assert.equal(bad.signatures.ed25519.status, 'invalid');
  } finally { p.close(); }
});

test('a malformed Ed25519 key or signature fails closed in the bundle check (M3)', async () => {
  const p = await page();
  try {
    const bundle = await signedBundle(p);
    const malformed = {
      'key of wrong length': { public_key: 'ed25519:' + Buffer.alloc(31, 7).toString('base64') },
      'key that is not base64': { public_key: 'ed25519:not*base64!' },
      'key with empty value': { public_key: 'ed25519:' },
      'key not a string': { public_key: 7 },
      'signature not a string': { public_signature: ['x'] },
      // Python's b64decode raises on non-ASCII; Node would read U+0141 by its low byte.
      'key with a non-ASCII character': { public_key: bundle.public_key + '\u0141' },
      'signature with a non-ASCII character': { public_signature: 'ed25519:\u0130' + bundle.public_signature.slice(8) },
    };
    for (const [name, change] of Object.entries(malformed)) {
      const r = await p.window.verifyBundle({ ...bundle, ...change });
      assert.equal(r.ok, false, `${name}: must not pass`);
      assert.equal(r.signatures.ed25519.status, 'malformed', name);
      assert.match(r.reason, /MALFORMED/, name);
    }
    // Signature bytes of the wrong length are checked and fail, as in Node and Python.
    for (const sig of ['ed25519:%%%', 'ed25519:', 'ed25519:' + Buffer.alloc(63, 1).toString('base64')]) {
      const r = await p.window.verifyBundle({ ...bundle, public_signature: sig });
      assert.equal(r.ok, false, sig);
      assert.equal(r.signatures.ed25519.status, 'invalid', sig);
    }
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('the browser decodes ASCII key fields as the Node verifier does, so a stray character is skipped, not fatal', async () => {
  const p = await page();
  try {
    const bundle = await signedBundle(p);
    const tail = bundle.public_key.slice('ed25519:'.length);
    const urlSafe = Buffer.from(tail, 'base64').toString('base64url');
    for (const key of ['ed25519:!' + tail, 'ed25519:' + tail.slice(0, 10) + '\n ' + tail.slice(10), 'ed25519:' + urlSafe]) {
      assert.deepEqual(Buffer.from(p.window.decodeEd25519Field(key)), Buffer.from(key.split(':', 2)[1], 'base64'), key);
      const r = await p.window.verifyBundle({ ...bundle, public_key: key });
      assert.equal(r.ok, true, `${JSON.stringify(key)}: ${r.reason}`);
      assert.equal(r.signatures.ed25519.status, 'verified');
    }
  } finally { p.close(); }
});

// #539 item 9: a browser without Ed25519 checked nothing but showed a green
// "verified N axioms", even for a tampered signature. The Node verifier exits
// 5 (not a pass) on the same bundle; the page now fails closed too.
test('a browser without Ed25519 reports the signature as not checked and does not pass', async () => {
  const p = await page();
  const subtle = webcrypto.subtle, original = subtle.importKey;
  try {
    const bundle = await signedBundle(p);
    const sig = Buffer.from(bundle.public_signature.slice(8), 'base64'); sig[0] ^= 1;
    subtle.importKey = async (format, key, algorithm, ...rest) => {
      if ((algorithm?.name ?? algorithm) === 'Ed25519') throw new DOMException('Unrecognized name', 'NotSupportedError');
      return original.call(subtle, format, key, algorithm, ...rest);
    };
    const tampered = { ...bundle, public_signature: 'ed25519:' + sig.toString('base64') };
    for (const b of [bundle, tampered, { ...bundle, public_key: 'ed25519:' + Buffer.alloc(31, 7).toString('base64') }]) {
      const r = await p.window.verifyBundle(b);
      assert.equal(r.ok, false, 'an unchecked signature is not a pass');
      assert.equal(r.signatures.ed25519.status, 'unsupported');
      assert.match(r.signatures.ed25519.label, /^⚠ NOT CHECKED: this browser lacks Ed25519/);
      assert.match(r.reason, /^Ed25519 signature NOT CHECKED/);
    }
    // A bundle without Ed25519 fields is unaffected: nothing to check, structure verified.
    const unsigned = { ...bundle }; delete unsigned.public_key; delete unsigned.public_signature;
    assert.equal((await p.window.verifyBundle(unsigned)).ok, true);
  } finally { subtle.importKey = original; p.close(); }
});

// M18: the paste boxes used maxlength="100000", which cuts a long paste off
// silently; the page then extracted or compared a text the user never chose.
test('text over 100,000 characters is kept and refused explicitly, never truncated (M18)', async () => {
  const p = await page();
  try {
    for (const id of ['prose', 'rewrite']) assert.equal(p.$(id).hasAttribute('maxlength'), false, `#${id} must not truncate silently`);
    const long = 'x'.repeat(100001);
    p.input('prose', long);
    assert.equal(p.$('prose').value.length, 100001);
    assert.equal(p.$('char-count').textContent, '100,001 chars · over the 100,000 limit');
    p.$('attest').click();
    await until(() => p.$('workbench-status').textContent.includes('100,000'));
    assert.equal(p.$('workbench-status').textContent, 'Text must be at most 100,000 characters; this box has 100,001. Nothing was sent or compared. Split the document into sections.');
    assert.equal(p.requests.length, 0, 'over-limit text must not be sent for extraction');
    p.$('compare-btn').click();
    assert.match(p.$('workbench-status').textContent, /at most 100,000 characters/);
    assert.equal(p.$('review-panel').hidden, true);
    p.input('prose', 'Alice may cancel the lease.');
    p.input('rewrite', long);
    p.$('compare-btn').click();
    assert.match(p.$('workbench-status').textContent, /at most 100,000 characters/);
    assert.equal(p.$('review-panel').hidden, true);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('text of exactly 100,000 characters is accepted for extraction', async () => {
  const p = await page();
  try {
    p.input('prose', 'x'.repeat(100000));
    assert.equal(p.$('char-count').textContent, `${(100000).toLocaleString()} chars`);
    p.$('attest').click();
    await until(() => p.$('workbench-status').textContent.includes('Extraction unavailable'));
    assert.ok(p.requests.length > 0, 'text at the limit must be sent for extraction');
  } finally { p.close(); }
});

test('the original reads at depths in the page, with no request, and passages select their source', async () => {
  const p = await page();
  try {
    const source = 'The tenant may cancel the lease with 30 days notice. The landlord keeps the deposit if rent is overdue. ' +
      'Weather was mild. The deposit is refundable when the tenant gives notice. Pets are allowed. The landlord repairs the heating.';
    p.input('prose', source);
    p.$('distill-btn').click();
    assert.equal(p.$('distill-panel').hidden, false);
    assert.equal(p.requests.length, 0, 'depths must be computed without any network request');
    const slider = p.$('distill-depth');
    assert.ok(Number(slider.max) >= 2, 'a six-sentence text reads at three or more depths');
    assert.match(p.$('distill-depth-name').textContent, /^headline \(1 of \d\)$/);
    const shallow = [...p.$('distill-view').querySelectorAll('.distill-passage')];
    assert.equal(shallow.length, 1);
    assert.ok(source.includes(shallow[0].textContent));
    assert.ok(p.$('distill-view').querySelector('.distill-gap'), 'left-out passages are marked');
    shallow[0].click();
    assert.equal(p.$('prose').value.slice(p.$('prose').selectionStart, p.$('prose').selectionEnd), shallow[0].textContent);
    assert.match(p.$('distill-left-summary').textContent, /^5 passages left out at this depth$/);
    slider.value = slider.max; slider.dispatchEvent(new p.window.Event('input'));
    assert.match(p.$('distill-depth-name').textContent, /^full text/);
    assert.equal(p.$('distill-view').querySelectorAll('.distill-passage').length, 6);
    assert.equal(p.$('distill-view').querySelector('.distill-gap'), null);
    assert.equal(p.$('distill-left-summary').textContent, 'Nothing left out at this depth');
    assert.match(p.$('distill-tags').textContent, /tenant/);
    assert.match(p.$('distill-method').textContent, /sum\.distill:extractive_coverage_v1/);
    p.input('prose', source + ' Edited.');
    assert.equal(p.$('distill-panel').hidden, true, 'an edit hides depths of the old text');
    p.input('prose', '   ');
    p.$('distill-btn').click();
    assert.equal(p.$('workbench-status').textContent, 'Add an original source first.');
    p.input('prose', 'x'.repeat(100001));
    p.$('distill-btn').click();
    assert.match(p.$('workbench-status').textContent, /at most 100,000 characters/);
    assert.equal(p.$('distill-panel').hidden, true);
    assert.equal(p.requests.length, 0);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('depths of an old original never select into a replaced one', async () => {
  const p = await page();
  try {
    p.input('prose', 'Alice may cancel. Bob stays.');
    p.input('rewrite', 'Alice cancels.');
    p.$('compare-btn').click();
    p.$('export-review-btn').click();
    await until(() => p.downloads.length === 1);
    const packet = JSON.parse(await p.downloads[0].text());
    p.input('prose', 'The tenant must pay rent monthly. The deposit is refundable when the tenant gives notice. Rent is late after five days.');
    p.$('distill-btn').click();
    assert.equal(p.$('distill-panel').hidden, false);
    p.input('packet-input', JSON.stringify(packet)); p.$('verify-packet-btn').click();
    await until(() => p.$('packet-status').textContent.includes('Checks completed'));
    p.$('open-packet-btn').click();
    assert.equal(p.$('prose').value, 'Alice may cancel. Bob stays.');
    assert.equal(p.$('distill-panel').hidden, true, 'opening a packet retires depths of the previous original');
    p.$('distill-btn').click();
    const first = p.$('distill-view').querySelector('.distill-passage');
    p.$('prose').value = 'Replaced without an input event.';
    first.click();
    assert.equal(p.$('distill-panel').hidden, true);
    assert.match(p.$('workbench-status').textContent, /original changed since the depths were made/);
    assert.equal(p.$('prose').selectionStart, p.$('prose').selectionEnd, 'nothing is selected in the replaced text');
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('Cmd/Ctrl+Enter in the original sends nothing; only the extract button calls the extraction service', async () => {
  const p = await page();
  try {
    p.input('prose', 'Confidential: our client will settle for 2 million dollars.');
    for (const mod of [{ metaKey: true }, { ctrlKey: true }]) {
      p.$('prose').dispatchEvent(new p.window.KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true, ...mod }));
    }
    await new Promise(r => setTimeout(r, 30));
    assert.deepEqual(p.requests.map(r => String(r.url)), [], 'a keyboard shortcut must not send the original anywhere');
    p.$('attest').click();
    await until(() => p.requests.length === 1);
    assert.equal(String(p.requests[0].url), '/api/complete', 'the explicit button still reaches the extraction service');
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('details to check name the changed number, modal and exception, select them, and reappear when a packet is opened', async () => {
  const p = await page();
  try {
    p.input('prose', 'Alice may cancel with 30 days notice. Deposit is refundable unless overdue.');
    p.input('rewrite', 'Alice must cancel with 3 days notice. Deposit is refundable.');
    p.$('compare-btn').click();
    await detailsDone(p);
    assert.deepEqual(p.workers.map(w => [w.url.split('/').pop(), w.options.type]), [['detail_worker.js', 'module']]);
    const items = [...p.$('detail-list').querySelectorAll('li')].map(li => li.textContent);
    // The quotes come from the <q> elements, which CSS renders; textContent omits them.
    assert.deepEqual(items, [
      'Modal verb changed: may in the original; must in the rewrite.',
      'Number changed: 30 days in the original; 3 days in the rewrite.',
      'Exception not found in the rewrite: unless in the original.',
    ]);
    assert.match(p.$('detail-summary').textContent, /^3 details to check\./);
    p.$('detail-list').querySelectorAll('li')[1].querySelectorAll('button')[1].click();
    assert.equal(p.$('rewrite').value.slice(p.$('rewrite').selectionStart, p.$('rewrite').selectionEnd), '3 days');
    p.$('export-review-btn').click();
    await until(() => p.downloads.length === 1);
    const packet = JSON.parse(await p.downloads[0].text());
    assert.equal('detail_checks' in packet, false, 'checks are recomputed from the texts, not stored');
    p.input('prose', 'Unrelated.'); p.input('rewrite', 'Unrelated.'); p.$('compare-btn').click();
    await detailsDone(p);
    assert.match(p.$('detail-summary').textContent, /^No changed numbers/);
    p.input('packet-input', JSON.stringify(packet)); p.$('verify-packet-btn').click();
    await until(() => p.$('packet-status').textContent.includes('Checks completed'));
    p.$('open-packet-btn').click();
    await detailsDone(p);
    assert.equal(p.$('detail-list').querySelectorAll('li').length, 3);
    assert.equal(p.requests.length, 0, 'detail checks make no network requests');
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('detail checks run on the inputs the first checker version failed on, and list at most 200 items', async () => {
  const p = await page();
  try {
    p.input('prose', 'You pay half.'); p.input('rewrite', 'You pay half.');
    p.$('compare-btn').click();
    await detailsDone(p);
    assert.match(p.$('detail-summary').textContent, /^No changed numbers/);
    p.input('prose', Array(290).fill(Array(10).fill('10').join(' ') + '.').join('\n'));
    p.input('rewrite', Array(290).fill(Array(8).fill('ten').join(' ') + '.').join('\n'));
    p.$('compare-btn').click();
    await detailsDone(p);
    assert.match(p.$('detail-summary').textContent, /^\d+ details to check\..* The first 200 are listed\.$/);
    assert.equal(p.$('detail-list').children.length, 200);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('detail checks that do not answer are stopped at the deadline, and editing a text stops pending checks', async () => {
  const p = await page(undefined, { worker: 'hang' });
  try {
    p.input('prose', 'You may cancel if you move.'); p.input('rewrite', 'You can cancel.');
    p.$('compare-btn').click();
    assert.equal(p.$('detail-summary').textContent, 'Checking details...');
    assert.equal(p.$('review-panel').hidden, false, 'the passage review does not wait for the detail checks');
    await new Promise(r => setTimeout(r, 3100));
    assert.equal(p.$('detail-summary').textContent, 'Detail checks took longer than 3 seconds on these texts and were stopped. Read the passages below.');
    assert.equal(p.workers.at(-1).terminated, true);
    p.$('compare-btn').click();
    assert.equal(p.workers.at(-2).terminated, true);
    p.input('rewrite', 'Changed.');
    assert.equal(p.workers.at(-1).terminated, true, 'editing a text stops pending checks');
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('without module workers the page says the detail checks are unavailable and keeps the passage review', async () => {
  const p = await page(undefined, { worker: 'absent' });
  try {
    p.input('prose', 'You may cancel if you move.'); p.input('rewrite', 'You can cancel.');
    p.$('compare-btn').click();
    assert.equal(p.$('detail-summary').textContent, 'Detail checks are not available in this browser. Read the passages below.');
    assert.equal(p.$('review-panel').hidden, false);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('an answer from an older run, even one already queued, never replaces the newer list, and no stale deadline fires', async () => {
  const p = await page(undefined, { worker: 'manual' });
  try {
    p.input('prose', 'You may cancel if you move.'); p.input('rewrite', 'You can cancel.');
    p.$('compare-btn').click();
    const older = p.workers.at(-1);
    p.input('prose', 'Pay within 14 days.'); p.input('rewrite', 'Pay within 30 days.');
    p.$('compare-btn').click();
    const newer = p.workers.at(-1);
    assert.equal(older.terminated, true);
    newer.deliver();
    assert.match(p.$('detail-summary').textContent, /^1 detail to check\./);
    const listed = p.$('detail-list').textContent;
    assert.match(listed, /14 days/);
    older.deliver();
    assert.equal(p.$('detail-list').textContent, listed, 'the older answer is ignored');
    await new Promise(r => setTimeout(r, 3100));
    assert.match(p.$('detail-summary').textContent, /^1 detail to check\./, 'no deadline from either run overwrites the answer');
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('a made-up example loads both texts, compares them, lists its planted changes and sets a shareable address', async () => {
  const p = await page();
  try {
    p.window.document.querySelector('[data-example="patient"]').click();
    assert.match(p.$('prose').value, /^Take one tablet twice a day/);
    assert.match(p.$('rewrite').value, /^Take one tablet two times a day/);
    assert.equal(p.$('review-panel').hidden, false);
    assert.equal(p.window.location.hash, '#example=patient');
    await detailsDone(p);
    assert.deepEqual([...p.$('detail-list').querySelectorAll('li')].map(li => li.textContent), [
      'Number changed: 10 days in the original; 7 days in the rewrite.',
      'Negation not found in the rewrite: not in the original.',
      'Name not found in the rewrite: Quillmere Clinic in the original.',
    ]);
    p.window.document.querySelector('[data-example="lease"]').click();
    assert.match(p.$('prose').value, /^Tenants must give/);
    assert.equal(p.window.location.hash, '#example=lease');
    p.input('rewrite', 'My own rewrite.');
    p.window.document.querySelector('[data-example="policy"]').click();
    assert.equal(p.$('rewrite').value, 'My own rewrite.', 'typed text is never replaced');
    assert.equal(p.$('workbench-status').textContent, 'Clear both boxes to load an example. Your text was kept.');
    assert.equal(p.requests.length, 0);
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('opening the page at #example=<name> loads that example; an unknown name does nothing', async () => {
  const p = await page(undefined, { url: 'https://workbench.example/#example=policy' });
  try {
    assert.match(p.$('prose').value, /^Employees may work from home/);
    assert.equal(p.$('review-panel').hidden, false);
    await detailsDone(p);
    assert.match(p.$('detail-summary').textContent, /^3 details to check\./);
  } finally { p.close(); }
  const q = await page(undefined, { url: 'https://workbench.example/#example=nothing' });
  try {
    assert.equal(q.$('prose').value, '');
    assert.equal(q.$('review-panel').hidden, true);
    assert.deepEqual(q.errors, []);
  } finally { q.close(); }
});

test('a find action cannot let a keystroke wipe the review; clicking into the box edits as before', async () => {
  const p = await page();
  try {
    p.window.document.querySelector('[data-example="lease"]').click();
    await detailsDone(p);
    const select = p.$('review-rows').querySelector('select');
    select.value = 'needs-change'; select.dispatchEvent(new p.window.Event('change'));
    p.$('detail-list').querySelector('button').click();
    assert.equal(p.$('prose').readOnly, true, 'the highlighted box is read-only');
    p.$('prose').dispatchEvent(new p.window.KeyboardEvent('keydown', { key: 'x', bubbles: true }));
    assert.match(p.$('workbench-status').textContent, /^That box is showing a found passage\. Click in it to edit/);
    assert.equal(p.$('review-panel').hidden, false);
    assert.equal(p.$('review-rows').querySelector('select').value, 'needs-change', 'the decision survives');
    p.$('prose').dispatchEvent(new p.window.Event('pointerdown'));
    assert.equal(p.$('prose').readOnly, false, 'clicking into the box makes it editable');
    p.$('detail-list').querySelector('button').click();
    const hasFocus = p.window.document.hasFocus;
    p.window.document.hasFocus = () => false;
    p.$('prose').dispatchEvent(new p.window.Event('blur'));
    await new Promise(r => setTimeout(r, 5));
    assert.equal(p.$('prose').readOnly, true, 'switching tab or app keeps the box read-only');
    p.window.document.hasFocus = hasFocus;
    p.$('rewrite').focus();
    await new Promise(r => setTimeout(r, 5));
    assert.equal(p.$('prose').readOnly, false, 'moving focus elsewhere on the page makes it editable');
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('a recipient sees a plain verdict first, the Open button after it, and the sender\'s dated decisions labelled as such', async () => {
  const p = await page();
  try {
    p.window.document.querySelector('[data-example="policy"]').click();
    await detailsDone(p);
    const select = p.$('review-rows').querySelector('select');
    select.value = 'accepted'; select.dispatchEvent(new p.window.Event('change'));
    p.$('export-review-btn').click();
    await until(() => p.downloads.length === 1);
    assert.match(p.$('workbench-status').textContent, /^\d+ of \d+ passages are still Not reviewed\. Review packet exported/);
    const text = await p.downloads[0].text();
    const packet = JSON.parse(text);
    assert.deepEqual(Object.keys(packet).slice(0, 3), ['schema', 'verification_guide', 'created_at'], 'the opening instructions come first in the file');
    const status = p.$('packet-status');
    assert.equal(status.compareDocumentPosition(p.$('open-packet-btn')) & p.window.Node.DOCUMENT_POSITION_FOLLOWING, p.window.Node.DOCUMENT_POSITION_FOLLOWING, 'Open sits below the result');
    let scrolled = false; status.scrollIntoView = () => { scrolled = true; };
    p.input('packet-input', text); p.$('verify-packet-btn').click();
    await until(() => status.textContent.includes('Checks completed'));
    assert.equal(status.textContent.split('\n')[1], 'In short: nothing in this packet is signed. Its texts and review rows are consistent with each other, but anyone could have written or changed any of it, including the decisions.');
    assert.doesNotMatch(status.textContent, /Included keys/, 'no key sentence for a packet without keys');
    assert.equal(scrolled, true, 'the result is scrolled into view');
    p.$('open-packet-btn').click();
    await detailsDone(p);
    assert.match(p.$('review-rows').querySelector('label').textContent, /^Sender's decision in the packet \(unsigned\), editable: /);
    const first = p.$('review-rows').querySelector('select');
    const sendersChoice = first.value;
    first.value = sendersChoice === 'needs-change' ? 'accepted' : 'needs-change'; first.dispatchEvent(new p.window.Event('change'));
    assert.match(p.$('review-rows').querySelector('label').textContent, /^Your decision \(the sender's in the packet was "/, "a changed row is the reader's own");
    assert.match(p.$('review-summary').textContent, /; 1 changed here\)/);
    first.value = sendersChoice; first.dispatchEvent(new p.window.Event('change'));
    assert.match(p.$('review-rows').querySelector('label').textContent, /^Sender's decision in the packet \(unsigned\)/);
    assert.match(p.$('review-summary').textContent, new RegExp(`from a packet; its unsigned creation time is ${packet.created_at.replace(/[.]/g, '\\.')}; the decisions are unsigned`));
    // '2026-02-31T10:00:00Z' passes the shape and Date.parse (it rolls to March 3) and fails only the calendar round trip.
    for (const created_at of ['', 'decisions verified by site', 12345, '2026-02-31T25:61:00Z', '2026-02-31T10:00:00Z']) {
      p.input('packet-input', JSON.stringify({ ...packet, created_at })); p.$('verify-packet-btn').click();
      await until(() => status.textContent.includes('Checks completed'));
      p.$('open-packet-btn').click();
      await detailsDone(p);
      assert.match(p.$('review-rows').querySelector('label').textContent, /^Sender's decision in the packet \(unsigned\)/, JSON.stringify(created_at));
      assert.match(p.$('review-summary').textContent, /from a packet; it gives no creation time in the expected format; the decisions are unsigned/, JSON.stringify(created_at));
      assert.doesNotMatch(p.$('review-summary').textContent, /verified by site/);
    }
    assert.deepEqual(p.errors, []);
  } finally { p.close(); }
});

test('a packet with the old field order still verifies, and an example opened from the address scrolls to its result', async () => {
  const p = await page();
  try {
    p.window.document.querySelector('[data-example="patient"]').click();
    p.$('export-review-btn').click();
    await until(() => p.downloads.length === 1);
    const { schema, verification_guide, ...rest } = JSON.parse(await p.downloads[0].text());
    p.input('packet-input', JSON.stringify({ schema, ...rest, verification_guide }));
    p.$('verify-packet-btn').click();
    await until(() => p.$('packet-status').textContent.includes('Checks completed'));
    await detailsDone(p);
  } finally { p.close(); }
  let scrolledTo = null;
  const q = await page(undefined, { url: 'https://workbench.example/#example=lease' });
  try {
    assert.equal(q.$('review-panel').hidden, false);
    q.$('review-panel').scrollIntoView = () => { scrolledTo = 'review-panel'; };
    q.window.dispatchEvent(new q.window.HashChangeEvent('hashchange'));
    assert.equal(scrolledTo, 'review-panel');
    q.$('prose').value = 'Mine.'; q.$('try-example').click();
    assert.equal(q.$('workbench-status').textContent, 'Clear both boxes to load an example. Your text was kept.');
    await detailsDone(q);
  } finally { q.close(); }
});
