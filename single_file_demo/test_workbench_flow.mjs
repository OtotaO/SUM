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

async function page(handler = async () => new Response('{}', { status: 503 })) {
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
      Object.defineProperty(window, 'crypto', { value: webcrypto });
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
    assert.equal(shown.split('\n')[1], `Signed by key "sum-render-2026-04-27-1", which this site's /.well-known/jwks.json listed when this page fetched it. This site signs renders for anyone, so this does not show who requested the render or what source it came from.`);
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
    assert.equal(shown.split('\n')[1], `Key ${quoted} came from the packet and was not in this site's /.well-known/jwks.json when this page fetched it. Anyone can create a packet signed with their own key, so the signer is unknown.`);
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
    assert.ok(site.split('\n')[1].startsWith(`Signed by key "${siteKid}", which this site's`));
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
