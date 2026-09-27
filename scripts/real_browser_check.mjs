#!/usr/bin/env node
// Real-browser check of the review page (single_file_demo/). NOT run in CI:
// it needs a local Chromium. CI covers the same behaviour in jsdom
// (single_file_demo/npm test); this script is for a person, before a deploy,
// to see the page work in a real engine under the real CSP.
//
//   CHROME=/path/to/chrome-or-headless-shell node scripts/real_browser_check.mjs [--shots DIR]
//
// It serves single_file_demo/ on a local port with the Content-Security-Policy
// from single_file_demo/_headers, drives the browser over the DevTools
// protocol (Node 22 has a global WebSocket), and prints PASS/FAIL per check.
// Exit code 1 if any check fails. Checks: first view, both example buttons,
// decisions, filters, hover, Read as views (each equal to the exact text),
// span jumps (read-only, Escape back), the edit guard, Compare again keeping
// decisions, export, packet check and open, the meaning-receipt sample, the
// bundle Ed25519 check on malformed keys, timings on 100,000-character input,
// no horizontal scroll at 390 and 360, and that every sentence the page
// prints for the adversarial pairs is one the independent oracle accepts.
import http from 'node:http';
import { spawn } from 'node:child_process';
import { readFileSync, existsSync, mkdtempSync, rmSync, writeFileSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, extname, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../single_file_demo/', import.meta.url));
const shotsAt = process.argv.indexOf('--shots');
const SHOTS = shotsAt > 0 ? process.argv[shotsAt + 1] : null;
if (SHOTS) mkdirSync(SHOTS, { recursive: true });
const CHROME = process.env.CHROME || [
  `${process.env.HOME}/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell`,
  '/usr/bin/chromium', '/usr/bin/google-chrome',
].find(p => existsSync(p));
if (!CHROME) { console.error('Set CHROME to a Chromium or headless-shell binary.'); process.exit(2); }
const { check, ADVERSARIAL } = await import(new URL('../single_file_demo/evidence_oracle.mjs', import.meta.url));
const E = await import(new URL('../single_file_demo/change_evidence.js', import.meta.url));
const { compareTexts } = await import(new URL('../single_file_demo/review_packet.js', import.meta.url));
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ---------------------------------------------------------------- static server with the page's CSP
const csp = /Content-Security-Policy:\s*(.+)/.exec(readFileSync(join(ROOT, '_headers'), 'utf8'))[1].trim();
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm', '.css': 'text/css' };
const server = http.createServer((req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^[/\\]+/, '') || 'index.html';
  const file = join(ROOT, path);
  if (!file.startsWith(ROOT) || !existsSync(file)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'Content-Type': TYPES[extname(file)] || 'application/octet-stream', 'Content-Security-Policy': csp, 'Cache-Control': 'no-store' });
  res.end(readFileSync(file));
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const URL_ROOT = `http://127.0.0.1:${server.address().port}/`;

// ---------------------------------------------------------------- DevTools driver
const profile = mkdtempSync(join(tmpdir(), 'sum-check-'));
const port = 9400 + Math.floor(Math.random() * 400);
const chrome = spawn(CHROME, ['--no-sandbox', '--hide-scrollbars', `--user-data-dir=${profile}`, `--remote-debugging-port=${port}`, 'about:blank'], { stdio: 'ignore' });
let targets = [];
for (let i = 0; i < 80 && !targets.length; i++) { try { targets = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).filter(t => t.type === 'page'); } catch {} await sleep(150); }
const ws = new WebSocket(targets[0].webSocketDebuggerUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
let seq = 0; const pending = new Map(); const problems = [];
ws.onmessage = ev => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { const { res, rej } = pending.get(m.id); pending.delete(m.id); m.error ? rej(new Error(JSON.stringify(m.error))) : res(m.result); return; }
  if (m.method === 'Runtime.exceptionThrown') problems.push('exception: ' + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text));
  if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error' && !/favicon/.test(m.params.entry.url || '')) problems.push('console: ' + m.params.entry.text);
  if (m.method === 'Network.requestWillBeSent' && new URL(m.params.request.url).pathname.startsWith('/api/')) problems.push('network request to ' + m.params.request.url);
};
const send = (method, params = {}) => new Promise((res, rej) => { const id = ++seq; pending.set(id, { res, rej }); ws.send(JSON.stringify({ id, method, params })); });
const run = async expr => { const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text); return r.result.value; };
await send('Page.enable'); await send('Runtime.enable'); await send('Log.enable'); await send('Network.enable');
const size = async (w, h, dark = false) => {
  await send('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile: w < 700 });
  await send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: dark ? 'dark' : 'light' }, { name: 'prefers-reduced-motion', value: 'reduce' }] });
};
const load = async () => { await send('Page.navigate', { url: URL_ROOT }); for (let i = 0; i < 60; i++) { await sleep(100); if (await run(`document.getElementById('review-panel') && !document.getElementById('review-panel').hidden`).catch(() => false)) return; } };
const center = sel => run(`(()=>{const e=document.querySelector(${JSON.stringify(sel)}); e.scrollIntoView({block:'center',behavior:'instant'}); const r=e.getBoundingClientRect(); return [r.left+r.width/2, r.top+r.height/2]})()`);
const click = async sel => { const [x, y] = await center(sel); for (const type of ['mouseMoved', 'mousePressed', 'mouseReleased']) await send('Input.dispatchMouseEvent', { type, x, y, button: 'left', clickCount: 1 }); await sleep(80); };
const hover = async sel => { const [x, y] = await center(sel); await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y }); await sleep(80); };
const key = async (k, code, keyCode, text) => { await send('Input.dispatchKeyEvent', { type: 'keyDown', key: k, code, windowsVirtualKeyCode: keyCode, text }); await send('Input.dispatchKeyEvent', { type: 'keyUp', key: k, code, windowsVirtualKeyCode: keyCode }); await sleep(60); };
const setBoxes = (a, b) => run(`(()=>{for (const [id, v] of [['prose', ${JSON.stringify(a)}], ['rewrite', ${JSON.stringify(b)}]]) { const t = document.getElementById(id); t.value = v; t.dispatchEvent(new Event('input', { bubbles: true })); } document.getElementById('edit-guard').hidden || document.getElementById('guard-yes').click(); })()`);
const shot = async name => { if (!SHOTS) return; const r = await send('Page.captureScreenshot', { format: 'png' }); writeFileSync(join(SHOTS, name), Buffer.from(r.data, 'base64')); };
// What a view shows: visible text of a blackline, without screen-reader-only text, letters or separators.
const VIEW = `(el => { const walk = n => { if (n.nodeType === 3) return n.data; if (n.nodeType !== 1) return ''; const cs = getComputedStyle(n);
  if (cs.display === 'none' || parseFloat(cs.fontSize) === 0 || n.classList.contains('vh') || n.classList.contains('sep') || n.classList.contains('block-label') || n.tagName === 'SUP') return '';
  return [...n.childNodes].map(walk).join(''); }; return walk(el); })`;

const results = [];
const ok = (name, cond, detail = '') => results.push(`${cond ? 'PASS' : 'FAIL'} ${name}${detail && !cond ? ' :: ' + detail : ''}`);

try {
  await size(1440, 1000);
  await load();
  ok('first view shows the compared lease example', await run(`document.getElementById('review-heading').textContent`) === '2 passages compared, 4 differences noted.');
  ok('first view reports the comparison', await run(`document.getElementById('workbench-status').textContent`) === 'Compared in this browser · 2 passages · 4 differences noted');
  ok('nothing is highlighted at rest', await run(`document.querySelectorAll('.hl').length`) === 0);
  await shot('page_1440_light.png');

  await click('[data-example="refund"]');
  ok('Refund policy fills both boxes and compares', await run(`document.getElementById('rewrite').value.startsWith('You can return') && document.getElementById('review-heading').textContent === '7 passages compared, 17 differences noted.'`));
  await click('#try-example');
  ok('Lease clause fills both boxes and compares', await run(`document.getElementById('rewrite').value === 'Alice can cancel the lease. The deposit is refundable.'`));

  await click('label[for="d-source-s1-needs-change"]');
  await click('label[for="d-source-s2-accepted"]');
  ok('decisions record and show', await run(`document.getElementById('review-record-count').textContent === '2 of 2' && getComputedStyle(document.querySelector('#p1 .initial'), '::after').content === '"‸"'`));
  ok('the checked stamp keeps its accessible name', await run(`(() => { const l = document.querySelector('label[for="d-source-s2-accepted"]'); return getComputedStyle(l.querySelector('.off')).textTransform !== 'uppercase' && getComputedStyle(l.querySelector('.on')).textTransform === 'uppercase'; })()`));

  await click('#ledger-list .filter[data-state="a-only"]');
  ok('a filter hides other notes', await run(`getComputedStyle(document.getElementById('n1a')).display === 'none' && getComputedStyle(document.getElementById('n1b')).display !== 'none'`));
  await hover('#n1b');
  ok('hovering a note outlines only its marks', await run(`[...document.querySelectorAll('.hl')].every(n => n.dataset.n === '1b') && document.querySelector('del[data-n="1b"]').classList.contains('hl')`));
  await shot('decided_filtered_1440_light.png');
  await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: 2, y: 2 });
  await click('#ledger-list .filter[data-state="a-only"]');

  await click('#compare-btn');
  ok('Compare again with no edit keeps the decisions', await run(`document.getElementById('d-source-s1-needs-change').checked && document.getElementById('d-source-s2-accepted').checked`));

  // Span jump: read-only box, a key cannot replace the selection, Escape returns.
  await click('#n1b .span-link');
  const before = await run(`document.getElementById('prose').value`);
  await key(' ', 'Space', 32, ' ');
  ok('a key after a span jump does not replace the selection', await run(`document.getElementById('prose').value`) === before && await run(`!document.getElementById('review-panel').hidden`));
  await key('Escape', 'Escape', 27);
  ok('Escape returns to the span button and releases the box', await run(`document.activeElement === document.querySelector('#n1b .span-link') && !document.getElementById('prose').readOnly`));

  // Edit guard: typing into a box with recorded decisions asks first.
  await run(`(() => { const t = document.getElementById('rewrite'); t.focus(); t.setSelectionRange(t.value.length, t.value.length); })()`);
  await key('x', 'KeyX', 88, 'x');
  ok('typing with recorded decisions asks first and changes nothing', await run(`!document.getElementById('edit-guard').hidden && document.getElementById('rewrite').value === 'Alice can cancel the lease. The deposit is refundable.'`));
  await click('#guard-no');

  // Export, then check and open the packet.
  await run(`window.__blobs = []; { const o = URL.createObjectURL; URL.createObjectURL = b => { window.__blobs.push(b); return o.call(URL, b); }; HTMLAnchorElement.prototype.click = function () {}; }`);
  await click('#export-review-btn');
  await sleep(300);
  const packet = JSON.parse(await run(`window.__blobs[0].text()`));
  ok('export writes literal-spans-v2 rows with only their five fields', packet.review.method === 'literal-spans-v2' && packet.review.rows.every(r => Object.keys(r).sort().join() === 'decision,id,kind,output,source'));
  await run(`window.scrollTo(0, 0)`);
  await click('.nav a.primary');
  await run(`(() => { const t = document.getElementById('packet-input'); t.value = ${JSON.stringify(JSON.stringify(packet))}; t.dispatchEvent(new Event('input', { bubbles: true })); })()`);
  await click('#verify-packet-btn');
  for (let i = 0; i < 40 && !(await run(`document.getElementById('packet-status').textContent.includes('Checks completed')`)); i++) await sleep(100);
  ok('the exported packet checks in the browser', await run(`document.getElementById('packet-status').textContent.includes('"review_spans": "recomputed"')`));
  await click('#open-packet-btn');
  await run(`document.getElementById('edit-guard').hidden || document.getElementById('guard-yes').click()`);
  ok('opening the packet shows its decisions', await run(`document.getElementById('d-source-s1-needs-change').checked && document.getElementById('d-source-s2-accepted').checked && document.getElementById('strip-tag').textContent === 'Opened packet'`));

  // Meaning receipt sample: the bound is issuer-asserted and rounded up; `controlled` is not shown.
  await run(`document.getElementById('meaning-annex').open = true`);
  await click('#meaning-sample-btn');
  for (let i = 0; i < 30 && !(await run(`document.getElementById('meaning-receipt-input').value.length > 0`)); i++) await sleep(100);
  await click('#meaning-verify-btn');
  for (let i = 0; i < 40 && !(await run(`/Stage A verified|Verification failed/.test(document.getElementById('meaning-verify-result').textContent)`)); i++) await sleep(100);
  const stageA = await run(`document.getElementById('meaning-verify-result').textContent`);
  ok('meaning receipt: Stage A verifies, bound rounded up and issuer-asserted, no controlled', /Stage A verified/.test(stageA) && stageA.includes('issuer-asserted bound, not replayed here: ≤ 0.6455') && !/controlled/.test(stageA), stageA.slice(0, 200));

  // Bundle Ed25519 in the browser's own WebCrypto: malformed keys fail closed.
  const m3 = await run(`(async () => {
    const bundle = await makeBundle([['alice', 'likes', 'cats']], 'T');
    const pair = await crypto.subtle.generateKey({ name: 'Ed25519' }, true, ['sign', 'verify']);
    const raw = new Uint8Array(await crypto.subtle.exportKey('raw', pair.publicKey));
    const sig = new Uint8Array(await crypto.subtle.sign('Ed25519', pair.privateKey, new TextEncoder().encode(bundle.canonical_tome + '|' + bundle.state_integer + '|' + bundle.timestamp)));
    const b64 = u => btoa(String.fromCharCode(...u));
    const good = { ...bundle, public_key: 'ed25519:' + b64(raw), public_signature: 'ed25519:' + b64(sig) };
    const v = async change => { const r = await verifyBundle({ ...good, ...change }); return r.ok + ':' + r.signatures.ed25519.status; };
    return [await v({}), await v({ public_key: 'ed25519:' + b64(raw.slice(0, 31)) }), await v({ public_key: 'ed25519:!!!' }), await v({ public_signature: 'ed25519:' + b64(new Uint8Array(64).fill(1)) })];
  })()`);
  ok('bundle check: valid verifies, malformed fails closed, wrong signature invalid', JSON.stringify(m3) === JSON.stringify(['true:verified', 'false:malformed', 'false:malformed', 'false:invalid']), JSON.stringify(m3));

  // Every sentence printed for the adversarial pairs is one the oracle accepts, and each view is exact.
  let pairsChecked = 0;
  for (const [a, b] of ADVERSARIAL) {
    if (!a.trim() || !b.trim()) continue;
    await setBoxes(a, b);
    await run(`document.getElementById('compare-btn').click()`);
    const page = await run(`(() => ({
      heading: document.getElementById('review-heading').textContent,
      statements: [...document.querySelectorAll('#review-rows .note .ns')].map(p => p.firstChild.textContent),
      inBoth: [...document.querySelectorAll('#review-rows .inboth .lit')].map(n => n.textContent),
      views: [...document.querySelectorAll('#review-rows article')].map(a => { const line = a.querySelector('.blackline'); return line ? { id: a.id } : null; }),
    }))()`);
    const review = compareTexts(a, b);
    const ev = E.buildEvidence(a, b, review);
    const want = ev.passages.flatMap(p => p.notes.map(n => E.noteStatement(n, p, a, b)));
    const bad = check(a, b);
    const same = JSON.stringify(page.statements) === JSON.stringify(want) && JSON.stringify(page.inBoth) === JSON.stringify(ev.passages.flatMap(p => p.inBoth.map(E.inBothText)));
    const headingTrue = /no literal differences/.test(page.heading) === (a === b);
    // Views in the real engine: switch each Read as view and compare with the exact passages.
    let viewsExact = true;
    for (const [view, side] of [['original', 'a'], ['rewrite', 'b']]) {
      await run(`document.getElementById('view-${view}').checked = true`);
      const shown = await run(`[...document.querySelectorAll('#review-rows article .blackline')].map(${VIEW})`);
      const expected = ev.passages.map(p => (side === 'a' ? p.a?.text : p.b?.text) || '');
      shown.forEach((text, i) => { if (text.replace(/‸/g, '') !== expected[i]) viewsExact = false; });
    }
    await run(`document.getElementById('view-marked').checked = true`);
    pairsChecked++;
    ok(`true statements and exact views: ${JSON.stringify(a).slice(0, 48)}`, same && !bad.length && headingTrue && viewsExact,
      `${bad.slice(0, 2).join(' | ')} same=${same} heading=${headingTrue} views=${viewsExact}`);
  }
  ok(`checked ${pairsChecked} adversarial pairs`, pairsChecked > 30);

  // Timings on 100,000-character input.
  // About 100,000 characters each side, under the 300-passage limit: each passage is two
  // copies of an example joined into one sentence.
  const unit = t => { const one = t.replace(/\. /g, '; ').replace(/\.$/, ''); return `${one}; ${one}. `; };
  const big = unit(E.EXAMPLES.refund.source).repeat(200).slice(0, 99990);
  const big2 = unit(E.EXAMPLES.refund.output).repeat(200).slice(0, 99990);
  ok('the 100,000-character texts are within the passage limit', compareTexts(big, big2).rows.length > 100);
  await setBoxes(E.EXAMPLES.lease.source, E.EXAMPLES.lease.output);
  const keystroke = await run(`(() => { const t = document.getElementById('prose'); t.value = ${JSON.stringify(big)}; const s = performance.now(); t.dispatchEvent(new Event('input', { bubbles: true })); return performance.now() - s; })()`);
  ok(`a keystroke handler on a 100,000-character box runs under 50 ms (${keystroke.toFixed(0)} ms)`, keystroke < 50);
  // The counts run a moment after typing stops: no task in the following 400 ms may block for 50 ms or more.
  const longest = await run(`new Promise(done => { const seen = []; const o = new PerformanceObserver(l => { for (const e of l.getEntries()) seen.push(e.duration); }); o.observe({ type: 'longtask' });
    const t = document.getElementById('prose'); t.value += ' more'; t.dispatchEvent(new Event('input', { bubbles: true }));
    setTimeout(() => { o.disconnect(); done(Math.max(0, ...seen)); }, 400); })`);
  ok(`the delayed count after a keystroke on 100,000 characters blocks under 50 ms (longest task ${longest.toFixed(0)} ms)`, longest < 50);
  await run(`(() => { const t = document.getElementById('rewrite'); t.value = ${JSON.stringify(big2)}; t.dispatchEvent(new Event('input', { bubbles: true })); })()`);
  await sleep(400);
  const compareMs = await run(`new Promise(done => { const s = performance.now(); const h = document.getElementById('review-heading'); document.getElementById('compare-btn').click();
    const wait = () => (!document.getElementById('review-panel').hidden || performance.now() - s > 15000 ? requestAnimationFrame(() => done(performance.now() - s)) : setTimeout(wait, 5)); wait(); })`);
  ok(`100,000 characters each side compare and render in under 2 s (${compareMs.toFixed(0)} ms)`, compareMs < 2000);
  const hostile = 'if '.repeat(33000);
  await setBoxes(hostile, 'x');
  const hostileMs = await run(`new Promise(done => { const s = performance.now(); document.getElementById('compare-btn').click();
    const wait = () => (!document.getElementById('review-panel').hidden || performance.now() - s > 15000 ? requestAnimationFrame(() => done(performance.now() - s)) : setTimeout(wait, 5)); wait(); })`);
  ok(`a hostile 99,000-character text renders in under 2 s (${hostileMs.toFixed(0)} ms)`, hostileMs < 2000);
  ok('long passages are drawn in part with an explicit button', await run(`[...document.querySelectorAll('.more-btn')].length > 0`));
  const packetBig = await run(`(async () => { const { compareTexts, makeReviewPacket } = await import('./review_packet.js'); return JSON.stringify(await makeReviewPacket({ source: ${JSON.stringify(big)}, output: ${JSON.stringify(big2)}, review: compareTexts(${JSON.stringify(big)}, ${JSON.stringify(big2)}) })); })()`);
  await run(`(() => { const t = document.getElementById('packet-input'); t.value = ${JSON.stringify(packetBig)}; t.dispatchEvent(new Event('input', { bubbles: true })); })()`);
  const openMs = await run(`new Promise(async done => { const s = performance.now(); document.getElementById('verify-packet-btn').click();
    const wait = () => (!document.getElementById('open-packet-btn').disabled || performance.now() - s > 15000 ? done(performance.now() - s) : setTimeout(wait, 5)); wait(); })`);
  ok(`checking a 200,000-character packet takes under 2 s (${openMs.toFixed(0)} ms)`, openMs < 2000);

  // Layout: no horizontal scroll at phone widths, even with long URLs and hashes.
  for (const w of [390, 360]) {
    await size(w, 900);
    const url = 'https://example.com/very/long/path/' + 'segment-'.repeat(12) + 'end?query=' + 'x'.repeat(40);
    await setBoxes(`Download the policy from ${url} before signing. The checksum is sha256-${'a1b2c3d4'.repeat(8)} and must match.`,
      `Download the policy from ${url.replace('end', 'fin')} before signing. The checksum is sha256-${'ffff'.repeat(16)} and should match.`);
    await run(`document.getElementById('compare-btn').click(); for (const d of document.querySelectorAll('details')) { d.style.display = ''; d.open = true; }`);
    const wide = await run(`document.documentElement.scrollWidth`);
    ok(`no horizontal scroll at ${w} with long tokens (${wide})`, wide <= w);
  }
  ok('no console errors, exceptions or /api requests', problems.length === 0, problems.join(' | '));
} catch (e) {
  results.push('FAIL harness error: ' + e.stack);
} finally {
  ws.close(); chrome.kill(); server.close();
  setTimeout(() => rmSync(profile, { recursive: true, force: true }), 300);
}
console.log(results.join('\n'));
const failed = results.filter(r => r.startsWith('FAIL')).length;
console.log(`\n${results.length - failed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
