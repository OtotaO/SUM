import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { sourceSpans, compareTexts, hashText, publicJwks, makeReviewPacket, verifyReviewPacket, checkRenderBinding } from './review_packet.js';
import { canonicalize } from './vendor/sum-verify-deps.js';

const fixture = JSON.parse(await readFile(new URL('../fixtures/render_receipts/source_render.json', import.meta.url)));
const keys = JSON.parse(await readFile(new URL('../fixtures/render_receipts/jwks_at_capture.json', import.meta.url)));
const signedRender = { receipt: fixture.render_receipt, triples: fixture.triples_used, sliders: fixture.quantized_sliders };
const source = 'Alice was born in 1990. Alice graduated in 2012.';
const packet = () => makeReviewPacket({ source, output: fixture.tome, review: compareTexts(source, fixture.tome), render: signedRender, jwks: keys });

test('source spans preserve original Unicode, qualification and offsets', () => {
  const text = '  😀 Alice may cancel, unless overdue.\nMarie Curie won two prizes! Final clause';
  const spans = sourceSpans(text);
  assert.equal(spans.length, 3);
  for (const span of spans) assert.equal(text.slice(span.start, span.end), span.text);
  assert.match(spans[0].text, /may cancel, unless overdue/);
  assert.equal(spans[1].text, 'Marie Curie won two prizes!');
});

test('changed qualifiers are never labeled verbatim or semantically preserved', () => {
  const review = compareTexts('Alice may cancel with 30 days notice.', 'Alice must cancel with 3 days notice.');
  assert.equal(review.rows[0].kind, 'changed-candidate');
  assert.match(review.scope, /meaning preservation are not measured/);
  assert.equal(review.rows[0].decision, 'unreviewed');
});

test('matches consume duplicate passages once and retain all unmatched spans', () => {
  const review = compareTexts('Alice likes cats. Alice likes cats. Bob owns dogs.', 'Alice likes cats. Mars is red.');
  assert.equal(review.rows.filter(r => r.kind === 'verbatim').length, 1);
  assert.equal(review.rows.filter(r => r.source).length, 3);
  assert.equal(review.rows.filter(r => r.output).length, 2);
});

test('empty rewrite retains the source as unmatched passages', () => {
  assert.equal(compareTexts('The lease may expire.', '').rows[0].kind, 'source-unmatched');
  assert.equal(compareTexts('', 'A new claim.').rows[0].kind, 'output-unmatched');
});

test('oversized text and excessive sentence counts fail explicitly', () => {
  assert.throws(() => compareTexts('x'.repeat(100001), ''), /100,000/);
  assert.throws(() => compareTexts('A. '.repeat(301), ''), /300/);
});

test('packet roundtrip verifies actual captured receipt and exact content', async () => {
  const result = await verifyReviewPacket(JSON.parse(JSON.stringify(await packet())));
  assert.equal(result.signature, 'verified');
  assert.equal(result.output_binding, 'verified');
  assert.equal(result.source_binding, 'not-signed');
  assert.equal(result.revocation, 'not-checked');
  assert.equal(result.freshness, 'not-checked');
});

test('valid receipt cannot authenticate substituted output with a recomputed unsigned hash', async () => {
  const p = await packet();
  p.output.text = 'Alice graduated in 2099.';
  p.output.hash = await hashText(p.output.text);
  p.review = compareTexts(p.source.text, p.output.text);
  await assert.rejects(verifyReviewPacket(p), /signed tome hash/);
});

test('valid receipt cannot authenticate substituted claims or slider settings', async () => {
  const p = await packet();
  p.render.triples[0][2] = '2099';
  await assert.rejects(verifyReviewPacket(p), /signed triples hash/);
  const q = await packet();
  q.render.sliders.density = 0.5;
  await assert.rejects(verifyReviewPacket(q), /Slider settings/);
});

test('packet rejects stale hashes, spans, unknown decisions and missing keys', async () => {
  const p = await packet(); p.source.text += '!';
  await assert.rejects(verifyReviewPacket(p), /hash mismatch/);
  const q = await packet(); q.review.rows[0].source.start++;
  await assert.rejects(verifyReviewPacket(q), /span or decision/);
  const r = await packet(); r.review.rows[0].decision = 'certified';
  await assert.rejects(verifyReviewPacket(r), /span or decision/);
  const s = await packet(); s.jwks = null;
  await assert.rejects(verifyReviewPacket(s), /Public keys/);
});

test('unsigned existing-rewrite packet never gains a signature verdict', async () => {
  const review = compareTexts('Original may be wrong.', 'Original is wrong.');
  review.rows[0].decision = 'needs-change';
  const p = await makeReviewPacket({ source: 'Original may be wrong.', output: 'Original is wrong.', review });
  const result = await verifyReviewPacket(p);
  assert.equal(result.signature, 'absent');
  assert.equal(result.human_decisions, 'unsigned');
  review.rows[0].decision = 'accepted';
  assert.equal(p.review.rows[0].decision, 'needs-change');
});

test('key export strips private material and unneeded key metadata', () => {
  const result = publicJwks({ keys: [{ ...keys.keys[0], d: 'private', password: 'secret', key_ops: ['sign'] }] });
  assert.equal(result.keys.length, 1);
  assert.ok(!('d' in result.keys[0]));
  assert.ok(!('password' in result.keys[0]));
  assert.ok(!('key_ops' in result.keys[0]));
});

test('captured receipt remains invalid after signature corruption', async () => {
  const render = structuredClone(signedRender);
  const pieces = render.receipt.jws.split('.');
  pieces[2] = (pieces[2][0] === 'A' ? 'B' : 'A') + pieces[2].slice(1);
  render.receipt.jws = pieces.join('.');
  await assert.rejects(checkRenderBinding(render, fixture.tome, keys));
});

test('source binding uses Python Unicode codepoint ordering across BMP and astral text', async () => {
  const pair = await crypto.subtle.generateKey({ name: 'Ed25519' }, true, ['sign', 'verify']);
  const publicKey = await crypto.subtle.exportKey('jwk', pair.publicKey);
  const triples = [['\u{10000}', 'likes', 'cats'], ['\uE000', 'likes', 'cats']];
  const payload = { ...fixture.render_receipt.payload,
    triples_hash: await hashText(canonicalize([triples[1], triples[0]])),
    tome_hash: await hashText('Unicode sample') };
  const header = Buffer.from(JSON.stringify({ alg: 'EdDSA', kid: 'unicode-test', b64: false, crit: ['b64'] })).toString('base64url');
  const signature = await crypto.subtle.sign('Ed25519', pair.privateKey, new TextEncoder().encode(header + '.' + canonicalize(payload)));
  const receipt = { schema: 'sum.render_receipt.v1', kid: 'unicode-test', payload, jws: header + '..' + Buffer.from(signature).toString('base64url') };
  const result = await checkRenderBinding({ receipt, triples, sliders: payload.sliders_quantized }, 'Unicode sample', { keys: [{ ...publicKey, kid: 'unicode-test' }] });
  assert.equal(result.triples_binding, 'verified');
});

// ---------------------------------------------------------------- key pin
// The packet and its jwks are unsigned. These packets are built the way the
// Worker signs (worker/src/receipt/sign.ts): a detached JWS whose protected
// header is {alg, kid, b64: false, crit: ['b64']}, Ed25519 over the ASCII of
// header + '.' + the JCS bytes of the payload, with the triples and tome
// hashes recomputed for the substituted output.
import * as rp from './review_packet.js';
import { createPrivateKey, createPublicKey } from 'node:crypto';

const SITE_KID = keys.keys[0].kid;
const ZERO_SEED_X = 'O2onvM62pC1io6jQKm8Nc2UyFXcd4kOmOsBIoYtZ2ik';
const forgedTome = 'Alice was born in 1990. Alice graduated in 2099.';
const forgedTriples = [['alice', 'born_in', '1990'], ['alice', 'graduated_in', '2099']];

async function freshKey(kid) {
  const pair = await crypto.subtle.generateKey({ name: 'Ed25519' }, true, ['sign', 'verify']);
  const { kty, crv, x } = await crypto.subtle.exportKey('jwk', pair.publicKey);
  return { privateKey: pair.privateKey, jwk: { kty, crv, x, kid, alg: 'EdDSA', use: 'sig' } };
}

async function signRender(privateKey, kid, tome = forgedTome, triples = forgedTriples) {
  const sorted = triples.map(t => [...t]).sort((a, b) => { for (let i = 0; i < 3; i++) if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1; return 0; });
  const payload = { ...fixture.render_receipt.payload, render_id: 'f0f0f0f0f0f0f0f0', signed_at: '2026-10-01T00:00:00.000Z',
    triples_hash: await hashText(canonicalize(sorted)), tome_hash: await hashText(tome) };
  const header = Buffer.from(JSON.stringify({ alg: 'EdDSA', kid, b64: false, crit: ['b64'] })).toString('base64url');
  const signature = await crypto.subtle.sign('Ed25519', privateKey, new TextEncoder().encode(header + '.' + canonicalize(payload)));
  const receipt = { schema: 'sum.render_receipt.v1', kid, payload, jws: header + '..' + Buffer.from(signature).toString('base64url') };
  return { receipt, triples, sliders: payload.sliders_quantized };
}

async function forgedPacket({ kid = 'sum-render-2026-10-01-1', extraKeys = [], signer = null } = {}) {
  const key = signer || await freshKey(kid);
  const render = await signRender(key.privateKey, kid);
  // JSON round trip: the verifier sees exactly what a pasted packet carries.
  return JSON.parse(JSON.stringify(await makeReviewPacket({ source, output: forgedTome, review: compareTexts(source, forgedTome), render, jwks: { keys: [key.jwk, ...extraKeys] } })));
}

test('key pin: the captured site receipt is a key this site publishes, and not-checked without site keys', async () => {
  const p = JSON.parse(JSON.stringify(await packet()));
  const pinned = await verifyReviewPacket(p, { siteJwks: keys });
  assert.equal(pinned.signature, 'verified');
  assert.equal(pinned.key_pin, 'site-key');
  assert.ok(!('key_warning' in pinned));
  assert.equal((await verifyReviewPacket(p)).key_pin, 'not-checked');
  assert.equal((await verifyReviewPacket(p, null)).key_pin, 'not-checked');
  assert.equal((await verifyReviewPacket(p, { siteJwks: null })).key_pin, 'not-checked');
});

test('key pin: an unsigned packet reports not-applicable and needs no site keys', async () => {
  const p = await makeReviewPacket({ source: 'A may be late.', output: 'A is late.', review: compareTexts('A may be late.', 'A is late.') });
  assert.equal((await verifyReviewPacket(p)).key_pin, 'not-applicable');
  assert.equal((await verifyReviewPacket(p, { siteJwks: keys })).key_pin, 'not-applicable');
});

test('key pin: a packet signed with a freshly generated key verifies but is not a site key', async () => {
  const p = await forgedPacket();
  const offline = await verifyReviewPacket(p);
  assert.equal(offline.signature, 'verified', 'the forged packet is internally consistent');
  assert.equal(offline.output_binding, 'verified');
  assert.equal(offline.key_pin, 'not-checked');
  const pinned = await verifyReviewPacket(p, { siteJwks: keys });
  assert.equal(pinned.signature, 'verified');
  assert.equal(pinned.key_pin, 'not-site-key');
  assert.equal(pinned.signer_identity, 'not-established');
  assert.equal(rp.pinPacketKey(p.jwks, keys, p.render.receipt.kid).key_pin, 'not-site-key');
});

test('key pin: a packet key that reuses a site key ID for a different key is rejected', async () => {
  const collision = /reuses key ID "sum-render-2026-04-27-1", which this site publishes for a different key/;
  // The receipt itself is signed under the site's key ID with another key.
  const p = await forgedPacket({ kid: SITE_KID });
  assert.equal((await verifyReviewPacket(p)).key_pin, 'not-checked', 'without site keys the collision cannot be seen');
  await assert.rejects(verifyReviewPacket(p, { siteJwks: keys }), collision);
  // An unused packet key that collides is rejected too.
  const decoy = await freshKey(SITE_KID);
  const q = await forgedPacket({ extraKeys: [decoy.jwk] });
  await assert.rejects(verifyReviewPacket(q, { siteJwks: keys }), collision);
});

test('key pin: two different keys under one key ID inside the packet are rejected', async () => {
  const other = await freshKey('sum-render-2026-10-01-1');
  const p = await forgedPacket({ extraKeys: [other.jwk] });
  await assert.rejects(verifyReviewPacket(p), /Ambiguous key ID: the packet carries different keys under key ID "sum-render-2026-10-01-1"/);
  await assert.rejects(verifyReviewPacket(p, { siteJwks: keys }), /Ambiguous key ID/);
  // A forged key listed after the real site key under the site's own kid. The
  // signature verifies against the first entry, so only the pin catches it.
  const forger = await freshKey(SITE_KID);
  const q = JSON.parse(JSON.stringify(await packet()));
  q.jwks.keys.push(forger.jwk);
  await assert.rejects(verifyReviewPacket(q), /Ambiguous key ID/);
  // Listed first instead, the forged key is the one selected and the signature fails.
  q.jwks.keys.reverse();
  await assert.rejects(verifyReviewPacket(q), /signature verification failed/);
  // Same kid and same x but another curve is different key material.
  const c = JSON.parse(JSON.stringify(await packet()));
  c.jwks.keys.push({ ...keys.keys[0], crv: 'X25519' });
  await assert.rejects(verifyReviewPacket(c), /Ambiguous key ID/);
  // An exact duplicate is not ambiguous.
  const r = JSON.parse(JSON.stringify(await packet()));
  r.jwks.keys.push({ ...r.jwks.keys[0] });
  assert.equal((await verifyReviewPacket(r, { siteJwks: keys })).key_pin, 'site-key');
});

test('key pin: the all-zero-seed test key is flagged and never pinned unless the site publishes it', async () => {
  // PKCS#8 wrapper for a raw 32-byte Ed25519 seed; the seed here is all zeros.
  const der = Buffer.concat([Buffer.from('302e020100300506032b657004220420', 'hex'), Buffer.alloc(32)]);
  const derived = createPublicKey(createPrivateKey({ key: der, format: 'der', type: 'pkcs8' })).export({ format: 'jwk' });
  assert.equal(derived.x, ZERO_SEED_X, 'O2onvM... is the public key of the zero seed');
  const goldenKeys = JSON.parse(await readFile(new URL('../fixtures/meaning_receipts/jwks.json', import.meta.url)));
  assert.equal(goldenKeys.keys[0].x, ZERO_SEED_X, 'the meaning goldens are signed with that key');
  assert.equal(rp.PUBLIC_TEST_KEY_X, ZERO_SEED_X);
  const kid = 'test-vector-zero-seed-DO-NOT-TRUST';
  const privateKey = await crypto.subtle.importKey('pkcs8', der, { name: 'Ed25519' }, false, ['sign']);
  const p = await forgedPacket({ kid, signer: { privateKey, jwk: { kty: 'OKP', crv: 'Ed25519', x: derived.x, kid } } });
  const offline = await verifyReviewPacket(p);
  assert.equal(offline.signature, 'verified');
  assert.equal(offline.key_pin, 'not-checked');
  assert.equal(offline.key_warning, 'public-test-vector-key');
  const pinned = await verifyReviewPacket(p, { siteJwks: keys });
  assert.equal(pinned.key_pin, 'not-site-key');
  assert.equal(pinned.key_warning, 'public-test-vector-key');
  // Only a site that itself publishes that exact key gets site-key, still flagged.
  const both = await verifyReviewPacket(p, { siteJwks: { keys: [...keys.keys, { kty: 'OKP', crv: 'Ed25519', x: ZERO_SEED_X, kid }] } });
  assert.equal(both.key_pin, 'site-key');
  assert.equal(both.key_warning, 'public-test-vector-key');
});

test('key pin: malformed site keys fail closed or stay unpinned, never site-key', async () => {
  const p = JSON.parse(JSON.stringify(await packet()));
  for (const siteJwks of ['{"keys":[]}', 42, true, [], [keys.keys[0]], {}, { keys: 'x' }, { keys: {} }, { keys: { 0: keys.keys[0], length: 1 } }]) {
    await assert.rejects(verifyReviewPacket(p, { siteJwks }), /Site public keys must be an object with a keys array/, JSON.stringify(siteJwks));
  }
  const site = keys.keys[0];
  const unpinned = [
    { keys: [] },
    { keys: [null, 7, 'x', [site], { kid: 5, kty: 'OKP', crv: 'Ed25519', x: site.x }, { kty: 'OKP', crv: 'Ed25519', x: site.x }] },
    { keys: [{ kty: 'RSA', kid: 'other-rsa', n: 'AQAB', e: 'AQAB' }] },
    { keys: [{ ...site, kid: site.kid.toUpperCase() }] },
    { keys: [{ ...site, kid: site.kid + ' ' }] },
    { keys: [{ ...site, kid: ' ' + site.kid }] },
  ];
  for (const siteJwks of unpinned) assert.equal((await verifyReviewPacket(p, { siteJwks })).key_pin, 'not-site-key', JSON.stringify(siteJwks));
  // The same key ID published for other material is a collision: exact strings, no trimming or case folding.
  for (const other of [{ ...site, x: site.x.toLowerCase() }, { kty: 'RSA', kid: site.kid, n: 'AQAB', e: 'AQAB' }]) {
    await assert.rejects(verifyReviewPacket(p, { siteJwks: { keys: [other] } }), /which this site publishes for a different key/, JSON.stringify(other));
  }
  assert.throws(() => rp.pinPacketKey(p.jwks, keys, 'no-such-kid'), /no key for key ID/);
  assert.throws(() => rp.pinPacketKey(null, keys, SITE_KID), /Packet public keys must be an object/);
});

test('key pin: a key ID is shown quoted, with every character outside printable ASCII escaped', async () => {
  assert.equal(rp.quoteKid(SITE_KID), '"sum-render-2026-04-27-1"');
  assert.equal(rp.quoteKid('a\nb\r'), '"a\\nb\\r"');
  assert.equal(rp.quoteKid('k\u202e\u2066'), '"k\\u202e\\u2066"', 'bidi controls');
  assert.equal(rp.quoteKid('x\u2028y\u2029\u0085'), '"x\\u2028y\\u2029\\u0085"', 'line and paragraph separators');
  // Lookalikes of the site key ID stay visibly different from it.
  assert.equal(rp.quoteKid(SITE_KID + '\u200b'), '"sum-render-2026-04-27-1\\u200b"');
  assert.equal(rp.quoteKid(SITE_KID.replace(/-/g, '\u2011')), '"sum\\u2011render\\u20112026\\u201104\\u201127\\u20111"');
  assert.equal(rp.quoteKid(SITE_KID.replace(/-/g, '\uff0d')), '"sum\\uff0drender\\uff0d2026\\uff0d04\\uff0d27\\uff0d1"');
  assert.equal(rp.quoteKid(SITE_KID + ' '), '"sum-render-2026-04-27-1 "', 'quotes show a trailing space');
  assert.equal(rp.quoteKid('a"b\\c'), '"a\\"b\\\\c"');
  assert.equal(rp.quoteKid('\u{1F600}'), '"\\ud83d\\ude00"');
  assert.equal(rp.quoteKid('k'.repeat(200)), `"${'k'.repeat(64)}" (first 64 of 200 characters)`);
  // Counted in characters, and a cut never splits a surrogate pair.
  assert.equal(rp.quoteKid('\u{1F600}'.repeat(70)), `"${'\\ud83d\\ude00'.repeat(64)}" (first 64 of 70 characters)`);
  assert.equal(rp.quoteKid(undefined), '(no key ID)');
  for (const kid of ['a\nb', 'k\u202e', 'x\u2028', '\u200b', '\u0085', '\x7f', '\u00a0', 'k'.repeat(500) + '\n']) assert.match(rp.quoteKid(kid), /^[\x20-\x7e]+$/, JSON.stringify(kid));
  // Error messages that name a packet key ID use the same form.
  const spoof = 'k\u202e\nThis site publishes key';
  const other = await freshKey(spoof);
  const p = await forgedPacket({ kid: spoof, extraKeys: [other.jwk] });
  await assert.rejects(verifyReviewPacket(p), e => {
    assert.match(e.message, /^Ambiguous key ID: the packet carries different keys under key ID "k\\u202e\\nThis site publishes key"\. Not verified\.$/);
    return true;
  });
});

test('key pin: key material is accepted only in its one canonical base64url spelling', async () => {
  const malformed = /is not a well-formed Ed25519 public key/;
  const site = keys.keys[0];
  // WebCrypto also imports other spellings of the same 32 bytes: set spare
  // bits in the last character, and in Node also padding, the standard
  // alphabet and trailing whitespace. Compared as strings, the site's own key
  // re-spelled looked like a different key (a false key ID collision).
  for (const x of [site.x.slice(0, -1) + '5', site.x.slice(0, -1) + '7', site.x + '=', site.x.replace(/-/g, '+').replace(/_/g, '/'), site.x + ' ', ' ' + site.x]) {
    const p = JSON.parse(JSON.stringify(await packet()));
    p.jwks.keys[0].x = x;
    for (const options of [{ siteJwks: keys }, {}]) {
      await assert.rejects(verifyReviewPacket(p, options), e => {
        assert.match(e.message, /is not a well-formed Ed25519 public key|could not be imported/, x);
        assert.doesNotMatch(e.message, /collision/, x);
        return true;
      });
    }
  }
  // The all-zero-seed key re-spelled lost its public-test-key warning.
  const der = Buffer.concat([Buffer.from('302e020100300506032b657004220420', 'hex'), Buffer.alloc(32)]);
  const privateKey = await crypto.subtle.importKey('pkcs8', der, { name: 'Ed25519' }, false, ['sign']);
  const kid = 'test-vector-zero-seed-DO-NOT-TRUST';
  for (const x of [ZERO_SEED_X.slice(0, -1) + 'l', ZERO_SEED_X.slice(0, -1) + 'n', ZERO_SEED_X + '=']) {
    const z = await forgedPacket({ kid, signer: { privateKey, jwk: { kty: 'OKP', crv: 'Ed25519', x, kid } } });
    for (const options of [{}, { siteJwks: keys }]) await assert.rejects(verifyReviewPacket(z, options), /is not a well-formed Ed25519 public key|could not be imported/, x);
  }
  // An unused malformed Ed25519 entry in the packet fails closed too.
  const extra = JSON.parse(JSON.stringify(await packet()));
  extra.jwks.keys.push({ kty: 'OKP', crv: 'Ed25519', x: 'AAAA', kid: 'unused' });
  await assert.rejects(verifyReviewPacket(extra), /Packet public keys: the key under key ID "unused" is not a well-formed Ed25519 public key/);
  // A site key in another spelling fails closed with a site message, not as a collision.
  for (const x of [site.x.slice(0, -1) + '5', site.x + ' ', site.x + '=']) {
    await assert.rejects(verifyReviewPacket(JSON.parse(JSON.stringify(await packet())), { siteJwks: { keys: [{ ...site, x }] } }),
      /^Error: Site public keys: the key under key ID "sum-render-2026-04-27-1" is not a well-formed Ed25519 public key/, x);
  }
  // Canonical spellings are unchanged: the site key and the zero-seed key.
  assert.equal(rp.pinPacketKey(keys, keys, SITE_KID).key_pin, 'site-key');
  assert.deepEqual(rp.pinPacketKey({ keys: [{ kty: 'OKP', crv: 'Ed25519', x: ZERO_SEED_X, kid }] }, null, kid), { key_pin: 'not-checked', key_warning: 'public-test-vector-key' });
});
