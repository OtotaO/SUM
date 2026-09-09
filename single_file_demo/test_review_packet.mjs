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
