import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { sourceSpans, sourceSpansV2, compareTexts, hashText, publicJwks, makeReviewPacket, verifyReviewPacket, checkRenderBinding,
  REVIEW_SCOPE, REVIEW_SCOPE_V2 } from './review_packet.js';
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
  for (const method of ['literal-spans-v1', 'literal-spans-v2']) {
    assert.throws(() => compareTexts('x'.repeat(100001), '', method), /100,000/);
    assert.throws(() => compareTexts('', 'x'.repeat(100001), method), /100,000/);
    assert.throws(() => compareTexts('Go. '.repeat(301), '', method), /300/);
  }
  // v1 splits "A. A. A." at every period; v2 reads a single capital letter as an initial.
  assert.throws(() => compareTexts('A. '.repeat(301), '', 'literal-spans-v1'), /300/);
  assert.throws(() => sourceSpansV2('x'.repeat(100001)), /100,000/);
  assert.throws(() => compareTexts('a', 'b', 'literal-spans-v9'), /Unsupported review method/);
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

// Two packets over the same texts, one per method. The text makes the two
// splitters disagree ("$7.95" and "Dr."), so each can only verify through its own.
const v1Source = 'Call Dr. Patel if a rash appears. Shipping costs $7.95 per order. Refunds follow.';
const v1Output = 'Call Dr. Patel if you get a rash. Shipping costs $7.95 per order. Refunds follow.';
async function methodPacket(method) {
  const review = compareTexts(v1Source, v1Output, method);
  review.rows[0].decision = 'accepted';
  return JSON.parse(JSON.stringify(await makeReviewPacket({ source: v1Source, output: v1Output, review })));
}

test('the two splitters really disagree on the method fixture', () => {
  assert.notDeepEqual(sourceSpans(v1Source), sourceSpansV2(v1Source));
});

test('existing literal-spans-v1 packets still verify through the frozen v1 splitter', async () => {
  const p = await methodPacket('literal-spans-v1');
  assert.equal(p.review.method, 'literal-spans-v1');
  assert.equal(p.review.scope, REVIEW_SCOPE);
  assert.equal(p.disclosure, REVIEW_SCOPE);
  const result = await verifyReviewPacket(p);
  assert.equal(result.review_spans, 'recomputed');
  assert.equal(result.signature, 'absent');
});

test('literal-spans-v2 packets verify through the v2 splitter', async () => {
  const p = await methodPacket('literal-spans-v2');
  assert.equal(p.review.method, 'literal-spans-v2');
  assert.equal(p.review.scope, REVIEW_SCOPE_V2);
  assert.equal(p.disclosure, REVIEW_SCOPE_V2, 'the disclosure names what this method measures');
  assert.equal((await verifyReviewPacket(p)).review_spans, 'recomputed');
  assert.equal(compareTexts(v1Source, v1Output).method, 'literal-spans-v2', 'new comparisons use v2');
});

test('a packet whose method, scope or spans do not match each other is rejected', async () => {
  const v1 = await methodPacket('literal-spans-v1'), v2 = await methodPacket('literal-spans-v2');
  // v2 rows relabelled as v1 (and the reverse) no longer match the recomputed spans.
  await assert.rejects(verifyReviewPacket({ ...v2, review: { ...v2.review, method: 'literal-spans-v1', scope: REVIEW_SCOPE } }), /Review spans|span or decision/);
  await assert.rejects(verifyReviewPacket({ ...v1, review: { ...v1.review, method: 'literal-spans-v2', scope: REVIEW_SCOPE_V2 } }), /Review spans|span or decision/);
  // A method with the other method's scope string.
  await assert.rejects(verifyReviewPacket({ ...v2, review: { ...v2.review, scope: REVIEW_SCOPE } }), /method or disclosure/);
  await assert.rejects(verifyReviewPacket({ ...v1, review: { ...v1.review, scope: REVIEW_SCOPE_V2 } }), /method or disclosure/);
  // Unknown or missing methods, including inherited property names.
  for (const method of ['literal-spans-v3', 'constructor', '__proto__', undefined, 7]) {
    await assert.rejects(verifyReviewPacket({ ...v2, review: { ...v2.review, method } }), /method or disclosure/);
  }
});

test('a review row with an added field is rejected', async () => {
  for (const method of ['literal-spans-v1', 'literal-spans-v2']) {
    const p = await methodPacket(method);
    p.review.rows[0].decided_at = '2026-09-27T00:00:00Z';
    await assert.rejects(verifyReviewPacket(p), /span or decision/);
    const q = await methodPacket(method);
    q.review.rows[0].evidence = [];
    await assert.rejects(verifyReviewPacket(q), /span or decision/);
  }
});

// A packet exported by the shipped literal-spans-v1 code (origin/main before
// literal-spans-v2), byte for byte. Its passages come from the v1 splitter,
// which reads "Dr." as a sentence end and "$7.95 per order." as a run-on.
const SHIPPED_V1_PACKET = "{\"schema\":\"sum.review_packet.v1\",\"created_at\":\"2026-09-26T12:00:00.000Z\",\"source\":{\"text\":\"Call Dr. Patel if a rash appears. Shipping costs $7.95 per order. Refunds follow.\",\"hash\":\"sha256-fc5d8d49488f8d99d985de10ff4ce84b94b650fbb97c64b9a7e2e5a383a16f03\"},\"output\":{\"text\":\"Call Dr. Patel if you get a rash. Shipping costs $7.95 per order.\",\"hash\":\"sha256-17ec232aefe502733c3cbc8ccf1b0b2307f1c7fa16c24e41e6f2bcaa110b105d\"},\"review\":{\"method\":\"literal-spans-v1\",\"offset_unit\":\"utf16-code-unit\",\"scope\":\"Literal sentence comparison with lexical suggestions. Paraphrases, entailment, factual accuracy and meaning preservation are not measured. Human decisions and the original source are unsigned.\",\"rows\":[{\"id\":\"source-s1\",\"kind\":\"verbatim\",\"source\":{\"id\":\"s1\",\"start\":0,\"end\":8,\"text\":\"Call Dr.\"},\"output\":{\"id\":\"s1\",\"start\":0,\"end\":8,\"text\":\"Call Dr.\"},\"decision\":\"needs-change\"},{\"id\":\"source-s2\",\"kind\":\"changed-candidate\",\"source\":{\"id\":\"s2\",\"start\":9,\"end\":33,\"text\":\"Patel if a rash appears.\"},\"output\":{\"id\":\"s2\",\"start\":9,\"end\":33,\"text\":\"Patel if you get a rash.\"},\"decision\":\"unreviewed\"},{\"id\":\"source-s3\",\"kind\":\"changed-candidate\",\"source\":{\"id\":\"s3\",\"start\":34,\"end\":81,\"text\":\"Shipping costs $7.95 per order. Refunds follow.\"},\"output\":{\"id\":\"s3\",\"start\":34,\"end\":65,\"text\":\"Shipping costs $7.95 per order.\"},\"decision\":\"unreviewed\"}]},\"render\":null,\"jwks\":null,\"disclosure\":\"Literal sentence comparison with lexical suggestions. Paraphrases, entailment, factual accuracy and meaning preservation are not measured. Human decisions and the original source are unsigned.\",\"verification_guide\":\"Open this packet in the SUM workbench packet verifier, or import verifyReviewPacket from single_file_demo/review_packet.js in Node. It recomputes text hashes and review spans. When a receipt and public keys are included, it verifies Ed25519 plus exact output, selected triples and slider bindings. The container, source, key ownership, review decisions and reviewer identity are not signed. Independently establish issuer key trust and revocation/freshness policy. No account or API key is needed. Keep the packet private if its source contains private material.\"}";

test('a packet exported by the shipped v1 code still verifies and still rejects tampering', async () => {
  const packet = JSON.parse(SHIPPED_V1_PACKET);
  assert.equal(packet.review.method, 'literal-spans-v1');
  const result = await verifyReviewPacket(packet);
  assert.equal(result.review_spans, 'recomputed');
  assert.equal(result.human_decisions, 'unsigned');
  const tampered = JSON.parse(SHIPPED_V1_PACKET);
  tampered.review.rows[1].source.end--;
  await assert.rejects(verifyReviewPacket(tampered), /span or decision/);
  const decided = JSON.parse(SHIPPED_V1_PACKET);
  decided.review.rows[0].decision = 'approved';
  await assert.rejects(verifyReviewPacket(decided), /span or decision/);
});
