// Node-runtime cross-runtime test for the meaning-risk + perspective
// receipt verifiers.
//
// The point: prove the Python-SIGNED golden meaning-risk and perspective
// receipts verify in Node — i.e. the cross-runtime claim is true for the
// NEW (differentiating) receipt family, not just render/transform. Plus
// tamper-rejection, schema/kid gates, and the disclosure invariant
// (exercised with a JS-signed, validly-signed but disclosure-free
// receipt, since a tampered receipt fails the signature first).
//
// Run: node single_file_demo/test_meaning_receipt_verify.js

import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import {
  verifyMeaningRiskReceipt,
  verifyPerspectiveReceipt,
  verifyMeaningEnvelope,
  MEANING_RISK_SCHEMA,
  ERROR_CLASSES,
} from "./meaning_receipt_verifier.js";
import { canonicalize } from "./vendor/sum-verify-deps.js";

let passed = 0;
function ok(cond, msg) {
  if (!cond) throw new Error("FAIL: " + msg);
  passed++;
}
async function rejects(fn, errorClass, msg) {
  try {
    await fn();
  } catch (e) {
    ok(e.errorClass === errorClass, `${msg} (got ${e.errorClass}, want ${errorClass})`);
    return;
  }
  throw new Error("FAIL: expected rejection: " + msg);
}

const root = fileURLToPath(new URL("..", import.meta.url));
const load = (p) => JSON.parse(readFileSync(root + p, "utf8"));

// ── 1. the Python-signed goldens verify in Node (the headline claim) ──
const mGolden = load("fixtures/meaning_receipts/meaning_risk_receipt.golden.json");
const mJwks = load("fixtures/meaning_receipts/jwks.json");
const m = await verifyMeaningRiskReceipt(mGolden, mJwks);
ok(m.verified === true, "meaning-risk golden verifies in Node");
ok(m.payload.risk_upper_bound_micro === 654992, "meaning payload intact");

const pGolden = load("fixtures/perspective_receipts/perspective_risk_receipt.golden.json");
const pJwks = load("fixtures/perspective_receipts/jwks.json");
const p = await verifyPerspectiveReceipt(pGolden, pJwks);
ok(p.verified === true, "perspective golden verifies in Node");
ok(Array.isArray(p.payload.groups) && p.payload.groups.length === 2, "perspective cohorts intact");
ok(p.payload.controls_all === false, "perspective controls_all preserved");

// ── 2. tamper → signature_invalid (the bytes change) ──
const mBad = JSON.parse(JSON.stringify(mGolden));
mBad.payload.risk_upper_bound_micro = 0;
await rejects(() => verifyMeaningRiskReceipt(mBad, mJwks), ERROR_CLASSES.SIGNATURE_INVALID, "tampered meaning bound");

const pBad = JSON.parse(JSON.stringify(pGolden));
pBad.payload.controls_all = true;
await rejects(() => verifyPerspectiveReceipt(pBad, pJwks), ERROR_CLASSES.SIGNATURE_INVALID, "forged perspective controls_all");

// ── 3. schema + kid gates ──
await rejects(() => verifyPerspectiveReceipt(mGolden, mJwks), ERROR_CLASSES.SCHEMA_UNKNOWN, "wrong-schema verifier");
await rejects(() => verifyMeaningRiskReceipt(mGolden, { keys: [] }), ERROR_CLASSES.UNKNOWN_KID, "empty JWKS");

// ── 4. disclosure invariant — JS-sign a validly-signed disclosure-free receipt ──
// (a tampered receipt fails the signature first, so we sign one.)
const key = await crypto.subtle.generateKey({ name: "Ed25519" }, true, ["sign", "verify"]);
const pubJwk = await crypto.subtle.exportKey("jwk", key.publicKey);
const kid = "js-test-key";
pubJwk.kid = kid; pubJwk.alg = "EdDSA"; pubJwk.use = "sig";
const jwks = { keys: [pubJwk] };

async function signMeaning(payload) {
  const protectedHeader = { alg: "EdDSA", kid, b64: false, crit: ["b64"] };
  const protB64 = Buffer.from(JSON.stringify(protectedHeader)).toString("base64url");
  const payloadBytes = new TextEncoder().encode(canonicalize(payload));
  const protBytes = new TextEncoder().encode(protB64 + ".");
  const signingInput = new Uint8Array(protBytes.length + payloadBytes.length);
  signingInput.set(protBytes, 0);
  signingInput.set(payloadBytes, protBytes.length);
  const sig = await crypto.subtle.sign({ name: "Ed25519" }, key.privateKey, signingInput);
  const sigB64 = Buffer.from(new Uint8Array(sig)).toString("base64url");
  return { schema: MEANING_RISK_SCHEMA, kid, payload, jws: `${protB64}..${sigB64}` };
}

const base = {
  scorer: "x", scorer_version: "1", loss_definition: "d", n: 2, method: "hoeffding",
  delta_micro: 50000, point_estimate_micro: 100000, risk_upper_bound_micro: 200000,
  losses_hash: "sha256-deadbeef", corpus_id: "c", transform: "t",
  not_covered: ["arrangement"], disclosure: "the proxy caveat",
  signed_at: "2026-06-07T00:00:00.000Z",
};
// well-formed → verifies
const good = await signMeaning(base);
ok((await verifyMeaningRiskReceipt(good, jwks)).verified === true, "JS-signed well-formed verifies");
// disclosure-free (validly signed) → disclosure_missing
const noNc = await signMeaning({ ...base, not_covered: [] });
await rejects(() => verifyMeaningRiskReceipt(noNc, jwks), ERROR_CLASSES.DISCLOSURE_MISSING, "empty not_covered");
const noDisc = await signMeaning({ ...base, disclosure: "   " });
await rejects(() => verifyMeaningRiskReceipt(noDisc, jwks), ERROR_CLASSES.DISCLOSURE_MISSING, "blank disclosure");

console.log(`meaning/perspective receipt verifier: ${passed} checks passed (Python-signed goldens verify in Node)`);
