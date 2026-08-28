// Runtime smoke test for the Worker's two receipt signers.
//
// WHY THIS FILE EXISTS. On 2026-07-31 PR #424 bumped jose 6.2.4 -> 6.2.5 in the
// worker-toolchain group. jose 6.2.5 made `CompactSign` reject a protected
// header carrying `b64: false`:
//
//     TypeError: use the flattened module for creating JWS with b64: false
//
// Both signReceipt() and signTransformReceipt() build exactly that header,
// because SUM render/transform receipts use a DETACHED payload. So from that
// merge onward every signing call on main threw. Nothing caught it: worker/ had
// no runtime tests at all, and the `worker-typecheck` job added in #441 runs
// `tsc --noEmit`, which passes because the header type is still legal. The
// live Worker kept working only because it had not been redeployed since before
// the bump.
//
// This test executes the real signers and verifies the result. It fails on the
// jose regression, and it fails on any change that alters the signed bytes.
//
// Run: node --experimental-strip-types test/sign_smoke.mjs
//      (or `npm run test:sign`)

import assert from "node:assert/strict";
import { generateKeyPair, exportJWK, flattenedVerify, importJWK } from "jose";
import canonicalize from "canonicalize";

import { signReceipt } from "../src/receipt/sign.ts";
import { signTransformReceipt } from "../src/receipt/transform_sign.ts";

const KID = "sign-smoke-test-key";
const { publicKey, privateKey } = await generateKeyPair("EdDSA", {
  crv: "Ed25519",
  extractable: true,
});
const privJwk = await exportJWK(privateKey);
const pubJwk = await exportJWK(publicKey);

let failures = 0;
function check(name, fn) {
  try {
    fn();
    console.log(`  ok   ${name}`);
  } catch (err) {
    failures += 1;
    console.log(`  FAIL ${name}\n       ${err.message}`);
  }
}

// A verifier that mirrors what a third party actually does: re-canonicalise the
// payload it received, then flattenedVerify the detached JWS against the JWK
// from /.well-known/jwks.json.
async function verifyDetached(jws, payload, jwk) {
  const parts = jws.split(".");
  assert.equal(parts.length, 3, `expected 3 JWS segments, got ${parts.length}`);
  assert.equal(parts[1], "", "middle segment must be empty (payload is detached)");
  const key = await importJWK(jwk, "EdDSA");
  const canonical = canonicalize(payload);
  assert.equal(typeof canonical, "string", "canonicalize returned undefined");
  return flattenedVerify(
    {
      protected: parts[0],
      payload: new TextEncoder().encode(canonical),
      signature: parts[2],
    },
    key,
  );
}

console.log("worker signer smoke test");

// ---- render receipt --------------------------------------------------------
const renderPayload = {
  schema_version: 1,
  tome_hash: "sha256:0000000000000000000000000000000000000000000000000000000000000000",
  triples_hash: "sha256:1111111111111111111111111111111111111111111111111111111111111111",
  density: 0.5,
  signed_at: "2026-08-28T00:00:00.000Z",
};

let renderReceipt;
try {
  renderReceipt = await signReceipt(renderPayload, privJwk, KID);
  console.log("  ok   signReceipt() did not throw");
} catch (err) {
  failures += 1;
  console.log(`  FAIL signReceipt() threw: ${err.message}`);
  console.log("       This is the jose >= 6.2.5 CompactSign/b64:false regression.");
}

if (renderReceipt) {
  check("render receipt carries kid", () => assert.equal(renderReceipt.kid, KID));
  check("render jws is a non-empty string", () => {
    assert.equal(typeof renderReceipt.jws, "string");
    assert.ok(renderReceipt.jws.length > 0);
  });
  check("render jws is detached (empty middle segment)", () => {
    const parts = renderReceipt.jws.split(".");
    assert.equal(parts.length, 3);
    assert.equal(parts[1], "");
  });
  check("render protected header is exactly the documented shape", () => {
    const hdr = JSON.parse(
      Buffer.from(renderReceipt.jws.split(".")[0], "base64url").toString("utf8"),
    );
    assert.deepEqual(hdr, { alg: "EdDSA", kid: KID, b64: false, crit: ["b64"] });
  });
  await (async () => {
    try {
      await verifyDetached(renderReceipt.jws, renderReceipt.payload, pubJwk);
      console.log("  ok   render receipt verifies against its public JWK");
    } catch (err) {
      failures += 1;
      console.log(`  FAIL render receipt did not verify: ${err.message}`);
    }
  })();
  await (async () => {
    const tampered = { ...renderReceipt.payload, density: 0.9 };
    try {
      await verifyDetached(renderReceipt.jws, tampered, pubJwk);
      failures += 1;
      console.log("  FAIL tampered render payload VERIFIED (must not)");
    } catch {
      console.log("  ok   tampered render payload is rejected");
    }
  })();
}

// ---- transform receipt -----------------------------------------------------
const transformPayload = {
  schema_version: 1,
  transform: "slider",
  input_hash: "sha256:2222222222222222222222222222222222222222222222222222222222222222",
  output_hash: "sha256:3333333333333333333333333333333333333333333333333333333333333333",
  parameters_hash: "sha256:4444444444444444444444444444444444444444444444444444444444444444",
  signed_at: "2026-08-28T00:00:00.000Z",
};

let transformReceipt;
try {
  transformReceipt = await signTransformReceipt(transformPayload, privJwk, KID);
  console.log("  ok   signTransformReceipt() did not throw");
} catch (err) {
  failures += 1;
  console.log(`  FAIL signTransformReceipt() threw: ${err.message}`);
  console.log("       This is the jose >= 6.2.5 CompactSign/b64:false regression.");
}

if (transformReceipt) {
  const jws = transformReceipt.jws;
  check("transform jws is detached (empty middle segment)", () => {
    const parts = String(jws).split(".");
    assert.equal(parts.length, 3);
    assert.equal(parts[1], "");
  });
  await (async () => {
    try {
      await verifyDetached(String(jws), transformReceipt.payload, pubJwk);
      console.log("  ok   transform receipt verifies against its public JWK");
    } catch (err) {
      failures += 1;
      console.log(`  FAIL transform receipt did not verify: ${err.message}`);
    }
  })();
}

console.log(failures === 0 ? "\nALL SIGNER SMOKE CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
