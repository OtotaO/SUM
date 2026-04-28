// Node smoke test for the v0.9.B render-receipt verifier.
//
// Iterates the receipt fixtures under fixtures/render_receipts/,
// runs verifyReceipt on each, asserts expected_outcome +
// expected_error_class match. Same fixture set will be consumed
// by the v0.9.C Python verifier; identical assertions across
// runtimes are the cross-runtime equivalence the K-style harness
// gives us for CanonicalBundle, applied to render receipts.
//
// Run:
//   node single_file_demo/test_render_receipt_verify.js
// Exit code: 0 on all-pass, 1 on any failure.
//
// This test exists alongside test_jcs.js / test_provenance.js /
// test_wasm.js to keep the single_file_demo's test triad pattern.

import { readFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { verifyReceipt, VerifyError } from "./receipt_verifier.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = join(__dirname, "..", "fixtures", "render_receipts");

// Skip the inputs the generator consumes; only iterate generated
// fixtures + the positive control.
const SKIP_FILES = new Set(["source_render.json", "jwks_at_capture.json"]);

const fixtureFiles = readdirSync(FIXTURES_DIR)
  .filter((f) => f.endsWith(".json") && !SKIP_FILES.has(f))
  .sort();

let pass = 0;
let fail = 0;
const failures = [];

console.log(`v0.9.B browser receipt-verifier smoke test`);
console.log(`fixtures: ${FIXTURES_DIR}`);
console.log(`count:    ${fixtureFiles.length}\n`);

for (const filename of fixtureFiles) {
  const fx = JSON.parse(readFileSync(join(FIXTURES_DIR, filename), "utf8"));
  const {
    name,
    expected_outcome,
    expected_error_class,
    receipt,
    jwks,
    revoked_kids,  // optional G3 revocation list
  } = fx;

  try {
    await verifyReceipt(receipt, jwks, revoked_kids);
    if (expected_outcome === "verify") {
      console.log(`  ✓ ${name} — verified as expected`);
      pass++;
    } else {
      console.log(
        `  ✗ ${name} — expected reject (${expected_error_class}), got verify`,
      );
      fail++;
      failures.push({ name, kind: "unexpectedly verified" });
    }
  } catch (e) {
    if (expected_outcome === "reject") {
      const actualClass =
        e instanceof VerifyError ? e.errorClass : "uncategorized_error";
      if (actualClass === expected_error_class) {
        console.log(`  ✓ ${name} — rejected with ${actualClass}`);
        pass++;
      } else {
        console.log(
          `  ✗ ${name} — expected ${expected_error_class}, got ${actualClass}: ${e.message}`,
        );
        fail++;
        failures.push({
          name,
          kind: "wrong error class",
          expected: expected_error_class,
          actual: actualClass,
          message: e.message,
        });
      }
    } else {
      console.log(`  ✗ ${name} — expected verify, got reject: ${e.message}`);
      fail++;
      failures.push({
        name,
        kind: "unexpectedly rejected",
        message: e.message,
      });
    }
  }
}

console.log(`\n${pass}/${pass + fail} fixtures passed`);
if (fail > 0) {
  console.error(`\n${fail} failure(s):`);
  for (const f of failures) {
    console.error(`  - ${f.name}: ${f.kind}`);
    if (f.expected) console.error(`      expected: ${f.expected}`);
    if (f.actual) console.error(`      actual:   ${f.actual}`);
    if (f.message) console.error(`      message:  ${f.message}`);
  }
  process.exit(1);
}
