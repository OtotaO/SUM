// Node smoke test for the browser transform-receipt verifier against
// the runtime-neutral fixture set under fixtures/transform_receipts/.
//
// Iterates each fixture, runs verifyTransformReceipt, asserts
// expected_outcome + expected_error_class match. The same fixtures
// are consumed by the Python verifier in
// Tests/test_transform_receipt_verifier_fixtures.py — byte-identical
// outcomes across both runtimes is the K-style cross-runtime
// equivalence guarantee, extended to the transform substrate.
//
// Run:
//   node single_file_demo/test_transform_receipt_fixtures.js
// Exit code: 0 on all-pass, 1 on any failure.

import { readFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { verifyTransformReceipt, VerifyError } from "./transform_receipt_verifier.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURES_DIR = join(__dirname, "..", "fixtures", "transform_receipts");

// Skip generator inputs/outputs; only iterate derived fixtures.
const SKIP_FILES = new Set(["source_receipt.json", "jwks_at_capture.json"]);

const fixtureFiles = readdirSync(FIXTURES_DIR)
  .filter((f) => f.endsWith(".json") && !SKIP_FILES.has(f))
  .sort();

let pass = 0;
let fail = 0;
const failures = [];

console.log(`Browser transform-receipt verifier smoke test`);
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
  } = fx;

  try {
    await verifyTransformReceipt(receipt, jwks);
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
