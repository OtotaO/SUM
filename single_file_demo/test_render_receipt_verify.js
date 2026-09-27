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

import { verifyReceipt, VerifyError, parseInstantMs } from "./receipt_verifier.js";

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

// Legacy revokedKids input: the served document, malformed shapes and
// the same-second instant case (parity with the Python verifier tests).
{
  const active = JSON.parse(readFileSync(join(FIXTURES_DIR, "revoked_kid_active.json"), "utf8"));
  const { receipt, jwks } = active;
  const entry = active.revoked_kids[0];
  const cases = [
    ["served document", { schema: "sum.revoked_kids.v1", revoked: active.revoked_kids }, "reject"],
    ["string input", "sum-render-2026-04-27-1", "reject"],
    ["number input", 5, "reject"],
    ["unrecognised object", { entries: [] }, "reject"],
    ["other schema", { schema: "some.other.list", revoked: [] }, "reject"],
    ["null schema", { schema: null, revoked: [] }, "reject"],
    ["null entry", [null], "reject"],
    ["string entry", ["sum-render-2026-04-27-1"], "reject"],
    ["entry without kid", [{}], "reject"],
    ["entry with numeric kid", [{ kid: 5, effective_revocation_at: "2026-01-01T00:00:00Z" }], "reject"],
    ["entry with empty kid", [{ kid: "", effective_revocation_at: "2026-01-01T00:00:00Z" }], "reject"],
    ["same second, whole-second effective", [{ ...entry, effective_revocation_at: "2026-04-27T00:45:16Z" }], "reject"],
    ["effective one second later", [{ ...entry, effective_revocation_at: "2026-04-27T00:45:17Z" }], "verify"],
    ["bare empty list form (spec 6.1)", { revoked: [] }, "verify"],
    ["served empty document", { schema: "sum.revoked_kids.v1", revoked: [] }, "verify"],
  ];
  for (const [name, revoked, expected] of cases) {
    let outcome;
    let cls;
    try {
      await verifyReceipt(receipt, jwks, revoked);
      outcome = "verify";
    } catch (e) {
      outcome = e instanceof VerifyError ? "reject" : `raw ${e && e.name}`;
      cls = e && e.errorClass;
    }
    const ok = outcome === expected && (expected === "verify" || cls === "revoked_kid");
    console.log(`  ${ok ? "✓" : "✗"} revokedKids: ${name} (${outcome}${cls ? ` ${cls}` : ""})`);
    if (ok) pass++;
    else {
      fail++;
      failures.push({ name: `revokedKids: ${name}`, kind: "revocation input", expected, actual: `${outcome} ${cls || ""}` });
    }
  }
}

// Shared instant grammar: the same table the Python verifier test runs.
{
  const table = JSON.parse(
    readFileSync(join(__dirname, "..", "Tests", "fixtures", "revocation_instants.json"), "utf8"),
  );
  const mismatches = table.cases.filter((c) => parseInstantMs(c.value) !== c.expected_ms);
  const ok = mismatches.length === 0;
  console.log(`  ${ok ? "✓" : "✗"} instant grammar: ${table.cases.length - mismatches.length}/${table.cases.length} shared cases`);
  if (ok) pass++;
  else {
    fail++;
    failures.push({ name: "instant grammar", kind: "parse mismatch", actual: JSON.stringify(mismatches.map((c) => c.value)) });
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
