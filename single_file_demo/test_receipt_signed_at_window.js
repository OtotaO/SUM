// Node smoke for the browser verifier's replay-window check.
//
// Mirrors Tests/test_receipt_signed_at_window.py — both runtimes
// must produce the same accept/reject outcomes given the same
// maxAgeSeconds / maxFutureSkewSeconds inputs.

import {
  verifyTransformReceipt,
  ERROR_CLASSES,
  VerifyError,
} from "./transform_receipt_verifier.js";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURE_PATH = join(
  __dirname,
  "..",
  "fixtures",
  "transform_receipts",
  "positive_control.json",
);

let pass = 0;
let fail = 0;

function ok(label) {
  pass++;
  console.log(`  ✓ ${label}`);
}
function nope(label, e) {
  fail++;
  console.log(`  ✗ ${label} — ${e}`);
}

async function main() {
  console.log("─── Replay-window check smoke (Node) ────────────────────────");
  const fx = JSON.parse(readFileSync(FIXTURE_PATH, "utf8"));
  const { receipt, jwks } = fx;

  // The positive_control fixture has signed_at = "2026-05-12T12:00:00.000Z"
  // — clearly more than 60 seconds old in any session that runs this
  // smoke after that date. So:
  //   maxAgeSeconds=null  → verify
  //   maxAgeSeconds=60    → reject (signed_at_out_of_window, "old")
  //   maxAgeSeconds=Inf   → verify
  // Fresh receipts and future-dated receipts are exercised in the
  // Python suite (Tests/test_receipt_signed_at_window.py) because
  // they require signing fresh receipts with a test key; this Node
  // smoke pins the static-fixture behaviour for cross-runtime parity.

  // 1) Default (no window) → verify
  try {
    const r = await verifyTransformReceipt(receipt, jwks);
    if (r.verified === true) ok("no window: fixture verifies");
    else nope("no window", `unexpected: ${JSON.stringify(r)}`);
  } catch (e) {
    nope("no window", e.message);
  }

  // 2) Short window → reject as too old
  try {
    await verifyTransformReceipt(receipt, jwks, { maxAgeSeconds: 60 });
    nope("short window: fixture rejected", "did not throw");
  } catch (e) {
    if (
      e instanceof VerifyError &&
      e.errorClass === ERROR_CLASSES.SIGNED_AT_OUT_OF_WINDOW
    ) {
      ok("short window: fixture rejected as signed_at_out_of_window");
    } else {
      nope(
        "short window",
        `wrong: ${e.errorClass || "uncategorised"}: ${e.message}`,
      );
    }
  }

  // 3) Very long window (1000 years) → verify
  const millennium = 1000 * 365 * 24 * 60 * 60;
  try {
    const r = await verifyTransformReceipt(receipt, jwks, {
      maxAgeSeconds: millennium,
    });
    if (r.verified === true) ok("millennium window: fixture verifies");
    else nope("millennium window", `unexpected: ${JSON.stringify(r)}`);
  } catch (e) {
    nope("millennium window", e.message);
  }

  console.log("");
  console.log(`Total: ${pass} pass, ${fail} fail`);
  if (fail > 0) process.exit(1);
}

main().catch((e) => {
  console.error("test harness crashed:", e);
  process.exit(1);
});
