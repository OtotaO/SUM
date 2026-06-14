// Differential cross-runtime verdict driver (used by
// Tests/test_differential_cross_runtime_fuzz.py).
//
// Reads JSONL from stdin; each line is {"receipt":..,"jwks":..,"schema":..}.
// Emits one JSON line per input describing this runtime's verdict:
//   {"v":true}                       -> verified
//   {"v":false,"c":"<errorClass>"}   -> rejected with a declared VerifyError class
//   {"crash":"<ExceptionType>"}      -> threw something UNDECLARED (a real crash)
//
// Pair with the Python verifier verdicts to assert the two runtimes agree on
// accept/reject (the "verify anywhere, get the same answer" contract) and never
// crash on adversarial input.
import { createInterface } from "node:readline";

const FAMILY = process.argv[2];
const here = new URL(".", import.meta.url);
const mod = {
  render: await import(new URL("receipt_verifier.js", here)),
  transform: await import(new URL("transform_receipt_verifier.js", here)),
  meaning: await import(new URL("meaning_receipt_verifier.js", here)),
}[FAMILY];
if (!mod) {
  process.stderr.write(`unknown family: ${FAMILY}\n`);
  process.exit(2);
}
const VerifyError = mod.VerifyError;

async function verify(rec, jwks, schema) {
  if (FAMILY === "render") return mod.verifyReceipt(rec, jwks);
  if (FAMILY === "transform") return mod.verifyTransformReceipt(rec, jwks);
  return mod.verifyMeaningEnvelope(rec, jwks, schema || mod.MEANING_RISK_SCHEMA);
}

const rl = createInterface({ input: process.stdin });
const out = [];
for await (const line of rl) {
  if (!line) continue;
  let obj;
  try {
    obj = JSON.parse(line);
  } catch {
    out.push(JSON.stringify({ crash: "InputParse" }));
    continue;
  }
  try {
    const res = await verify(obj.receipt, obj.jwks, obj.schema);
    out.push(JSON.stringify({ v: res && res.verified === true }));
  } catch (e) {
    if (VerifyError && e instanceof VerifyError) {
      out.push(JSON.stringify({ v: false, c: e.errorClass }));
    } else {
      out.push(JSON.stringify({ crash: (e && e.constructor && e.constructor.name) || String(e) }));
    }
  }
}
process.stdout.write(out.join("\n"));
