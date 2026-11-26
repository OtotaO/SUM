// Cross-runtime gate for the LLM system prompt.
//
// WHY THIS FILE EXISTS. `buildSystemPrompt` in src/render/axis_prompts.ts is
// the Worker twin of Python's `build_system_prompt`
// (sum_engine_internal/ensemble/tome_sliders.py), and that file's header asks
// for the two prompts to stay equal byte for byte. Nothing checked it: a fragment or
// the fact-preservation clause edited on one side only would change what the
// model is told on that side alone, and tsc cannot see a changed string.
//
// The fixture has one row per quantized combination of the four prompt axes
// (5 bins each, 625 rows): the slider values and the first 16 hex characters
// of sha256 over the UTF-8 Python prompt. It is generated FROM the Python
// reference by test/generate_prompt_fixture.py, and Tests/test_tome_sliders.py
// fails if the committed file no longer matches what Python produces.
//
// Run: node --experimental-strip-types test/prompt_smoke.mjs
//      (or `npm run test:prompts`)

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { register } from "node:module";
register("./ts_resolve.mjs", import.meta.url);
const { buildSystemPrompt } = await import("../src/render/axis_prompts.ts");

const FIXTURE_URL = new URL("./fixtures/system_prompt_digests.json", import.meta.url);
const fixture = JSON.parse(readFileSync(FIXTURE_URL, "utf8"));

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

// Same digest as generate_prompt_fixture.py::prompt_digest. The prompt holds a
// non-ASCII dash, so the UTF-8 encoding is stated on both sides.
const digest = (text) => createHash("sha256").update(text, "utf8").digest("hex").slice(0, 16);

console.log("worker system prompt cross-runtime test");
console.log(`  fixture: ${fixture.schema} (reference: ${fixture.reference_runtime})`);

check("fixture axes and bin centres are the ones the Worker quantizes to", () => {
  assert.deepEqual(fixture.axes, ["length", "formality", "audience", "perspective"]);
  // render.ts snapToBin with 5 bins returns (idx + 0.5) / 5.
  assert.deepEqual(fixture.bin_centres, [0, 1, 2, 3, 4].map((i) => (i + 0.5) / 5));
});

check("fixture has all 625 combinations, each a distinct prompt", () => {
  assert.equal(fixture.rows.length, 625);
  assert.equal(new Set(fixture.rows.map((row) => row.slice(0, 4).join(","))).size, 625);
  assert.equal(new Set(fixture.rows.map((row) => row[4])).size, 625);
});

const mismatches = [];
for (const row of fixture.rows) {
  const sliders = { density: fixture.density };
  fixture.axes.forEach((axis, i) => { sliders[axis] = row[i]; });
  const worker = digest(buildSystemPrompt(sliders));
  if (worker !== row[4]) mismatches.push(`${JSON.stringify(sliders)} worker ${worker} python ${row[4]}`);
}
check(`every system prompt matches Python byte for byte (${fixture.rows.length} rows)`, () => {
  assert.equal(mismatches.length, 0, `${mismatches.length} differ; first: ${mismatches.slice(0, 3).join("; ")}`);
});

console.log(
  failures === 0 ? "\nALL SYSTEM PROMPT CROSS-RUNTIME CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`,
);
process.exit(failures === 0 ? 0 : 1);
