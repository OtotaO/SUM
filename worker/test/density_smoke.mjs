// Cross-runtime gate for the density slider's axiom subsetting.
//
// WHY THIS FILE EXISTS. `applyDensity` in src/render/axis_prompts.ts is the
// Worker twin of Python's `apply_density`
// (sum_engine_internal/ensemble/tome_sliders.py, driven through
// slider_renderer.py::_axiom_key). Both sort the `s||p||o` axiom keys and
// keep the leading floor(N * density). Python sorts with the builtin
// `sorted()`, which compares by Unicode codepoint. The Worker sorted with
// `String.prototype.localeCompare`, which compares by ICU collation. Those
// are different orders, so the two runtimes kept DIFFERENT subsets of the
// same triples at the same density, and the surviving subset is what the
// deterministic tome is built from, whose hash is a signed receipt field
// (`tome_hash` in routes/render.ts). Same input, same sliders, two hashes.
//
// This is the same class of leak that src/receipt/sign.ts::hashTriples was
// fixed for in the v0.9.A.1 review pass: a JS-native string comparison that
// looks like Python's but is not.
//
// The fixture is generated FROM the Python reference; see
// fixtures/density_sort/generate_fixture.py.
//
// Run: node --experimental-strip-types test/density_smoke.mjs
//      (or `npm run test:density`)

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { applyDensity } from "../src/render/axis_prompts.ts";

const FIXTURE_URL = new URL(
  "../../fixtures/density_sort/apply_density_cross_runtime_v1.json",
  import.meta.url,
);
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

// Mirrors slider_renderer.py::_axiom_key and axis_prompts.ts::keyOf.
const keyOf = (t) => `${t[0]}||${t[1]}||${t[2]}`;

console.log("worker density cross-runtime test");
console.log(`  fixture: ${fixture.schema} (reference: ${fixture.reference_runtime})`);

const triples = fixture.triples.map((t) => [t[0], t[1], t[2]]);

check("fixture carries at least two density cases", () => {
  assert.ok(Array.isArray(fixture.cases));
  assert.ok(fixture.cases.length >= 2);
});

for (const { density, expected_kept_keys: expected } of fixture.cases) {
  check(`density ${density} keeps the Python subset, in Python order`, () => {
    const kept = applyDensity(triples, density).map(keyOf);
    assert.deepEqual(
      kept,
      expected,
      `\n       worker: ${JSON.stringify(kept)}` +
        `\n       python: ${JSON.stringify(expected)}`,
    );
  });
}

// Order-independent restatement of the same guard: even where the two
// comparators agree on the COUNT, they must agree on the membership.
for (const { density, expected_kept_keys: expected } of fixture.cases) {
  check(`density ${density} keeps the Python subset as a set`, () => {
    const kept = new Set(applyDensity(triples, density).map(keyOf));
    const want = new Set(expected);
    assert.equal(kept.size, want.size, `kept ${kept.size} keys, expected ${want.size}`);
    for (const k of want) {
      assert.ok(kept.has(k), `missing ${JSON.stringify(k)}; got ${JSON.stringify([...kept])}`);
    }
  });
}

// Boundary behaviour the Python twin also guarantees.
check("empty input yields empty output", () => {
  assert.deepEqual(applyDensity([], 0.5), []);
});
check("density 0 yields empty output", () => {
  assert.deepEqual(applyDensity(triples, 0.0), []);
});

console.log(
  failures === 0 ? "\nALL DENSITY CROSS-RUNTIME CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`,
);
process.exit(failures === 0 ? 0 : 1);
