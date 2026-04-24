#!/usr/bin/env node
/**
 * test_wasm.js — proves the committed sum_core.wasm produces primes
 * byte-identical to the Python / Node / browser reference.
 *
 * Runs in Node (no browser needed); instantiates the WASM from disk,
 * calls sum_get_deterministic_prime on the cross-runtime fixture set,
 * asserts every result matches the authoritative values from the
 * standalone verifier's self-test (standalone_verifier/verify.js
 * --self-test vectors 3).
 *
 * If this ever fails, either the .wasm committed in this directory
 * drifted from a freshly-built core-zig/ (rebuild: `cd core-zig &&
 * zig build wasm && cp zig-out/bin/sum_core.wasm ../single_file_demo/`),
 * or the prime-derivation contract changed on one side. Either way,
 * fix the cause, don't mask the symptom.
 *
 * Zero npm dependencies. Part of the demo's self-test suite
 * alongside test_jcs.js and test_provenance.js.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const WASM_PATH = path.join(__dirname, 'sum_core.wasm');

// Fixture set: each row is [axiom_key, expected_prime_decimal]. The
// expected values come from verify.js --self-test (vectors 3) and are
// the ground truth for Python ↔ Node ↔ Browser ↔ WASM agreement.
const FIXTURES = [
  ['alice||likes||cats',         '14326936561644797201'],
  ['bob||knows||python',         '12933559861697884259'],
  ['earth||orbits||sun',         '10246101339925224733'],
];

async function main() {
  const bytes = fs.readFileSync(WASM_PATH);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const exp = instance.exports;

  // Sanity check the exported surface before running fixtures.
  for (const name of ['memory', 'wasm_alloc_bytes', 'wasm_free_bytes', 'sum_get_deterministic_prime']) {
    if (!(name in exp)) {
      console.error(`  ❌ missing WASM export: ${name}`);
      process.exit(1);
    }
  }

  let passed = 0;
  let failed = 0;

  for (const [key, expected] of FIXTURES) {
    const encoded = new TextEncoder().encode(key);
    const ptr = exp.wasm_alloc_bytes(encoded.length);
    if (!ptr) {
      console.error(`  ❌ ${key}: wasm_alloc_bytes returned null`);
      failed++;
      continue;
    }
    try {
      new Uint8Array(exp.memory.buffer, ptr, encoded.length).set(encoded);
      const raw = exp.sum_get_deterministic_prime(ptr, encoded.length);
      // u64 reinterpret — see sum_core_wasm.js for rationale.
      const big = ((typeof raw === 'bigint' ? raw : BigInt(raw)) & 0xffffffffffffffffn);
      const got = big.toString();
      if (got === expected) {
        console.log(`  ✅ ${key} → ${got}`);
        passed++;
      } else {
        console.log(`  ❌ ${key} → ${got} (expected ${expected})`);
        failed++;
      }
    } finally {
      exp.wasm_free_bytes(ptr, encoded.length);
    }
  }

  console.log(`\nsum_core.wasm self-test: ${passed} passed, ${failed} failed`);
  process.exit(failed === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error('  ❌ unexpected error:', err);
  process.exit(2);
});
