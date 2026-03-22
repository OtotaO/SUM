#!/usr/bin/env node
/**
 * Independent Semantic Witness — Phase 16
 *
 * Standalone Node.js verifier that consumes a Phase 15 Canonical ABI bundle,
 * independently reconstructs the Gödel State Integer from canonical tome lines,
 * and proves: JS_Reconstructed_State === Python_Exported_State
 *
 * This proves the Gödel Integer is NOT a Python-specific artifact.
 *
 * Usage:
 *   node verify.js <bundle.json>
 *   node verify.js --self-test
 *
 * Requirements: Node.js 16+ (BigInt + crypto built-ins). Zero npm dependencies.
 *
 * Honesty notes:
 *   - This verifies cross-runtime SEMANTIC STATE EQUIVALENCE.
 *   - HMAC signatures are NOT verified here (shared-secret, not public witness).
 *   - Collision resolution: Python uses sympy.nextprime with a while-loop for
 *     collisions. This verifier implements the same deterministic derivation.
 *     In practice, SHA-256 collision on 8-byte prefix is astronomically rare.
 *
 * Author: ototao
 * License: Apache License 2.0
 */

'use strict';

const fs = require('fs');
const crypto = require('crypto');

// ─── BigInt Math Utilities ─────────────────────────────────────────

/**
 * Modular exponentiation: (base^exp) % mod using BigInt.
 */
function modPow(base, exp, mod) {
  base = ((base % mod) + mod) % mod;
  let result = 1n;
  while (exp > 0n) {
    if (exp & 1n) {
      result = (result * base) % mod;
    }
    exp >>= 1n;
    base = (base * base) % mod;
  }
  return result;
}

/**
 * Deterministic Miller-Rabin primality test.
 *
 * For n < 3,317,044,064,679,887,385,961,981 the witness set
 * {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37} is provably sufficient
 * for a deterministic (zero-error) primality test.
 *
 * All our SHA-256-derived seeds are 64-bit, well within this bound.
 */
function isPrime(n) {
  if (n < 2n) return false;
  if (n < 4n) return true;
  if (n % 2n === 0n || n % 3n === 0n) return false;

  // Small primes fast-path
  const smallPrimes = [5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n, 41n, 43n];
  for (const sp of smallPrimes) {
    if (n === sp) return true;
    if (n % sp === 0n) return false;
  }

  // Write n-1 as 2^r * d
  let d = n - 1n;
  let r = 0n;
  while (d % 2n === 0n) {
    d >>= 1n;
    r++;
  }

  // Deterministic witnesses for n < 3.3e24
  const witnesses = [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n];

  for (const a of witnesses) {
    if (a >= n) continue;

    let x = modPow(a, d, n);
    if (x === 1n || x === n - 1n) continue;

    let composite = true;
    for (let i = 0n; i < r - 1n; i++) {
      x = (x * x) % n;
      if (x === n - 1n) {
        composite = false;
        break;
      }
    }
    if (composite) return false;
  }
  return true;
}

/**
 * Find the next prime >= n+1.
 * Mirrors Python's sympy.nextprime(n).
 */
function nextPrime(n) {
  if (n < 2n) return 2n;
  let candidate = n + 1n;
  if (candidate % 2n === 0n) candidate++;
  while (!isPrime(candidate)) {
    candidate += 2n;
  }
  return candidate;
}

/**
 * GCD of two BigInts.
 */
function gcd(a, b) {
  a = a < 0n ? -a : a;
  b = b < 0n ? -b : b;
  while (b > 0n) {
    [a, b] = [b, a % b];
  }
  return a;
}

/**
 * LCM of two BigInts.
 */
function lcm(a, b) {
  if (a === 0n || b === 0n) return 0n;
  return (a / gcd(a, b)) * b;
}

// ─── Canonical ABI Parser ──────────────────────────────────────────

/**
 * Derive a deterministic prime for an axiom key.
 *
 * Mirrors Python's GodelStateAlgebra._deterministic_prime():
 *   SHA-256(axiom_key) → first 8 bytes big-endian → seed → nextprime(seed)
 */
function derivePrime(axiomKey) {
  const hash = crypto.createHash('sha256').update(axiomKey, 'utf-8').digest();
  const seedHex = hash.subarray(0, 8).toString('hex');
  const seed = BigInt('0x' + seedHex);
  return nextPrime(seed);
}

/**
 * Parse canonical tome lines and extract axiom keys.
 *
 * Canonical format (v1.0.0):
 *   @canonical_version: 1.0.0
 *   # Title
 *   ## Subject
 *   The {subject} {predicate} {object}.
 *
 * Only lines matching "The {s} {p} {o}." are extracted.
 */
function parseCanonicalTome(tomeText) {
  const axioms = [];
  const lines = tomeText.split('\n');

  for (const line of lines) {
    const trimmed = line.trim();
    // Match canonical fact lines: "The subject predicate object."
    const match = trimmed.match(/^The\s+(\S+)\s+(\S+)\s+(\S+)\.$/);
    if (match) {
      const [, subject, predicate, object] = match;
      const axiomKey = `${subject.toLowerCase()}||${predicate.toLowerCase()}||${object.toLowerCase()}`;
      axioms.push(axiomKey);
    }
  }

  return axioms;
}

/**
 * Reconstruct the Gödel State Integer from a list of axiom keys.
 */
function reconstructState(axiomKeys) {
  let state = 1n;
  const primeMap = new Map(); // axiom → prime (for collision detection)

  for (const key of axiomKeys) {
    if (primeMap.has(key)) continue; // deduplicate

    let prime = derivePrime(key);

    // Collision resolution: if two different axioms produce the same prime,
    // advance to the next prime. Mirrors Python's while-loop.
    while (primeMap.has(prime.toString()) && primeMap.get(prime.toString()) !== key) {
      prime = nextPrime(prime);
    }

    primeMap.set(key, prime);
    primeMap.set(prime.toString(), key); // reverse map for collision check
    state = lcm(state, prime);
  }

  return state;
}

// ─── Self-Test Mode ────────────────────────────────────────────────

function runSelfTest() {
  console.log('═══════════════════════════════════════════════════════════');
  console.log('  Independent Semantic Witness — Self-Test');
  console.log('═══════════════════════════════════════════════════════════\n');

  let passed = 0;
  let failed = 0;

  function assert(label, actual, expected) {
    if (actual === expected) {
      console.log(`  ✅ ${label}`);
      passed++;
    } else {
      console.log(`  ❌ ${label}`);
      console.log(`     Expected: ${expected}`);
      console.log(`     Actual:   ${actual}`);
      failed++;
    }
  }

  // Vector 1: Canonical line parsing
  console.log('── Canonical Line Parsing ──');
  const axioms = parseCanonicalTome('The alice likes cats.');
  assert('Parse "The alice likes cats."', axioms[0], 'alice||likes||cats');
  assert('Parse count', axioms.length, 1);

  // Vector 2: SHA-256 seed derivation
  console.log('\n── SHA-256 Seed Derivation ──');
  const hash = crypto.createHash('sha256').update('alice||likes||cats', 'utf-8').digest();
  const seedHex = hash.subarray(0, 8).toString('hex');
  assert('SHA-256 first 8 bytes', seedHex, 'c6d380e53c64fca9');
  const seed = BigInt('0x' + seedHex);
  assert('Seed value', seed.toString(), '14326936561644797097');

  // Vector 3: Prime derivation
  console.log('\n── Prime Derivation ──');
  const prime1 = derivePrime('alice||likes||cats');
  assert('Prime for alice||likes||cats', prime1.toString(), '14326936561644797201');

  const prime2 = derivePrime('bob||knows||python');
  assert('Prime for bob||knows||python', prime2.toString(), '12933559861697884259');

  const prime3 = derivePrime('earth||orbits||sun');
  assert('Prime for earth||orbits||sun', prime3.toString(), '10246101339925224733');

  // Vector 4: Mini state reconstruction
  console.log('\n── State Reconstruction ──');
  const expectedState = BigInt('1898585074409907150524167558344558620554613878579045806247');
  const miniTome = [
    '@canonical_version: 1.0.0',
    '# Test Tome',
    '',
    '## Alice',
    '',
    'The alice likes cats.',
    '',
    '## Bob',
    '',
    'The bob knows python.',
    '',
    '## Earth',
    '',
    'The earth orbits sun.',
    '',
  ].join('\n');

  const parsedAxioms = parseCanonicalTome(miniTome);
  assert('Parsed axiom count', parsedAxioms.length, 3);

  const state = reconstructState(parsedAxioms);
  assert('Reconstructed state', state.toString(), expectedState.toString());
  assert('State digit count', state.toString().length, 58);

  // Summary
  console.log(`\n═══════════════════════════════════════════════════════════`);
  console.log(`  Self-Test: ${passed} passed, ${failed} failed`);
  console.log('═══════════════════════════════════════════════════════════');

  return failed === 0;
}

// ─── Bundle Verification ───────────────────────────────────────────

function verifyBundle(bundlePath) {
  console.log('═══════════════════════════════════════════════════════════');
  console.log('  Independent Semantic Witness — Phase 16');
  console.log('  Cross-Runtime Canonical ABI Verification');
  console.log('═══════════════════════════════════════════════════════════\n');

  // 1. Read bundle
  let bundleJson;
  try {
    bundleJson = fs.readFileSync(bundlePath, 'utf-8');
  } catch (e) {
    console.error(`❌ Cannot read bundle file: ${bundlePath}`);
    process.exit(2);
  }

  let bundle;
  try {
    bundle = JSON.parse(bundleJson);
  } catch (e) {
    console.error('❌ Invalid JSON in bundle file.');
    process.exit(2);
  }

  // 2. Validate bundle structure
  const required = ['canonical_tome', 'state_integer', 'canonical_format_version', 'bundle_version'];
  for (const field of required) {
    if (!(field in bundle)) {
      console.error(`❌ Missing required field: ${field}`);
      process.exit(2);
    }
  }

  console.log(`  Bundle Version:    ${bundle.bundle_version}`);
  console.log(`  Canonical Version: ${bundle.canonical_format_version}`);
  console.log(`  Branch:            ${bundle.branch || 'unknown'}`);
  console.log(`  Timestamp:         ${bundle.timestamp || 'unknown'}`);
  console.log(`  Is Delta:          ${bundle.is_delta || false}`);

  // 3. Version gate
  if (bundle.canonical_format_version !== '1.0.0') {
    console.error(`\n❌ Unsupported canonical format version: ${bundle.canonical_format_version}`);
    console.error('   This witness only supports version 1.0.0.');
    process.exit(3);
  }

  // 4. Parse canonical tome
  const axiomKeys = parseCanonicalTome(bundle.canonical_tome);
  console.log(`  Axioms Discovered: ${axiomKeys.length}`);
  if (bundle.axiom_count !== undefined) {
    console.log(`  Axioms Expected:   ${bundle.axiom_count}`);
  }

  // 5. Reconstruct state
  console.log('\n  Reconstructing Gödel State Integer...');
  const reconstructed = reconstructState(axiomKeys);
  const exported = BigInt(bundle.state_integer);

  const reconDigits = reconstructed.toString().length;
  const exportDigits = exported.toString().length;

  console.log(`  Exported Digits:       ${exportDigits}`);
  console.log(`  Reconstructed Digits:  ${reconDigits}`);

  // 6. Compare
  const match = reconstructed === exported;

  console.log('\n═══════════════════════════════════════════════════════════');
  if (match) {
    console.log('  ✅ WITNESS VERIFICATION PASSED');
    console.log('');
    console.log('  The JavaScript-reconstructed Gödel State Integer');
    console.log('  exactly matches the Python-exported state.');
    console.log('');
    console.log('  This proves:');
    console.log('  • Canonical semantic content is runtime-independent');
    console.log('  • Deterministic prime derivation reproduces across languages');
    console.log('  • The Gödel Integer is NOT a Python-specific artifact');
    console.log('');
    console.log('  Caveats:');
    console.log('  • HMAC signature NOT verified (shared-secret, not public witness)');
    console.log('  • Collision resolution assumed collision-free (SHA-256 8-byte prefix)');
  } else {
    console.log('  ❌ WITNESS VERIFICATION FAILED');
    console.log('');
    console.log('  The reconstructed state does NOT match the exported state.');
    console.log(`  Exported:      ${exported.toString().substring(0, 40)}...`);
    console.log(`  Reconstructed: ${reconstructed.toString().substring(0, 40)}...`);
  }
  console.log('═══════════════════════════════════════════════════════════');

  return match;
}

// ─── Main ──────────────────────────────────────────────────────────

function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log('Usage:');
    console.log('  node verify.js <bundle.json>   — Verify a canonical bundle');
    console.log('  node verify.js --self-test      — Run deterministic test vectors');
    process.exit(0);
  }

  if (args[0] === '--self-test') {
    const ok = runSelfTest();
    process.exit(ok ? 0 : 1);
  }

  const ok = verifyBundle(args[0]);
  process.exit(ok ? 0 : 1);
}

main();
