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

// ─── Shared primality + prime-derivation primitives ─────────────────
//
// All BigInt math and prime derivation moved to ./math so this
// verifier and single_file_demo/godel.js share a single source of
// truth. See standalone_verifier/math.js for the rationale and the
// full surface (modPow, gcd, lcm, sqrt, isPrime, nextPrime, the
// BPSW stack, derivePrime, derivePrimeV2).
//
// crypto.createHash stays imported at the top of this file because
// the self-test (runSelfTest) includes an explicit "SHA-256 first 8
// bytes equals c6d380e53c64fca9 for 'alice||likes||cats'" regression
// check that exercises the hashing primitive directly — separate from
// derivePrime's SHA-256 use, which is now inside math.js.
const {
  modPow,
  gcd,
  lcm,
  sqrt,
  isPrime,
  nextPrime,
  isPrimeBPSW,
  nextPrimeBPSW,
  jacobi,
  selfridgeParams,
  strongLucasTest,
  derivePrime,
  derivePrimeV2,
} = require('./math');

// ─── Canonical ABI Parser ──────────────────────────────────────────

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
    // Per CANONICAL_ABI_SPEC §3.2: subject and predicate are \S+ (single
    // token each); OBJECT is .+ (greedy) and MAY contain whitespace
    // ("nobel prizes", "computer algorithm", "printing press"). The
    // Python Ouroboros parser at internal/ensemble/ouroboros.py line 108
    // uses the same shape; this parser must match or cross-runtime
    // reconstruction drifts silently on any multi-word-object corpus.
    const match = trimmed.match(/^The\s+(\S+)\s+(\S+)\s+(.+)\.$/);
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
 *
 * @param {string[]} axiomKeys - Axiom keys to reconstruct from.
 * @param {string} scheme - Prime derivation scheme: 'sha256_64_v1' or 'sha256_128_v2'.
 */
function reconstructState(axiomKeys, scheme = 'sha256_64_v1') {
  // Select derivation function based on scheme
  let derive;
  let nextP;
  if (scheme === 'sha256_64_v1') {
    derive = derivePrime;
    nextP = nextPrime;
  } else if (scheme === 'sha256_128_v2') {
    derive = derivePrimeV2;
    nextP = nextPrimeBPSW;
  } else {
    throw new Error(`Unknown prime scheme: ${scheme}. Known schemes: sha256_64_v1, sha256_128_v2`);
  }

  let state = 1n;
  const primeMap = new Map(); // axiom → prime (for collision detection)

  for (const key of axiomKeys) {
    if (primeMap.has(key)) continue; // deduplicate

    let prime = derive(key);

    // Collision resolution: if two different axioms produce the same prime,
    // advance to the next prime. Mirrors Python's while-loop.
    // In v2, collisions are a hard failure (PrimeCollisionError in Python).
    if (scheme === 'sha256_128_v2') {
      // v2: hard fail on collision
      if (primeMap.has(prime.toString()) && primeMap.get(prime.toString()) !== key) {
        throw new Error(`PrimeCollisionError: v2 collision for '${key}' with existing '${primeMap.get(prime.toString())}'. This is a fatal v2 error.`);
      }
    } else {
      // v1: advance to next prime on collision
      while (primeMap.has(prime.toString()) && primeMap.get(prime.toString()) !== key) {
        prime = nextP(prime);
      }
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

// ─── v2 Self-Test Mode ─────────────────────────────────────────────

function runV2SelfTest() {
  console.log('═══════════════════════════════════════════════════════════');
  console.log('  Stage 3 — sha256_128_v2 Cross-Runtime Parity Test');
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

  // Frozen v2 reference vectors (must match Python exactly)
  const v2Vectors = [
    { key: 'alice||likes||cats', prime: '264285332112933860981052902103273947671' },
    { key: 'bob||knows||python', prime: '238582068730743173113692744107846045503' },
    { key: 'earth||orbits||sun', prime: '189007209170893135023962148948466996823' },
    { key: 'quantum||entangles||photon', prime: '75919499181718715751351316207293075217' },
    { key: 'water||contains||hydrogen', prime: '29728997747738460826164635775038413403' },
  ];

  console.log('── v2 Prime Derivation (16-byte seed + BPSW) ──');
  for (const vec of v2Vectors) {
    const prime = derivePrimeV2(vec.key);
    assert(`v2 prime for ${vec.key}`, prime.toString(), vec.prime);
  }

  // BPSW sanity checks
  console.log('\n── BPSW Sanity Checks ──');
  assert('BPSW: 2 is prime', isPrimeBPSW(2n).toString(), 'true');
  assert('BPSW: 3 is prime', isPrimeBPSW(3n).toString(), 'true');
  assert('BPSW: 4 is not prime', isPrimeBPSW(4n).toString(), 'false');
  assert('BPSW: 104729 is prime', isPrimeBPSW(104729n).toString(), 'true');
  assert('BPSW: 104730 is not prime', isPrimeBPSW(104730n).toString(), 'false');

  // Verify v2 primes are actually prime by BPSW
  console.log('\n── v2 Primes Are Prime ──');
  for (const vec of v2Vectors) {
    assert(`BPSW confirms ${vec.key}`, isPrimeBPSW(BigInt(vec.prime)).toString(), 'true');
  }

  // Verify v1 vectors still pass
  console.log('\n── v1 Backward Compatibility ──');
  const v1Vectors = [
    { key: 'alice||likes||cats', prime: '14326936561644797201' },
    { key: 'bob||knows||python', prime: '12933559861697884259' },
    { key: 'earth||orbits||sun', prime: '10246101339925224733' },
  ];
  for (const vec of v1Vectors) {
    const prime = derivePrime(vec.key);
    assert(`v1 prime for ${vec.key}`, prime.toString(), vec.prime);
  }

  console.log(`\n═══════════════════════════════════════════════════════════`);
  console.log(`  v2 Parity Test: ${passed} passed, ${failed} failed`);
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

  // 3b. Scheme validation
  const KNOWN_SCHEMES = ['sha256_64_v1', 'sha256_128_v2'];
  const scheme = bundle.prime_scheme || 'sha256_64_v1';
  if (!KNOWN_SCHEMES.includes(scheme)) {
    console.error(`\n❌ Unknown prime scheme: ${scheme}`);
    console.error(`   Known schemes: ${KNOWN_SCHEMES.join(', ')}`);
    process.exit(3);
  }
  console.log(`  Prime Scheme:      ${scheme}`);

  // 4. Parse canonical tome
  const axiomKeys = parseCanonicalTome(bundle.canonical_tome);
  console.log(`  Axioms Discovered: ${axiomKeys.length}`);
  if (bundle.axiom_count !== undefined) {
    console.log(`  Axioms Expected:   ${bundle.axiom_count}`);
  }

  // 5. Reconstruct state (scheme-aware)
  console.log(`\n  Reconstructing Gödel State Integer using ${scheme}...`);
  let reconstructed;
  try {
    reconstructed = reconstructState(axiomKeys, scheme);
  } catch (e) {
    console.error(`\n❌ Reconstruction failed: ${e.message}`);
    process.exit(4);
  }
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
    console.log(`  Scheme:  ${scheme}`);
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
    if (scheme === 'sha256_64_v1') {
      console.log('  • Collision resolution assumed collision-free (SHA-256 8-byte prefix)');
    } else {
      console.log('  • v2 collisions are fatal (no advancement loop)');
    }
  } else {
    console.log('  ❌ WITNESS VERIFICATION FAILED');
    console.log('');
    console.log(`  Scheme: ${scheme}`);
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

  if (args[0] === '--v2-test') {
    const ok = runV2SelfTest();
    process.exit(ok ? 0 : 1);
  }

  const ok = verifyBundle(args[0]);
  process.exit(ok ? 0 : 1);
}

main();
