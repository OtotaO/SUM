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

// ─── BPSW Primality Test (Stage 3 — sha256_128_v2) ─────────────────
//
// Baillie-PSW = strong base-2 Miller-Rabin + Strong Lucas (Selfridge A).
// No known BPSW pseudoprime exists. Used for 128-bit+ primes where the
// 12-witness deterministic M-R is no longer provably correct.

/**
 * Jacobi symbol (a/n) using the algorithm from Cohen's CANT.
 * Returns -1, 0, or 1.
 */
function jacobi(a, n) {
  if (n <= 0n || n % 2n === 0n) throw new Error('jacobi: n must be odd positive');
  a = ((a % n) + n) % n;
  let result = 1;
  while (a !== 0n) {
    while (a % 2n === 0n) {
      a /= 2n;
      const nMod8 = n % 8n;
      if (nMod8 === 3n || nMod8 === 5n) result = -result;
    }
    [a, n] = [n, a];
    if (a % 4n === 3n && n % 4n === 3n) result = -result;
    a = a % n;
  }
  return n === 1n ? result : 0;
}

/**
 * Selfridge Method A: find first D in {5, -7, 9, -11, 13, ...}
 * where Jacobi(D|n) = -1. Returns {D, P, Q}.
 */
function selfridgeParams(n) {
  let d = 5n;
  let sign = 1n;
  for (let i = 0; i < 100; i++) {
    const j = jacobi(d, n);
    if (j === -1) {
      return { D: d, P: 1n, Q: (1n - d) / 4n };
    }
    if (j === 0 && (d < 0n ? -d : d) < n) {
      // n has a factor
      return null;
    }
    // Sequence: 5, -7, 9, -11, 13, -15, ...
    sign = -sign;
    d = sign * ((d < 0n ? -d : d) + 2n);
  }
  return null; // should not happen for valid input
}

/**
 * Strong Lucas probable prime test.
 * Returns true if n is a strong Lucas probable prime with Selfridge A params.
 */
function strongLucasTest(n) {
  if (n < 2n) return false;
  if (n === 2n) return true;
  if (n % 2n === 0n) return false;

  // Perfect square check (Lucas test requires D not be a perfect square factor)
  const sqrtN = sqrt(n);
  if (sqrtN * sqrtN === n) return false;

  const params = selfridgeParams(n);
  if (params === null) return false;

  const { D, P, Q } = params;

  // Compute n - Jacobi(D|n) = n + 1 (since Jacobi = -1 by construction)
  const nPlus1 = n + 1n;

  // Write n+1 = 2^s * d where d is odd
  let d = nPlus1;
  let s = 0n;
  while (d % 2n === 0n) {
    d >>= 1n;
    s++;
  }

  // Lucas sequence U_d, V_d (mod n) using the standard doubling method
  let U = 1n;
  let V = P;
  let Qk = Q;
  const bits = d.toString(2);

  // Start from the second bit (MSB already processed as U=1, V=P)
  for (let i = 1; i < bits.length; i++) {
    // Double: U_{2k} = U_k * V_k, V_{2k} = V_k^2 - 2*Q^k
    U = (U * V) % n;
    V = (V * V - 2n * Qk) % n;
    Qk = (Qk * Qk) % n;

    if (bits[i] === '1') {
      // Step: U_{k+1} = (P*U + V) / 2, V_{k+1} = (D*U + P*V) / 2
      const newU = (P * U + V);
      const newV = (D * U + P * V);
      U = newU % 2n === 0n ? (newU / 2n) % n : ((newU + n) / 2n) % n;
      V = newV % 2n === 0n ? (newV / 2n) % n : ((newV + n) / 2n) % n;
      Qk = (Qk * Q) % n;
    }
  }

  // Normalize U and V to be positive
  U = ((U % n) + n) % n;
  V = ((V % n) + n) % n;

  // Strong Lucas test: U_d ≡ 0 (mod n) OR V_{d*2^r} ≡ 0 for some 0 <= r < s
  if (U === 0n || V === 0n) return true;

  for (let r = 1n; r < s; r++) {
    V = (V * V - 2n * Qk) % n;
    V = ((V % n) + n) % n;
    Qk = (Qk * Qk) % n;
    if (V === 0n) return true;
  }

  return false;
}

/**
 * Integer square root using Newton's method (BigInt).
 */
function sqrt(n) {
  if (n < 0n) throw new Error('sqrt of negative');
  if (n < 2n) return n;
  let x = n;
  let y = (x + 1n) / 2n;
  while (y < x) {
    x = y;
    y = (x + n / x) / 2n;
  }
  return x;
}

/**
 * BPSW primality test: strong base-2 Miller-Rabin + Strong Lucas.
 *
 * This is the approved primality test for sha256_128_v2.
 * It is deterministic in execution (no random witnesses) with no
 * known counterexamples, but is not a formal proof of primality.
 */
function isPrimeBPSW(n) {
  if (n < 2n) return false;
  if (n < 4n) return true;
  if (n % 2n === 0n || n % 3n === 0n) return false;

  // Small primes fast-path
  const smallPrimes = [5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n, 41n, 43n];
  for (const sp of smallPrimes) {
    if (n === sp) return true;
    if (n % sp === 0n) return false;
  }

  // Phase 1: Strong base-2 Miller-Rabin
  let d = n - 1n;
  let r = 0n;
  while (d % 2n === 0n) {
    d >>= 1n;
    r++;
  }

  let x = modPow(2n, d, n);
  if (x !== 1n && x !== n - 1n) {
    let composite = true;
    for (let i = 1n; i < r; i++) {
      x = (x * x) % n;
      if (x === n - 1n) { composite = false; break; }
    }
    if (composite) return false;
  }

  // Phase 2: Strong Lucas
  return strongLucasTest(n);
}

/**
 * Next prime using BPSW for 128-bit inputs.
 */
function nextPrimeBPSW(n) {
  if (n < 2n) return 2n;
  let candidate = n + 1n;
  if (candidate % 2n === 0n) candidate++;
  while (!isPrimeBPSW(candidate)) {
    candidate += 2n;
  }
  return candidate;
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
 * Derive a deterministic prime for an axiom key (v1: sha256_64_v1).
 *
 * Mirrors Python's GodelStateAlgebra._deterministic_prime_v1():
 *   SHA-256(axiom_key) → first 8 bytes big-endian → seed → nextprime(seed)
 */
function derivePrime(axiomKey) {
  const hash = crypto.createHash('sha256').update(axiomKey, 'utf-8').digest();
  const seedHex = hash.subarray(0, 8).toString('hex');
  const seed = BigInt('0x' + seedHex);
  return nextPrime(seed);
}

/**
 * Derive a deterministic prime for an axiom key (v2: sha256_128_v2).
 *
 * SHA-256(axiom_key) → first 16 bytes big-endian → seed → nextprime(seed)
 * Uses BPSW instead of 12-witness M-R.
 */
function derivePrimeV2(axiomKey) {
  const hash = crypto.createHash('sha256').update(axiomKey, 'utf-8').digest();
  const seedHex = hash.subarray(0, 16).toString('hex');
  const seed = BigInt('0x' + seedHex);
  return nextPrimeBPSW(seed);
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
