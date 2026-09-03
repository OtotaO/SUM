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
 *   - Ed25519 signatures (if present) ARE verified here via Node's
 *     WebCrypto SubtleCrypto — the same API the browser demo uses. A
 *     tampered bundle with a valid tome but a mismatched signature
 *     fails, and so does a bundle whose public key or signature cannot
 *     be decoded: a signature that could not be checked never passes.
 *     Requires Node 18.4+; on older Node nothing can be checked, so
 *     the verifier says so and exits 5 ("could not verify"), which is
 *     neither a pass nor a proven failure. See the verdict rule above
 *     verifyEd25519 and the exit-code table in the usage text.
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
  CANONICAL_LINE_REGEX,
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
    // Canonical-ABI line regex lives in math.js (single source of truth
    // for JS); see its comment for the \S+/\S+/.+ invariant and the
    // rationale. Using the shared export here means this parser cannot
    // drift from the one in single_file_demo/index.html or any future
    // JS consumer — the bug fixed in commit 2e4188c cannot resurface.
    const match = trimmed.match(CANONICAL_LINE_REGEX);
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

// ─── Ed25519 Signature Verification ────────────────────────────────
//
// Uses Node's WebCrypto SubtleCrypto — the same API the browser demo
// uses (see single_file_demo/index.html::verifyEd25519InBrowser). By
// sharing the surface, the Python ↔ Node ↔ Browser trust triangle is
// symmetric: any bundle verifiable in one is verifiable in the other
// two.
//
// Node 18.4+ ships Ed25519 in WebCrypto. On older Node, importKey
// throws and we report 'unsupported' — never a false ✓, and (since
// the split below) never a green exit either.
//
// ── The verdict rule ──────────────────────────────────────────────
//
// verifyEd25519 returns exactly one of five statuses, and this is the
// whole contract. Keep it in sync with the banner labels below and
// with the exit codes documented in main().
//
//   'verified'    → PASS (exit 0). The signature checks out against
//                   the embedded public key.
//   'invalid'     → FAIL (exit 1). The signature was checked and does
//                   not match. Forgery or post-signing tampering.
//   'malformed'   → FAIL (exit 1). The public key or the signature
//                   could not be decoded or imported, so NO check was
//                   possible. An unusable key is not an excuse to
//                   pass: fail closed. Python's `_ed25519_verify`
//                   returns False on the same inputs, so both
//                   runtimes reject, in the same rejection class.
//   'absent'      → PASS structurally (exit 0), labelled 'absent'.
//                   An unsigned bundle is not a forgery; this witness
//                   reports the state reconstruction and says the
//                   bundle carries no public signature. That
//                   semantics is deliberate and unchanged.
//   'unsupported' → FAIL CLOSED with a distinct exit code (5) and an
//                   explicit warning. Reserved for the ONE case it
//                   honestly describes: this Node build genuinely has
//                   no Ed25519 in WebCrypto. Nothing was checked, so
//                   a script must be able to tell "could not verify"
//                   from "verified" — hence 5 rather than 0 or 1.
//
// Before the split, every failure inside the try block (an
// unparseable public key, a non-string field, a key of the wrong
// length) returned 'unsupported', and the banner printed a green PASS
// carrying the false excuse "Node lacks Ed25519 in WebCrypto" on a
// Node that has it. Pinned by A8 in
// scripts/verify_cross_runtime_adversarial.py.

function decodeEd25519Field(s) {
  // Deliberately mirrors Python's CanonicalCodec._ed25519_verify:
  // split on the first colon, base64-decode the tail, validate
  // nothing else. Python does not check the "ed25519:" prefix, so
  // neither do we — a stricter Node would reject bundles Python
  // accepts, which is an accept-side asymmetry, not a hardening.
  // Buffer.from(..., 'base64') is lenient (it drops junk characters
  // rather than throwing), so a wrong-length or garbage field is
  // caught downstream by importKey, not here.
  if (typeof s !== 'string') throw new TypeError('Ed25519 field is not a string');
  const b64 = s.split(':', 2)[1] || '';
  return new Uint8Array(Buffer.from(b64, 'base64'));
}

// Does THIS runtime have Ed25519 in WebCrypto at all? Probed with a
// syntactically valid 32-byte key so the answer depends only on the
// runtime, never on the bundle. Used to tell a genuinely missing
// primitive apart from a key this runtime could not import — matching
// on error names or messages would couple the verdict to Node's
// wording, which is not a stable contract.
async function ed25519PrimitiveAvailable(subtle) {
  try {
    await subtle.importKey(
      'raw', new Uint8Array(32), { name: 'Ed25519' }, false, ['verify']
    );
    return true;
  } catch (e) {
    return false;
  }
}

async function verifyEd25519(bundle) {
  if (!bundle.public_signature || !bundle.public_key) return 'absent';

  const subtle = (crypto.webcrypto && crypto.webcrypto.subtle) || null;
  if (!subtle) return 'unsupported';

  let pubBytes, sigBytes;
  try {
    pubBytes = decodeEd25519Field(bundle.public_key);
    sigBytes = decodeEd25519Field(bundle.public_signature);
  } catch (e) {
    return 'malformed';
  }

  let key;
  try {
    key = await subtle.importKey(
      'raw', pubBytes, { name: 'Ed25519' }, false, ['verify']
    );
  } catch (e) {
    // Two unrelated failures land here: the runtime has no Ed25519, or
    // the key bytes are not an importable Ed25519 key. Only the first
    // one may be called 'unsupported'.
    return (await ed25519PrimitiveAvailable(subtle)) ? 'malformed' : 'unsupported';
  }

  const payload = new TextEncoder().encode(
    `${bundle.canonical_tome}|${bundle.state_integer}|${bundle.timestamp}`
  );
  try {
    const ok = await subtle.verify({ name: 'Ed25519' }, key, sigBytes, payload);
    return ok ? 'verified' : 'invalid';
  } catch (e) {
    // The key imported, so the primitive exists; a throw here means the
    // signature bytes were unusable. Note that a merely wrong-length
    // signature does NOT throw — WebCrypto returns false for it, which
    // is 'invalid', the same verdict Python reaches. Only a hard throw
    // reaches this branch.
    return 'malformed';
  }
}

// ─── Bundle Verification ───────────────────────────────────────────
//
// Exit codes. These are the machine-readable half of the verdict and
// scripts branch on them, so they are append-only: never reuse or
// renumber one. Also documented in the usage text printed by main().
const EXIT_OK = 0;                    // verified, or structurally valid + unsigned
const EXIT_VERIFICATION_FAILED = 1;   // state mismatch, or Ed25519 invalid/malformed
const EXIT_BAD_INPUT = 2;             // unreadable file, bad JSON, missing field
const EXIT_UNSUPPORTED_FORMAT = 3;    // unknown canonical format version or prime scheme
const EXIT_RECONSTRUCTION_ERROR = 4;  // state reconstruction threw
const EXIT_ED25519_UNCHECKABLE = 5;   // this runtime cannot check Ed25519 at all

async function verifyBundle(bundlePath) {
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
    process.exit(EXIT_BAD_INPUT);
  }

  let bundle;
  try {
    bundle = JSON.parse(bundleJson);
  } catch (e) {
    console.error('❌ Invalid JSON in bundle file.');
    process.exit(EXIT_BAD_INPUT);
  }

  // 2. Validate bundle structure
  const required = ['canonical_tome', 'state_integer', 'canonical_format_version', 'bundle_version'];
  for (const field of required) {
    if (!(field in bundle)) {
      console.error(`❌ Missing required field: ${field}`);
      process.exit(EXIT_BAD_INPUT);
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
    process.exit(EXIT_UNSUPPORTED_FORMAT);
  }

  // 3b. Scheme validation
  const KNOWN_SCHEMES = ['sha256_64_v1', 'sha256_128_v2'];
  const scheme = bundle.prime_scheme || 'sha256_64_v1';
  if (!KNOWN_SCHEMES.includes(scheme)) {
    console.error(`\n❌ Unknown prime scheme: ${scheme}`);
    console.error(`   Known schemes: ${KNOWN_SCHEMES.join(', ')}`);
    process.exit(EXIT_UNSUPPORTED_FORMAT);
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
    process.exit(EXIT_RECONSTRUCTION_ERROR);
  }
  const exported = BigInt(bundle.state_integer);

  const reconDigits = reconstructed.toString().length;
  const exportDigits = exported.toString().length;

  console.log(`  Exported Digits:       ${exportDigits}`);
  console.log(`  Reconstructed Digits:  ${reconDigits}`);

  // 6. Compare state integer
  const match = reconstructed === exported;

  // 7. Ed25519 signature (optional) — verified with Node SubtleCrypto.
  //    If present and invalid, the whole verify fails even when state
  //    reconstruction matches; a bundle with a tampered tome and a
  //    patched state_integer could otherwise pass step 6 while carrying
  //    a stale Ed25519 signature, and that must be caught here.
  const ed25519Status = await verifyEd25519(bundle);
  let ed25519Label;
  if (ed25519Status === 'verified') ed25519Label = '✓ verified (Node SubtleCrypto)';
  else if (ed25519Status === 'invalid') ed25519Label = '✗ INVALID — signature does not match public key';
  else if (ed25519Status === 'malformed') ed25519Label = '✗ MALFORMED — public key or signature could not be decoded; signature NOT checked';
  else if (ed25519Status === 'unsupported') ed25519Label = '⚠ NOT CHECKED — this Node lacks Ed25519 in WebCrypto (upgrade to ≥18.4)';
  else ed25519Label = 'absent';

  // Only 'verified' and 'absent' are pass verdicts. 'invalid' and
  // 'malformed' are failures; 'unsupported' is not a pass either,
  // because nothing was checked (see the verdict rule above).
  const overallMatch =
    match && (ed25519Status === 'verified' || ed25519Status === 'absent');
  // Exit-code precedence: a structural mismatch is the strongest claim
  // we can make, so it wins; otherwise an unchecked signature reports
  // "could not verify" rather than "verification failed".
  const exitCode = !match
    ? EXIT_VERIFICATION_FAILED
    : ed25519Status === 'unsupported'
      ? EXIT_ED25519_UNCHECKABLE
      : overallMatch
        ? EXIT_OK
        : EXIT_VERIFICATION_FAILED;

  console.log('\n═══════════════════════════════════════════════════════════');
  if (overallMatch) {
    console.log('  ✅ WITNESS VERIFICATION PASSED');
    console.log('');
    console.log(`  Scheme:   ${scheme}`);
    console.log(`  Ed25519:  ${ed25519Label}`);
    console.log('  The JavaScript-reconstructed Gödel State Integer');
    console.log('  exactly matches the Python-exported state.');
    console.log('');
    console.log('  This proves:');
    console.log('  • Canonical semantic content is runtime-independent');
    console.log('  • Deterministic prime derivation reproduces across languages');
    console.log('  • The Gödel Integer is NOT a Python-specific artifact');
    if (ed25519Status === 'verified') {
      console.log('  • Ed25519 public-key attestation holds cross-runtime');
    }
    console.log('');
    console.log('  Caveats:');
    console.log('  • HMAC signature NOT verified (shared-secret, not public witness)');
    if (scheme === 'sha256_64_v1') {
      console.log('  • Collision resolution assumed collision-free (SHA-256 8-byte prefix)');
    } else {
      console.log('  • v2 collisions are fatal (no advancement loop)');
    }
  } else {
    if (exitCode === EXIT_ED25519_UNCHECKABLE) {
      console.log('  ⚠️  WITNESS VERIFICATION INCOMPLETE — Ed25519 NOT CHECKED');
    } else {
      console.log('  ❌ WITNESS VERIFICATION FAILED');
    }
    console.log('');
    console.log(`  Scheme:   ${scheme}`);
    console.log(`  Ed25519:  ${ed25519Label}`);
    if (!match) {
      console.log('  The reconstructed state does NOT match the exported state.');
      console.log(`  Exported:      ${exported.toString().substring(0, 40)}...`);
      console.log(`  Reconstructed: ${reconstructed.toString().substring(0, 40)}...`);
    }
    if (ed25519Status === 'invalid') {
      console.log('  The embedded Ed25519 signature does NOT match the public key.');
      console.log('  Either the tome was tampered with after signing, or the');
      console.log('  public_key field was swapped. Do not trust this bundle.');
    }
    if (ed25519Status === 'malformed') {
      console.log('  The embedded Ed25519 public key or signature could not be');
      console.log('  decoded or imported, so the signature was NOT checked.');
      console.log('  This is a broken bundle, not a missing runtime primitive:');
      console.log('  a field that will not decode, or a key that will not');
      console.log('  import where Ed25519 IS available, fails on any runtime.');
      console.log('  A signature that cannot be checked is not a signature.');
      console.log('  Do not trust this bundle.');
    }
    if (ed25519Status === 'unsupported') {
      console.log('  This Node build has no Ed25519 in WebCrypto, so the embedded');
      console.log('  signature could NOT be checked. The state reconstruction');
      console.log(`  ${match ? 'DID match' : 'did NOT match'}, but the provenance claim is unverified.`);
      console.log(`  Exiting ${EXIT_ED25519_UNCHECKABLE} ("could not verify"), which is NOT exit 0`);
      console.log('  ("verified") and NOT exit 1 ("verification failed").');
      console.log('  Upgrade to Node ≥ 18.4, or verify with `sum verify`.');
    }
  }
  console.log('═══════════════════════════════════════════════════════════');

  // The exit status must carry the SAME verdict the banner just printed.
  // Returning the structural `match` here let a bundle with intact content
  // and a forged Ed25519 signature print FAILED and exit 0, so any script
  // reading the exit code accepted the forgery. Pinned by A7 in
  // scripts/verify_cross_runtime_adversarial.py.
  return exitCode;
}

// ─── Main ──────────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log('Usage:');
    console.log('  node verify.js <bundle.json>   — Verify a canonical bundle');
    console.log('  node verify.js --self-test      — Run deterministic test vectors');
    console.log('  node verify.js --v2-test        — Run sha256_128_v2 parity vectors');
    console.log('');
    console.log('Exit codes:');
    console.log(`  ${EXIT_OK}  verified (or structurally valid and carrying no signature)`);
    console.log(`  ${EXIT_VERIFICATION_FAILED}  verification FAILED: state mismatch, or Ed25519 invalid/malformed`);
    console.log(`  ${EXIT_BAD_INPUT}  bad input: unreadable file, invalid JSON, missing required field`);
    console.log(`  ${EXIT_UNSUPPORTED_FORMAT}  unsupported canonical format version or prime scheme`);
    console.log(`  ${EXIT_RECONSTRUCTION_ERROR}  state reconstruction error`);
    console.log(`  ${EXIT_ED25519_UNCHECKABLE}  COULD NOT VERIFY: this Node lacks Ed25519 in WebCrypto`);
    console.log('     (nothing was checked; this is neither a pass nor a proven failure)');
    process.exit(EXIT_OK);
  }

  if (args[0] === '--self-test') {
    const ok = runSelfTest();
    process.exit(ok ? 0 : 1);
  }

  if (args[0] === '--v2-test') {
    const ok = runV2SelfTest();
    process.exit(ok ? 0 : 1);
  }

  // verifyBundle returns the exit code itself, not a boolean: exit 5
  // ("Ed25519 could not be checked at all") is a third outcome that no
  // boolean can carry, and collapsing it into 0 is exactly the
  // fail-open this verifier had.
  const code = await verifyBundle(args[0]);
  process.exit(code);
}

main().catch((err) => {
  console.error('❌ verify.js: unexpected error:', err);
  process.exit(EXIT_BAD_INPUT);
});
