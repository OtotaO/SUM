/**
 * GödelStateAlgebra — JavaScript MINT-side port.
 *
 * Minimal surface needed by the v0 single-file demo: given a list of
 * (subject, predicate, object) triples, produce the same Gödel state
 * integer Python produces via
 * ``internal.algorithms.semantic_arithmetic.GodelStateAlgebra.encode_chunk_state``.
 *
 * Only the ``sha256_64_v1`` scheme is implemented here. It's what
 * seed_v1 and seed_v2 use and what current SUM production emits.
 * The sha256_128_v2 BPSW path lives in standalone_verifier/verify.js;
 * this module stays small and demo-focused.
 *
 * Contract locked by scripts/verify_prime_derivation_cross_runtime.py:
 * for every axiom key SUM might emit, ``mintPrime(key)`` here produces
 * the same BigInt as Python's ``GodelStateAlgebra.get_or_mint_prime``.
 * If that contract breaks, every cross-runtime Gödel state verification
 * breaks silently downstream.
 *
 * Zero dependencies. Pure ES2015+. Works in Node ≥ 14 (BigInt) and every
 * browser with WebCrypto subtle + BigInt (all modern browsers as of 2023).
 *
 * For the sync path (Node), uses built-in ``crypto.createHash``. For the
 * async path (browser), use ``mintPrimeAsync`` which goes through
 * ``crypto.subtle.digest``.
 */
'use strict';

// ─── BigInt arithmetic primitives ────────────────────────────────────

/** Modular exponentiation (base^exp) % mod, BigInt. */
function modPow(base, exp, mod) {
  base = ((base % mod) + mod) % mod;
  let result = 1n;
  while (exp > 0n) {
    if (exp & 1n) result = (result * base) % mod;
    exp >>= 1n;
    base = (base * base) % mod;
  }
  return result;
}

/** BigInt GCD via Euclidean algorithm. */
function gcd(a, b) {
  a = a < 0n ? -a : a;
  b = b < 0n ? -b : b;
  while (b > 0n) [a, b] = [b, a % b];
  return a;
}

/** BigInt LCM = |a*b| / gcd(a,b); 0 if either operand is 0. */
function lcm(a, b) {
  if (a === 0n || b === 0n) return 0n;
  return (a / gcd(a, b)) * b;
}

// ─── Deterministic Miller-Rabin primality ────────────────────────────
//
// For n < 3,317,044,064,679,887,385,961,981 the witness set
// {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37} is provably sufficient
// for a zero-error test. All sha256_64_v1 seeds fit well within that
// bound (max 2^64 ≈ 1.8e19 « 3.3e24).

const _SMALL_PRIMES = [5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n, 41n, 43n];
const _MR_WITNESSES = [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n];

function isPrime(n) {
  if (n < 2n) return false;
  if (n < 4n) return true;
  if (n % 2n === 0n || n % 3n === 0n) return false;

  for (const sp of _SMALL_PRIMES) {
    if (n === sp) return true;
    if (n % sp === 0n) return false;
  }

  let d = n - 1n;
  let r = 0n;
  while (d % 2n === 0n) {
    d >>= 1n;
    r++;
  }

  for (const a of _MR_WITNESSES) {
    if (a >= n) continue;
    let x = modPow(a, d, n);
    if (x === 1n || x === n - 1n) continue;
    let composite = true;
    for (let i = 0n; i < r - 1n; i++) {
      x = (x * x) % n;
      if (x === n - 1n) { composite = false; break; }
    }
    if (composite) return false;
  }
  return true;
}

/** Smallest prime p with p > n (strictly greater; if n itself is prime,
 *  returns the next one up). Matches Python's ``sympy.nextprime(n)``
 *  semantics — verified by the cross-runtime harness. */
function nextPrime(n) {
  let candidate = n + 1n;
  if (candidate < 2n) return 2n;
  if (candidate === 2n) return 2n;
  if (candidate % 2n === 0n) candidate++;
  while (!isPrime(candidate)) candidate += 2n;
  return candidate;
}

// ─── Prime derivation (sha256_64_v1) ─────────────────────────────────

/**
 * Derive a deterministic prime for an axiom key via the sha256_64_v1
 * scheme. Mirrors Python's ``GodelStateAlgebra._deterministic_prime_v1``:
 *
 *   SHA-256(axiom_key) → first 8 bytes big-endian → seed → nextprime(seed)
 *
 * The axiom_key is expected to be in the form "subject||predicate||object"
 * with all three components already canonicalized (lowercased, lemmatized
 * where appropriate). See ``axiomKey`` for the canonical join.
 */
function derivePrime(axiomKey) {
  const { createHash } = require('crypto');
  const hash = createHash('sha256').update(axiomKey, 'utf-8').digest();
  const seedHex = hash.subarray(0, 8).toString('hex');
  const seed = BigInt('0x' + seedHex);
  return nextPrime(seed);
}

/**
 * Async WebCrypto variant. Use in browser environments where Node's
 * ``crypto.createHash`` is not present. Returns the same BigInt as
 * ``derivePrime`` in Node.
 */
async function derivePrimeAsync(axiomKey) {
  const subtle = (globalThis.crypto && globalThis.crypto.subtle) || null;
  if (!subtle) return derivePrime(axiomKey);
  const enc = new TextEncoder();
  const buf = await subtle.digest('SHA-256', enc.encode(axiomKey));
  const bytes = new Uint8Array(buf);
  let seedHex = '';
  for (let i = 0; i < 8; i++) seedHex += bytes[i].toString(16).padStart(2, '0');
  const seed = BigInt('0x' + seedHex);
  return nextPrime(seed);
}

/** Python-API-compatible alias. */
const mintPrime = derivePrime;

// ─── Axiom key formatting ────────────────────────────────────────────

/**
 * Produce the canonical axiom key ``"s||p||o"`` from a triple.
 * Each component is stripped and lowercased — matching the Python
 * side's `_norm_key`/`canonical` convention. Subjects are expected
 * to be underscore-joined by the extractor for multi-word cases; this
 * helper does not re-join spaces (that is the extractor's job).
 */
function axiomKey(subject, predicate, object_) {
  return `${subject.trim().toLowerCase()}||${predicate.trim().toLowerCase()}||${object_.trim().toLowerCase()}`;
}

// ─── State encoding ──────────────────────────────────────────────────

/**
 * Encode a list of triples as a Gödel state integer (LCM of all minted
 * primes).
 *
 * Matches Python's ``GodelStateAlgebra.encode_chunk_state`` on the
 * sha256_64_v1 scheme — verified by the cross-runtime harness.
 *
 * @param {Array<[string,string,string]>} triples
 * @returns {BigInt}
 */
function encodeChunkState(triples) {
  let state = 1n;
  for (const [s, p, o] of triples) {
    const key = axiomKey(s, p, o);
    const prime = derivePrime(key);
    state = lcm(state, prime);
  }
  return state;
}

module.exports = {
  // Arithmetic
  modPow,
  gcd,
  lcm,
  // Primality
  isPrime,
  nextPrime,
  // Derivation
  derivePrime,
  derivePrimeAsync,
  mintPrime,
  // Axiom API
  axiomKey,
  encodeChunkState,
};
