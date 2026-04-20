#!/usr/bin/env node
/**
 * SUM — shared JavaScript math primitives (BigInt, primality, prime derivation).
 *
 * Single source of truth for every Node.js consumer in this repository.
 * Before this module, standalone_verifier/verify.js and
 * single_file_demo/godel.js each carried their own copies of modPow,
 * gcd, lcm, isPrime, and nextPrime — three hundred lines of duplicated
 * BigInt math that had to be kept byte-identical to Python's
 * ``GodelStateAlgebra._deterministic_prime_v1`` or the cross-runtime
 * state-integer reconstruction would drift silently. Consolidating into
 * one module means one bug fix reaches every consumer and one code-review
 * pass covers the entire primality surface.
 *
 * Scope — what lives here:
 *
 *   BigInt arithmetic:     modPow, gcd, lcm, sqrt
 *   Primality (v1, fast):  isPrime, nextPrime
 *                          12-witness deterministic Miller-Rabin,
 *                          provably correct for n < 3.3e24 — which
 *                          every sha256_64_v1 seed (≤ 2^64 ≈ 1.8e19)
 *                          fits inside with twelve orders of magnitude
 *                          of headroom.
 *   Primality (v2, BPSW):  isPrimeBPSW, nextPrimeBPSW,
 *                          plus its helpers jacobi, selfridgeParams,
 *                          strongLucasTest. Strong base-2 Miller-Rabin
 *                          composed with Selfridge-A Strong Lucas. No
 *                          known counterexample.
 *   Prime derivation:      derivePrime (sha256_64_v1) and
 *                          derivePrimeV2 (sha256_128_v2). These are the
 *                          *only* callers of Node's ``crypto.createHash``
 *                          on the math path; the rest is pure BigInt.
 *
 * Scope — what does NOT live here:
 *
 *   - WebCrypto (browser) variants of derivePrime. These live in
 *     single_file_demo/godel.js as ``derivePrimeAsync`` because they
 *     depend on globalThis.crypto.subtle which is a browser-side API.
 *   - State encoding / axiom key formatting (encodeChunkState, axiomKey).
 *     Those are consumer-facing MINT surface; they live in godel.js.
 *   - Bundle parsing, tome parsing, signature verification, and CLI
 *     driver code. Those are verification-consumer-specific and stay in
 *     verify.js.
 *   - The in-browser inlined copy in single_file_demo/index.html. The
 *     single-file HTML artifact CANNOT import; its copy is deliberately
 *     duplicated, and scripts/verify_cross_runtime.py is the guard that
 *     ensures all three implementations (Python, Node, Browser inlined)
 *     continue to produce the same state integer for the same axiom set.
 *
 * Requirements: Node.js ≥ 16 (BigInt + crypto.createHash). Zero npm
 * dependencies — same contract verify.js advertised before the split.
 *
 * Usage:
 *   const { modPow, isPrime, nextPrime, derivePrime, derivePrimeV2,
 *           lcm, gcd, sqrt, isPrimeBPSW, nextPrimeBPSW,
 *           jacobi, selfridgeParams, strongLucasTest } = require('./math');
 *
 * License: Apache License 2.0.
 */
'use strict';

const crypto = require('crypto');

// ─── BigInt arithmetic primitives ─────────────────────────────────────

/** Modular exponentiation: (base^exp) % mod using BigInt. */
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

/** GCD of two BigInts (Euclidean algorithm). */
function gcd(a, b) {
  a = a < 0n ? -a : a;
  b = b < 0n ? -b : b;
  while (b > 0n) {
    [a, b] = [b, a % b];
  }
  return a;
}

/** LCM of two BigInts. Returns 0n if either operand is 0n. */
function lcm(a, b) {
  if (a === 0n || b === 0n) return 0n;
  return (a / gcd(a, b)) * b;
}

/** Integer square root using Newton's method (BigInt). */
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

// ─── Deterministic Miller-Rabin (sha256_64_v1) ────────────────────────
//
// For n < 3,317,044,064,679,887,385,961,981 the witness set
// {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37} is provably sufficient
// for a zero-error test. All sha256_64_v1 seeds (max 2^64 ≈ 1.8e19)
// fit well within this bound.

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
      if (x === n - 1n) {
        composite = false;
        break;
      }
    }
    if (composite) return false;
  }
  return true;
}

/** Smallest prime strictly greater than n. Mirrors Python's sympy.nextprime. */
function nextPrime(n) {
  if (n < 2n) return 2n;
  let candidate = n + 1n;
  if (candidate % 2n === 0n) candidate++;
  while (!isPrime(candidate)) {
    candidate += 2n;
  }
  return candidate;
}

// ─── BPSW primality (sha256_128_v2) ───────────────────────────────────
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
 * where Jacobi(D|n) = -1. Returns {D, P, Q} or null if n has a factor.
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
      return null;
    }
    sign = -sign;
    d = sign * ((d < 0n ? -d : d) + 2n);
  }
  return null;
}

/**
 * Strong Lucas probable prime test with Selfridge-A parameters.
 * Returns true if n passes the strong Lucas test.
 */
function strongLucasTest(n) {
  if (n < 2n) return false;
  if (n === 2n) return true;
  if (n % 2n === 0n) return false;

  const sqrtN = sqrt(n);
  if (sqrtN * sqrtN === n) return false;

  const params = selfridgeParams(n);
  if (params === null) return false;

  const { D, P, Q } = params;
  const nPlus1 = n + 1n;

  let d = nPlus1;
  let s = 0n;
  while (d % 2n === 0n) {
    d >>= 1n;
    s++;
  }

  let U = 1n;
  let V = P;
  let Qk = Q;
  const bits = d.toString(2);

  for (let i = 1; i < bits.length; i++) {
    U = (U * V) % n;
    V = (V * V - 2n * Qk) % n;
    Qk = (Qk * Qk) % n;

    if (bits[i] === '1') {
      const newU = (P * U + V);
      const newV = (D * U + P * V);
      U = newU % 2n === 0n ? (newU / 2n) % n : ((newU + n) / 2n) % n;
      V = newV % 2n === 0n ? (newV / 2n) % n : ((newV + n) / 2n) % n;
      Qk = (Qk * Q) % n;
    }
  }

  U = ((U % n) + n) % n;
  V = ((V % n) + n) % n;

  if (U === 0n || V === 0n) return true;

  for (let r = 1n; r < s; r++) {
    V = (V * V - 2n * Qk) % n;
    V = ((V % n) + n) % n;
    Qk = (Qk * Qk) % n;
    if (V === 0n) return true;
  }

  return false;
}

/** BPSW primality: strong base-2 M-R + Strong Lucas. */
function isPrimeBPSW(n) {
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

  let x = modPow(2n, d, n);
  if (x !== 1n && x !== n - 1n) {
    let composite = true;
    for (let i = 1n; i < r; i++) {
      x = (x * x) % n;
      if (x === n - 1n) { composite = false; break; }
    }
    if (composite) return false;
  }

  return strongLucasTest(n);
}

/** Next prime using BPSW. Used by sha256_128_v2 derivation. */
function nextPrimeBPSW(n) {
  if (n < 2n) return 2n;
  let candidate = n + 1n;
  if (candidate % 2n === 0n) candidate++;
  while (!isPrimeBPSW(candidate)) {
    candidate += 2n;
  }
  return candidate;
}

// ─── Prime derivation from axiom keys ─────────────────────────────────

/**
 * Derive a deterministic prime for an axiom key (sha256_64_v1).
 *
 * Mirrors Python's ``GodelStateAlgebra._deterministic_prime_v1``:
 *   SHA-256(axiom_key) → first 8 bytes big-endian → seed → nextprime(seed)
 *
 * Byte-identical output verified by scripts/verify_godel_cross_runtime.py
 * (12 fixture axiom keys, Python ↔ Node byte-level agreement on the
 * resulting BigInt).
 */
function derivePrime(axiomKey) {
  const hash = crypto.createHash('sha256').update(axiomKey, 'utf-8').digest();
  const seedHex = hash.subarray(0, 8).toString('hex');
  const seed = BigInt('0x' + seedHex);
  return nextPrime(seed);
}

/**
 * Derive a deterministic prime for an axiom key (sha256_128_v2).
 *
 * SHA-256(axiom_key) → first 16 bytes big-endian → seed → nextPrimeBPSW(seed)
 *
 * Scope note: v2 extends the seed from 8 to 16 bytes and uses BPSW
 * instead of 12-witness M-R, pushing the collision probability and the
 * primality-test provable bound out of concern for corpora with millions
 * of axioms. Not yet exercised by any shipped consumer; the plumbing is
 * here for when sha256_128_v2 is promoted to CURRENT_SCHEME.
 */
function derivePrimeV2(axiomKey) {
  const hash = crypto.createHash('sha256').update(axiomKey, 'utf-8').digest();
  const seedHex = hash.subarray(0, 16).toString('hex');
  const seed = BigInt('0x' + seedHex);
  return nextPrimeBPSW(seed);
}

module.exports = {
  // Arithmetic
  modPow,
  gcd,
  lcm,
  sqrt,
  // Primality — v1
  isPrime,
  nextPrime,
  // Primality — v2 (BPSW stack)
  jacobi,
  selfridgeParams,
  strongLucasTest,
  isPrimeBPSW,
  nextPrimeBPSW,
  // Derivation
  derivePrime,
  derivePrimeV2,
};
