/**
 * GödelStateAlgebra — JavaScript MINT-side surface.
 *
 * Thin wrapper over ``standalone_verifier/math.js``. Exposes the API
 * shape SUM's Node-side consumers want (axiomKey, encodeChunkState,
 * mintPrime alias) plus the browser-specific async variant of
 * derivePrime that uses WebCrypto when available.
 *
 * Contract locked by scripts/verify_godel_cross_runtime.py: for every
 * axiom key SUM might emit, ``mintPrime(key)`` here produces the same
 * BigInt as Python's ``GodelStateAlgebra.get_or_mint_prime``. The
 * shared math.js module makes that contract cheap to maintain — fix
 * a primality bug once, every Node consumer inherits the fix.
 *
 * Only the sha256_64_v1 scheme is surfaced here (what seed_v1 / seed_v2
 * and current SUM production use). The sha256_128_v2 path is available
 * in math.js (derivePrimeV2, nextPrimeBPSW) for when it's promoted to
 * CURRENT_SCHEME; adding it to this surface is a one-line re-export.
 *
 * Zero dependencies. Pure ES2015+. Works in Node ≥ 14 (BigInt +
 * require('crypto')) and every browser with WebCrypto subtle + BigInt.
 */
'use strict';

const {
  gcd,
  lcm,
  modPow,
  isPrime,
  nextPrime,
  derivePrime,
  derivePrimeV2,
} = require('../standalone_verifier/math');

/**
 * Async WebCrypto variant of derivePrime. Use in browser environments
 * where Node's ``crypto.createHash`` is not present (math.js's sync
 * derivePrime requires Node). Returns the same BigInt as the sync path
 * on identical input — verified by the cross-runtime harness.
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

/**
 * Produce the canonical axiom key ``"s||p||o"`` from a triple.
 *
 * Each component is stripped and lowercased — matching the Python
 * side's ``_norm_key`` / canonical convention. Subjects are expected
 * to already be underscore-joined by the extractor for multi-word
 * cases; this helper does not re-join spaces (that is the extractor's
 * job, for both the sieve and the LLM adapter).
 */
function axiomKey(subject, predicate, object_) {
  return `${subject.trim().toLowerCase()}||${predicate.trim().toLowerCase()}||${object_.trim().toLowerCase()}`;
}

/**
 * Encode a list of triples as a Gödel state integer (LCM of all minted
 * primes).
 *
 * Matches Python's ``GodelStateAlgebra.encode_chunk_state`` on the
 * sha256_64_v1 scheme — verified by the cross-runtime harness.
 * Idempotent under duplicates (``lcm(p, p) == p``).
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
  // Re-exported arithmetic (so consumers don't need to know about math.js).
  modPow,
  gcd,
  lcm,
  // Re-exported primality.
  isPrime,
  nextPrime,
  // Derivation — sync (Node) + async (browser WebCrypto).
  derivePrime,
  derivePrimeAsync,
  derivePrimeV2,
  mintPrime,
  // MINT-side API.
  axiomKey,
  encodeChunkState,
};
