#!/usr/bin/env node
/**
 * Gödel CLI: read a JSON payload from stdin, perform a requested
 * operation, write the result to stdout.
 *
 * Payload shapes:
 *
 *   {"op": "mint", "axiom_key": "alice||like||cat", "scheme"?: "sha256_64_v1"}
 *     → writes the BigInt prime as a decimal string
 *
 *   {"op": "encode", "triples": [["alice","like","cat"], ...], "scheme"?: "sha256_64_v1"}
 *     → writes the BigInt state integer as a decimal string
 *
 * The optional `scheme` field selects the prime-derivation scheme:
 *   - "sha256_64_v1"  (default) — SHA-256 → first 8 bytes  → nextprime (Miller-Rabin)
 *   - "sha256_128_v2"           — SHA-256 → first 16 bytes → nextprime (BPSW)
 *
 * Used by scripts/verify_godel_cross_runtime.py and
 * scripts/verify_godel_v2_cross_runtime.py. Exit 0 on success, 1 on
 * error.
 */
'use strict';

const { mintPrime, derivePrimeV2, lcm } = require('./godel');

const SCHEMES = {
  sha256_64_v1: { derive: mintPrime },
  sha256_128_v2: { derive: derivePrimeV2 },
};

function _derivePrime(scheme, axiomKey) {
  const entry = SCHEMES[scheme];
  if (!entry) {
    throw new Error(
      `unknown scheme: ${scheme}. Known: ${Object.keys(SCHEMES).join(', ')}`
    );
  }
  return entry.derive(axiomKey);
}

function _axiomKey(subject, predicate, object_) {
  return (
    String(subject).trim().toLowerCase() + '||' +
    String(predicate).trim().toLowerCase() + '||' +
    String(object_).trim().toLowerCase()
  );
}

function _encodeChunk(triples, scheme) {
  let state = 1n;
  for (const [s, p, o] of triples) {
    const key = _axiomKey(s, p, o);
    const prime = _derivePrime(scheme, key);
    state = lcm(state, prime);
  }
  return state;
}

let input = '';
process.stdin.setEncoding('utf-8');
process.stdin.on('data', (c) => { input += c; });
process.stdin.on('end', () => {
  let payload;
  try {
    payload = JSON.parse(input);
  } catch (e) {
    console.error(`godel_cli: JSON parse failed: ${e.message}`);
    process.exit(1);
  }
  try {
    const scheme = payload.scheme || 'sha256_64_v1';
    if (!SCHEMES[scheme]) {
      throw new Error(
        `unknown scheme: ${scheme}. Known: ${Object.keys(SCHEMES).join(', ')}`
      );
    }
    if (payload.op === 'mint') {
      const prime = _derivePrime(scheme, String(payload.axiom_key));
      process.stdout.write(prime.toString());
    } else if (payload.op === 'encode') {
      const triples = payload.triples;
      if (!Array.isArray(triples)) {
        throw new Error('encode: triples must be an array of [s,p,o] tuples');
      }
      const state = _encodeChunk(triples, scheme);
      process.stdout.write(state.toString());
    } else {
      throw new Error(`unknown op: ${payload.op}`);
    }
  } catch (e) {
    console.error(`godel_cli: ${e.message}`);
    process.exit(1);
  }
});
