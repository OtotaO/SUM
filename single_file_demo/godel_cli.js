#!/usr/bin/env node
/**
 * Gödel CLI: read a JSON payload from stdin, perform a requested
 * operation, write the result to stdout.
 *
 * Payload shapes:
 *
 *   {"op": "mint", "axiom_key": "alice||like||cat"}
 *     → writes the BigInt prime as a decimal string
 *
 *   {"op": "encode", "triples": [["alice","like","cat"], ...]}
 *     → writes the BigInt state integer as a decimal string
 *
 * Used by scripts/verify_prime_derivation_cross_runtime.py and
 * scripts/verify_state_cross_runtime.py. Exit 0 on success, 1 on error.
 */
'use strict';

const { mintPrime, encodeChunkState } = require('./godel');

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
    if (payload.op === 'mint') {
      const prime = mintPrime(String(payload.axiom_key));
      process.stdout.write(prime.toString());
    } else if (payload.op === 'encode') {
      const triples = payload.triples;
      if (!Array.isArray(triples)) {
        throw new Error('encode: triples must be an array of [s,p,o] tuples');
      }
      const state = encodeChunkState(triples);
      process.stdout.write(state.toString());
    } else {
      throw new Error(`unknown op: ${payload.op}`);
    }
  } catch (e) {
    console.error(`godel_cli: ${e.message}`);
    process.exit(1);
  }
});
