#!/usr/bin/env node
/**
 * JCS CLI: read one JSON value from stdin, write its RFC 8785 canonical
 * UTF-8 bytes to stdout (raw, no newline). Used by
 * scripts/verify_jcs_byte_identity.py to check byte-level agreement
 * with the Python canonicalizer.
 *
 * Usage:
 *   echo '{"b":1,"a":2}' | node single_file_demo/jcs_cli.js
 *   → writes bytes for '{"a":2,"b":1}' to stdout
 *
 * Exit 0 on success, 1 on parse failure or canonicalization error.
 */
'use strict';

const { canonicalize } = require('./jcs');

let input = '';
process.stdin.setEncoding('utf-8');
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
  let parsed;
  try {
    parsed = JSON.parse(input);
  } catch (e) {
    console.error(`jcs_cli: JSON.parse failed: ${e.message}`);
    process.exit(1);
  }
  try {
    const bytes = canonicalize(parsed);
    process.stdout.write(Buffer.from(bytes));
  } catch (e) {
    console.error(`jcs_cli: canonicalize failed: ${e.message}`);
    process.exit(1);
  }
});
