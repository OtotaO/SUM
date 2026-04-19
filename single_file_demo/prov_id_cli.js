#!/usr/bin/env node
/**
 * prov_id CLI: read one JSON object from stdin (a ProvenanceRecord shape),
 * validate it, compute the prov_id, and write it to stdout.
 *
 * Used by scripts/verify_prov_id_cross_runtime.py to check that Python and
 * JavaScript produce byte-identical prov_ids for the same input fields.
 *
 * Exit 0 on success, 1 on parse / validation / hash failure.
 */
'use strict';

const { provenanceRecord, computeProvId } = require('./provenance');

let input = '';
process.stdin.setEncoding('utf-8');
process.stdin.on('data', (chunk) => { input += chunk; });
process.stdin.on('end', () => {
  let fields;
  try {
    fields = JSON.parse(input);
  } catch (e) {
    console.error(`prov_id_cli: JSON.parse failed: ${e.message}`);
    process.exit(1);
  }
  try {
    const rec = provenanceRecord(fields);
    process.stdout.write(computeProvId(rec));
  } catch (e) {
    console.error(`prov_id_cli: ${e.name}: ${e.message}`);
    process.exit(1);
  }
});
