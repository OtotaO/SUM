// Runs the detail checks over the frozen corpus and writes the flags that score.mjs reads.
// Usage: node fixtures/detail_flags_eval/run.mjs <corpus.json> <flags-out.json>
import { readFileSync, writeFileSync } from 'node:fs';
import { detailFlags } from '../../single_file_demo/detail_flags.js';

const [corpusPath, outPath] = process.argv.slice(2);
if (!corpusPath || !outPath) { console.error('usage: node run.mjs <corpus.json> <flags-out.json>'); process.exit(2); }
const { pairs } = JSON.parse(readFileSync(corpusPath, 'utf8'));
const flags = {};
for (const pair of pairs) flags[pair.id] = detailFlags(pair.original, pair.rewrite).flags;
writeFileSync(outPath, JSON.stringify(flags, null, 1) + '\n');
