#!/usr/bin/env bash
# Render once, then verify the SAME bundle bytes in three runtimes.
#
# This is the minimal end-to-end of SUM's cross-runtime trust loop:
#   1. Render/attest with the PUBLISHED sum-engine (PyPI) -> a bundle.
#   2. Verify that bundle with the Python verifier (sum verify).
#   3. Verify the SAME bytes with the Node standalone_verifier (no npm
#      dependencies), proving the Gödel state integer is not a
#      Python-specific artifact.
#   4. (Browser, manual) the same bytes verify in single_file_demo/.
#
# Run from the repo root:  bash examples/verify-example/run.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
BUNDLE="$HERE/bundle.json"

# ── Runtime 1: render with the published package ────────────────────
# `sum` is the console script from `pip install 'sum-engine[sieve]'`.
# The [sieve] extra is the deterministic, offline (spaCy) extractor —
# no API key, no network. `sum attest` extracts (subject, predicate,
# object) triples and emits a signed CanonicalBundle on stdout.
echo "== 1. render (published sum-engine) =="
sum attest --extractor sieve < "$HERE/prose.txt" > "$BUNDLE"
echo "wrote $BUNDLE"

# ── Runtime 2: verify in Python ─────────────────────────────────────
echo
echo "== 2. verify in Python (sum verify) =="
sum verify --input "$BUNDLE"

# ── Runtime 3: verify the SAME bytes in Node ────────────────────────
# Zero npm dependencies; reconstructs the Gödel state integer from the
# canonical tome and checks it matches the Python-exported state.
echo
echo "== 3. verify in Node (standalone_verifier) =="
node "$REPO_ROOT/standalone_verifier/verify.js" "$BUNDLE"

# ── Runtime 4 (browser, manual) ─────────────────────────────────────
echo
echo "== 4. (browser) open single_file_demo/index.html and paste the bundle =="
echo "   The SAME bytes verify under WebCrypto in Chrome / Firefox / Safari."
