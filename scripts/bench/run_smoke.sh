#!/usr/bin/env bash
# Smoke run for slider_drift_bench: 1 doc, all 5 axes × 5 positions = 25 cells.
# Output: /tmp/slider_smoke.jsonl + summary on stderr.
set -e
cd "$(dirname "$0")/../.."
python -m Tests.benchmarks.slider_drift_bench \
  --corpus scripts/bench/corpora/seed_tiny.json \
  --max-docs 1 \
  --out /tmp/slider_smoke.jsonl
