#!/usr/bin/env bash
# Real-prose run for slider_drift_bench: 8 multi-fact paragraphs ×
# 5 axes × 5 positions = 200 cells. Estimated cost ~$0.30 in tokens.
# Output: /tmp/slider_paragraphs.jsonl + summary on stderr.
set -e
cd "$(dirname "$0")/../.."
python -m Tests.benchmarks.slider_drift_bench \
  --corpus scripts/bench/corpora/seed_paragraphs.json \
  --out /tmp/slider_paragraphs.jsonl
