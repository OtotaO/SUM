#!/usr/bin/env bash
# Scale-verification run for slider_drift_bench: 16 multi-paragraph
# documents (200-400 words each, expected 15-30 source triples post-
# extraction). 5 axes × 5 positions = 400 cells. Estimated cost
# ~$1.50 with NLI audit. Wall clock ~3-4 min at concurrency=16.
#
# Used to confirm whether the v0.4-verified preservation claim
# (median 1.000, p10 0.818 on 4-12 triple docs) generalises to
# longer documents with 15-30 triples each.
set -e
cd "$(dirname "$0")/../.."
python -m Tests.benchmarks.slider_drift_bench \
  --corpus scripts/bench/corpora/seed_long_paragraphs.json \
  --out /tmp/slider_long_paragraphs.jsonl
