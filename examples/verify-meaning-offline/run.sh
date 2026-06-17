#!/usr/bin/env bash
# Replay SUM's binding-gate meaning-loss bound — fully offline.
#
# No numpy / scipy / torch, no GPU, no network. The only install is the
# dependency-light verifier SDK; the signed golden (BillSum, CC0 public domain)
# ships in this repo. Run from a checkout:
#
#     pip install 'sum-engine[verify]'
#     examples/verify-meaning-offline/run.sh
#
# A `verified: true` + `replayed: true` verdict means the committed per-pair
# losses hash to the receipt's anchor and re-certify to its stated bound by
# exact integer equality — checked on your machine, trusting nobody. Read the
# `proxy_caveat`: that PASS is a cryptographic fact, not proof meaning was
# preserved (the bound is over a proxy that tracks human judgment only modestly).
set -euo pipefail
cd "$(dirname "$0")/../.."

python -m sum_verify \
  fixtures/meaning_receipts_billsum/meaning_risk_receipt.billsum.golden.json \
  --jwks   fixtures/meaning_receipts_billsum/jwks.json \
  --losses fixtures/meaning_receipts_billsum/losses_billsum.json
