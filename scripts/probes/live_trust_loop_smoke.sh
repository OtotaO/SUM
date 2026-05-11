#!/usr/bin/env bash
#
# live_trust_loop_smoke.sh — end-to-end adversarial probe against the live
# SUM hosted demo at sum-demo.ototao.workers.dev (or any deployment URL
# passed as the first argument).
#
# Why this script exists: the local pytest suite and the cross-runtime
# K/A gates prove the trust loop against fixtures. They do NOT prove it
# against the production deployment. This probe set runs a 9-fixture
# adversarial pass against a live /api/render call:
#
#   1. POST sample triples with off-center sliders → trigger LLM path
#   2. Confirm response carries a signed render_receipt with non-canonical provider
#   3. Fetch JWKS; confirm Ed25519 OKP shape, kid matches receipt
#   4. Python verify_receipt() accepts the genuine receipt
#   5. Tamper payload.model         → must reject (bad_signature)
#   6. Tamper payload.provider      → must reject (bad_signature)
#   7. Tamper payload.sliders_quantized.density → must reject (bad_signature)
#   8. Tamper kid to an unknown one → must reject (no key in JWKS)
#   9. Tamper payload.tome_hash     → must reject (bad_signature)
#   10. Recompute sha256(served_tome) and compare to payload.tome_hash
#
# Exit 0 if all 10 fixtures pass; non-zero with the first failing fixture
# named on stderr. Pipe-friendly for CI gates.
#
# Requirements (caller's responsibility):
#   - curl on PATH
#   - python3 on PATH
#   - `pip install 'sum-engine[receipt-verify]'` resolvable from the
#     python3 binary's environment (i.e. importable as
#     `sum_engine_internal.render_receipt`)
#
# Usage:
#   bash scripts/probes/live_trust_loop_smoke.sh
#   bash scripts/probes/live_trust_loop_smoke.sh https://my-deploy.workers.dev
#
# Author: ototao
# License: Apache-2.0

set -euo pipefail

DEMO_URL="${1:-https://sum-demo.ototao.workers.dev}"
TMP_DIR="${TMPDIR:-/tmp}/sum-trust-loop-probe-$$"
mkdir -p "$TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "─── Live trust-loop probe against ${DEMO_URL} ─────────────────────"

# ─── Fixture 1: render call exercising LLM path ──────────────────────
echo "[1/10] POST /api/render with off-center sliders + force_render"
curl -fsS -m 60 -X POST "${DEMO_URL}/api/render" \
    -H 'content-type: application/json' \
    -d '{
      "triples":[
        ["alice","likes","cats"],
        ["bob","owns","dog"],
        ["carol","reads","books"]
      ],
      "slider_position":{
        "density":1.0,"length":0.9,"formality":0.9,
        "audience":0.5,"perspective":0.5
      },
      "force_render":true
    }' > "${TMP_DIR}/render.json"

# ─── Fixture 2: response shape sanity ─────────────────────────────────
echo "[2/10] Validate response shape"
python3 - <<'PY' "${TMP_DIR}/render.json"
import json, sys
with open(sys.argv[1]) as f:
    r = json.load(f)
assert "tome" in r and r["tome"], "response missing tome"
assert "render_receipt" in r, "response missing render_receipt"
rr = r["render_receipt"]
assert rr.get("schema") == "sum.render_receipt.v1", \
    f"unexpected receipt schema: {rr.get('schema')!r}"
assert rr.get("kid"), "receipt missing kid"
p = rr.get("payload", {})
assert p.get("provider") != "canonical-path", \
    "off-center sliders should have triggered LLM path, not canonical-path"
assert p.get("digital_source_type") == "trainedAlgorithmicMedia"
print(f"  provider={p['provider']} model={p['model']} kid={rr['kid']}")
PY

# ─── Fixture 3: JWKS shape + kid match ────────────────────────────────
echo "[3/10] Fetch JWKS and verify kid match"
curl -fsS -m 10 "${DEMO_URL}/.well-known/jwks.json" > "${TMP_DIR}/jwks.json"
python3 - <<'PY' "${TMP_DIR}/jwks.json" "${TMP_DIR}/render.json"
import json, sys
with open(sys.argv[1]) as f:
    jwks = json.load(f)
with open(sys.argv[2]) as f:
    receipt_kid = json.load(f)["render_receipt"]["kid"]
keys = jwks.get("keys", [])
assert keys, "JWKS has no keys"
matching = [k for k in keys if k.get("kid") == receipt_kid]
assert matching, f"no JWKS key for receipt kid={receipt_kid!r}"
k = matching[0]
assert k.get("kty") == "OKP" and k.get("crv") == "Ed25519" and k.get("alg") == "EdDSA", \
    f"unexpected key shape: {k}"
print(f"  kid={receipt_kid} kty=OKP crv=Ed25519 alg=EdDSA ✓")
PY

# ─── Fixtures 4-9: tamper suite via verify_receipt ────────────────────
echo "[4/10] verify_receipt accepts genuine receipt"
echo "[5/10] tamper payload.model         → must reject"
echo "[6/10] tamper payload.provider      → must reject"
echo "[7/10] tamper sliders_quantized     → must reject"
echo "[8/10] tamper kid to unknown        → must reject"
echo "[9/10] tamper payload.tome_hash     → must reject"
python3 - <<'PY' "${TMP_DIR}/render.json" "${TMP_DIR}/jwks.json"
import copy
import json
import sys
import warnings

warnings.filterwarnings("ignore")  # joserfc EdDSA deprecation warning

from sum_engine_internal.render_receipt import verify_receipt

with open(sys.argv[1]) as f:
    response = json.load(f)
with open(sys.argv[2]) as f:
    jwks = json.load(f)
receipt = response["render_receipt"]

failed = []

def expect(label, accept_expected, mutate=None):
    r = copy.deepcopy(receipt)
    if mutate:
        mutate(r)
    try:
        verify_receipt(r, jwks)
        outcome = "ACCEPT"
    except Exception as e:
        outcome = f"REJECT ({type(e).__name__})"
    if accept_expected and not outcome.startswith("ACCEPT"):
        failed.append(f"  {label}: expected ACCEPT, got {outcome}")
    elif not accept_expected and outcome.startswith("ACCEPT"):
        failed.append(f"  {label}: expected REJECT, got {outcome} (signature held under mutation!)")

expect("genuine", True)
expect("tamper-model",     False, lambda r: r["payload"].__setitem__("model", "attacker"))
expect("tamper-provider",  False, lambda r: r["payload"].__setitem__("provider", "openai"))
expect("tamper-sliders",   False, lambda r: r["payload"]["sliders_quantized"].__setitem__("density", 0.1))
expect("tamper-kid",       False, lambda r: r.__setitem__("kid", "attacker-key-2026"))
expect("tamper-tome-hash", False, lambda r: r["payload"].__setitem__("tome_hash", "sha256-" + "0"*64))

if failed:
    print("  ✗ FAIL:", file=sys.stderr)
    for line in failed:
        print(line, file=sys.stderr)
    sys.exit(1)
print("  ✓ all 6 verify_receipt fixtures behaved correctly")
PY

# ─── Fixture 10: served tome ↔ payload.tome_hash ──────────────────────
echo "[10/10] sha256(served tome) == payload.tome_hash"
python3 - <<'PY' "${TMP_DIR}/render.json"
import hashlib, json, sys
with open(sys.argv[1]) as f:
    r = json.load(f)
tome = r["tome"]
expected = "sha256-" + hashlib.sha256(tome.encode("utf-8")).hexdigest()
actual = r["render_receipt"]["payload"]["tome_hash"]
if expected != actual:
    print(f"  ✗ FAIL: served tome does NOT match payload.tome_hash", file=sys.stderr)
    print(f"    expected={expected}", file=sys.stderr)
    print(f"    actual  ={actual}", file=sys.stderr)
    sys.exit(1)
print(f"  ✓ tome_hash matches recomputed sha256(tome): {actual[:30]}…")
PY

echo "─── Live trust-loop probe PASSED (10/10) ──────────────────────────"
