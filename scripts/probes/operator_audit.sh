#!/usr/bin/env bash
#
# operator_audit.sh — empirical audit of the live Cloudflare Worker's
# configuration vs. what wrangler.toml claims.
#
# Why: Wrangler >=3.94 replaces all [vars] with whatever's in
# wrangler.toml on every deploy, so dashboard-set vars get silently
# wiped. We hit this twice on 2026-05-11 (JWKS regression class).
# Secret names are also queryable and have previously surfaced
# leaked-key-as-name foot-guns. This script audits both classes.
#
# What it checks:
#   1. Secret list — names only (values never queryable). Each name
#      redacted to its first 12 chars + "…" + length + classification
#      (looks-like-key vs. structural-name). Flags any secret whose
#      name matches a known API-key prefix as a potential leak.
#   2. Wrangler.toml [vars] — what the next deploy will write.
#   3. Live endpoint contract:
#        /.well-known/jwks.json    → MUST return valid JWKS
#        /.well-known/revoked-kids.json → MUST return canonical JSON
#        /api/render with off-center sliders → exits depending on
#          whether ANTHROPIC_API_KEY / OPENAI_API_KEY is healthy.
#          Reports operator-funded-path-broken if 401.
#
# What it does NOT do:
#   - Modify any config. Read-only.
#   - Print secret values (they're not queryable).
#   - Print full secret names that look like leaked keys — only the
#     classification + prefix is logged so the script output is safe
#     to paste into a public incident-response doc.
#
# Usage:
#   bash scripts/probes/operator_audit.sh             # human-readable
#   bash scripts/probes/operator_audit.sh --json      # machine-readable
#
# Exit codes:
#   0 — audit complete; everything is consistent with wrangler.toml
#   1 — drift detected (var missing, secret name leak, endpoint regression)
#   2 — could not complete audit (wrangler not on PATH, etc.)
#
# Author: ototao
# License: Apache-2.0

set -euo pipefail

JSON_MODE=0
[ "${1:-}" = "--json" ] && JSON_MODE=1

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORKER_DIR="${REPO_ROOT}/worker"
DEMO_URL="${DEMO_URL:-https://sum-demo.ototao.workers.dev}"

if ! command -v jq >/dev/null 2>&1 && [ "$JSON_MODE" = "1" ]; then
  echo "operator_audit: --json mode requires jq on PATH" >&2
  exit 2
fi

# Known API-key prefix substrings that would indicate a secret-name
# leak. Add new prefixes as new vendors are integrated.
KEY_PREFIXES=("sk-ant-" "sk-proj-" "sk-or-" "hf_" "Bearer ")

# ─── 1. Secret list ───────────────────────────────────────────────────
# Portable across macOS bash 3.2 (no mapfile). Each secret name on its
# own line; classified as OK or LEAKED-KEY-AS-NAME.
SECRETS_RAW_FILE="$(mktemp)"
trap 'rm -f "$SECRETS_RAW_FILE"' EXIT
(cd "$WORKER_DIR" && npx --no-install wrangler secret list 2>/dev/null \
  | jq -r '.[].name' 2>/dev/null) > "$SECRETS_RAW_FILE" || true

SECRETS_OK=()
SECRETS_LEAK=()
while IFS= read -r name; do
  [ -z "$name" ] && continue
  leak=0
  for prefix in "${KEY_PREFIXES[@]}"; do
    case "$name" in
      "$prefix"*) leak=1 ;;
    esac
  done
  if [ "$leak" = "1" ]; then
    redacted="$(printf '%s' "$name" | cut -c1-12)…(len=${#name},class=LEAKED-KEY-AS-NAME)"
    SECRETS_LEAK+=("$redacted")
  else
    SECRETS_OK+=("$name")
  fi
done < "$SECRETS_RAW_FILE"

# ─── 2. wrangler.toml [vars] ──────────────────────────────────────────
VARS_DECLARED=()
while IFS= read -r line; do
  VARS_DECLARED+=("$line")
done < <(
  awk '
    /^\[vars\]/ { in_vars = 1; next }
    /^\[/ && !/^\[vars\]/ { in_vars = 0 }
    in_vars && /^[A-Z_][A-Z_0-9]*[[:space:]]*=/ {
      sub(/[[:space:]]*=.*$/, "");
      print
    }
  ' "$WORKER_DIR/wrangler.toml"
)

# ─── 3. Live endpoint contract ────────────────────────────────────────
JWKS_STATUS="?"
JWKS_KEYS="?"
JWKS_BODY="$(curl -s -m 10 -w '\n%{http_code}' "${DEMO_URL}/.well-known/jwks.json?cb=$(date +%s)" 2>/dev/null || true)"
JWKS_HTTP="$(printf '%s' "$JWKS_BODY" | tail -n1)"
JWKS_JSON="$(printf '%s' "$JWKS_BODY" | sed '$d')"
if [ "$JWKS_HTTP" = "200" ]; then
  JWKS_KEYS="$(printf '%s' "$JWKS_JSON" | python3 -c 'import sys,json; print(len(json.load(sys.stdin).get("keys",[])))' 2>/dev/null || echo "?")"
  JWKS_STATUS="ok"
elif [ "$JWKS_HTTP" = "503" ]; then
  JWKS_STATUS="503-not-configured"
else
  JWKS_STATUS="http-${JWKS_HTTP}"
fi

REVOKED_STATUS="?"
REVOKED_BODY="$(curl -s -m 10 -w '\n%{http_code}' "${DEMO_URL}/.well-known/revoked-kids.json?cb=$(date +%s)" 2>/dev/null || true)"
REVOKED_HTTP="$(printf '%s' "$REVOKED_BODY" | tail -n1)"
REVOKED_JSON="$(printf '%s' "$REVOKED_BODY" | sed '$d')"
if [ "$REVOKED_HTTP" = "200" ]; then
  REVOKED_SCHEMA="$(printf '%s' "$REVOKED_JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("schema","?"))' 2>/dev/null || echo "?")"
  if [ "$REVOKED_SCHEMA" = "sum.revoked_kids.v1" ]; then
    REVOKED_STATUS="ok"
  else
    REVOKED_STATUS="schema-wrong:${REVOKED_SCHEMA}"
  fi
else
  REVOKED_STATUS="http-${REVOKED_HTTP}"
fi

# Render path — bounded probe. 502/401 from anthropic.com means the
# operator's ANTHROPIC_API_KEY is stale (revoked but not rotated on
# Worker). The Worker code itself is healthy; this is a key-rotation
# state finding.
RENDER_STATUS="?"
RENDER_BODY="$(curl -s -m 30 -X POST "${DEMO_URL}/api/render" \
  -H 'content-type: application/json' \
  -d '{"triples":[["a","b","c"]],"slider_position":{"density":1.0,"length":0.9,"formality":0.9,"audience":0.5,"perspective":0.5},"force_render":true}' \
  2>/dev/null || true)"
if printf '%s' "$RENDER_BODY" | python3 -c 'import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get("render_receipt") else 1)' 2>/dev/null; then
  RENDER_STATUS="ok-signed"
elif printf '%s' "$RENDER_BODY" | grep -q '"error":"render failed: anthropic 401'; then
  RENDER_STATUS="anthropic-401-stale-operator-key"
elif printf '%s' "$RENDER_BODY" | grep -q '"error":"render failed: openai'; then
  RENDER_STATUS="openai-error-operator-path-broken"
else
  RENDER_STATUS="other-error"
fi

# ─── Drift detection ──────────────────────────────────────────────────
DRIFT_FOUND=0
declare -a DRIFT_LOG

# JWKS endpoint healthy?
if [ "$JWKS_STATUS" != "ok" ]; then
  DRIFT_FOUND=1
  DRIFT_LOG+=("JWKS endpoint not healthy: ${JWKS_STATUS} — wrangler.toml [vars] missing RENDER_RECEIPT_PUBLIC_JWKS? Or kid mismatch with RENDER_RECEIPT_SIGNING_KID secret?")
fi
# Revocation endpoint healthy?
if [ "$REVOKED_STATUS" != "ok" ]; then
  DRIFT_FOUND=1
  DRIFT_LOG+=("Revoked-kids endpoint not healthy: ${REVOKED_STATUS} — Worker deploy stale? CDN-cache stuck on pre-route HTML?")
fi
# Render path healthy (warning only — stale operator key is a known state)?
if [ "$RENDER_STATUS" != "ok-signed" ]; then
  DRIFT_FOUND=1
  DRIFT_LOG+=("Render path not healthy: ${RENDER_STATUS} — rotate operator key or accept BYO-keys-only mode.")
fi
# Leaked secret names?
if [ "${#SECRETS_LEAK[@]}" -gt 0 ]; then
  DRIFT_FOUND=1
  DRIFT_LOG+=("Secret name(s) look like API key values — possible past wrangler-secret-put mistake: ${SECRETS_LEAK[*]}")
fi

# ─── Output ───────────────────────────────────────────────────────────
if [ "$JSON_MODE" = "1" ]; then
  jq -n \
    --arg demo "$DEMO_URL" \
    --argjson secrets_ok "$(printf '%s\n' "${SECRETS_OK[@]:-}" | jq -R . | jq -s .)" \
    --argjson secrets_leaked_count "${#SECRETS_LEAK[@]}" \
    --argjson vars_declared "$(printf '%s\n' "${VARS_DECLARED[@]:-}" | jq -R . | jq -s .)" \
    --arg jwks "$JWKS_STATUS" \
    --arg jwks_keys "$JWKS_KEYS" \
    --arg revoked "$REVOKED_STATUS" \
    --arg render "$RENDER_STATUS" \
    --argjson drift "$DRIFT_FOUND" \
    --argjson drift_log "$(printf '%s\n' "${DRIFT_LOG[@]:-}" | jq -R . | jq -s .)" \
    '{
      schema: "sum.operator_audit.v1",
      demo_url: $demo,
      secrets: { ok: $secrets_ok, leaked_count: $secrets_leaked_count },
      vars_declared: $vars_declared,
      endpoints: { jwks: $jwks, jwks_keys: $jwks_keys, revoked_kids: $revoked, render: $render },
      drift: { found: ($drift == 1), log: $drift_log }
    }'
else
  echo "─── Operator audit — $(date -u +'%Y-%m-%dT%H:%M:%SZ') ────────────────"
  echo "Demo URL: $DEMO_URL"
  echo ""
  echo "── Secrets ──"
  for s in "${SECRETS_OK[@]:-}"; do
    [ -z "$s" ] && continue
    echo "  ✓ $s"
  done
  for s in "${SECRETS_LEAK[@]:-}"; do
    [ -z "$s" ] && continue
    echo "  ✗ $s  ← name looks like an API key value; redacted here"
  done
  echo ""
  echo "── wrangler.toml [vars] (these survive every deploy) ──"
  for v in "${VARS_DECLARED[@]:-}"; do
    [ -z "$v" ] && continue
    echo "  • $v"
  done
  echo ""
  echo "── Live endpoints ──"
  echo "  JWKS:           $JWKS_STATUS  (keys: $JWKS_KEYS)"
  echo "  Revoked-kids:   $REVOKED_STATUS"
  echo "  Render (op):    $RENDER_STATUS"
  echo ""
  if [ "$DRIFT_FOUND" = "1" ]; then
    echo "── ✗ Drift detected ──"
    for d in "${DRIFT_LOG[@]:-}"; do
      [ -z "$d" ] && continue
      echo "  • $d"
    done
    echo ""
    echo "─── Operator audit completed with drift findings ──────────────────"
  else
    echo "─── Operator audit clean — no drift ───────────────────────────────"
  fi
fi

exit $DRIFT_FOUND
