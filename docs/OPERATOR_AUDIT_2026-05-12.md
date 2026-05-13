# Operator Audit — 2026-05-12

**Tool:** `scripts/probes/operator_audit.sh`
**Live deployment audited:** https://sum-demo.ototao.workers.dev
**Worker version at audit time:** `cd9ad611-81cb-4f7f-aba3-dd23bff287e3`

This is a point-in-time audit of the live Cloudflare Worker's configuration vs. what `worker/wrangler.toml` declares. It exists because the 2026-05-11 session surfaced **two production regressions in two deploys**: Wrangler ≥3.94 silently overwrites dashboard-set `[vars]` with whatever's in `wrangler.toml`, and an API key value was accidentally stored as a secret *name* in the past. The audit script catches both classes pre-emptively and is meant to be re-run before any future deploy or after any incident.

---

## 1. What the audit checks

| Check | Drift signal | Severity |
|---|---|---|
| Secret names — flagged if they match known API-key prefixes (`sk-ant-`, `sk-proj-`, `sk-or-`, `hf_`, `Bearer `) | Past `wrangler secret put` mistake where the key value was passed as the name | Critical — key value exposed in secret-listing surface |
| `wrangler.toml [vars]` enumeration | Decides what the next deploy will overwrite at the Worker level | Informational |
| `/.well-known/jwks.json` returns valid JWKS | Indicates `RENDER_RECEIPT_PUBLIC_JWKS` either pinned in `wrangler.toml` or set via `wrangler deploy --var` | High — receipt verification breaks without |
| `/.well-known/revoked-kids.json` returns canonical `sum.revoked_kids.v1` JSON | Indicates Worker deployed at or after `f4c75bf` (G3 revocation MVP) and CF edge cache isn't holding stale HTML | Medium — revocation surface dark without |
| `POST /api/render` with off-center sliders produces a signed receipt | Indicates operator's `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` is healthy, not revoked | High — operator-funded render path broken without |

The script never queries secret *values* (they're not queryable via wrangler). Leaked-name secrets are reported with the name truncated to its first 12 characters + length + class so output is safe to paste into a public incident-response doc.

Run via Make target:

```bash
make probe-operator-audit                          # human-readable
DEMO=https://other-deploy.workers.dev \
  make probe-operator-audit                        # alternate deploy
bash scripts/probes/operator_audit.sh --json       # machine-readable
```

Exit codes: 0 = clean, 1 = drift detected, 2 = audit could not complete (wrangler not on PATH).

---

## 2. Findings — 2026-05-12

| Check | State | Action required |
|---|---|---|
| `ANTHROPIC_API_KEY` (real secret) | ✓ Present | None on naming; *value is stale*, see render-path finding |
| `RENDER_RECEIPT_SIGNING_JWK` | ✓ Present | None |
| `RENDER_RECEIPT_SIGNING_KID` | ✓ Present | None |
| **`sk-ant-api03…(len=108,class=LEAKED-KEY-AS-NAME)`** | ✗ Secret whose *name* is a full Anthropic API key value | **Operator: `npx wrangler secret delete '<full-name>'` then rotate the underlying Anthropic key at console.anthropic.com if not already revoked.** Tracked in this session's commit history; key was rotated on 2026-05-11; secret-name deletion still pending. |
| `wrangler.toml [vars]` declared: `SUM_DEFAULT_MODEL_ANTHROPIC`, `SUM_DEFAULT_MODEL_OPENAI` | (on this branch, pre-PR #208 merge) | After PR #208 merges, will also include `RENDER_RECEIPT_PUBLIC_JWKS` — eliminating the per-deploy `--var` requirement |
| `/.well-known/jwks.json` | ✓ HTTP 200, 1 key (`kid=sum-render-2026-04-27-1`) | None — currently served from a `wrangler deploy --var` invocation; will move to wrangler.toml pin on PR #208 merge |
| `/.well-known/revoked-kids.json` | ✓ HTTP 200, schema `sum.revoked_kids.v1` | None |
| `POST /api/render` (operator path) | ✗ Anthropic API returns 401 — operator's `ANTHROPIC_API_KEY` is the revoked key | **Operator: `wrangler secret put ANTHROPIC_API_KEY` with the new rotated value.** BYO-keys path (settings panel) works around this in the meantime. |

**Two operator actions remain** to close all drift:

1. Delete the orphaned leaked-name secret. The full name is the raw value of the (now-revoked) Anthropic key; obtain it by running `npx wrangler secret list` locally, then:
   ```
   cd worker && npx wrangler secret delete '<full-name-from-secret-list>'
   ```
   The secret-name string is not committed to this repo for the obvious reason that committing it triggers GitHub's push-protection scanner — the key is revoked at console.anthropic.com but its string value still matches the API-key pattern.
2. After upstream rotation at console.anthropic.com, install the new key:
   ```
   cd worker && npx wrangler secret put ANTHROPIC_API_KEY
   # paste the new sk-ant-… value at the prompt
   ```

After both, re-run `make probe-operator-audit` — should exit 0.

---

## 3. Forward policy

### What can drift, and how we now catch it

| Drift class | Detection | Mitigation |
|---|---|---|
| Dashboard-set `[vars]` wiped on next `wrangler deploy` | Audit script's endpoint contract checks | Pin all required vars in `wrangler.toml` (RENDER_RECEIPT_PUBLIC_JWKS done in PR #208) |
| Secret value rotated upstream but not updated on Worker | Audit script's `/api/render` 401 check | `wrangler secret put` after every upstream rotation |
| Secret name accidentally set to a key value | Audit script's prefix-match check | Always use `wrangler secret put NAME` (positional) + paste value on stdin prompt — never pipe the key as the name |
| Worker code deployed but CDN cache stuck on pre-deploy HTML | Audit script's endpoint schema check + cache-buster query | Worker handlers set `cache-control: no-store` on dynamic routes; static `/.well-known/*` routes are matched before asset binding via `run_worker_first` |
| `RENDER_RECEIPT_SIGNING_JWK` (private signing key) compromised | Out of scope for this audit (private key never queryable); detection is via the JWKS-rotation cadence + `docs/INCIDENT_RESPONSE.md` case 1 | Rotate key + bump kid + update `RENDER_RECEIPT_PUBLIC_JWKS` in `wrangler.toml` |

### When to run

- **Before every `wrangler deploy`** — catches what the next deploy will overwrite
- **After every secret rotation** — confirms `wrangler secret put` reached the live Worker
- **Periodically (weekly?) on the deployed Worker** — catches CDN-cache or upstream-rotation drift before a funder reviewer notices
- **On every Worker-deploy CI run** — would catch regressions automatically; not yet wired into `.github/workflows/deploy-worker.yml` (follow-up)

### Forward-compat policy for new secrets

Any new secret must:
1. Have a documented purpose in `worker/wrangler.toml` comments.
2. Be set via `wrangler secret put NAME` followed by a stdin prompt — **never** pipe the key value as the positional `NAME` argument.
3. Have its purpose noted in `docs/THREAT_MODEL.md` if it's a key-material or auth-bearing secret.
4. Be enumerable by the audit script (the prefix-match logic will need a new entry if the vendor's keys have a distinctive prefix).

---

## 4. Cross-references

- [`worker/wrangler.toml`](../worker/wrangler.toml) — the `[vars]` block is the source of truth for what survives deploys.
- [`docs/THREAT_MODEL.md`](THREAT_MODEL.md) — attack-surface table; the 2026-05-11 BYO-keys arc added three rows; the secret-name-leak class added in this audit doc as a forward-policy item.
- [`docs/INCIDENT_RESPONSE.md`](INCIDENT_RESPONSE.md) — case 1 (render-key compromise) is the runbook for the leaked-name finding; case 2 (JWKS drift) is the runbook for the `[vars]` regression class.
- [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) — the wire spec the JWKS endpoint serves keys for.
- [`scripts/probes/operator_audit.sh`](../scripts/probes/operator_audit.sh) — the source for this audit; rerunnable any time.
