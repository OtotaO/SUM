# Incident response runbook

Operational reflexes for the eight failure modes the existing trust surface admits are possible. Each case has the same four fields:

- **Detection** — the signal that surfaces this case. Where you'd see it first.
- **Containment** — what to do in the first hour to stop the failure spreading.
- **Revocation / correction** — specific commands + files to update; the durable fix.
- **Public disclosure wording** — what to write in the advisory / changelog / status post.

This doc exists so the response is executed from a written reflex, not improvised under pressure. If you're reading this in the middle of an incident: take the runbook line by line. If you find a step that's wrong or missing, fix the doc in the same PR as the incident response so the next person has it.

For external-reporter intake (researcher → operator), see [`SECURITY.md`](../SECURITY.md). This file is for operators after a confirmed finding.

---

## 1. Render signing key suspected compromised

The Ed25519 key under the active `kid` (currently `sum-render-2026-04-27-1`, served from `sum-demo.ototao.workers.dev/.well-known/jwks.json`) is suspected leaked / extracted / used by an unauthorised party. This is the highest-severity case — every render receipt is only as trustworthy as this key.

**Detection.** Any of: (a) anomalous `signed_at` timestamps in receipts you didn't issue; (b) Cloudflare Workers Secrets Store audit log shows access from an unexpected source; (c) a researcher reports they can produce signatures verifying under your `kid` without server access; (d) the Worker source repo shows commits you didn't author touching the receipt-signing path.

**Containment.** Immediately:
1. Rotate the signing key. Generate a new keypair locally:
   ```
   npx tsx scripts/cert/gen_render_keypair.ts sum-render-<YYYY-MM-DD>-1
   ```
2. Upload the new private JWK as a Worker secret:
   ```
   wrangler secret put RENDER_RECEIPT_SIGNING_JWK < /tmp/render_receipt_private.jwk
   ```
3. Add the new public JWK to `RENDER_RECEIPT_PUBLIC_JWKS` (alongside the compromised old one — don't drop the old one yet, downstream verifiers may still be holding receipts that reference it).
4. Update `RENDER_RECEIPT_SIGNING_KID` to the new kid.
5. Wipe the local tempfile: `rm -P /tmp/render_receipt_private.jwk` (macOS / BSD) or `shred -u` (Linux).
6. Deploy: `gh workflow run deploy-worker.yml`.

After containment, every new render uses the new kid; old receipts (issued with the compromised kid) still exist in the wild and CAN be re-issued by the compromiser until the old kid is on the revocation list.

**Revocation / correction.** G3 revocation MVP shipped — publish the compromised kid to `/.well-known/revoked-kids.json` via the `RENDER_RECEIPT_REVOKED_KIDS` Worker secret:

```bash
# Build the revocation list (sum.revoked_kids.v1 schema):
cat <<'JSON' > /tmp/revoked_kids.json
{
  "schema": "sum.revoked_kids.v1",
  "issued_at": "<UTC ISO-8601 timestamp NOW>",
  "revoked": [
    {
      "kid": "<compromised-kid>",
      "effective_revocation_at": "<UTC ISO-8601 of suspected first-compromise>",
      "reason": "compromise"
    }
  ]
}
JSON

# Push to the Worker:
wrangler secret put RENDER_RECEIPT_REVOKED_KIDS < /tmp/revoked_kids.json
rm -P /tmp/revoked_kids.json   # macOS / BSD; or `shred -u` on Linux

# Deploy:
gh workflow run deploy-worker.yml
```

Verifiers fetching `/.well-known/revoked-kids.json` and passing it to `verify_receipt(receipt, jwks, revoked_kids=...)` will reject receipts under the compromised kid whose `signed_at` ≥ `effective_revocation_at` with the `revoked_kid` error class — distinct from `signature_invalid` so the operator-side distinction is visible at the consumer.

Receipts signed BEFORE `effective_revocation_at` retain their original validity (revocation invalidates future trust, not past). This preserves the audit trail for legitimate historical renders signed before the compromise window. Set `effective_revocation_at` to the suspected first-compromise timestamp; receipts before that survive, receipts after reject.

The rotation grace window (case 1's containment step 7) covers the JWKS-side hygiene; the revocation list covers the operator-intent side. Both should be set on a real compromise — JWKS removal stops new responses being issued under the kid; revocation list tells consumers to stop trusting cached / archived receipts under it. See [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) §6.1.

**Public disclosure wording.**

> SUM render-signing key `<old-kid>` was rotated on `<date>` after credible evidence the key may have been compromised. Receipts issued under this kid before `<date>` may have been signed by an unauthorised party. Receipts issued from `<date>` onward use kid `<new-kid>` and are unaffected. Verifiers should treat receipts under `<old-kid>` with `signed_at` after `<date>` as suspect. We are coordinating with [reporter, if applicable]; no further customer action is required unless you maintain an archive of receipts you cryptographically rely on, in which case please re-fetch from the issuer to obtain a fresh receipt.

---

## 2. JWKS served wrong / stale / revoked key

The `/.well-known/jwks.json` endpoint serves a JWKS that does not match the actual signing key the Worker is using, OR serves a revoked key as still-valid, OR drops a kid before its rotation grace window has elapsed.

**Detection.** Any of: (a) verifier rejection rate spikes on `unknown_kid` for receipts that should still verify; (b) a researcher reports the JWKS contents don't match the key the Worker signs with; (c) `wrangler tail` shows the Worker emitting a `kid` in the protected header that's not present in JWKS.

**Containment.**
1. Fetch the current JWKS: `curl https://sum-demo.ototao.workers.dev/.well-known/jwks.json`.
2. Confirm against the Worker secret (`wrangler secret list`) which kid is supposed to be active.
3. If JWKS is missing the active kid: redeploy with the correct `RENDER_RECEIPT_PUBLIC_JWKS` value (the Worker reads this var at request time; redeploy isn't strictly required if the var was changed via `wrangler secret put`, but a redeploy provides a clean reset).
4. If JWKS contains a revoked kid: remove it from `RENDER_RECEIPT_PUBLIC_JWKS` and redeploy.

**Revocation / correction.** Same as case 1's revocation/correction path once G3 ships. Until then, the JWKS contents are the entire revocation mechanism.

**Public disclosure wording.**

> SUM's JWKS endpoint at `https://sum-demo.ototao.workers.dev/.well-known/jwks.json` served `<wrong content>` between `<start>` and `<end>` UTC. Verifiers fetching JWKS during that window would have `<rejected valid receipts | accepted revoked-key receipts>`. The endpoint has been corrected as of `<recovery time>`. Verifiers should re-fetch JWKS to clear cached state.

---

## 3. PyPI release suspected compromised

A published `sum-engine` wheel or sdist on PyPI either: (a) was uploaded by something other than the `OtotaO/SUM` Trusted Publisher workflow; (b) carries a PEP 740 attestation that doesn't pass `scripts/verify_pypi_attestation.py`; (c) has SHA-256 hashes that diverge from the local `dist_hashes.json` we built and shipped.

**Detection.** Any of: (a) `verify-pypi` post-publish detection job (the alarm-not-gate at the end of `publish-pypi.yml`) fails on a recent release; (b) a downstream user reports `pip install sum-engine==X.Y.Z` produces bytes that don't match published SHA-256 hashes; (c) the Trusted Publisher identity in the attestation doesn't match `OtotaO/SUM` + `publish-pypi.yml`.

**Containment.**
1. Yank the suspect release immediately:
   ```
   # Use the PyPI web UI: https://pypi.org/manage/project/sum-engine/release/<version>/
   # Click "Yank release" with a reason. This prevents new pip installs but
   # doesn't delete the file (you can't, by design — yanking is the
   # PyPI-supported intermediate state).
   ```
2. Disable the GitHub Actions environment for `pypi` (Settings → Environments → pypi → Disable) until the compromise is understood. This blocks any further publish runs.
3. Rotate the GitHub Actions OIDC trust by deleting + re-creating the Trusted Publisher entry on PyPI (Account Settings → Publishing → remove `OtotaO/SUM` → re-add). Forces a fresh OIDC handshake on the next legitimate publish.

**Revocation / correction.**
1. Re-publish a corrected version with the next patch number (`0.3.X+1`).
2. Document in `CHANGELOG.md` that the compromised version was yanked and superseded.
3. If the compromise propagated through a downstream ecosystem, file CVEs and request affected downstream registries to flag the compromised version.

**Public disclosure wording.**

> `sum-engine X.Y.Z` published on `<date>` was yanked on `<recovery date>` after we detected the published artifact did not match the bytes our release pipeline produced. The corrected release is `X.Y.Z+1`. If you installed `X.Y.Z`, run `pip install --upgrade sum-engine` and verify with `pip install sum-engine[receipt-verify] && python -m scripts.verify_pypi_attestation --index prod --version X.Y.Z+1 …`. We are coordinating with PyPI on the original publish path.

---

## 4. GitHub Actions workflow compromised

A workflow file under `.github/workflows/` is modified to do something other than its intended job — most commonly, exfiltrating a secret or publishing an artifact that wasn't built from the legitimate source.

**Detection.** Any of: (a) `vendor-byte-equivalence` job in `quantum-ci.yml` fails on bytes the pinned dep versions should produce (signal: someone modified `scripts/vendor/build.js` or `package-lock.json` without committing the regenerated bundle); (b) StepSecurity Harden-Runner audit log (R0.3, when it lands) shows anomalous network egress from a release job; (c) a workflow run produces an artifact whose hash doesn't match what the source tree should produce.

**Containment.**
1. Disable the affected workflow: Settings → Actions → Disable workflow.
2. Revoke any secrets the workflow had access to (`wrangler secret list` for Worker secrets; PyPI Trusted Publisher entries for publish workflows).
3. `git log --oneline .github/workflows/` to see recent changes to workflow files; revert or hard-fix any unauthorised modifications.

**Revocation / correction.**
1. Re-pin all third-party actions by full SHA (R0.3 task — `actions/checkout@v4` becomes `actions/checkout@a5ac7e51b41094c92402da3b24376905380afc29`).
2. Run OpenSSF Scorecard against the repo and address any findings the compromise relied on.
3. Audit the GitHub Actions audit log for the suspect time window.

**Public disclosure wording.**

> SUM's GitHub Actions workflow `<name>` was modified between `<start>` and `<end>` UTC by `<actor, if known>`. Artifacts produced during this window MAY contain code from sources other than `OtotaO/SUM`. Specifically: `<list of suspect artifacts + hashes>`. Users who installed during this window should `<upgrade-or-reinstall instructions>`. We have rotated `<list of secrets>`, re-pinned all third-party actions, and added `<additional hardening>` to prevent recurrence.

---

## 5. Worker deploy compromised

The deployed Worker bundle at `sum-demo.ototao.workers.dev` differs from what the source tree at `OtotaO/SUM:main` would produce.

**Detection.** Any of: (a) the live `/api/render` produces receipts whose `provider` or `model` fields don't match what the source code emits; (b) the live `/.well-known/jwks.json` serves keys from a different rotation than the local `RENDER_RECEIPT_PUBLIC_JWKS` value; (c) `curl` of a static asset (e.g. `index.html`) returns content that doesn't match the file at HEAD.

**Containment.**
1. Rotate the Cloudflare API token used by the deploy workflow (Cloudflare dashboard → My Profile → API Tokens → revoke + create new).
2. Update the GitHub repo secret `CLOUDFLARE_API_TOKEN` with the new token.
3. Re-deploy the Worker from a clean checkout: `cd worker && npm ci && npx wrangler deploy --compatibility-date <pinned>`.
4. Verify the redeploy: hash the deployed bundle, compare against a local `wrangler deploy --dry-run` artifact.

**Revocation / correction.**
1. Once the trust manifest (R0.2) lands, every release publishes a signed `sum.trust_root.v1.json` that names the expected Worker bundle hash. Verifiers MUST cross-check the deployed bytes against the manifest.
2. Until R0.2 lands, the only revocation path is the rotation+redeploy above.

**Public disclosure wording.**

> The SUM demo Worker at `sum-demo.ototao.workers.dev` served bundle `<hash>` between `<start>` and `<end>` UTC. The expected bundle for that period was `<expected hash>`. We have redeployed from a clean checkout and rotated the Cloudflare API token. If you depend on receipts issued during this window, please re-render with the corrected Worker — the affected receipts are listed in advisory `<id>`.

---

## 6. Benchmark claim later found wrong

A measured number in `docs/PROOF_BOUNDARY.md`, `docs/SLIDER_CONTRACT.md`, `CHANGELOG.md`, or the README turns out to be wrong on review (the bench had a bug, the corpus was tainted, the metric was misconfigured, etc.).

**Detection.** Any of: (a) re-running the bench produces materially different numbers; (b) a researcher reports the methodology has a flaw; (c) a code reviewer notices the bench's reported number doesn't match the JSONL artifact's contents.

**Containment.** This is not a security incident — it's an honesty incident. Don't deploy hotfixes; fix the doc.

1. Stop using the wrong number in any new commit, README, blog post, tweet, or external surface.
2. Re-run the bench with the corrected methodology and capture the new number.
3. Open a doc-only PR replacing the wrong number with the new one + a short note explaining what was wrong.

**Revocation / correction.**
1. Update `CHANGELOG.md` with a `### Corrected` entry naming the wrong number, the corrected number, and the methodology flaw.
2. Update `docs/PROOF_BOUNDARY.md` if the wrong claim was at the proved/measured/designed boundary.
3. Cross-reference the correction in any other doc that quoted the wrong number.

**Public disclosure wording.**

> A benchmark number SUM published on `<date>` was incorrect. The original claim of `<X>` should have been `<Y>`; the methodology error was `<short description>`. The number is corrected as of commit `<sha>` and `CHANGELOG.md` carries the full record. We are not aware of downstream decisions made on the basis of the wrong number; if you depend on it, the corrected value is in `<canonical doc>`.

The truthfulness contract in `docs/PROOF_BOUNDARY.md` §5 means correcting a wrong number is the right move regardless of how visible the error was — the doc is its own arbiter of when correction is required.

---

## 7. LLM provider silently changes served-model behavior

The Anthropic API (production) returns responses that no longer match the contract SUM's Worker assumes — the `model` field returns a different snapshot than requested, response format changes, content-policy filtering is applied that wasn't before, etc.

**Detection.** Any of: (a) the receipt-aware audit (`scripts/bench/runners/receipt_audit.py`) shows `payload.model` returning an unexpected value; (b) `payload.model` ends with `_inferred` (the Worker's fallback when the API doesn't echo a model), suggesting Anthropic stopped echoing model IDs; (c) bench fact-preservation drops on cells that used to verify cleanly with no other change.

**Containment.**
1. Pin the model snapshot harder: update `SUM_DEFAULT_MODEL_ANTHROPIC` (Worker secret) to a known-good explicit dated snapshot (e.g. `claude-haiku-4-5-20251001` rather than the moving `claude-haiku-latest`).
2. If receipts continue to reference the unexpected model, document this as a known divergence in `CHANGELOG.md` and decide whether the new model snapshot is acceptable for the bench's contract.

**Revocation / correction.** This is provider-side; SUM cannot rotate Anthropic. The receipts SUM has already issued are still cryptographically valid — they assert "this issuer signed a render attributing this output to this provider's API at this time." If the provider changed semantics, that's true and visible in the receipt's `model` field. Update doc surfaces (PROOF_BOUNDARY §2.6, the live demo's preservation copy if needed) to reflect the new measurement.

**Public disclosure wording.**

> Anthropic's `<model>` snapshot used by the SUM render Worker shifted behavior on `<date>`. Receipts issued before `<date>` reflect the original behavior; receipts after reflect the new behavior. SUM's bench numbers in `docs/SLIDER_CONTRACT.md` were re-measured on `<date>` and have been updated to the post-shift values. The receipts themselves remain cryptographically valid; the receipt's `payload.model` field is the load-bearing record of which model served each render.

---

## 8. Canonicalisation bug discovered in Python / Node / browser verifier

The Python `jcs` module, the JS `canonicalize` library, or the verifier's handling of either, produces different bytes for the same JSON value across runtimes. A correctly-signed receipt that should verify across all three runtimes is rejected by one or more.

**Detection.** Any of: (a) `vendor-byte-equivalence` CI job fails on the receipt-verifier smoke even though the vendored bundle bytes match; (b) a fixture under `fixtures/render_receipts/` produces different `expected_error_class` values across the JS Node smoke and the Python pytest run; (c) a researcher reports a JSON value that canonicalises differently between the two implementations.

**Containment.**
1. Pin the failing fixture as a permanent regression test in both `Tests/test_render_receipt_verifier.py` and `single_file_demo/test_render_receipt_verify.js`.
2. Identify which side is wrong: hand-canonicalise per RFC 8785 spec letter and compare. Whichever output diverges from the spec is the bug; the other side's behavior is the contract.

**Revocation / correction.**
1. Patch the diverging implementation. For Python, that's `sum_engine_internal/infrastructure/jcs.py`. For JS, the vendored `canonicalize` library — file an upstream issue at `github.com/erdtman/canonicalize` and pin a workaround in `scripts/vendor/build.js` (transform the upstream output) until the upstream fix lands.
2. Update `docs/PROOF_BOUNDARY.md` §1.8's "proved on adversarial inputs across runtimes" claim with a footnote naming the bug + fix commit.

**Public disclosure wording.**

> A canonicalisation divergence between SUM's Python and JS receipt verifiers was discovered on `<date>`. The bug caused `<X>` receipts to verify in one runtime but not the other. A fix landed in commit `<sha>` and is in releases `>= sum-engine X.Y.Z` and Worker deploy `<id>`. Receipts issued before the fix that hit the divergence are still cryptographically valid — the bug was in the verifier, not the signer. Affected verifiers should upgrade. The `vendor-byte-equivalence` and Python pytest CI jobs now both include a regression fixture for this case.

---

## Cross-references

- [`SECURITY.md`](../SECURITY.md) — researcher → operator intake (the inbound side).
- [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) — receipt wire format + key rotation cadence (§6).
- [`docs/TRUST_ROOT_FORMAT.md`](TRUST_ROOT_FORMAT.md) — R0.2 trust-root manifest spec; gives cases 3, 4, and 5 a single signed artifact downstream verifiers can cross-check against once a real release-signing key is in production.
- [`docs/TRANSPARENCY_ANCHOR.md`](TRANSPARENCY_ANCHOR.md) — R0.5 transparency-anchor design; once implementation lands, the anchor history is the external timestamped record for "what was served when" — the recovery surface for post-hoc tampering detection on cases 2, 3, and 5.
- [`docs/MODEL_CALL_EVIDENCE_FORMAT.md`](MODEL_CALL_EVIDENCE_FORMAT.md) — R0.5 sidecar design; the `request_body_hash` + `provider_response_id_hash` fields are the forensic binding that detects case 7 once implementation lands.
- [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) — what's proved vs measured vs designed; arbiter of when a correction is required.
- [`docs/NEXT_SESSION_PLAYBOOK.md`](NEXT_SESSION_PLAYBOOK.md) G3 — revocation + crypto-agility track that, once shipped, will give cases 1 and 2 a `revoked_kids.json` surface to publish to (instead of relying on JWKS-list timing alone).

## On the discipline

Incident response is one of those things every project says it has and few have actually rehearsed. The four-field structure here exists so the response is executed from a written reflex, not improvised under pressure. If you're reading this in a real incident: take the runbook line by line. If you find a step that's wrong or missing, fix the doc in the same PR as the incident response so the next person has it.
