# CI_WIRING_TO_PASTE_2026-05-17.md

**Status: operator-paste artifact.** Engine session lacks `workflow` OAuth scope and cannot push `.github/workflows/quantum-ci.yml` changes directly. This document stages the exact YAML block to add to that workflow + how to commit it locally. Deferred from PR #220 (cross-runtime fixture K-matrix for `sum.transform_receipt.v1`) per the operator-paste fallback rule.

## What to add

The cross-runtime fixture-set smokes for `sum.transform_receipt.v1` already exist as `single_file_demo/test_transform_receipt_fixtures.js` (JS) and `Tests/test_transform_receipt_verifier_fixtures.py` (Python). They run under `make pre-push` locally but are NOT yet wired into CI's `release-machinery-validation` job.

The mirror block follows the existing render-receipt smoke pair at lines 268–288 of `.github/workflows/quantum-ci.yml`.

## Where to insert

Open `.github/workflows/quantum-ci.yml`. Find the line:

```yaml
        python -m pytest Tests/test_render_receipt_verifier.py -q
```

(currently at line 288). Insert the new two-step block IMMEDIATELY AFTER that line, BEFORE the `Trust-root manifest round-trip — Python (R0.2)` step.

## The block — copy verbatim

```yaml
    - name: Transform-receipt-verifier smoke — JS (20 fixtures)
      # Cross-runtime K-matrix extension for sum.transform_receipt.v1.
      # Same shape as the render-receipt smoke above; consumes
      # fixtures/transform_receipts/ via the deterministic-test-key
      # generator. Caught a real verifier bug on first wiring (double-
      # decoded protectedHeader); keeping it in CI prevents regression.
      run: node single_file_demo/test_transform_receipt_fixtures.js

    - name: Transform-receipt-verifier smoke — Python (same 20 fixtures)
      # Paired with the JS smoke above. Both runtimes MUST produce
      # byte-identical accept/reject + error_class outcomes on every
      # fixture for the cross-runtime trust-triangle claim to extend
      # from render receipts to transform receipts.
      run: python -m pytest Tests/test_transform_receipt_verifier_fixtures.py -q
```

## Commit + push

From any local clone with `workflow` OAuth scope:

```bash
git checkout -b ci/wire-transform-receipt-fixtures
# (paste the block as described above)
git add .github/workflows/quantum-ci.yml
git commit -m "ci: wire cross-runtime fixture smokes for sum.transform_receipt.v1"
git push -u origin ci/wire-transform-receipt-fixtures
gh pr create --title "ci: wire cross-runtime fixture smokes for sum.transform_receipt.v1" \
  --body "Operator-paste from docs/CI_WIRING_TO_PASTE_2026-05-17.md. Engine session
couldn't push workflow files directly (OAuth scope). The two smokes
already run under make pre-push locally; this gates them in CI."
gh pr merge --squash --auto
```

## Why this matters

Both smokes already pass locally — `make pre-push` step 3 includes `Tests/test_transform_receipt_verifier_fixtures.py`. The CI wiring closes the gap so a future change that regresses the cross-runtime byte-equivalence of `sum.transform_receipt.v1` is caught in CI on every PR, not just on local pre-push runs.

Until this lands, regression of the transform-receipt fixture set is locally caught but not gated. The render-receipt analogue HAS been CI-gated since v0.9.C; this brings the transform-receipt path to parity.

## After this lands

Delete this file. It's a one-time paste artifact, not a durable doc.

## Pointers

- PR #220 — the cross-runtime fixture set that this CI block protects.
- `fixtures/transform_receipts/README.md` — the fixture-format spec.
- `single_file_demo/test_transform_receipt_fixtures.js` — the JS smoke.
- `Tests/test_transform_receipt_verifier_fixtures.py` — the Python smoke.
