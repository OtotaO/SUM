# DOGFOOD_FINDINGS_2026-05-29.md

**F14 capture — pyproject.toml's declared `[sieve]` floor (`spacy>=3.7.0`) was no longer empirically operable.** Caught while implementing the F13 follow-up tech-debt — the floor-version CI matrix named in [`DOGFOOD_FINDINGS_2026-05-28.md`](DOGFOOD_FINDINGS_2026-05-28.md). The matrix was supposed to *prevent* future F13-shape failures; the act of writing it surfaced an existing one.

Companion to:
- [`DOGFOOD_FINDINGS_2026-05-17.md`](DOGFOOD_FINDINGS_2026-05-17.md) — F1–F7
- [`DOGFOOD_FINDINGS_2026-05-18.md`](DOGFOOD_FINDINGS_2026-05-18.md) — F8–F11
- [`DOGFOOD_FINDINGS_2026-05-28.md`](DOGFOOD_FINDINGS_2026-05-28.md) — F13 (F12 not used for dogfood; commit-message only for the NIM-bench-runner retry hardening)

## Pass setup

- Not a user-facing dogfood pass — caught while drafting the floor CI matrix locally
- Trigger: tried to pin `spacy==3.7.0` (the declared `[sieve]` floor) and install + smoke-test
- Method: same install sequence the CI will run; verified locally before any commit

## F14 — `[sieve]` declared spacy floor `>=3.7.0` is broken; actual operable floor is `3.8.0` [severity: HIGH; fixed 2026-05-29]

`pip install sum-engine[sieve]` with `spacy==3.7.0` pinned succeeds. `python -m spacy download en_core_web_sm` succeeds. Then **`sum attest`** fails at runtime:

```
sum: spaCy model 'en_core_web_sm' missing; downloading (~50 MB, one-time)…
Collecting https://github.com/explosion/spacy-models/releases/download/-en_core_web_sm/-en_core_web_sm.tar.gz
  ERROR: HTTP error 404 while getting … /download/-en_core_web_sm/-en_core_web_sm.tar.gz
```

Two failures stacked:

1. **`spacy.load("en_core_web_sm")` fails** even though `python -m spacy download en_core_web_sm` just succeeded — because the model spacy.io's compatibility table now serves for `en_core_web_sm` is a 3.8-series model, and the spacy 3.7.0 runtime cannot load it.
2. **The auto-download fallback in `sum_engine_internal/algorithms/syntactic_sieve.py` builds a malformed URL** with a leading `-` where a model version should be. Spacy 3.7.0's compatibility lookup returns no entry for itself (the 3.7 row was rotated out), so the version interpolates as empty, producing `download/-en_core_web_sm/-en_core_web_sm.tar.gz` and a clean 404.

Net: every fresh install of `sum-engine[sieve]` that respects the declared floor (spacy 3.7.0) hits an unrecoverable runtime failure at first `sum attest`. The declared floor in pyproject.toml lagged the actual operable floor.

**Workaround at the time:** install `spacy==3.8.0` or later.

**Fix (shipped 2026-05-29):** bump `[sieve]` floor in pyproject.toml: `spacy>=3.7.0` → `spacy>=3.8.0`. Verified locally: full attest → verify round-trip exits 0 on spacy 3.8.0 + click 8.0.0 (the new pinned floors).

**Why caught:** The F13 follow-up tech-debt named in `docs/DOGFOOD_FINDINGS_2026-05-28.md` was a CI floor-version matrix. Writing that matrix required choosing the pin: I chose the *declared* floor first per spec, ran it locally to verify before commit, and it failed. That's the discipline from the live-state-probing memory (`feedback_live_state_probing.md`): empirical probe of the actual install path, not source-presence at the spec level.

Without that discipline I would have shipped a CI job that fails on day 1, OR (worse) one that silently used a pinned-too-new floor to avoid the failure — green-theater inversion.

## Adjacent change in the same PR — floor-venv smoke CI gate

`.github/workflows/quantum-ci.yml` gains a new job `pip install sum-engine (floor venv smoke)`, sibling of the existing latest-resolution smoke. The 2-axis signal:

| floor smoke | latest smoke | Interpretation |
|---|---|---|
| PASS | PASS | clean |
| PASS | FAIL | upstream broke at HEAD; CI caught it before users (F13/F14 shape) |
| FAIL | PASS | declared floor is stale; raise pyproject floor (F14 itself) |
| FAIL | FAIL | full upstream meltdown; investigate |

Both jobs are independent (different venvs, different pins). Either failing blocks merge. Neither is allowed to silently degrade.

## Caveat — what this fix does NOT do

- It does NOT pin transitive Python deps in a lockfile. A user running `pip install` still gets pip's resolver behavior; the only guarantee is the floor declared in pyproject. If pip resolves to something higher and that fails, users still get a fresh F-finding. The latest-resolution CI smoke catches this; users may still get to it first.
- It does NOT prevent the next spacy major from breaking sum_engine_internal/algorithms/syntactic_sieve.py's auto-download path. That path is fragile by design (subprocess to spacy CLI). A more durable fix is documented as future tech debt: stop relying on auto-download in the sieve runtime; require users to run `spacy download` explicitly and surface a friendly error if missing. Not closed here.

## Summary table — F-findings so far

| F | Topic | Severity | Status |
|---:|---|---|---|
| F1 | `transformers` `FutureWarning` poisons stdout JSON pipes | HIGH | fixed in v0.7.0 |
| F2 | Sieve extraction too conservative for journalist use | MEDIUM | open |
| F3 | PyPI v0.6.0 lacked `transform` subcommand | CRITICAL | fixed (v0.7.0) |
| F4 | `sum attest` output incompatible with `sum transform apply compose` | HIGH | fixed 2026-05-22 (PR #251) |
| F5 | Scenario A pipeline broken end-to-end | HIGH | resolved by F3 + F4 |
| F6 | Misc friction from 2026-05-17 pass | LOW | absorbed |
| F7 | `beta.chat.completions.parse` degenerate on non-OpenAI providers | HIGH | fixed 2026-05-18 (PR #241) |
| F8 | `perspective` axis silently flipped third-person to first-person | HIGH | open |
| F9 | `perspective` axis at 0.9 not adversarial-reviewer stance | MEDIUM | open |
| F10 | formality and audience axes are highly collinear | MEDIUM | open |
| F11 | `engineer-precise` produced bulleted list, not prose | LOW | open |
| F12 | (not used for dogfood — CHANGELOG-only) | n/a | fixed (PRs #246/#247/#249) |
| F13 | `pip install sum-engine[sieve]` ImportError on fresh venv (spacy/click) | CRITICAL | fixed 2026-05-28 (PR #256) |
| F14 | `[sieve]` declared spacy floor `>=3.7.0` broken at runtime | HIGH | fixed 2026-05-29 (this PR) |

## Loop closure

F13 and F14 are the same shape — upstream-dep-rot caught at the install boundary. F13 was caught by the existing fresh-venv-smoke job (latest resolution). F14 was caught by the act of writing the floor-matrix that closes F13's tech-debt callout. Both are now guarded by CI gates. The next dep-rot of this shape will fire one of the two jobs and block merge before reaching users.

Standing direction (memory `project_direction_2026-05-11`) unchanged: wait + dogfood. The CI gate is the dogfood loop's pre-emption, not its substitute.

## Pointers

- `pyproject.toml` `[sieve]` extra — floor bump + click pin
- `.github/workflows/quantum-ci.yml` — new `pypi-install-smoke-floor` job
- `docs/DOGFOOD_FINDINGS_2026-05-28.md` — F13 + the tech-debt callout this PR closes
- `feedback_live_state_probing.md` (memory) — the discipline that caught F14 during write
