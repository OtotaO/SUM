# DOGFOOD_FINDINGS_2026-05-28.md

**Third dogfood-friction capture — `pip install sum-engine[sieve]` blocked on a fresh venv by upstream dep rot.** Companion to [`DOGFOOD_FINDINGS_2026-05-17.md`](DOGFOOD_FINDINGS_2026-05-17.md) (F1–F7) and [`DOGFOOD_FINDINGS_2026-05-18.md`](DOGFOOD_FINDINGS_2026-05-18.md) (F8–F11). F12 was used in commit messages for the NIM rate-limit retry hardening on the bench T1 runner (PRs #246/#247/#249); that work is captured in CHANGELOG `[Unreleased]` and is not journalist-facing, so it does not appear here. F13 below is the first user-facing finding since 2026-05-18.

## Pass setup

- Not a full Scenario A dogfood — caught while running `make smoke` and the `pip install sum-engine (fresh venv smoke)` CI job during PR #254's merge cycle on 2026-05-28
- The smoke is the same surface a cold journalist hits at step 0: `pip install sum-engine[sieve]`

## F13 — `pip install sum-engine[sieve]` fails on fresh venv: spacy 3.8 + typer 0.13 + click dep rot [severity: CRITICAL; fixed 2026-05-28]

`pip install sum-engine[sieve]` succeeds. First spacy import then raises:

```
File ".../spacy/cli/_util.py", line 18, in <module>
    from click import NoSuchOption
ModuleNotFoundError: No module named 'click'
```

Trigger: `spacy/__init__.py` imports `spacy/cli/__init__.py` at module load, which cascades to `spacy/cli/_util.py` which directly imports click. spacy declares `typer` as a dep but not `click`. Until typer 0.12.x, typer pulled click transitively; **typer ≥ 0.13 dropped click**. spacy 3.8 hasn't caught up. Net: every fresh install of `sum-engine[sieve]` after the typer 0.13 release is broken at first spacy use.

**Workaround at the time:** `pip install click>=8` after `pip install sum-engine[sieve]`.

**Fix (shipped 2026-05-28, PR #256):** added `click>=8.0` to the `[sieve]` optional-dependency group in `pyproject.toml`. Local repro of the full CI smoke step with the fix:

```bash
python -m venv /tmp/sumtest
/tmp/sumtest/bin/pip install ".[sieve]"
/tmp/sumtest/bin/python -m spacy download en_core_web_sm   # exit 0
/tmp/sumtest/bin/sum --version                              # "sum 0.7.0"
echo "Alice likes cats." | /tmp/sumtest/bin/sum attest --extractor=sieve > /tmp/b.json
/tmp/sumtest/bin/sum verify --input /tmp/b.json             # ok=true
```

CI: `pip install sum-engine (fresh venv smoke)` job now passes again on main.

**Why it matters:** This is **Scenario A's step 0** — a journalist following the DOGFOOD_QUICKSTART runs `pip install --upgrade 'sum-engine[openai,sieve,receipt-verify]'` and then hits an ImportError on first `sum attest`. The crypto trust loop, the canonical bytes, the slider — none of it reachable. Charter §3 names dogfood as load-bearing; a step-0 install failure is the worst possible position on the dogfood loop because it stops the loop before any signal can be generated.

**Caught by:** the `pip install sum-engine (fresh venv smoke)` CI job failing on main (post-PR #255 merge), not by a user. Significant: the CI gate caught it before any dogfood user did. The earlier outcome-coherence pass in PR #253 would not have caught this — it was a source-presence audit at the doc level, not an empirical live-install probe. PR #256 closed the gap.

**Caveat — what this finding does NOT say:** there is no automated guard against future upstream-dep-rot of this shape. The next time spacy adds a transitive dep that disappears from its `typer` chain, the same failure recurs. A more durable fix would be a CI matrix that pins spacy + typer at the floor versions in `pyproject.toml`; that is named as future tech debt, not closed in this pass.

## Summary table — F-findings so far

| F | Topic | Severity | Status |
|---:|---|---|---|
| F1 | `transformers` `FutureWarning` poisons stdout JSON pipes | HIGH | fixed in v0.7.0 |
| F2 | Sieve extraction too conservative for journalist use | MEDIUM | open (workaround: LLM extractor) |
| F3 | PyPI v0.6.0 lacked `transform` subcommand | CRITICAL | fixed (v0.7.0 ships transform) |
| F4 | `sum attest` output incompatible with `sum transform apply compose` | HIGH | fixed 2026-05-22 (PR #251) |
| F5 | Scenario A pipeline broken end-to-end | HIGH | resolved by F3 + F4 |
| F6 | Misc friction from the 2026-05-17 mechanical pass | LOW | absorbed |
| F7 | `beta.chat.completions.parse` returns degenerate output on non-OpenAI providers | HIGH | fixed 2026-05-18 (PR #241) |
| F8 | `perspective` axis silently flipped third-person to first-person | HIGH | open (axis prompt-hardening backlog) |
| F9 | `perspective` axis at 0.9 doesn't produce adversarial-reviewer stance | MEDIUM | open |
| F10 | formality and audience axes are highly collinear | MEDIUM | open (capability-region work, T2) |
| F11 | `engineer-precise` produced bulleted list, not prose | LOW | open |
| F12 | (not used for dogfood — commit/CHANGELOG only; NIM rate-limit retry on T1 bench) | n/a | fixed (PRs #246/#247/#249) |
| F13 | `pip install sum-engine[sieve]` fails on fresh venv (spacy/click dep rot) | CRITICAL | fixed 2026-05-28 (PR #256) |

## Loop closure

F13 was caught by the install-smoke CI gate before reaching a dogfood user. **Standing direction (memory `project_direction_2026-05-11`): wait + dogfood.** F13's fix means the next user-side dogfood pass is unblocked at step 0; the wait continues.

## Pointers

- PR #256 — `fix(packaging): add click to sieve extra` — the dep fix
- `docs/DOGFOOD_QUICKSTART.md` lines 23–30 + 183 — refreshed in PR #255 to cite v0.7.0
- `docs/DOGFOOD_FINDINGS_2026-05-17.md` — F1–F7 baseline
- `docs/DOGFOOD_FINDINGS_2026-05-18.md` — F8–F11 from the slider-variant pass
