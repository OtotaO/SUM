---
description: autonomous hardening loop — inspired by karpathy/autoresearch
---

# SUM Autonomous Hardening Protocol

This is the operating manual for an AI agent performing autonomous fortress
hardening on the SUM codebase. The human writes this file and sets the
direction. The agent executes experiments indefinitely until interrupted.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch):
"you're not programming the code — you're programming the program."

## Setup

1. **Read context files** for full situational awareness:
   - `docs/THREAT_MODEL.md` — what's protected, what's not
   - `docs/PROOF_BOUNDARY.md` — proven vs measured vs aspirational
   - `docs/CANONICAL_ABI_SPEC.md` — normative protocol
   - `Tests/` directory — current test suite
   - `scripts/verify_fortress.py` — the single-metric gate

2. **Establish the baseline**:
   ```bash
   source .venv/bin/activate
   python -m pytest Tests/ -q 2>&1 | tail -3    # total test count
   python scripts/verify_fortress.py 2>&1        # fortress check
   ```

3. **Record baseline** in `experiments.tsv`.

## The Single Metric

The gate is **`scripts/verify_fortress.py`**: all checks must pass.

Secondary metric: **test count** — monotonically increasing.

A change is a **keep** if:
- All fortress checks still pass (exit code 0)
- Test count ≥ previous test count
- Full `pytest` regression passes with 0 failures

A change is a **discard** if any of the above fail.

## What You CAN Do

- Add new tests (property, adversarial, witness, extraction)
- Add new fortress checks to `verify_fortress.py`
- Harden existing modules (input validation, error handling)
- Close attack surfaces listed in `THREAT_MODEL.md`
- Improve documentation honesty (qualify claims, add caveats)
- Add CLI tools to `scripts/`

## What You CANNOT Do

- Remove existing passing tests
- Weaken existing security mechanisms
- Add external network dependencies
- Modify frozen fixtures (`Tests/fixtures/`)
- Push to production without all tests passing

## Experiment Loop

// turbo-all

LOOP:

1. **Identify target**: Read `THREAT_MODEL.md` for open attack surfaces,
   or `PROOF_BOUNDARY.md` for aspirational properties to prove.

2. **Plan the experiment**: One focused change (e.g., "add timestamp
   validation to bundle import" or "test collision resolution across
   two independent algebra instances").

3. **Implement**: Write the code change + its tests.

4. **Run verification**:
   ```bash
   # The gate
   python -m pytest Tests/ -q 2>&1 | tail -3
   python scripts/verify_fortress.py 2>&1
   ```

5. **Evaluate**:
   - If test count increased AND all pass → **keep**
   - If test count same AND all pass AND closes a surface → **keep**
   - If any test fails → **discard** (revert changes)

6. **Log result** in `experiments.tsv`:
   ```
   commit	tests	checks	status	description
   ```

7. **Commit** (if keeping):
   ```bash
   git add -A && git commit -m "hardening: <description>"
   ```

8. **Push** periodically (every 3-5 keeps):
   ```bash
   git push origin main
   ```

9. **Repeat from step 1**.

## Logging Format

`experiments.tsv` is tab-separated with 5 columns:

```
commit	tests	checks	status	description
a1b2c3d	245	15/15	keep	baseline
b2c3d4e	248	15/15	keep	add timestamp format validation
c3d4e5f	248	14/15	discard	broke ref vector check
```

- `commit`: short hash (7 chars), or `-------` for discards
- `tests`: total pytest count
- `checks`: fortress checks passed/total
- `status`: `keep`, `discard`, or `crash`
- `description`: one-line summary

## Priority Queue

Ordered by impact. Work top-to-bottom:

1. Bundle timestamp format validation (reject non-ISO timestamps)
2. Cross-instance collision resolution verification
3. Semantic tome ordering stability proof
4. Node.js verifier Ed25519 awareness
5. Bundle version negotiation (graceful downgrade)
6. Extraction confidence scoring skeleton
7. Rate limiting skeleton for API endpoints
