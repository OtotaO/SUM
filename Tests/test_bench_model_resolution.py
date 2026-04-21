"""Resolution contract for SUM_BENCH_MODEL + per-role overrides.

The bench harness historically required four separate env vars
(SUM_BENCH_FACTSCORE_MODEL, SUM_BENCH_MINICHECK_MODEL,
SUM_BENCH_GENERATOR_MODEL, SUM_BENCH_EXTRACTOR_MODEL), forcing the
common case (one model for every role) to repeat itself four times
in every CI config and every developer shell. SUM_BENCH_MODEL is the
Occam cut: one var in the common case, role-specific overrides when
you actually need them.

These tests pin:
  * SUM_BENCH_MODEL alone fills every role.
  * A role-specific env var still overrides SUM_BENCH_MODEL.
  * Mixed (global + one override) produces the expected split.
  * Neither set → SystemExit with a useful message.
  * Unpinned model id (no date suffix) → SystemExit regardless of source.
  * --no-llm bypasses LLM model resolution entirely.

Author: ototao
License: Apache License 2.0
"""

from __future__ import annotations

import pytest

from scripts.bench.run_bench import _ROLE_OVERRIDE_ENV, _resolve_model_snapshots


_ALL_ROLES = tuple(_ROLE_OVERRIDE_ENV.keys())
_ALL_ENV_VARS = ("SUM_BENCH_MODEL",) + tuple(_ROLE_OVERRIDE_ENV.values())


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts with zero bench-env-vars set."""
    for var in _ALL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_no_llm_skips_model_resolution():
    snapshots = _resolve_model_snapshots(no_llm=True)
    # Only the deterministic sieve pseudo-snapshot remains.
    assert snapshots == {"sum.sieve": "deterministic_v1"}


def test_single_sum_bench_model_fills_every_role(monkeypatch):
    monkeypatch.setenv("SUM_BENCH_MODEL", "gpt-4o-mini-2024-07-18")
    snapshots = _resolve_model_snapshots(no_llm=False)
    for role in _ALL_ROLES:
        assert snapshots[role] == "gpt-4o-mini-2024-07-18"


def test_role_override_takes_precedence(monkeypatch):
    monkeypatch.setenv("SUM_BENCH_MODEL", "gpt-4o-mini-2024-07-18")
    monkeypatch.setenv("SUM_BENCH_MINICHECK_MODEL", "gpt-4o-2024-08-06")
    snapshots = _resolve_model_snapshots(no_llm=False)
    assert snapshots["minicheck"] == "gpt-4o-2024-08-06"
    # Other roles still fall back to the global default.
    for role in ("factscore", "generator", "extractor"):
        assert snapshots[role] == "gpt-4o-mini-2024-07-18"


def test_all_role_vars_still_work_without_global(monkeypatch):
    """Backwards-compat: runs that set all four role vars keep working."""
    monkeypatch.setenv("SUM_BENCH_FACTSCORE_MODEL", "gpt-4o-mini-2024-07-18")
    monkeypatch.setenv("SUM_BENCH_MINICHECK_MODEL", "gpt-4o-mini-2024-07-18")
    monkeypatch.setenv("SUM_BENCH_GENERATOR_MODEL", "gpt-4o-2024-08-06")
    monkeypatch.setenv("SUM_BENCH_EXTRACTOR_MODEL", "gpt-4o-2024-08-06")
    snapshots = _resolve_model_snapshots(no_llm=False)
    assert snapshots["factscore"] == "gpt-4o-mini-2024-07-18"
    assert snapshots["minicheck"] == "gpt-4o-mini-2024-07-18"
    assert snapshots["generator"] == "gpt-4o-2024-08-06"
    assert snapshots["extractor"] == "gpt-4o-2024-08-06"


def test_empty_env_raises_systemexit_mentioning_global_var():
    """The error message must steer users to SUM_BENCH_MODEL (the
    Occam-minimal path) rather than only listing the four role vars."""
    with pytest.raises(SystemExit, match="SUM_BENCH_MODEL"):
        _resolve_model_snapshots(no_llm=False)


def test_unpinned_model_id_rejected(monkeypatch):
    monkeypatch.setenv("SUM_BENCH_MODEL", "gpt-4o")  # unpinned suspect
    with pytest.raises(SystemExit, match="looks unpinned"):
        _resolve_model_snapshots(no_llm=False)


def test_unpinned_role_override_rejected(monkeypatch):
    monkeypatch.setenv("SUM_BENCH_MODEL", "gpt-4o-mini-2024-07-18")
    monkeypatch.setenv("SUM_BENCH_EXTRACTOR_MODEL", "gpt-4")  # unpinned suspect
    with pytest.raises(SystemExit, match="looks unpinned"):
        _resolve_model_snapshots(no_llm=False)


def test_whitespace_only_treated_as_missing(monkeypatch):
    monkeypatch.setenv("SUM_BENCH_MODEL", "   ")
    with pytest.raises(SystemExit, match="SUM_BENCH_MODEL"):
        _resolve_model_snapshots(no_llm=False)
