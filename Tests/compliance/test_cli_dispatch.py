"""Cross-regime CLI dispatch contract tests.

Pins the substrate-level invariants that span every compliance
regime — the things that broke when the dispatch was a hardcoded
``if args.regime == "eu-ai-act-article-12"`` and were the reason
this PR refactored to a dispatch dict.

Three contracts pinned here:

  C1. ``_COMPLIANCE_REGIMES`` keys (the description registry)
      exactly equal ``_compliance_validators()`` keys (the dispatch
      registry). A mismatch means either a regime is listed without
      a validator (CLI lies to the user about coverage) or a
      validator is wired without listing (no description, no
      ``sum compliance regimes`` surface).
  C2. Every registered regime returns a
      ``sum.compliance_report.v1`` shape from
      ``cmd_compliance_check``. Catches a future regime that
      accidentally returns a different schema.
  C3. ``cmd_compliance_check`` exit code is 0 iff the report is
      ok with no parse errors, 1 otherwise — pipe-friendly for
      CI gates. Same contract documented in the wire-spec docs.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pytest

from sum_cli.main import (
    _COMPLIANCE_REGIMES,
    _compliance_validators,
    cmd_compliance_check,
)


# ─── C1: registry consistency ─────────────────────────────────────────


def test_compliance_regimes_keys_match_validators_keys():
    """The two registries must agree on which regimes exist. A
    drift here is a wiring bug — either the description registry
    or the dispatch table is stale."""
    described = set(_COMPLIANCE_REGIMES)
    wired = set(_compliance_validators())
    assert described == wired, (
        f"_COMPLIANCE_REGIMES vs _compliance_validators registry drift:\n"
        f"  described but not wired: {sorted(described - wired)}\n"
        f"  wired but not described: {sorted(wired - described)}"
    )


def test_each_validator_is_callable_returning_validation_report():
    """Each entry in the dispatch table is a callable accepting
    ``rows`` and returning a ``ValidationReport`` — the regime-
    agnostic substrate contract."""
    from sum_engine_internal.compliance.report import ValidationReport
    validators = _compliance_validators()
    assert validators, "no compliance validators wired"
    for regime_id, validate in validators.items():
        report = validate([])  # empty input — every validator must accept
        assert isinstance(report, ValidationReport), (
            f"regime {regime_id!r} validator returned "
            f"{type(report).__name__}, expected ValidationReport"
        )
        assert report.regime == regime_id, (
            f"regime {regime_id!r} validator returned report.regime="
            f"{report.regime!r}; the validator's REGIME constant must "
            f"match its dispatch key"
        )


# ─── C2: report schema is regime-agnostic ─────────────────────────────


def test_every_regime_returns_compliance_report_v1_schema(tmp_path):
    """Drive ``cmd_compliance_check`` for every registered regime
    against an empty audit log; every regime emits a
    ``sum.compliance_report.v1`` JSON object."""
    audit = tmp_path / "empty.jsonl"
    audit.write_text("")

    for regime_id in _COMPLIANCE_REGIMES:
        out_buf = io.StringIO()
        old = sys.stdout
        sys.stdout = out_buf
        try:
            rc = cmd_compliance_check(argparse.Namespace(
                regime=regime_id, audit_log=str(audit), pretty=False,
            ))
        finally:
            sys.stdout = old

        assert rc == 0, (
            f"regime {regime_id!r}: empty audit log should pass; "
            f"got rc={rc}"
        )
        payload = json.loads(out_buf.getvalue())
        assert payload["schema"] == "sum.compliance_report.v1", (
            f"regime {regime_id!r} returned schema "
            f"{payload['schema']!r}, expected sum.compliance_report.v1"
        )
        assert payload["regime"] == regime_id


# ─── C3: exit-code contract ───────────────────────────────────────────


def test_unknown_regime_exits_2(tmp_path):
    """An unknown regime is a usage error (rc=2), distinct from
    ``ok=false`` violations (rc=1) and ``ok=true`` (rc=0)."""
    audit = tmp_path / "audit.jsonl"
    audit.write_text("")
    rc = cmd_compliance_check(argparse.Namespace(
        regime="not-a-real-regime", audit_log=str(audit), pretty=False,
    ))
    assert rc == 2


def test_violations_exit_code_is_1(tmp_path):
    """A regime that finds violations on the input must return rc=1
    so CI gates can `|| exit 1` reliably."""
    audit = tmp_path / "audit.jsonl"
    # Schema violation — fires R1 in every record-keeping regime
    audit.write_text(json.dumps({"schema": "wrong.schema"}) + "\n")

    for regime_id in _COMPLIANCE_REGIMES:
        out_buf = io.StringIO()
        old = sys.stdout
        sys.stdout = out_buf
        try:
            rc = cmd_compliance_check(argparse.Namespace(
                regime=regime_id, audit_log=str(audit), pretty=False,
            ))
        finally:
            sys.stdout = old
        assert rc == 1, (
            f"regime {regime_id!r}: violations should exit 1 for CI "
            f"gate compatibility; got rc={rc}"
        )
