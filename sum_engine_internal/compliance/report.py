"""Regime-agnostic ``ValidationReport`` shape.

Every compliance validator returns the same ``ValidationReport``
dataclass so downstream consumers (CLI, dashboard, retention
pipeline) can ingest reports across regimes without per-regime
adapters. New regimes add new ``rule_id`` strings; consumers stay
the same.

Schema: ``sum.compliance_report.v1`` — additive; new optional
fields may appear in future minor versions.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


_REPORT_SCHEMA = "sum.compliance_report.v1"


@dataclass(frozen=True)
class Violation:
    """A single per-row rule violation.

    ``rule_id`` is a regime-stable identifier (e.g.
    ``"eu-ai-act-art-12.required-fields"``). ``row_index`` is the
    0-based position of the offending row in the input audit-log
    stream so a human can ``head -n+1 -c +<index>`` to find it.
    """
    rule_id: str
    row_index: int
    operation: str | None
    message: str
    row: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationReport:
    """Aggregate verdict for a regime over an audit-log stream."""
    schema: str
    regime: str
    rows_examined: int
    violations: tuple[Violation, ...]

    @property
    def ok(self) -> bool:
        return len(self.violations) == 0

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    def violations_by_rule(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in self.violations:
            out[v.rule_id] = out.get(v.rule_id, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "regime": self.regime,
            "rows_examined": self.rows_examined,
            "ok": self.ok,
            "violation_count": self.violation_count,
            "violations_by_rule": self.violations_by_rule(),
            "violations": [asdict(v) for v in self.violations],
        }


def make_report(regime: str, rows_examined: int, violations: list[Violation]) -> ValidationReport:
    return ValidationReport(
        schema=_REPORT_SCHEMA,
        regime=regime,
        rows_examined=rows_examined,
        violations=tuple(violations),
    )
