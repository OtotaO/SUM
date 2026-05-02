"""Compliance validators on top of ``sum.audit_log.v1``.

The audit log is regime-agnostic foundation infrastructure; a
specific compliance regime is implemented as a downstream consumer
that tails the JSONL stream, validates per-regime required fields,
and emits a verdict.

This package collects the first-party validators. Each regime is a
separate module so users can compose only what they need.

  - :mod:`eu_ai_act_article_12` — record-keeping for high-risk AI
    systems (Regulation (EU) 2024/1689, Article 12).

A regime-agnostic ``ValidationReport`` shape lives at
:mod:`sum_engine_internal.compliance.report` so future validators
can share consumers (``sum compliance check`` CLI, dashboard
ingest, retention pipelines) without coupling to a specific regime.
"""
from __future__ import annotations

from sum_engine_internal.compliance.report import (
    ValidationReport,
    Violation,
)

__all__ = [
    "ValidationReport",
    "Violation",
]
