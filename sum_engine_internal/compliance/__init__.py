"""Compliance validators on top of ``sum.audit_log.v1``.

The audit log is regime-agnostic foundation infrastructure; a
specific compliance regime is implemented as a downstream consumer
that tails the JSONL stream, validates per-regime required fields,
and emits a verdict.

This package collects the first-party validators. Each regime is a
separate module so users can compose only what they need.

  - :mod:`eu_ai_act_article_12` — record-keeping for high-risk AI
    systems (Regulation (EU) 2024/1689, Article 12).
  - :mod:`gdpr_article_30` — Records of Processing Activities
    (Regulation (EU) 2016/679, Article 30); pins the per-row floor
    enabling Art 30 reporting. The controller separately maintains
    record-set-scope metadata (Art 30(1)(a)–(g)) out-of-band.
  - :mod:`hipaa_164_312_b` — HIPAA Security Rule § 164.312(b)
    Audit Controls (Technical Safeguards); pins the per-row form
    requirements for an audit recording that supports examination
    of activity in systems that contain or use ePHI.
  - :mod:`iso_27001_8_15` — ISO/IEC 27001:2022 Annex A.8.15
    (Logging); pins the per-row form floor required for an audit
    log to count as a "produced" log under A.8.15.
  - :mod:`soc_2_cc_7_2` — SOC 2 Trust Services Criteria CC7.2
    (System Operations); pins the per-row floor required to
    enable the monitoring criterion.
  - :mod:`pci_dss_4_req_10` — PCI DSS v4.0 Requirement 10 (Log
    and Monitor All Access to System Components and Cardholder
    Data); pins the per-row content visible in audit_log.v1
    against Req 10.2.2 + 10.6. The user_id structural gap is
    named explicitly in the wire-spec doc.

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
