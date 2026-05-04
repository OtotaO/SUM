"""Cross-regime tests for the shared compliance predicate library.

Sprint 3 of the intensification path to arXiv extracted
``_is_iso8601_utc`` (previously duplicated byte-identically across
six regime modules) into ``sum_engine_internal/compliance/
_predicates.py``. This file pins two contracts:

  P1. **Predicate behaviour** — the shared ``is_iso8601_utc``
      handles the contract cases the regime tests exercise via
      their R3 timestamp-iso8601-utc rules. Regression here would
      flip every regime's R3 verdict in lockstep, so the test is
      load-bearing for all six regimes simultaneously.

  P2. **Single source of truth** — exactly one definition of
      ``is_iso8601_utc`` exists in the compliance package. Six
      imports, one definition. A regression where someone re-
      adds the duplicate in a regime module surfaces here.

Adding a new shared predicate is allowed; this test file should
grow alongside.
"""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

from sum_engine_internal.compliance._predicates import is_iso8601_utc


# ─── P1: predicate behaviour ──────────────────────────────────────────


class TestIsIso8601Utc:
    """Pin the predicate's contract. Each case mirrors the kind of
    timestamp the per-regime R3 tests exercise — a regression here
    flips every regime's R3 verdict in lockstep."""

    def test_basic_zulu_timestamp_passes(self):
        assert is_iso8601_utc("2026-05-03T12:34:56Z")

    def test_zulu_with_milliseconds_passes(self):
        assert is_iso8601_utc("2026-05-03T12:34:56.789Z")

    def test_zulu_with_microseconds_passes(self):
        assert is_iso8601_utc("2026-05-03T12:34:56.123456Z")

    def test_offset_timezone_rejected(self):
        """Conservative contract: only Z-suffixed UTC accepted.
        +00:00 is *semantically* UTC but a different format."""
        assert not is_iso8601_utc("2026-05-03T12:34:56+00:00")
        assert not is_iso8601_utc("2026-05-03T12:34:56-00:00")
        assert not is_iso8601_utc("2026-05-03T12:34:56+02:00")

    def test_unparseable_body_rejected(self):
        """Non-timestamp text with a Z suffix is still rejected
        because the body must parse as a datetime."""
        assert not is_iso8601_utc("not-a-timestamp-Z")
        assert not is_iso8601_utc("Z")
        assert not is_iso8601_utc("hello-Z")

    def test_missing_z_suffix_rejected(self):
        assert not is_iso8601_utc("2026-05-03T12:34:56")
        assert not is_iso8601_utc("2026-05-03 12:34:56")

    def test_non_string_rejected(self):
        assert not is_iso8601_utc(None)
        assert not is_iso8601_utc(0)
        assert not is_iso8601_utc(1234567890)
        assert not is_iso8601_utc(["2026-05-03T12:34:56Z"])
        assert not is_iso8601_utc({"timestamp": "2026-05-03T12:34:56Z"})

    def test_empty_string_rejected(self):
        assert not is_iso8601_utc("")

    def test_human_readable_date_rejected(self):
        assert not is_iso8601_utc("May 3, 2026")
        assert not is_iso8601_utc("2026/05/03")


# ─── P2: single source of truth ───────────────────────────────────────


def test_only_one_definition_in_compliance_package():
    """The compliance package has exactly one definition of
    ``is_iso8601_utc``. Six imports, one definition. If a regime
    module re-introduces a duplicate (intentionally or accidentally),
    this test surfaces it.

    Implementation: search the package directory for ``def
    is_iso8601_utc`` (the canonical name) and the legacy
    ``def _is_iso8601_utc`` (which used to exist in every regime
    module before Sprint 3); only the former should appear, and
    only once, in ``_predicates.py``.
    """
    package_dir = Path(
        importlib.import_module("sum_engine_internal.compliance").__file__
    ).parent

    canonical_definitions: list[Path] = []
    legacy_definitions: list[Path] = []
    for py_file in package_dir.glob("*.py"):
        text = py_file.read_text()
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("def is_iso8601_utc"):
                canonical_definitions.append((py_file, line_no))
            elif stripped.startswith("def _is_iso8601_utc"):
                legacy_definitions.append((py_file, line_no))

    assert len(canonical_definitions) == 1, (
        f"is_iso8601_utc must have exactly one definition in the "
        f"compliance package; found {len(canonical_definitions)}: "
        f"{canonical_definitions}"
    )
    assert canonical_definitions[0][0].name == "_predicates.py", (
        f"is_iso8601_utc's single definition must live in "
        f"compliance/_predicates.py; found in {canonical_definitions[0]}"
    )
    assert len(legacy_definitions) == 0, (
        f"Legacy `_is_iso8601_utc` (underscored) should not reappear "
        f"in any regime module after Sprint 3; found "
        f"{len(legacy_definitions)}: {legacy_definitions}"
    )


def test_all_six_regimes_import_the_shared_predicate():
    """Each per-regime validator imports ``is_iso8601_utc`` from
    the shared ``_predicates`` module. Catches a regime drifting
    back into a private copy."""
    regime_modules = [
        "eu_ai_act_article_12",
        "gdpr_article_30",
        "hipaa_164_312_b",
        "iso_27001_8_15",
        "soc_2_cc_7_2",
        "pci_dss_4_req_10",
    ]
    for name in regime_modules:
        mod = importlib.import_module(
            f"sum_engine_internal.compliance.{name}"
        )
        # The function must be in the module's namespace AND it must be
        # the same object as the one in _predicates (proving import,
        # not re-definition).
        assert hasattr(mod, "is_iso8601_utc"), (
            f"regime {name!r} must import is_iso8601_utc from "
            f"compliance._predicates"
        )
        assert mod.is_iso8601_utc is is_iso8601_utc, (
            f"regime {name!r}'s is_iso8601_utc must be the shared "
            f"predicate (object identity), not a re-definition"
        )


# ─── Cross-regime contract: predicate fixes propagate ────────────────


def test_predicate_fix_would_propagate_across_all_regimes():
    """A behavioural change to ``is_iso8601_utc`` flows into every
    regime's R3 rule simultaneously — the property that motivated
    Sprint 3's extraction. Pinned by running the predicate against
    a representative timestamp through *each regime module's
    validate function* and confirming consistent R3 behaviour.
    """
    from sum_engine_internal.compliance import (
        eu_ai_act_article_12 as ev,
        gdpr_article_30 as gv,
        hipaa_164_312_b as hv,
        iso_27001_8_15 as iv,
        soc_2_cc_7_2 as sv,
        pci_dss_4_req_10 as pv,
    )

    base_row = {
        "schema": "sum.audit_log.v1",
        "operation": "attest",
        "cli_version": "0.5.0",
        "source_uri": "sha256:abc",
        "axiom_count": 1,
        "state_integer_digits": 10,
        "ok": True,
        "mode": "local-deterministic",
    }

    # A timestamp that the shared predicate currently rejects:
    # offset-style "+00:00" instead of "Z". Every regime should fire
    # its R3 rule.
    bad_row = {**base_row, "timestamp": "2026-05-03T12:34:56+00:00"}

    rule_ids_by_regime = {
        "eu-ai-act-art-12": "eu-ai-act-art-12.timestamp-iso8601-utc",
        "gdpr-art-30": "gdpr-art-30.timestamp-iso8601-utc",
        "hipaa-164-312-b": "hipaa-164-312-b.timestamp-iso8601-utc",
        "iso-27001-8-15": "iso-27001-8-15.timestamp-iso8601-utc",
        "soc-2-cc-7-2": "soc-2-cc-7-2.timestamp-iso8601-utc",
        "pci-dss-4-req-10": "pci-dss-4-req-10.timestamp-iso8601-utc",
    }

    for module, expected_rule in [
        (ev, "eu-ai-act-art-12.timestamp-iso8601-utc"),
        (gv, "gdpr-art-30.timestamp-iso8601-utc"),
        (hv, "hipaa-164-312-b.timestamp-iso8601-utc"),
        (iv, "iso-27001-8-15.timestamp-iso8601-utc"),
        (sv, "soc-2-cc-7-2.timestamp-iso8601-utc"),
        (pv, "pci-dss-4-req-10.timestamp-iso8601-utc"),
    ]:
        report = module.validate([bad_row])
        rule_ids = {v.rule_id for v in report.violations}
        assert expected_rule in rule_ids, (
            f"{module.REGIME}: predicate rejection for "
            f"'2026-05-03T12:34:56+00:00' should fire R3 ({expected_rule}); "
            f"got rules {sorted(rule_ids)}"
        )
