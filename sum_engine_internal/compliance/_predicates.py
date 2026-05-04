"""Shared predicates for the per-regime compliance validators.

Six regime validators (EU AI Act Art 12, GDPR Art 30, HIPAA
§ 164.312(b), ISO 27001 A.8.15, SOC 2 CC7.2, PCI DSS v4.0 Req 10)
each previously carried a byte-identical copy of
``_is_iso8601_utc``. With six regimes the duplication is load-
bearing technical debt: a fix to the timestamp predicate (e.g.
accepting ``+00:00`` as equivalent to ``Z``, or tightening the
parser to reject empty fractional components) currently requires
editing six files in lockstep. **Sprint 3 of the intensification
path to arXiv extracts the predicate here.**

Why a private module (``_predicates``, leading underscore). Each
regime's *rules* and *rule_id* strings are intentionally regime-
specific — downstream dashboards filter on rule_id, so they're
part of the regime's stable contract. The *predicates* (e.g.
"is this a parseable ISO 8601 UTC timestamp?") are intentionally
regime-shared — a fix here applies to every consumer simultaneously.
The leading underscore signals "internal to the compliance package",
not part of the public API.

Predicates are pure functions over a single value with no side
effects, no exceptions, no logging. A predicate returns ``bool``;
it does not return ``Violation`` or any regime-specific structure.
The regime module is responsible for wrapping a False predicate
return in a regime-specific ``Violation`` with the appropriate
``rule_id`` and statutory message.

Adding a new predicate. Three checks before adding:

  (a) **Three or more regimes need it.** A predicate used by only
      one or two regimes stays in those regimes' modules; lifting
      prematurely creates a shared dependency without buying the
      DRY win.
  (b) **The contract is stable across regimes.** If two regimes
      need slightly different timestamp semantics, two predicates
      live here, not one with regime-specific switches.
  (c) **The predicate has zero side effects.** No reads, no
      writes, no logging. A False return is the only signal.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


def is_iso8601_utc(s: Any) -> bool:
    """Is ``s`` a parseable ISO 8601 UTC string ending in ``Z``?

    Returns ``True`` iff:
      - ``s`` is a ``str``,
      - ``s`` ends with the literal ``"Z"``,
      - replacing the trailing ``"Z"`` with ``"+00:00"`` yields
        a string that ``datetime.fromisoformat`` accepts.

    The trailing-``Z`` requirement is intentional: the regimes use
    UTC-only timestamps for chronological reporting, and accepting
    ``+00:00`` as equivalent would let mixed-format streams through.
    Fixing this universally requires touching every regime's tests
    in lockstep, so the contract is conservative.

    Used by all six per-regime validators. A future fix or
    relaxation here applies to every regime simultaneously.

    >>> is_iso8601_utc("2026-05-03T12:34:56Z")
    True
    >>> is_iso8601_utc("2026-05-03T12:34:56.789Z")
    True
    >>> is_iso8601_utc("2026-05-03T12:34:56+00:00")  # not Z-suffixed
    False
    >>> is_iso8601_utc("not-a-timestamp-Z")
    False
    >>> is_iso8601_utc(None)
    False
    >>> is_iso8601_utc(1234567890)
    False
    """
    if not isinstance(s, str):
        return False
    if not s.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(s.replace("Z", "+00:00"))
        return True
    except (ValueError, TypeError):
        return False
