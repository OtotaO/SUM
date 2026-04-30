"""Regression test for the cold-install extractor probe.

The previous implementation of ``_pick_extractor`` probed via
``spacy.load("en_core_web_sm")`` which raises ``OSError`` when the
model is not installed. The exception was caught by a broad
``except Exception``, the probe returned "no extractor available",
and the user got a SystemExit on a fresh ``pip install
'sum-engine[sieve]'`` — even though [sieve] had just installed
spaCy itself.

The fix routes the probe through ``DeterministicSieve()`` whose
constructor catches ``OSError`` and runs ``python -m spacy
download en_core_web_sm`` automatically before retrying
``spacy.load``. Same UX as ``--extractor sieve`` always had.

This test pins the behaviour: when spaCy is importable but the
model-load path raises ``OSError``, the probe MUST trigger the
auto-download path (as evidenced by ``DeterministicSieve.__init__``
being called) rather than silently returning a fallthrough.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import sys
import types
from unittest import mock

import pytest


def test_pick_extractor_routes_through_sieve_constructor(monkeypatch):
    """The probe MUST construct ``DeterministicSieve()`` so its
    auto-downloader fires on cold installs. A direct ``spacy.load``
    probe (the previous behaviour) would skip the auto-download.
    """
    from sum_cli import main as sum_main

    construction_attempts: list[bool] = []

    class _StubSieve:
        def __init__(self):
            construction_attempts.append(True)

    monkeypatch.setattr(
        "sum_engine_internal.algorithms.syntactic_sieve.DeterministicSieve",
        _StubSieve,
    )

    # spaCy must be importable for the sieve branch to be reached.
    # If it's not installed in the test env, skip — we can't probe
    # the cold-install path without it. (CI installs sum-engine[sieve]
    # so this branch runs there.)
    try:
        import spacy  # noqa: F401
    except ImportError:
        pytest.skip("spaCy not installed; cannot exercise the sieve branch")

    result = sum_main._pick_extractor(None)
    assert result == "sieve"
    assert construction_attempts == [True], (
        "_pick_extractor must construct DeterministicSieve to trigger "
        "its auto-download fallback; otherwise cold installs fail."
    )


def test_pick_extractor_falls_back_to_llm_when_sieve_construction_fails(monkeypatch):
    """If sieve construction raises (e.g., spaCy auto-download
    itself fails — air-gapped install with no PyPI access), the
    probe MUST fall through to LLM if OPENAI_API_KEY is set."""
    from sum_cli import main as sum_main

    class _BrokenSieve:
        def __init__(self):
            raise RuntimeError(
                "simulated: spaCy model download failed (offline?)"
            )

    monkeypatch.setattr(
        "sum_engine_internal.algorithms.syntactic_sieve.DeterministicSieve",
        _BrokenSieve,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    try:
        import spacy  # noqa: F401
    except ImportError:
        pytest.skip("spaCy not installed; cannot exercise the sieve branch")

    assert sum_main._pick_extractor(None) == "llm"


def test_pick_extractor_systemexit_when_no_extractor_available(monkeypatch):
    """If neither sieve nor LLM is reachable, the SystemExit must
    carry the install hint string so users see actionable guidance,
    not a stack trace."""
    from sum_cli import main as sum_main

    class _BrokenSieve:
        def __init__(self):
            raise RuntimeError("simulated")

    monkeypatch.setattr(
        "sum_engine_internal.algorithms.syntactic_sieve.DeterministicSieve",
        _BrokenSieve,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        import spacy  # noqa: F401
    except ImportError:
        pytest.skip("spaCy not installed; cannot exercise the sieve branch")

    with pytest.raises(SystemExit) as excinfo:
        sum_main._pick_extractor(None)
    msg = str(excinfo.value)
    assert "no extractor available" in msg
    assert "sum-engine[sieve]" in msg
    assert "sum-engine[llm]" in msg
    assert "--extractor" in msg


def test_pick_extractor_honors_override(monkeypatch):
    """Explicit --extractor must short-circuit the probe entirely.
    The override bypasses determinism / availability checks so a
    user can intentionally exercise the LLM path even when sieve is
    available, or vice versa."""
    from sum_cli import main as sum_main

    # No spaCy probe should occur when override is set; verify by
    # ensuring DeterministicSieve.__init__ is NOT called.
    construction_attempts: list[bool] = []

    class _ShouldNotConstruct:
        def __init__(self):
            construction_attempts.append(True)

    monkeypatch.setattr(
        "sum_engine_internal.algorithms.syntactic_sieve.DeterministicSieve",
        _ShouldNotConstruct,
    )

    assert sum_main._pick_extractor("llm") == "llm"
    assert sum_main._pick_extractor("sieve") == "sieve"
    assert construction_attempts == [], (
        "override path must NOT probe via DeterministicSieve()"
    )
