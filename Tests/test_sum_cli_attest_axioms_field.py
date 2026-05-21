"""F4 regression: `sum attest` emits an `axioms` field consumable by
`sum transform apply compose`.

Before this guard, attest's bundle had `axiom_count` (a scalar) but no
list of axioms — so the dogfood quickstart's Scenario A
(attest → compose → slider) failed at step 4 with
"compose: bundle dict must have 'triples' or 'axioms' key".

This test pins that the bundle now carries an `axioms` field of
`{subject, predicate, object}` dicts AND that compose accepts it
without a round-trip through canonical_tome parsing.

Source: docs/DOGFOOD_FINDINGS_2026-05-17.md, finding F4.
License: Apache License 2.0
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from sum_cli.main import cmd_attest
from sum_engine_internal.transforms.compose import _bundle_triples


def _run_attest(text: str, tmp_path: Path) -> dict:
    in_path = tmp_path / "in.txt"
    in_path.write_text(text)
    args = argparse.Namespace(
        input=str(in_path),
        extractor="sieve",
        model=None,
        source=None,
        branch="main",
        title="F4 Regression",
        signing_key=None,
        ed25519_key=None,
        ledger=None,
        pretty=False,
        verbose=False,
    )
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        assert cmd_attest(args) == 0, f"attest failed: {buf.getvalue()}"
    finally:
        sys.stdout = old
    return json.loads(buf.getvalue())


def test_attest_emits_axioms_field(tmp_path: Path) -> None:
    bundle = _run_attest(
        "The sun is a star. Mars is a planet. Pluto is a dwarf planet.",
        tmp_path,
    )
    assert "axioms" in bundle, "F4 regression: bundle missing axioms field"
    assert isinstance(bundle["axioms"], list)
    assert len(bundle["axioms"]) == bundle["axiom_count"], (
        "axioms length must match axiom_count"
    )
    for ax in bundle["axioms"]:
        assert set(ax.keys()) == {"subject", "predicate", "object"}, (
            f"axiom shape mismatch: {ax}"
        )
        assert all(isinstance(v, str) and v for v in ax.values()), (
            f"axiom must have non-empty string fields: {ax}"
        )


def test_attest_axioms_consumable_by_compose(tmp_path: Path) -> None:
    """The dogfood Scenario A end-to-end shape: compose._bundle_triples
    must accept the attest output without raising."""
    bundle = _run_attest(
        "Water boils at one hundred celsius. Ice melts at zero celsius.",
        tmp_path,
    )
    triples = _bundle_triples(bundle)
    assert len(triples) == bundle["axiom_count"]
    for s, p, o in triples:
        assert s and p and o
        assert isinstance(s, str) and isinstance(p, str) and isinstance(o, str)
