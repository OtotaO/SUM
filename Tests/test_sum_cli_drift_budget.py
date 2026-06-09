"""`sum drift-budget` — the multi-hop chain readout CLI.

Exercised with the deterministic, dependency-free `lexical` scorer so the
test needs neither a judge model nor numpy.
"""
from __future__ import annotations

import io
import json
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def _cap():
    out, err = io.StringIO(), io.StringIO()
    with patch("sys.stdout", out), patch("sys.stderr", err):
        yield out, err


def run(argv):
    from sum_cli.main import main

    with _cap() as (out, err):
        rc = main(argv)
    return rc, out.getvalue(), err.getvalue()


def _chain(tmp_path):
    files = []
    for i, text in enumerate(
        [
            "The cat sat on the mat. The dog ran in the park.",
            "The cat sat on the mat.",
            "A cat sat.",
        ]
    ):
        p = tmp_path / f"x{i}.txt"
        p.write_text(text)
        files.append(str(p))
    return files


def test_drift_budget_json(tmp_path):
    files = _chain(tmp_path)
    rc, out, _ = run(["drift-budget", *files, "--scorer", "lexical", "--json"])
    assert rc == 0
    d = json.loads(out)
    assert d["n_hops"] == 2
    assert len(d["hops"]) == 2
    # additive budget is the sum of per-hop losses
    assert d["additive_budget"] == round(sum(h["loss"] for h in d["hops"]), 6)
    # scope must declare this is a measurement, not a bound
    assert "not a certified bound" in d["scope"]
    assert "slack" in d and "additive_is_conservative" in d


def test_drift_budget_human_readable(tmp_path):
    files = _chain(tmp_path)
    rc, out, _ = run(["drift-budget", *files, "--scorer", "lexical"])
    assert rc == 0
    assert "Drift budget" in out
    assert "additive budget" in out
    assert "end-to-end loss" in out
    assert "most expensive" in out  # the attribution marker


def test_drift_budget_needs_two_texts(tmp_path):
    files = _chain(tmp_path)
    rc, _, err = run(["drift-budget", files[0], "--scorer", "lexical"])
    assert rc == 2
    assert "at least 2" in err


def test_drift_budget_missing_file_rc2(tmp_path):
    files = _chain(tmp_path)
    rc, _, _ = run(["drift-budget", files[0], "/no/such/file.txt", "--scorer", "lexical"])
    assert rc == 2
