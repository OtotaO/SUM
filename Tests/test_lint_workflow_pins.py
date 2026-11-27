"""scripts/lint_workflow_pins.py reads only the one-line `- uses: owner/repo@ref`
form; any other line holding the word `uses` outside a comment fails, so a tag
pin cannot pass by being written another way (a flow mapping, an anchor, a
block scalar, a complex key). A key spelled with a YAML escape is not seen."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.lint_workflow_pins import UNREADABLE, lint_workflow

SHA = "3d3c42e5aac5ba805825da76410c181273ba90b1"


def _lint(tmp_path: Path, body: str):
    path = tmp_path / "wf.yml"
    path.write_text(body, encoding="utf-8")
    return lint_workflow(path)


@pytest.mark.parametrize("line", [
    f"    - uses: actions/checkout@{SHA}  # v7.0.1",
    f"      uses: actions/setup-python@{SHA}",
    "    - uses: ./local-action",
    "    - uses: docker://alpine:3.20",
    "    # - uses: actions/checkout@v4",
    "    - name: plain step",
])
def test_lines_that_pass(tmp_path, line):
    assert _lint(tmp_path, line + "\n") == []


@pytest.mark.parametrize("line", [
    "    - uses: actions/checkout@v4",
    "    - uses: actions/checkout@main",
])
def test_a_tag_pin_in_the_one_line_form_fails(tmp_path, line):
    (lineno, target, ref), = _lint(tmp_path, line + "\n")
    assert (lineno, target) == (1, "actions/checkout")
    assert ref in ("v4", "main")


@pytest.mark.parametrize("line", [
    "    - {uses: actions/checkout@v4}",                                  # flow mapping
    "    steps: [{name: x, uses: actions/upload-artifact@main}]",         # flow sequence
    "      uses: >-",                                                     # block scalar
    "      uses: &co actions/cache@v4",                                   # anchor
    "      uses: *co",                                                    # alias
    "    - uses: actions/checkout",                                       # no @ref
    '    - "uses": "actions/checkout@v4"',                                # quoted key
    "    ? uses",                                                         # complex key
    "    - ? uses",                                                       # complex key in a list
    "    ? uses # a comment",                                             # complex key, commented
    "    - { ? uses",                                                     # explicit key in a flow mapping
    "    ? &k uses",                                                      # anchor on a complex key
    "    ? !!str uses",                                                   # tag on a complex key
    "    x: &k uses",                                                     # the anchor an alias key reads
])
def test_other_spellings_of_a_uses_key_fail(tmp_path, line):
    failures = _lint(tmp_path, line + "\n")
    assert [(f[0], f[1]) for f in failures] == [(1, UNREADABLE)], failures


def test_a_complex_key_fails_on_its_key_line(tmp_path):
    body = "    ? uses\n    : actions/checkout@v4\n"
    assert [(f[0], f[1]) for f in _lint(tmp_path, body)] == [(1, UNREADABLE)]


@pytest.mark.parametrize("line", [
    f"    - uses: 'actions/checkout@{SHA}'",       # quoted SHA: refused, write it unquoted
    '      run: echo "uses: x"',                   # the text uses: inside a run script
    "    - name: a step that uses a cache",         # the word in a name
])
def test_spellings_refused_by_design(tmp_path, line):
    assert _lint(tmp_path, line + "\n") != []


def test_the_repo_workflows_pass():
    root = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    for path in sorted(root.glob("*.y*ml")):
        assert lint_workflow(path) == [], path.name
