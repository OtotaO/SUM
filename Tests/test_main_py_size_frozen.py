"""sum_cli/main.py does not grow (playbook Working rules: freeze list).

The freeze names "the size of sum_cli/main.py". A recalled rule has a miss
rate; this test does not. 4954 lines is the size on 2026-10-23, when the
check was added. Shrinking is fine; growing needs the operator to lift the
freeze and raise this number in the same PR.
"""
from pathlib import Path

CEILING = 4954


def test_main_py_does_not_grow():
    path = Path(__file__).resolve().parents[1] / "sum_cli" / "main.py"
    lines = len(path.read_text("utf-8").splitlines())
    assert lines <= CEILING, f"sum_cli/main.py has {lines} lines; the freeze ceiling is {CEILING}"
