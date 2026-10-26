"""The Fortress gate must not pass when the Node.js witness cannot run."""

import json

import pytest

from scripts import verify_fortress


def test_missing_node_fails_the_gate(monkeypatch, capsys):
    real_run = verify_fortress.subprocess.run

    def run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "node":
            raise FileNotFoundError("node")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(verify_fortress.subprocess, "run", run)
    monkeypatch.setattr(verify_fortress, "passed", 0)
    monkeypatch.setattr(verify_fortress, "total", 0)
    monkeypatch.setattr("sys.argv", ["verify_fortress.py", "--json"])

    with pytest.raises(SystemExit) as exc:
        verify_fortress.main()

    assert exc.value.code == 1
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["all_pass"] is False
    assert summary["total"] == 21
    assert summary["passed"] == 20
