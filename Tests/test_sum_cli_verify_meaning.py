"""`sum verify-meaning` — the external-party verify on-ramp (dogfood F21).

Verifies the committed meaning-risk + perspective goldens via the CLI a
third party would actually run, and that tampering / wrong schema fail
with the right exit codes.
"""
from __future__ import annotations

import io
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("joserfc", reason="[receipt-verify] not installed")

_REPO = Path(__file__).resolve().parents[1]


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


_MEANING = str(_REPO / "fixtures/meaning_receipts/meaning_risk_receipt.golden.json")
_MEANING_JWKS = str(_REPO / "fixtures/meaning_receipts/jwks.json")
_PERSP = str(_REPO / "fixtures/perspective_receipts/perspective_risk_receipt.golden.json")
_PERSP_JWKS = str(_REPO / "fixtures/perspective_receipts/jwks.json")


def test_verify_meaning_golden():
    rc, out, _ = run(["verify-meaning", _MEANING, "--jwks", _MEANING_JWKS])
    assert rc == 0
    v = json.loads(out)
    assert v["verified"] is True
    assert v["schema"] == "sum.meaning_risk_receipt.v1"
    assert "risk_upper_bound" in v and "not_covered" in v


def test_verify_perspective_golden():
    rc, out, _ = run(["verify-meaning", _PERSP, "--jwks", _PERSP_JWKS])
    assert rc == 0
    v = json.loads(out)
    assert v["verified"] is True
    assert v["schema"] == "sum.perspective_risk_receipt.v1"
    assert {c["group_id"] for c in v["cohorts"]} == {"plain", "technical"}
    assert v["controls_all"] is False


def test_tampered_receipt_rc1(tmp_path):
    r = json.loads(Path(_MEANING).read_text())
    r["payload"]["risk_upper_bound_micro"] = 0
    p = tmp_path / "bad.json"; p.write_text(json.dumps(r))
    rc, out, _ = run(["verify-meaning", str(p), "--jwks", _MEANING_JWKS])
    assert rc == 1
    assert json.loads(out)["verified"] is False


def test_unknown_schema_rc2(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"schema": "sum.render_receipt.v1", "kid": "k", "payload": {}, "jws": "a..b"}))
    rc, _, err = run(["verify-meaning", str(p), "--jwks", _MEANING_JWKS])
    assert rc == 2
    assert "verify-meaning handles" in err


def test_missing_file_rc2():
    rc, _, err = run(["verify-meaning", "/nonexistent.json", "--jwks", _MEANING_JWKS])
    assert rc == 2
    assert "cannot read" in err


# --- the binding-gate goldens replay via --losses on their COMMITTED losses ---
# Regression for the adoption-sim bug (2026-06-09): the committed losses files
# are metadata-wrapped (`{"judge": .., "losses": [..]}`), but `--losses` used to
# require a bare array — so a third party pointing `--losses` at the project's
# own flagship fixture got `could not convert string to float: 'judge'`. The
# advertised "replays offline" on-ramp was broken on the golden. `--losses` now
# unwraps the `losses` key; Stage-B replay (no judge) must reproduce the bound.

@pytest.mark.parametrize("fdir,bound_micro", [
    ("meaning_receipts_billsum", 645438),
    ("meaning_receipts_translation", 412359),
])
def test_committed_golden_replays_via_wrapped_losses_file(fdir, bound_micro):
    base = _REPO / "fixtures" / fdir
    receipt = str(next(base.glob("*.golden.json")))
    jwks = str(base / "jwks.json")
    losses = str(next(base.glob("losses_*.json")))  # the wrapped {judge, losses:[..]} file
    rc, out, _ = run(["verify-meaning", receipt, "--jwks", jwks, "--losses", losses])
    v = json.loads(out)
    assert rc == 0 and v["verified"] is True
    assert v["replayed"] is True  # Stage B actually ran (not signature-only)
    assert round(v["risk_upper_bound"] * 1e6) == bound_micro
