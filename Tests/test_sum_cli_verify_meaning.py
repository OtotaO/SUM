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
    """Signature-only verify: the bound is labelled issuer-asserted, no
    `controlled` decision is shown, and the scope fields are echoed."""
    rc, out, _ = run(["verify-meaning", _MEANING, "--jwks", _MEANING_JWKS])
    assert rc == 0
    v = json.loads(out)
    assert v["verified"] is True
    assert v["schema"] == "sum.meaning_risk_receipt.v1"
    assert v["replayed"] is False and "not_covered" in v
    assert "risk_upper_bound" not in v and "controlled" not in v
    pl = json.loads(Path(_MEANING).read_text())["payload"]
    assert round(v["issuer_asserted_risk_upper_bound"] * 1e6) == pl["risk_upper_bound_micro"]
    # This golden predates the scope fields: echoed as not_declared.
    assert v["statistical_scope"] == "not_declared"
    assert v["sampling_status"] == "not_declared"
    assert (v["n"], v["method"], v["delta"]) == (pl["n"], pl["method"], 0.05)


def test_verify_perspective_golden():
    rc, out, _ = run(["verify-meaning", _PERSP, "--jwks", _PERSP_JWKS])
    assert rc == 0
    v = json.loads(out)
    assert v["verified"] is True
    assert v["schema"] == "sum.perspective_risk_receipt.v1"
    assert v["replayed"] is False
    assert {c["group_id"] for c in v["cohorts"]} == {"plain", "technical"}
    assert all("issuer_asserted_risk_upper_bound" in c for c in v["cohorts"])
    assert "controls_all" not in v  # a decision only rides a replayed bound
    assert v["simultaneous"] is False
    assert (v["n"], v["delta"], v["statistical_scope"]) == (12, 0.05, "not_declared")


def _perspective_side_band(tmp_path):
    """Recompute the perspective golden's committed evidence exactly as
    fixtures/perspective_receipts/generate_fixtures.py does (lexical scorer,
    deterministic, offline)."""
    from sum_engine_internal.research.meaning import LexicalCoverageScorer, score_pairs
    corpus = json.loads(Path(_PERSP).with_name("corpus_2026-06-07.json").read_text())
    triples = corpus["triples"]
    losses = score_pairs([(t["source"], t["rendering"]) for t in triples],
                         LexicalCoverageScorer())
    lp, gp = tmp_path / "losses.json", tmp_path / "groups.json"
    lp.write_text(json.dumps(list(losses)))
    gp.write_text(json.dumps([t["cohort"] for t in triples]))
    return str(lp), str(gp)


def test_perspective_replay_reports_controls_all(tmp_path):
    lp, gp = _perspective_side_band(tmp_path)
    rc, out, _ = run(["verify-meaning", _PERSP, "--jwks", _PERSP_JWKS,
                      "--losses", lp, "--group-ids", gp])
    assert rc == 0, out
    v = json.loads(out)
    assert v["replayed"] is True and v["controls_all"] is False
    assert all("risk_upper_bound" in c for c in v["cohorts"])


@pytest.mark.parametrize("flag", ["--losses", "--group-ids"])
def test_perspective_half_side_band_is_usage_error(tmp_path, flag):
    """With only one of --losses / --group-ids nothing is replayed; the old
    CLI still printed replayed:true. Now a usage error naming both flags."""
    lp, gp = _perspective_side_band(tmp_path)
    rc, out, err = run(["verify-meaning", _PERSP, "--jwks", _PERSP_JWKS,
                        flag, lp if flag == "--losses" else gp])
    assert rc == 2 and out == ""
    assert "--group-ids" in err and "--losses" in err


@pytest.mark.parametrize("fixture,target", [
    ("meaning_receipts_billsum/meaning_risk_receipt.billsum.golden.json",
     "sum.perspective_risk_receipt.v1"),
    ("chain_receipts_billsum/chain_receipt.billsum.golden.json",
     "sum.perspective_risk_receipt.v1"),
    ("chain_receipts_billsum/chain_receipt.billsum.golden.json",
     "sum.meaning_risk_receipt.v1"),
    ("perspective_receipts/perspective_risk_receipt.golden.json",
     "sum.meaning_risk_receipt.v1"),
])
def test_relabelled_receipt_rejected_cleanly(tmp_path, fixture, target):
    """`schema` is outside the signature. A genuine receipt of another family
    relabelled toward verify-meaning's schemas must be rejected with rc 1 and
    a verdict naming the shape gate, never verified:true or a crash."""
    src = _REPO / "fixtures" / fixture
    r = json.loads(src.read_text())
    r["schema"] = target
    p = tmp_path / "relabelled.json"
    p.write_text(json.dumps(r))
    rc, out, _ = run(["verify-meaning", str(p), "--jwks", str(src.with_name("jwks.json"))])
    assert rc == 1
    v = json.loads(out)
    assert v["verified"] is False
    assert v["error"] == "MeaningReceiptDisclosureError"
    assert "another receipt family" in v["detail"]


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
    assert v["controlled"] is True  # the decision rides a replayed bound
    assert "issuer_asserted_risk_upper_bound" not in v
