"""The issuer example (`examples/issue_meaning_receipt.py`) must mint a
receipt that the shipped verifier accepts and replays — the issue → verify
loop end-to-end. Uses the deterministic `lexical` scorer so the test needs no
judge model.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="[research] certifier")
pytest.importorskip("joserfc", reason="[verify] / sign path")

_REPO = Path(__file__).resolve().parents[1]
_EXAMPLE = _REPO / "examples" / "issue_meaning_receipt.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("issue_meaning_receipt", _EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_issue_then_verify_roundtrips(tmp_path):
    pairs = [
        ["The committee approved the budget on Tuesday.", "The committee approved the budget."],
        ["Rates rose half a point, the biggest jump this year.", "Rates rose half a point this year."],
        ["The policy applies to all staff from January.", "The policy applies to all staff in January."],
    ]
    pf = tmp_path / "pairs.json"
    pf.write_text(json.dumps(pairs))
    out = tmp_path / "out"

    mod = _load_example()
    rc = mod.main([
        str(pf), "--out", str(out), "--scorer", "lexical",
        "--corpus-id", "test-corpus", "--transform", "summarize:test",
    ])
    assert rc == 0

    receipt = json.loads((out / "receipt.json").read_text())
    jwks = json.loads((out / "jwks.json").read_text())
    losses_file = json.loads((out / "losses.json").read_text())
    assert receipt["schema"] == "sum.meaning_risk_receipt.v1"
    assert "losses" in losses_file and len(losses_file["losses"]) == 3
    # the private key is written and is NOT in the public jwks
    assert (out / "private_jwk.json").exists()
    assert all("d" not in k for k in jwks["keys"])

    # the shipped verifier accepts it AND replays the bound off the losses
    from sum_verify import MeaningReceiptReplayError, verify

    payload = verify(receipt, jwks, losses=losses_file)  # wrapped losses → library unwrap
    assert payload["corpus_id"] == "test-corpus"
    assert payload["risk_upper_bound_micro"] == receipt["payload"]["risk_upper_bound_micro"]

    # tampering the side-band losses is rejected
    bad = list(losses_file["losses"])
    bad[0] = min(1.0, bad[0] + 0.5)
    with pytest.raises(MeaningReceiptReplayError):
        verify(receipt, jwks, losses=bad)
