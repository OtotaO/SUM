"""The transparency log's chain verifies on every PR, and the tool's
append/verify mechanics hold (docs/TRANSPARENCY_LOG.md).

The committed log is load-bearing the moment anyone relies on a witnessed
receipt — so CI recomputes the whole chain (and the witnessed files'
canonical hashes) on every run, exactly like the goldens replay.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "witness_receipt", REPO / "scripts" / "witness_receipt.py"
)
witness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_spec and witness)


def test_committed_log_chain_verifies():
    report = witness.verify_log(check_files=True)
    assert report["ok"], report["problems"]
    assert report["n_entries"] >= 2  # the two binding-gate goldens


def test_committed_log_witnesses_both_goldens():
    entries = witness.read_log()
    sources = {e["source_path"] for e in entries}
    assert (
        "fixtures/meaning_receipts_billsum/"
        "meaning_risk_receipt.billsum.golden.json" in sources
    )
    assert (
        "fixtures/meaning_receipts_translation/"
        "meaning_risk_receipt.translation.golden.json" in sources
    )


def test_append_then_verify_roundtrip(tmp_path):
    log = tmp_path / "log.jsonl"
    receipt = tmp_path / "r.json"
    receipt.write_text(json.dumps({"schema": "sum.test.v1", "kid": "k",
                                   "payload": {"a": 1}, "jws": "x"}))
    e1 = witness.append_entry(receipt, log_path=log, note="first")
    assert e1["seq"] == 1 and e1["prev_entry_hash"] is None
    receipt2 = tmp_path / "r2.json"
    receipt2.write_text(json.dumps({"schema": "sum.test.v1", "kid": "k",
                                    "payload": {"a": 2}, "jws": "y"}))
    e2 = witness.append_entry(receipt2, log_path=log)
    assert e2["seq"] == 2 and e2["prev_entry_hash"] == e1["entry_hash"]
    report = witness.verify_log(log, check_files=False)
    assert report["ok"] and report["n_entries"] == 2


def test_append_is_once_per_receipt(tmp_path):
    log = tmp_path / "log.jsonl"
    receipt = tmp_path / "r.json"
    receipt.write_text(json.dumps({"schema": "sum.test.v1", "kid": "k",
                                   "payload": {"a": 1}, "jws": "x"}))
    witness.append_entry(receipt, log_path=log)
    with pytest.raises(ValueError, match="already witnessed"):
        witness.append_entry(receipt, log_path=log)


def test_tamper_is_detected(tmp_path):
    log = tmp_path / "log.jsonl"
    for i in range(3):
        r = tmp_path / f"r{i}.json"
        r.write_text(json.dumps({"schema": "sum.test.v1", "kid": "k",
                                 "payload": {"i": i}, "jws": "x"}))
        witness.append_entry(r, log_path=log)
    lines = log.read_text().splitlines()
    # Edit the middle entry's note-free body (flip its receipt_hash).
    doctored = json.loads(lines[1])
    doctored["receipt_hash"] = "sha256-" + "0" * 64
    lines[1] = json.dumps(doctored, separators=(",", ":"))
    log.write_text("\n".join(lines) + "\n")
    report = witness.verify_log(log, check_files=False)
    assert not report["ok"]
    # Both the edited entry's own hash AND the next entry's chain break.
    assert any("entry_hash mismatch" in p for p in report["problems"])
    assert any("chain broken" in p for p in report["problems"])


def test_hash_shape_matches_receipt_family():
    entries = witness.read_log()
    for e in entries:
        assert e["receipt_hash"].startswith("sha256-")
        assert len(e["receipt_hash"]) == 7 + 64
        assert e["entry_hash"].startswith("sha256-")


def test_non_envelope_is_refused(tmp_path):
    bad = tmp_path / "not_a_receipt.json"
    bad.write_text(json.dumps({"hello": "world"}))
    with pytest.raises(ValueError, match="no 'schema' field"):
        witness.append_entry(bad, log_path=tmp_path / "log.jsonl")


def test_duplicated_line_is_flagged(tmp_path):
    """A byte-identical duplicated entry must be flagged — the verify walk
    positions entries by enumerate, not list.index (which returns the
    FIRST equal dict and would satisfy the seq check for a duplicate)."""
    log = tmp_path / "log.jsonl"
    receipt = tmp_path / "r.json"
    receipt.write_text(json.dumps({"schema": "sum.test.v1", "kid": "k",
                                   "payload": {"a": 1}, "jws": "x"}))
    witness.append_entry(receipt, log_path=log)
    line = log.read_text().strip()
    log.write_text(line + "\n" + line + "\n")  # replay the same line
    report = witness.verify_log(log, check_files=False)
    assert not report["ok"]
    assert any("non-monotonic" in p or "chain broken" in p
               for p in report["problems"])
