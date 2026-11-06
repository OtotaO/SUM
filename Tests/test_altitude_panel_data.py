"""The demo page's altitude panel data (``single_file_demo/altitude_rungs.json``)
— structural lock so the committed asset can't rot away from what the panel's
inline JS reads and what the page's honesty labels promise.

The JSON is a committed MEASUREMENT artifact (NLI judge, machine-pinned; see
its generator's docstring). These tests do NOT re-run the judge — they lock
structure, provenance linkage, and the honesty fields, torch-free, in CI.
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "single_file_demo" / "altitude_rungs.json"


def _load():
    return json.loads(_DATA.read_text("utf-8"))


def test_altitude_data_has_ladder_shape():
    d = _load()
    rungs = d["rungs"]
    assert 4 <= len(rungs) <= 6  # the plan's 4-6 detents
    # rung 0 is the source: no loss, 0 compression
    assert rungs[0]["meaning_loss"] is None
    assert rungs[0]["compression_pct"] == 0
    # every later rung carries the fields the panel JS reads
    for r in rungs[1:]:
        for field in (
            "label", "note", "text", "words", "meaning_loss",
            "compression_pct", "source_claims", "preserved_claims",
            "dropped_claims", "added_claims",
        ):
            assert field in r, f"rung {r.get('label')} missing {field}"
        assert 0.0 <= r["meaning_loss"] <= 1.0
    # compression strictly deepens down the ladder
    comps = [r["compression_pct"] for r in rungs]
    assert comps == sorted(comps) and len(set(comps)) == len(comps)


def test_altitude_document_is_in_the_witnessed_chain_corpus():
    """The panel's story is 'this bill is one of the 32 in the signed
    chain' — lock that the document really is, and that the source text is
    byte-identical to the committed corpus."""
    d = _load()
    corpus = json.loads(
        (
            _REPO / "fixtures" / "meaning_receipts_billsum"
            / "corpus_billsum_test_first64.json"
        ).read_text("utf-8")
    )
    doc_id = d["document"]["id"]
    idx = next(i for i, p in enumerate(corpus["pairs"]) if p["id"] == doc_id)
    assert idx < 32  # the chain binds the first 32
    assert d["rungs"][0]["text"] == corpus["pairs"][idx]["source"]
    assert d["rungs"][1]["text"] == corpus["pairs"][idx]["rendering"]


def test_altitude_chain_linkage_matches_committed_chain():
    """The chain_id and quoted bounds in the panel data must match the
    committed chain receipt exactly (no drifting prose numbers)."""
    d = _load()
    chain = json.loads(
        (
            _REPO / "fixtures" / "chain_receipts_billsum"
            / "chain_receipt.billsum.golden.json"
        ).read_text("utf-8")
    )
    pl = chain["payload"]
    assert d["chain_receipt"]["chain_id"] == pl["chain_id"]
    note = d["chain_receipt"]["note"]
    # every number quoted in the note is the receipt's own, in micro units
    assert f"{pl['hops'][0]['risk_upper_bound_micro'] / 1e6:.6f}" in note
    assert f"{pl['hops'][1]['risk_upper_bound_micro'] / 1e6:.6f}" in note
    assert f"{pl['budget_micro'] / 1e6:.6f}" in note
    assert f"{pl['end_to_end']['risk_upper_bound_micro'] / 1e6:.6f}" in note


def test_altitude_chain_wording_claims_no_confidence():
    """The 32 bills are a fixed prefix, not a random draw, so the page must
    not attach a confidence level or 'certified' to the chain (README and
    the chain fixture say no confidence reading applies)."""
    d = _load()
    html = (_REPO / "single_file_demo" / "index.html").read_text("utf-8")
    for text in (d["chain_receipt"]["note"], d["scope"], html):
        for phrase in ("certified chain", "joint confidence", "95% per hop", "(1-delta)"):
            assert phrase not in text, phrase
    assert "no confidence reading applies" in d["chain_receipt"]["note"]


def test_altitude_scope_is_honest():
    """The scope string must carry the measurement-not-guarantee framing and
    the proxy blindness disclosure, and name the judge."""
    d = _load()
    scope = d["scope"].lower()
    assert "measurement" in scope
    assert "not a guarantee" in scope
    assert "arrangement" in scope  # the not_covered blindness list
    assert d["scorer"].startswith("bidirectional-entailment[nli:")


def test_altitude_panel_wired_into_page():
    """index.html actually fetches the asset and carries the panel + its
    honesty line (a deploy of the JSON without the panel, or vice versa,
    fails here)."""
    html = (_REPO / "single_file_demo" / "index.html").read_text("utf-8")
    assert 'fetch("altitude_rungs.json")' in html
    assert 'id="altitude-panel"' in html
    assert "measured, not certified" in html


def test_claims_checked_against_part_of_the_bill_are_not_called_unsupported():
    """The NLI judge reads at most its token window of each pair. When the
    bill does not fit, an added claim was checked against the bill's start
    only, so the page must not call it unsupported by the bill (review
    finding C1/CH-1: a claim the bill supports later was listed as added /
    unsupported)."""
    d = _load()
    html = (_REPO / "single_file_demo" / "index.html").read_text("utf-8")
    assert "versus the full bill" not in html
    assert "within the judge's 512-token reading window" in html
    assert "not found in the part of the bill the judge read" in html
    assert "512 tokens" in d["scope"] and "not as unsupported" in d["scope"]
    for rung in d["rungs"][1:]:
        assert "added_checked_against_partial_source" in rung, rung["label"]
        if rung["added_claims"] and rung["added_checked_against_partial_source"]:
            assert rung["judge_token_limit"] == 512, rung["label"]
    # The bill is longer than the window on this document, so every rung's
    # added claims carry the partial-source marker.
    assert all(r["added_checked_against_partial_source"] for r in d["rungs"][1:])


def test_chain_note_calls_the_values_descriptive_not_bounds():
    note = _load()["chain_receipt"]["note"]
    assert " bound " not in note and "budget" not in note
    assert note.endswith(".")
