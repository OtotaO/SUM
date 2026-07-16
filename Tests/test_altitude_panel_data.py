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
    """The panel's story is 'this bill is one of the 32 in the certified
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
