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
    """The panel's story is 'this bill is one of the 32 covered by the signed
    chain receipt' — lock that the document really is, and that the source
    text is byte-identical to the committed corpus."""
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


def test_altitude_chain_note_is_descriptive_and_matches_the_generator():
    """The note states bound values with the delta each was computed at, as the
    receipt records them. It must not read as a confidence statement ("95%",
    "joint confidence"), and the committed JSON must equal what the generator
    writes, so a regeneration cannot quietly bring the old wording back."""
    import importlib.util

    d = _load()
    chain = json.loads(
        (
            _REPO / "fixtures" / "chain_receipts_billsum"
            / "chain_receipt.billsum.golden.json"
        ).read_text("utf-8")
    )
    pl = chain["payload"]
    note = d["chain_receipt"]["note"]
    for hop in pl["hops"]:
        assert f"delta {hop['delta_micro'] / 1e6:.2f}" in note
    assert f"joint delta {pl['joint_delta_micro'] / 1e6:.2f}" in note
    assert "confidence" not in note.replace("not a confidence statement", "")
    assert "%" not in note
    spec = importlib.util.spec_from_file_location(
        "_altitude_gen", _REPO / "single_file_demo" / "generate_altitude_rungs.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert note == gen.CHAIN_NOTE
    assert d["scope"] == gen.SCOPE


def test_altitude_scope_is_honest():
    """The scope string must say this is a measurement on one bill and not a
    bound for other documents, carry the proxy blindness disclosure, and name
    the judge."""
    d = _load()
    scope = d["scope"].lower()
    assert "measurement" in scope
    assert "not a bound for other documents" in scope
    assert "arrangement" in scope  # the not_covered blindness list
    assert d["scorer"].startswith("bidirectional-entailment[nli:")


def test_altitude_panel_wired_into_page():
    """index.html actually fetches the asset and carries the panel + its
    honesty line (a deploy of the JSON without the panel, or vice versa,
    fails here)."""
    html = (_REPO / "single_file_demo" / "index.html").read_text("utf-8")
    assert 'fetch("altitude_rungs.json")' in html
    assert 'id="altitude-panel"' in html
    assert "measured on one bill" in html


def test_page_copy_makes_no_overclaim():
    """The page's own copy and code never call SUM's outputs certified,
    faithful, guaranteed, compliant or verified-true. A signature shows which
    key signed which bytes; a comparison is literal string evidence.

    Scope: index.html, workbench.js, change_evidence.js and
    altitude_rungs.json. Not scanned: sample_meaning_risk_receipt.json, which
    the meaning-receipt box loads on request. It is a signed receipt, so its
    wording (it contains "CERTIFICATE") cannot be edited without breaking the
    signature, and the box labels its bound as issuer-asserted.

    The one exemption is the extraction prompt sent to the model (it asks the
    model to abstain on a clause it cannot represent faithfully); it is never
    rendered."""
    import re

    demo = _REPO / "single_file_demo"
    html = (demo / "index.html").read_text("utf-8")
    prompt = re.search(r"const CLAUDE_PROMPT_TEMPLATE = `.*?`;", html, re.S)
    assert prompt, "extraction prompt not found; update this exemption"
    html = html.replace(prompt.group(0), "")
    banned = re.compile(r"certif|faithful|guarantee|compliant|verified-true", re.I)
    for name, text in (
        ("index.html", html),
        ("altitude_rungs.json", _DATA.read_text("utf-8")),
        ("workbench.js", (demo / "workbench.js").read_text("utf-8")),
        ("change_evidence.js", (demo / "change_evidence.js").read_text("utf-8")),
    ):
        hits = sorted({m.group(0) for m in banned.finditer(text)})
        assert not hits, f"{name} uses {hits}"
