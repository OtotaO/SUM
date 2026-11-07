"""The demo page's "Tags and back" card data
(``single_file_demo/tags_roundtrip_panel.json``).

The JSON is a copy of part of one recorded run (``fixtures/tags_roundtrip/
efso_vaults_1/results.json``). These tests regenerate it in memory and require
the committed file to match byte for byte, and check every number against the
results file directly, so the card cannot drift from the recorded run. No
model is called.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEMO = _REPO / "single_file_demo"
_DATA = _DEMO / "tags_roundtrip_panel.json"
_FIXTURE = _REPO / "fixtures" / "tags_roundtrip" / "efso_vaults_1"
_RESULTS = _FIXTURE / "results.json"


def _gen():
    spec = importlib.util.spec_from_file_location("_tags_panel_gen", _DEMO / "generate_tags_panel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _panel():
    return json.loads(_DATA.read_text("utf-8"))


def _results():
    return json.loads(_RESULTS.read_text("utf-8"))


def test_committed_json_regenerates_byte_for_byte():
    gen = _gen()
    assert gen.serialize(gen.build()) == _DATA.read_bytes(), (
        "tags_roundtrip_panel.json is stale: run python single_file_demo/generate_tags_panel.py"
    )


def test_generated_from_is_the_results_file():
    d = _panel()
    assert d["generated_from"]["sha256"] == hashlib.sha256(_RESULTS.read_bytes()).hexdigest()
    res = _results()
    assert hashlib.sha256(d["source"]["text"].encode("utf-8")).hexdigest() == res["input"]["sha256"]
    assert d["source"]["sha256"] == res["input"]["sha256"]
    assert d["source"]["words"] == res["input"]["words"]


def test_every_number_equals_results_json():
    d, res = _panel(), _results()
    assert [(r["condition"], r["tag_count"]) for r in d["rows"]] == [
        (r["condition"], r["tag_count"]) for r in res["rungs"]
    ]
    for row, rung in zip(d["rows"], res["rungs"]):
        assert row["tag_words"] == rung["tag_words"]
        assert row["samples"] == len(rung["samples"])
        assert [t["tag"] for t in row["tags"]] == [
            t if isinstance(t, str) else t["tag"] for t in rung["tags"]
        ]
        for t, src in zip(row["tags"], rung["tags"]):
            assert t["school"] == ("topic" if isinstance(src, str) else src["school"])
        for judge in ("gist", "exact"):
            for side in ("recall", "fidelity"):
                assert row["summary"][judge][side] == {
                    k: rung["summary"][judge][side][k] for k in ("mean", "min", "max")
                }
        sample = next(s for s in rung["samples"] if s["seed"] == 7)
        shown = row["shown_sample"]
        assert shown["seed"] == 7
        assert shown["rewrite"] == sample["rewrite"]
        assert shown["rewrite_words"] == sample["rewrite_words"]
        for judge in ("gist", "exact"):
            for k, v in shown["scores"][judge].items():
                assert v == sample["scores"][judge][k], (row["condition"], row["tag_count"], judge, k)
    for school in ("topic", "form", "fact"):
        assert len(d["tags"][school]) == len(res["tags"][school])
        for t, src in zip(d["tags"][school], res["tags"][school]):
            for k in ("tag", "school", "kind", "rank", "evidence_found"):
                assert t[k] == src[k]
    assert d["judges"]["exact"]["threshold"] == res["judges"]["exact"]["threshold"] == 0.9
    assert d["judges"]["gist"]["threshold"] == float.fromhex(res["judges"]["gist"]["threshold_float_hex"]) == 0.5
    assert d["model"] == res["model"] == "gpt-4o-mini-2024-07-18"
    assert d["scope"] == res["scope"]


def test_source_metadata_matches_fixture_readme():
    readme = (_FIXTURE / "README.md").read_text("utf-8")
    src = _panel()["source"]
    for field in ("title", "url", "published"):
        assert src[field] in readme, field
    assert "agreed" in readme


def test_card_wired_into_page_with_honest_wording():
    html = (_DEMO / "index.html").read_text("utf-8")
    assert 'fetch("tags_roundtrip_panel.json")' in html
    assert 'id="tags-panel"' in html
    assert "Tags and back: a recorded example" in html
    assert "recorded run, measured on one text" in html
    assert "this card calls no model" in html
    # The page elsewhere calls /api/render and /api/complete; the card must not claim the page is model-free.
    assert "nothing on this page calls a model" not in html
    # The mixed rows used a different rewrite prompt; the card must say so.
    assert "Follow the form tags for voice and structure" in html
    # The main prompt differences and the README's confounds must be named on the card.
    assert '"as plain prose"' in html
    assert "carry more tag words" in html
    assert "not independent" in html
    # Every topic-vs-mixed prompt difference the card names must be real.
    rj = json.loads((_REPO / "fixtures/tags_roundtrip/efso_vaults_1/results.json").read_text("utf-8"))
    assert "as plain prose" in rj["prompts"]["rewrite"] and "as plain prose" not in rj["prompts"]["rewrite_mixed"]
    # The plain-language lines quote the judges' settings and the model; keep them equal to the data.
    d = _panel()
    assert d["model"] in html
    assert "cosine similarity of at least 0.5" in html
    assert "at least 0.9 from one window of at most 80 tokens" in html
    assert "at most 80 tokens" in d["judges"]["exact"]["window"]
    start = html.index('id="tags-panel"')
    card = html[start:html.index("</script>", start)]
    for word in ("guarantee", "certified", "faithful", "compliant", "—"):
        assert word not in card, word
    assert "innerHTML" not in card
