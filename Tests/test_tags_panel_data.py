"""The demo page's "Tags and back" card data
(``single_file_demo/tags_roundtrip_panel.json``).

The JSON copies part of the first recorded run on each worked example
(``fixtures/tags_roundtrip/efso_vaults_N/results.json``), with the cut where
that run scored it, and the "tags alone" scores of the 32 topic tags from the
equal-budget run (``fixtures/tags_roundtrip/equal_budget/``). These tests
regenerate it in memory and require the committed file to match byte for byte,
and check every number against the results files directly, so the card cannot
drift from the recorded runs. No model is called.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_DEMO = _REPO / "single_file_demo"
_DATA = _DEMO / "tags_roundtrip_panel.json"
_TAGS = _REPO / "fixtures" / "tags_roundtrip"
_IDS = ("efso_vaults_1", "efso_vaults_2")


def _gen():
    spec = importlib.util.spec_from_file_location("_tags_panel_gen", _DEMO / "generate_tags_panel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _panel():
    return json.loads(_DATA.read_text("utf-8"))


def _text(text_id):
    return next(t for t in _panel()["texts"] if t["id"] == text_id)


def _results(text_id):
    return json.loads((_TAGS / text_id / "results.json").read_text("utf-8"))


def test_committed_json_regenerates_byte_for_byte():
    gen = _gen()
    assert gen.serialize(gen.build()) == _DATA.read_bytes(), (
        "tags_roundtrip_panel.json is stale: run python single_file_demo/generate_tags_panel.py"
    )


def test_both_texts_in_order():
    assert [t["id"] for t in _panel()["texts"]] == list(_IDS)


@pytest.mark.parametrize("text_id", _IDS)
def test_generated_from_is_the_results_file(text_id):
    d, res = _text(text_id), _results(text_id)
    assert d["generated_from"]["sha256"] == hashlib.sha256(
        (_TAGS / text_id / "results.json").read_bytes()).hexdigest()
    assert hashlib.sha256(d["source"]["text"].encode("utf-8")).hexdigest() == res["input"]["sha256"]
    assert d["source"]["sha256"] == res["input"]["sha256"]
    assert d["source"]["words"] == res["input"]["words"]


@pytest.mark.parametrize("text_id", _IDS)
def test_every_number_equals_results_json(text_id):
    d, res = _text(text_id), _results(text_id)
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
        # The cut: present exactly where the run scored it, every number copied.
        if "cut_summary" not in rung:
            assert row["cut"] is None
            continue
        cs, cut = rung["cut_summary"], row["cut"]
        assert cut["summary"]["kept_sentences_mean"] == cs["kept_sentences_mean"]
        assert cut["summary"]["nonempty"] == cs["nonempty"]
        for judge in ("gist", "exact"):
            assert set(cut["summary"][judge]) == set(cs[judge])
            for side, v in cs[judge].items():
                assert cut["summary"][judge][side] == {k: v[k] for k in ("mean", "min", "max")}
        sc = sample["cut"]
        assert cut["shown"]["text"] == sc["text"]
        assert cut["shown"]["kept_sentences"] == sc["kept_sentences"]
        assert cut["shown"]["rewrite_sentences"] == sc["rewrite_sentences"]
        if sc["scores"] is None:
            assert cut["shown"]["scores"] is None
        else:
            for judge in ("gist", "exact"):
                for k in ("recall", "fidelity"):
                    assert cut["shown"]["scores"][judge][k] == sc["scores"][judge][k]
    if "cut" in res:
        assert d["cut"]["rule"] == res["cut"]["rule"]
        assert d["cut"]["threshold"] == res["cut"]["threshold"] == 0.5
        assert d["cut"]["note"] == res.get("cut_note", "")
    else:
        assert d["cut"] is None
    for school in ("topic", "form", "fact"):
        assert len(d["tags"][school]) == len(res["tags"][school])
        for t, src in zip(d["tags"][school], res["tags"][school]):
            for k in ("tag", "school", "kind", "rank", "evidence_found"):
                assert t[k] == src[k]
    assert d["judges"]["exact"]["threshold"] == res["judges"]["exact"]["threshold"] == 0.9
    assert d["judges"]["gist"]["threshold"] == float.fromhex(res["judges"]["gist"]["threshold_float_hex"]) == 0.5
    assert d["model"] == res["model"] == "gpt-4o-mini-2024-07-18"
    assert d["scope"] == res["scope"]


def test_only_the_second_text_has_the_cut():
    assert _text("efso_vaults_1")["cut"] is None
    assert all(r["cut"] is None for r in _text("efso_vaults_1")["rows"])
    rows2 = _text("efso_vaults_2")["rows"]
    assert [r["cut"] is not None for r in rows2] == [r["condition"] != "control" for r in rows2]


@pytest.mark.parametrize("text_id", _IDS)
def test_tags_alone_are_the_same_32_topic_tags(text_id):
    alone = _text(text_id)["tags_alone"]
    path = _REPO / alone["from"]["path"]
    assert alone["from"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    rung = next(r for r in json.loads(path.read_text("utf-8"))["rungs"] if r["condition"] == alone["from"]["condition"])
    row = next(r for r in _text(text_id)["rows"] if r["condition"] == "topic" and r["tag_count"] == 32)
    assert alone["row"] == {"condition": "topic", "tag_count": 32}
    assert rung["tags"] == [t["tag"] for t in row["tags"]]
    assert alone["gist_recall"] == rung["notes_only_scores"]["gist"]["recall"]
    assert alone["exact_recall"] == rung["notes_only_scores"]["exact"]["recall"]
    # What the card's gist caveat says about this row is true here: the list alone is above every rewrite.
    assert alone["gist_recall"] > row["summary"]["gist"]["recall"]["max"]
    # ...and its other claim: each rewrite is credited with some sentence the tags alone are not.
    alone_missed = set(rung["notes_only_scores"]["gist"]["not_recovered"])
    rung_full = next(r for r in _results(text_id)["rungs"] if r["condition"] == "topic" and r["tag_count"] == 32)
    for sample in rung_full["samples"]:
        assert alone_missed - set(sample["scores"]["gist"]["not_recovered"])


@pytest.mark.parametrize("text_id,consent", [("efso_vaults_1", "agreed"), ("efso_vaults_2", "sent it")])
def test_source_metadata_matches_fixture_readme(text_id, consent):
    readme = (_TAGS / text_id / "README.md").read_text("utf-8")
    src = _text(text_id)["source"]
    for field in ("title", "url", "published"):
        assert src[field] in readme, field
    assert consent in readme


def test_text_notes_match_the_readmes():
    r1 = (_TAGS / "efso_vaults_1" / "README.md").read_text("utf-8")
    r2 = (_TAGS / "efso_vaults_2" / "README.md").read_text("utf-8")
    assert "played no part in choosing the judge settings" in r2
    assert "played no part in choosing the judge settings or the cut rule." in " ".join(_text("efso_vaults_2")["notes"])
    assert "fell from 0.23 to 0.16" in " ".join(r1.split())
    n1 = " ".join(_text("efso_vaults_1")["notes"])
    assert "chosen on this text" in n1 and "This run did not score the cut." in n1


def test_card_wired_into_page_with_honest_wording():
    html = (_DEMO / "index.html").read_text("utf-8")
    assert 'fetch("tags_roundtrip_panel.json")' in html
    assert 'id="tags-panel"' in html
    assert "Tags and back: recorded examples" in html
    assert "recorded runs, each measured on one text" in html
    assert "this card calls no model" in html
    # The page elsewhere calls /api/render and /api/complete; the card must not claim the page is model-free.
    assert "nothing on this page calls a model" not in html
    # The mixed rows used a different rewrite prompt; the card must say so.
    assert "Follow the form tags for voice and structure" in html
    # The main prompt differences and the README's confounds must be named on the card.
    assert '"as plain prose"' in html
    assert "carry more tag words" in html
    assert "not independent" in html
    # Every topic-vs-mixed prompt difference the card names must be real, for both texts.
    for text_id in _IDS:
        prompts = _results(text_id)["prompts"]
        assert "as plain prose" in prompts["rewrite"] and "as plain prose" not in prompts["rewrite_mixed"]
        assert "Follow the form tags for voice and structure" in prompts["rewrite_mixed"]
    # The gist caveat and the cut's two bars are on the card.
    assert "The gist judge also credits a bare list" in html
    assert "the original is not consulted" in html
    assert "only at 0.9" in html and "probability 0.5 or more" in html
    # The plain-language lines quote the judges' settings and the model; keep them equal to the data.
    for t in _panel()["texts"]:
        assert t["model"] in html
        assert "at most 80 tokens" in t["judges"]["exact"]["window"]
    assert "cosine similarity of at least 0.5" in html
    assert "at least 0.9 from one window of at most 80 tokens" in html
    start = html.index('id="tags-panel"')
    card = html[start:html.index("</script>", start)]
    for word in ("guarantee", "certified", "faithful", "compliant", "—"):
        assert word not in card, word
    assert "innerHTML" not in card
