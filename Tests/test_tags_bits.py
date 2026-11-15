"""scripts/tags_bits.py with a word-overlap stub in place of the language model,
so CI needs no weights; plus checks on the committed controls and results."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts import tags_bits as tb

ROOT = Path(__file__).resolve().parent.parent
WORD = re.compile(r"[a-z0-9]+")


def stub_bits(context: str, target: str) -> float:
    """One bit per target word the context does not contain."""
    seen = set(WORD.findall(context.lower()))
    return float(sum(1 for w in WORD.findall(target.lower()) if w not in seen))


def test_render_null_and_notes():
    assert tb.render([]) == tb.HEADER + tb.NULL_NOTES + tb.FOOTER
    assert tb.render(["a b", "c"]) == tb.HEADER + "- a b\n- c" + tb.FOOTER


def test_gains_and_summary_arithmetic():
    units = ["alpha beta", "gamma delta"]
    null = [stub_bits(tb.render([]), u) for u in units]
    g = tb.gains(stub_bits, units, ["alpha", "delta gamma"], null)
    assert g == [1.0, 2.0]
    s = tb.summary(g, sum(null))
    assert s == {"gain_bits": 3.0, "share": 0.75, "units_gaining": 2}


def test_checks_ratios():
    rows = [{"condition": c, "gain_bits": v, "share": v / 100}
            for c, v in [("fact@60", 10.0), ("negated:fact@60", 9.0),
                         ("swapped:fact@60", 5.0), ("unrelated", -1.0), ("filler", 0.0)]]
    c = tb.checks({"rows": rows, "self": {"mean_unit_ratio": 0.8}})
    assert c["negated_over_true"] == pytest.approx(0.9)
    assert c["swapped_over_true"] == pytest.approx(0.5)
    assert c["unrelated_share"] == pytest.approx(-0.01)
    assert c["self_mean_unit_ratio"] == 0.8


@pytest.mark.parametrize("text", tb.TEXTS)
def test_run_with_stub_is_deterministic_and_complete(text):
    out = tb.run(text, stub_bits, repeat=2)
    assert out["method"] == "sum.tags_bits.v1"
    assert out["max_difference_between_runs_bits"] == 0.0
    names = [r["condition"] for r in out["rows"]]
    rungs = json.loads((ROOT / f"fixtures/tags_roundtrip/equal_budget/results_{text}.json")
                       .read_text(encoding="utf-8"))["rungs"]
    for r in rungs:
        if r["condition"] != "control":
            assert r["condition"] in names
    for c in ("negated:fact@60", "swapped:fact@60", "unrelated", "filler"):
        assert c in names
    assert len(out["null_bits"]) == len(out["units"])
    # The stub gives a sentence its full bits when it is its own note.
    assert out["self"]["mean_unit_ratio"] == pytest.approx(1.0)


def test_source_sha_is_enforced(monkeypatch, tmp_path):
    real = tb.ROOT
    fake = tmp_path / "repo"
    for rel in ("fixtures/tags_roundtrip/equal_budget", "fixtures/tags_roundtrip/efso_vaults_1",
                "fixtures/tags_bits"):
        (fake / rel).mkdir(parents=True)
    for rel in ("fixtures/tags_roundtrip/equal_budget/results_efso_vaults_1.json",
                "fixtures/tags_roundtrip/equal_budget/results_efso_vaults_2.json",
                "fixtures/tags_bits/controls.json"):
        (fake / rel).write_bytes((real / rel).read_bytes())
    (fake / "fixtures/tags_roundtrip/efso_vaults_1/source.md").write_text("changed", encoding="utf-8")
    monkeypatch.setattr(tb, "ROOT", fake)
    with pytest.raises(SystemExit):
        tb.load_inputs("efso_vaults_1")


CONTROLS = json.loads((ROOT / "fixtures/tags_bits/controls.json").read_text(encoding="utf-8"))["texts"]


@pytest.mark.parametrize("text", tb.TEXTS)
def test_controls_line_up_with_the_recorded_fact_tags(text):
    fact = [r for r in json.loads((ROOT / f"fixtures/tags_roundtrip/equal_budget/results_{text}.json")
                                  .read_text(encoding="utf-8"))["rungs"]
            if r["condition"] == tb.FACT_RUNG][0]["tags"]
    c = CONTROLS[text]
    assert len(c["negated"]) == len(c["swapped"]) == len(fact)
    for true, neg, swap in zip(fact, c["negated"], c["swapped"]):
        assert re.search(r"\b(never|not)\b", neg), neg
        assert neg != true and swap != true
        assert WORD.findall(swap.lower())[0] == WORD.findall(true.lower())[0]


@pytest.mark.parametrize("text", tb.TEXTS)
def test_committed_results_match_inputs(text):
    path = ROOT / f"fixtures/tags_bits/results_{text}.json"
    if not path.exists():
        pytest.skip("not run yet")
    out = json.loads(path.read_text(encoding="utf-8"))
    assert out["method"] == tb.METHOD_ID
    assert out["scorer"]["model_revision"] == tb.MODEL_REVISION
    for rel, sha in out["inputs"].items():
        assert tb.sha256_file(ROOT / rel) == sha, rel
    by = {r["condition"]: r for r in out["rows"]}
    assert out["checks"]["negated_over_true"] == pytest.approx(
        by["negated:fact@60"]["gain_bits"] / by["fact@60"]["gain_bits"])
    for r in out["rows"]:
        assert sum(r["per_unit_gain_bits"]) == pytest.approx(r["gain_bits"])
        assert r["share"] == pytest.approx(r["gain_bits"] / out["null_total_bits"])
