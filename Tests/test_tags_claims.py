"""scripts/tags_claims.py with a word-overlap stub in place of the NLI judge,
so CI needs no weights; plus checks on the committed claims and results."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts import tags_bits as tb
from scripts import tags_claims as tc
from scripts.tags_roundtrip import EXACT_THRESHOLD, _norm

ROOT = Path(__file__).resolve().parent.parent
WORD = re.compile(r"[a-z0-9]+")
CLAIMS = json.loads((ROOT / tc.CLAIMS).read_text(encoding="utf-8"))


def stub_probs(premise: str, hypotheses):
    """Share of a claim's words that the premise contains."""
    seen = set(WORD.findall(premise.lower()))
    out = []
    for h in hypotheses:
        words = WORD.findall(h.lower())
        out.append(sum(w in seen for w in words) / len(words) if words else 0.0)
    return out


@pytest.mark.parametrize("text", tb.TEXTS)
def test_claims_are_well_formed_and_quoted(text):
    source = _norm((ROOT / f"fixtures/tags_roundtrip/{text}/source.md").read_text(encoding="utf-8"))
    claims = CLAIMS["texts"][text]
    ids = [c["id"] for c in claims] + [c["id"] for c in CLAIMS["dropped"][text]]
    assert len(ids) == len(set(ids))
    for c in claims:
        assert 4 <= len(c["claim"].split()) <= 16, c["claim"]
        assert len(c["quote"].split()) <= 25, c["quote"]
        assert _norm(c["quote"]) in source, c["quote"]
        assert isinstance(c["checked"], bool)


def test_measure_and_checks_arithmetic():
    claims = [{"id": "a", "claim": "alpha beta"}, {"id": "b", "claim": "gamma delta"}]
    conds = [{"condition": n, "group": g, "notes": notes} for n, g, notes in [
        ("fact@60", "recorded", ["alpha beta", "gamma"]),
        ("negated:fact@60", "control", ["alpha never"]),
        ("swapped:fact@60", "control", ["alpha beta"]),
        ("unrelated", "control", ["x"]), ("filler", "control", ["a note"]),
        ("other_text:fact@60", "same_series", ["gamma delta"]),
        ("source", "self", ["alpha beta gamma delta"]),
        ("no_tags_rewrite:t:seed7", "floor", ["beta"]),
        ("no_tags_rewrite:t:seed8", "floor", ["epsilon zeta"])]]
    claims.append({"id": "e", "claim": "epsilon zeta"})
    rows = tc.measure(stub_probs, claims, conds, 0.9)
    by = {r["condition"]: r for r in rows}
    assert by["fact@60"]["credited_ids"] == ["a"]
    split = tc.firm(rows, claims)
    assert split == {"loose_ids": ["e"], "firm_claims": 2}
    assert by["source"]["firm_recall"] == 1.0 and by["source"]["recall"] == pytest.approx(2 / 3)
    c = tc.checks(rows, split, len(claims))
    assert c["fact_firm_credited"] == 1 and c["negated_firm_credited"] == 0
    assert c["swapped_firm_credited"] == 1 and c["other_text_max_firm_credited"] == 1
    assert c["loose_share"] == pytest.approx(1 / 3)
    assert tc.credited(None, 0.9) is False and tc.credited(0.9, 0.9) is True


@pytest.mark.parametrize("text", tb.TEXTS)
def test_run_with_stub_is_deterministic_and_complete(text):
    out = tc.run(text, stub_probs, EXACT_THRESHOLD, repeat=2)
    assert out["method"] == "sum.tags_claims.v1"
    assert out["max_difference_between_runs"] == 0.0
    names = [r["condition"] for r in out["rows"]]
    for c in ("fact@60", "negated:fact@60", "swapped:fact@60", "unrelated", "filler", "source"):
        assert c in names
    assert sum(n.startswith("no_tags_rewrite:") for n in names) == 6
    assert out["claims"] == len(CLAIMS["texts"][text])
    assert tc.CLAIMS.as_posix() in out["inputs"]


@pytest.mark.parametrize("text", tb.TEXTS)
def test_committed_results_match_inputs(text):
    path = ROOT / f"fixtures/tags_claims/results_{text}.json"
    if not path.exists():
        pytest.skip("not run yet")
    out = json.loads(path.read_text(encoding="utf-8"))
    assert out["method"] == tc.METHOD_ID
    for rel, sha in out["inputs"].items():
        assert tb.sha256_file(ROOT / rel) == sha, rel
    ids = [c["id"] for c in CLAIMS["texts"][text]]
    t = out["judge"]["threshold"]
    for r in out["rows"]:
        assert len(r["probs"]) == len(ids)
        assert r["credited_ids"] == [i for i, p in zip(ids, r["probs"]) if tc.credited(p, t)]
        assert r["recall"] == pytest.approx(r["credited"] / len(ids))
        assert set(r["firm_credited_ids"]) == set(r["credited_ids"]) - set(out["loose_ids"])
    loose = {i for r in out["rows"] if r["group"] == "floor" for i in r["credited_ids"]}
    assert sorted(loose) == out["loose_ids"]
    assert out["checks"] == pytest.approx(tc.checks(out["rows"], {"loose_ids": out["loose_ids"],
                                                                  "firm_claims": out["firm_claims"]}, len(ids)))


def test_model_pass_record_matches_claims():
    path = ROOT / "fixtures/tags_claims/model_pass_efso_vaults_2.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["claims_sha256"] == tb.sha256_file(ROOT / tc.CLAIMS)
    claims = {c["id"]: c for c in CLAIMS["texts"][rec["text"]]}
    assert rec["claims_read"] == len(claims)
    for p in rec["proposals"]:
        assert claims[p["id"]]["claim"] == p["claim"] and claims[p["id"]]["quote"] == p["quote"]
        assert p["proposed"] and len(p["reasons"]) == 2
