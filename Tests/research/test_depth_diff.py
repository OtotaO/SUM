"""Tests for RenderFrontier.depth_diff (the semantic depth diff).

Deterministic: a subset-entailment stub judge (no [judge] extra needed), valid
here because every rung is EXTRACTIVE (verbatim source sentences dropped), so
'is this claim preserved' == 'are its words present'. These lock the connect-
the-pieces arithmetic: per-rung kept/dropped/added via the existing explain(),
and loss-per-compression as a finite difference over a REAL compression
coordinate (1 - words(rendering)/words(source)) — never the index position.
"""
import pytest

from sum_engine_internal.research.frontier import RenderFrontier, RungDiff
from sum_engine_internal.research.meaning.meaning_loss import (
    EntailmentScorer,
    LexicalCoverageScorer,
)

_STOP = set("a an the of to and or is are was were in on for with that this it its as by".split())


def _content(t):
    return {w.strip(".,;:!?").lower() for w in t.split()} - _STOP


def _subset_entails(premise, hypothesis):
    h = _content(hypothesis)
    return bool(h) and h.issubset(_content(premise))


def _judge():
    return EntailmentScorer(entails=_subset_entails, judge_name="stub-subset", judge_version="t")


SOURCE = (
    "The lease begins on March 1 and runs for twelve months. "
    "Rent is 1800 dollars due on the first of each month. "
    "A late fee of 75 dollars applies after the fifth day. "
    "The landlord may enter with 24 hours written notice. "
    "Either party may terminate with 60 days notice."
)
S1 = "The lease begins on March 1 and runs for twelve months."
S2 = "Rent is 1800 dollars due on the first of each month."


def _frontier(renderings):
    j = _judge()
    return RenderFrontier.from_renderings(SOURCE, renderings, j), j


def test_one_rung_per_point_and_first_lambda_is_none():
    f, j = _frontier([("full", {}, SOURCE), ("brief", {}, S1 + " " + S2)])
    rungs = f.depth_diff(j)
    assert len(rungs) == 2
    assert all(isinstance(r, RungDiff) for r in rungs)
    assert rungs[0].loss_per_compression is None  # no left neighbour
    assert rungs[0].meaning_loss == 0.0 and rungs[0].dropped_claims == ()


def test_dropped_and_added_decomposition():
    # brief drops 3 source sentences; headline keeps S1 verbatim + 1 fabrication.
    f, j = _frontier([
        ("full", {}, SOURCE),
        ("brief", {}, S1 + " " + S2),
        ("headline", {}, S1 + " Pets are not allowed under any circumstances."),
    ])
    rungs = f.depth_diff(j)
    assert len(rungs[1].dropped_claims) == 3        # late fee, entry, termination
    assert rungs[1].added_claims == ()
    assert len(rungs[2].added_claims) == 1          # the fabricated 'Pets...' sentence
    assert "Pets are not allowed" in rungs[2].added_claims[0]


def test_lambda_is_over_compression_not_index():
    # Two rungs with the SAME drop but DIFFERENT lengths must give a lambda that
    # depends on the compression delta, not on the unit index spacing.
    f, j = _frontier([("full", {}, SOURCE), ("brief", {}, S1 + " " + S2)])
    rungs = f.depth_diff(j)
    src_w = len(SOURCE.split())
    comp = 1.0 - len((S1 + " " + S2).split()) / src_w
    expected = (rungs[1].meaning_loss - rungs[0].meaning_loss) / comp
    assert rungs[1].loss_per_compression == pytest.approx(expected)
    # index spacing would have divided by position step 1.0; compression != 1.0 here.
    assert comp != pytest.approx(1.0)


def test_lambda_none_when_compression_unchanged():
    # Two renderings of identical length (a paraphrase-length swap): Δcompression
    # ~ 0, so the slope is undefined and must be None, not a divide-by-zero/inf.
    same_len = "Rent is 1800 dollars due on the first of each month."  # same word count as S1
    assert len(same_len.split()) == len(S1.split())
    f, j = _frontier([("a", {}, S1), ("b", {}, same_len)])
    rungs = f.depth_diff(j)
    assert rungs[1].loss_per_compression is None


def test_single_rung_lambda_none():
    f, j = _frontier([("only", {}, SOURCE)])
    rungs = f.depth_diff(j)
    assert len(rungs) == 1 and rungs[0].loss_per_compression is None


def test_lexical_scorer_rejected():
    f, _ = _frontier([("full", {}, SOURCE)])
    with pytest.raises(TypeError, match="entailment judge"):
        f.depth_diff(LexicalCoverageScorer())  # no .explain → rejected


def test_as_dict_is_measured_not_certified():
    f, j = _frontier([("full", {}, SOURCE), ("brief", {}, S1)])
    d = f.depth_diff(j)[1].as_dict()
    # carries the honest notes, never a guarantee / (1-δ) / certified field.
    assert "NOT a metric" in d["meaning_loss_note"]
    assert "NOT a rate-distortion derivative" in d["loss_per_compression_note"]
    blob = repr(d).lower()
    assert "guarantee" not in blob and "certified" not in blob and "1-delta" not in blob


def test_cli_depth_diff_json(tmp_path, monkeypatch):
    """The `sum depth-diff` render path, deterministically (stub judge, no
    [judge] extra), so the CLI is covered even where the NLI model is absent."""
    import argparse
    import contextlib
    import io
    import json as _json

    import sum_cli.main as m

    src = tmp_path / "s.txt"
    src.write_text(SOURCE, encoding="utf-8")
    v1 = tmp_path / "detailed.txt"
    v1.write_text(S1 + " " + S2, encoding="utf-8")
    monkeypatch.setattr(m, "_load_meaning_scorer", lambda name: (_judge(), None))
    ns = argparse.Namespace(source=str(src), version=[str(v1)], scorer="nli", json=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = m.cmd_depth_diff(ns)
    assert rc == 0
    out = _json.loads(buf.getvalue())
    assert out["scorer"].startswith("bidirectional-entailment")
    assert "not a certified bound" in out["scope"]
    assert out["rungs"][0]["label"] == "detailed.txt"
    assert "NOT a metric" in out["rungs"][0]["meaning_loss_note"]


def test_cli_depth_diff_requires_a_version(tmp_path, monkeypatch):
    import argparse

    import sum_cli.main as m

    src = tmp_path / "s.txt"
    src.write_text(SOURCE, encoding="utf-8")
    monkeypatch.setattr(m, "_load_meaning_scorer", lambda name: (_judge(), None))
    ns = argparse.Namespace(source=str(src), version=[], scorer="nli", json=True)
    assert m.cmd_depth_diff(ns) == 2  # no --version → rc 2
