"""The golden sum.perspective_risk_receipt.v1 fixture — Python side.

Pins the committed perspective golden: it verifies + replays per-cohort
from the committed corpus, and the generator is byte-stable. The Node
side (the cross-runtime proof) is single_file_demo/test_meaning_receipt_verify.js.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("joserfc", reason="[receipt-verify] not installed")

from sum_engine_internal.research.meaning import (
    LexicalCoverageScorer,
    certify_meaning_risk_by_group,
    score_pairs,
    verify_perspective_risk_receipt,
)

_REPO = Path(__file__).resolve().parents[2]
_FIX = _REPO / "fixtures" / "perspective_receipts"


def _load(name):
    return json.loads((_FIX / name).read_text("utf-8"))


@pytest.fixture(scope="module")
def corpus():
    return _load("corpus_2026-06-07.json")


@pytest.fixture(scope="module")
def evidence(corpus):
    triples = corpus["triples"]
    losses = score_pairs(
        [(t["source"], t["rendering"]) for t in triples], LexicalCoverageScorer()
    )
    cohorts = [t["cohort"] for t in triples]
    return losses, cohorts


def test_golden_verifies_and_replays_per_cohort(evidence):
    losses, cohorts = evidence
    golden = _load("perspective_risk_receipt.golden.json")
    jwks = _load("jwks.json")
    out = verify_perspective_risk_receipt(
        golden, jwks, losses=losses, group_ids=cohorts
    )
    assert {g["group_id"] for g in out["groups"]} == {"plain", "technical"}
    assert out["n"] == 12


def test_golden_reports_controls_all_honestly():
    pl = _load("perspective_risk_receipt.golden.json")["payload"]
    # the technical cohort (denser sources) is not controlled at 0.5 →
    # controls_all is honestly False even if the plain cohort passes
    assert pl["controls_all"] is False


def test_golden_is_byte_stable():
    spec = importlib.util.spec_from_file_location(
        "_persp_gen", _FIX / "generate_fixtures.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    receipt, jwks = mod.build()
    assert receipt == _load("perspective_risk_receipt.golden.json")
    assert jwks == _load("jwks.json")
