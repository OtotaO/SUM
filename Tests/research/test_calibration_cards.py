"""Calibration cards are hard-locked to their committed measurement artifacts.

A `sum.calibration_card.v1` card is the product surface for judge validity
(docs/CALIBRATION_CARDS.md). Its entire value is that it CANNOT drift from
the measurement it cites — so this test recomputes every number in every
card from the committed result artifact and asserts equality. A card edit
without a matching measurement, or a re-measurement without a card refresh,
fails CI here.

No torch / datasets / network: this only reads committed JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CARDS_DIR = REPO / "fixtures" / "calibration_cards"

REQUIRED_FIELDS = {
    "schema",
    "signed",
    "card_id",
    "title",
    "what_this_is",
    "corpus",
    "aggregation_level",
    "measured_at",
    "results",
    "provenance",
    "scope_limits",
    "reading_guidance",
}


def _cards() -> list[Path]:
    found = sorted(CARDS_DIR.glob("*.json"))
    assert found, f"no calibration cards found under {CARDS_DIR}"
    return found


@pytest.mark.parametrize("card_path", _cards(), ids=lambda p: p.stem)
def test_card_shape(card_path: Path):
    card = json.loads(card_path.read_text())
    missing = REQUIRED_FIELDS - set(card)
    assert not missing, f"{card_path.name} missing fields: {sorted(missing)}"
    assert card["schema"] == "sum.calibration_card.v1"
    assert card["signed"] is False, (
        "cards are unsigned BY DESIGN (measurements about the instrument, "
        "not certified document properties) — see docs/CALIBRATION_CARDS.md"
    )
    assert card["scope_limits"], "a card without scope limits is marketing"
    assert card["results"], "a card must carry at least one result"
    # Every provenance pointer must exist in the repo.
    prov = card["provenance"]
    assert (REPO / prov["script"]).is_file(), prov["script"]
    if "result_artifact" in prov:
        assert (REPO / prov["result_artifact"]).is_file(), prov["result_artifact"]


def test_summeval_pooled_numbers_match_artifact():
    card = json.loads(
        (CARDS_DIR / "summeval_pooled_2026-07-02.json").read_text()
    )
    artifact = json.loads(
        (
            REPO
            / "Tests"
            / "benchmarks"
            / "meaning_proxy_human_calibration.result.json"
        ).read_text()
    )
    for row in card["results"]:
        measured = artifact["scorers"][row["scorer"]]["meaning_composite"]
        assert row["spearman"] == measured["spearman"], row["scorer"]
        assert row["n"] == measured["n"], row["scorer"]


@pytest.mark.parametrize("half,card_name", [
    ("xsum", "frank_xsum_2026-07-02.json"),
    ("cnndm", "frank_cnndm_2026-07-02.json"),
])
def test_frank_numbers_match_artifact(half: str, card_name: str):
    card = json.loads((CARDS_DIR / card_name).read_text())
    artifact = json.loads(
        (REPO / "Tests" / "benchmarks" / "frank_results.json").read_text()
    )
    scorers = artifact["halves"][half]["scorers"]
    for row in card["results"]:
        measured = scorers[row["scorer"]]
        assert row["spearman"] == measured["spearman"], row["scorer"]
        assert row["spearman_ci95"] == measured["spearman_ci95"], row["scorer"]
        assert row["n"] == measured["n"], row["scorer"]


def test_system_level_card_matches_proof_boundary_prose():
    """The system-level card cites PROOF_BOUNDARY (no committed result JSON
    for the recompute script yet) — pin the cited range and highlight to the
    exact prose so either drifting alone fails."""
    card = json.loads(
        (CARDS_DIR / "summeval_system_level_2026-07-02.json").read_text()
    )
    row = card["results"][0]
    assert row["spearman_range"] == [0.57, 0.75]
    assert row["n_systems"] == 16
    boundary = (REPO / "docs" / "PROOF_BOUNDARY.md").read_text()
    assert "0.57" in boundary and "0.75" in boundary
    assert "permutation test" in boundary
    assert "0.70" in row["highlight"] and "[0.49, 0.83]" in row["highlight"]
    assert "0.70, article-bootstrap 95% CI [0.49, 0.83]" in boundary
