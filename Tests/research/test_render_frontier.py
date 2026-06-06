"""Tests for the render-frontier abstraction — the workbench backend.

The frontier is the single object the slider / number box / scrubber /
API are all views over. These pin: construction + scoring, the ordered
faithful→compressed path, the scrubber's position mapping, the named-
point lookup, serialisation, and the injected-render-fn path.
"""
from __future__ import annotations

import pytest

from sum_engine_internal.research.frontier import FrontierPoint, RenderFrontier
from sum_engine_internal.research.meaning.meaning_loss import (
    LexicalCoverageScorer,
)


SOURCE = (
    "The treaty was signed in Vienna in 1815. Delegates from the great "
    "powers redrew the map of Europe. The settlement held for decades."
)

# Ordered most-faithful → most-compressed (the compression control).
_RENDERINGS = [
    ("full", {"density": 1.0}, SOURCE),
    ("medium", {"density": 0.6},
     "The treaty was signed in Vienna in 1815. The great powers redrew Europe."),
    ("tight", {"density": 0.3}, "Vienna 1815 treaty redrew Europe."),
    ("tag", {"density": 0.1}, "Vienna treaty"),
]


def _frontier():
    return RenderFrontier.from_renderings(SOURCE, _RENDERINGS, LexicalCoverageScorer())


# ── construction + scoring ────────────────────────────────────────────


def test_builds_one_point_per_rendering():
    f = _frontier()
    assert len(f) == 4
    assert f.scorer_name == "lexical-coverage-bidirectional"
    assert f.scorer_version == "1"


def test_faithful_point_is_identity_zero_loss():
    f = _frontier()
    assert f.faithful.label == "full"
    assert f.faithful.meaning_loss == 0.0  # rendering == source


def test_loss_generally_rises_with_compression():
    f = _frontier()
    # not strictly monotone (measured, not controlled), but the most
    # compressed point must lose more than the faithful one
    assert f.compressed.meaning_loss > f.faithful.meaning_loss


def test_all_losses_bounded():
    f = _frontier()
    assert all(0.0 <= x <= 1.0 for x in f.losses())


def test_positions_span_unit_interval():
    f = _frontier()
    positions = [p.position for p in f.points]
    assert positions[0] == 0.0
    assert positions[-1] == 1.0
    assert positions == sorted(positions)


# ── scrubber ──────────────────────────────────────────────────────────


def test_scrub_endpoints():
    f = _frontier()
    assert f.scrub(0.0).label == "full"
    assert f.scrub(1.0).label == "tag"


def test_scrub_clamps_out_of_range():
    f = _frontier()
    assert f.scrub(-5.0).label == "full"
    assert f.scrub(99.0).label == "tag"


def test_scrub_midpoint_picks_an_interior_point():
    f = _frontier()
    mid = f.scrub(0.5)
    assert mid.label in {"medium", "tight"}  # round(0.5*3) = 2 → 'tight'


def test_scrub_single_point_frontier():
    f = RenderFrontier.from_renderings(
        SOURCE, [("only", {"density": 1.0}, SOURCE)], LexicalCoverageScorer()
    )
    assert f.scrub(0.0).label == "only"
    assert f.scrub(1.0).label == "only"
    assert f.faithful is f.compressed


# ── named point lookup (a perspective is a labelled point) ────────────


def test_at_label():
    f = _frontier()
    assert f.at("tight").params == {"density": 0.3}


def test_at_missing_label_raises():
    f = _frontier()
    with pytest.raises(KeyError):
        f.at("nonexistent")


# ── fact_preservation passthrough ─────────────────────────────────────


def test_fact_preservation_passthrough():
    f = RenderFrontier.from_renderings(
        SOURCE, _RENDERINGS, LexicalCoverageScorer(),
        fact_preservations=[1.0, 0.9, 0.7, 0.4],
    )
    assert f.faithful.fact_preservation == 1.0
    assert f.compressed.fact_preservation == 0.4


def test_fact_preservation_length_mismatch_raises():
    with pytest.raises(ValueError, match="fact_preservations"):
        RenderFrontier.from_renderings(
            SOURCE, _RENDERINGS, LexicalCoverageScorer(),
            fact_preservations=[1.0, 0.9],  # too few
        )


# ── injected render_fn ────────────────────────────────────────────────


def test_from_render_fn():
    def fake_render(source: str, params):
        # deterministic stand-in renderer: truncate by 'density'
        words = source.split()
        keep = max(1, int(len(words) * params["density"]))
        return " ".join(words[:keep])

    settings = [("full", {"density": 1.0}), ("half", {"density": 0.5}),
                ("tag", {"density": 0.05})]
    f = RenderFrontier.from_render_fn(SOURCE, settings, fake_render, LexicalCoverageScorer())
    assert len(f) == 3
    assert f.faithful.meaning_loss == 0.0  # density=1.0 keeps all words
    assert f.compressed.meaning_loss > 0.0


# ── serialisation (the API / MCP / CURL view) ─────────────────────────


def test_as_dict_shape():
    f = _frontier()
    d = f.as_dict()
    assert d["scorer"] == "lexical-coverage-bidirectional"
    assert d["n"] == 4
    assert "measurement_note" in d  # honest: per-doc measurement, not a bound
    assert len(d["points"]) == 4
    p0 = d["points"][0]
    assert set(p0) >= {"label", "params", "rendering", "meaning_loss", "position"}


def test_point_as_dict_omits_none_fact_preservation():
    p = FrontierPoint("x", {"density": 1.0}, "r", 0.1, 0.0)
    assert "fact_preservation" not in p.as_dict()


# ── empty ─────────────────────────────────────────────────────────────


def test_empty_frontier_raises():
    with pytest.raises(ValueError, match="at least one"):
        RenderFrontier.from_renderings(SOURCE, [], LexicalCoverageScorer())
