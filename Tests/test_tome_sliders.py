"""Tests for TomeSliders + controlled rendering."""
from __future__ import annotations

import pytest

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
from sum_engine_internal.ensemble.tome_sliders import TomeSliders, apply_density


# ─── TomeSliders dataclass ────────────────────────────────────────────


class TestTomeSliders:
    def test_defaults(self) -> None:
        s = TomeSliders()
        assert s.density == 1.0
        assert s.length == 0.5
        assert s.formality == 0.5
        assert s.audience == 0.5
        assert s.perspective == 0.5

    def test_rejects_density_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            TomeSliders(density=1.5)
        with pytest.raises(ValueError):
            TomeSliders(density=-0.1)

    def test_rejects_length_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            TomeSliders(length=1.5)

    def test_is_frozen(self) -> None:
        s = TomeSliders()
        with pytest.raises(Exception):
            s.density = 0.5  # type: ignore[misc]

    def test_requires_extrapolator_balanced(self) -> None:
        assert TomeSliders().requires_extrapolator() is False
        assert TomeSliders(density=0.5).requires_extrapolator() is False

    def test_requires_extrapolator_when_tilted(self) -> None:
        assert TomeSliders(length=0.9).requires_extrapolator() is True
        assert TomeSliders(formality=0.0).requires_extrapolator() is True
        assert TomeSliders(audience=1.0).requires_extrapolator() is True
        assert TomeSliders(perspective=0.1).requires_extrapolator() is True

    def test_header_line(self) -> None:
        s = TomeSliders(density=0.5, length=0.8, formality=0.2)
        h = s.header_line()
        assert "density=0.500" in h
        assert "length=0.800" in h
        assert "formality=0.200" in h
        assert "audience=0.500" in h
        assert "perspective=0.500" in h


# ─── apply_density ────────────────────────────────────────────────────


class TestApplyDensity:
    def test_full_density_returns_all_sorted(self) -> None:
        assert apply_density(["c", "a", "b"], 1.0) == ["a", "b", "c"]

    def test_zero_density_returns_empty(self) -> None:
        assert apply_density(["a", "b", "c"], 0.0) == []

    def test_half_density_takes_first_half(self) -> None:
        assert apply_density(["a", "b", "c", "d"], 0.5) == ["a", "b"]

    def test_empty_input_returns_empty(self) -> None:
        assert apply_density([], 0.5) == []

    def test_deterministic(self) -> None:
        r1 = apply_density(["b", "a", "c", "d", "e"], 0.6)
        r2 = apply_density(["a", "b", "c", "d", "e"], 0.6)
        assert r1 == r2

    def test_density_above_one_clamps(self) -> None:
        assert apply_density(["a", "b"], 2.0) == ["a", "b"]

    def test_density_below_zero_clamps(self) -> None:
        assert apply_density(["a", "b"], -0.5) == []

    def test_density_rounds_down(self) -> None:
        # 3 elements × 0.7 = 2.1 → floor to 2
        assert apply_density(["a", "b", "c"], 0.7) == ["a", "b"]


# ─── Integration: generate_controlled on AutoregressiveTomeGenerator ──


def _make_gen_with_axioms(triples: list[tuple[str, str, str]]) -> tuple[
    AutoregressiveTomeGenerator, int
]:
    algebra = GodelStateAlgebra()
    for s, p, o in triples:
        algebra.get_or_mint_prime(s, p, o)
    state = algebra.encode_chunk_state(triples)
    return AutoregressiveTomeGenerator(algebra), state


class TestGenerateControlled:
    def test_full_density_matches_all_axioms(self) -> None:
        gen, state = _make_gen_with_axioms([
            ("alice", "likes", "cat"),
            ("bob", "owns", "dog"),
            ("carol", "plays", "piano"),
        ])
        out = gen.generate_controlled(state, TomeSliders(density=1.0))
        assert "alice" in out.lower()
        assert "bob" in out.lower()
        assert "carol" in out.lower()

    def test_half_density_drops_later_axioms(self) -> None:
        gen, state = _make_gen_with_axioms([
            ("alice", "likes", "cat"),
            ("bob", "owns", "dog"),
            ("carol", "plays", "piano"),
            ("dave", "writes", "book"),
        ])
        out = gen.generate_controlled(state, TomeSliders(density=0.5))
        # Lexicographic first half of {alice||..., bob||..., carol||..., dave||...}
        # = {alice||..., bob||...}
        assert "alice" in out.lower()
        assert "bob" in out.lower()
        assert "carol" not in out.lower()
        assert "dave" not in out.lower()

    def test_zero_density_produces_empty_indicator(self) -> None:
        gen, state = _make_gen_with_axioms([("alice", "likes", "cat")])
        out = gen.generate_controlled(state, TomeSliders(density=0.0))
        assert "No axioms survive" in out
        assert "density=0.000" in out

    def test_default_sliders_equivalent_to_full_canonical(self) -> None:
        gen, state = _make_gen_with_axioms([
            ("alice", "likes", "cat"),
            ("bob", "owns", "dog"),
        ])
        out = gen.generate_controlled(state)
        assert "alice" in out.lower()
        assert "bob" in out.lower()

    def test_header_contains_slider_record(self) -> None:
        gen, state = _make_gen_with_axioms([("alice", "likes", "cat")])
        out = gen.generate_controlled(
            state,
            TomeSliders(density=1.0, length=0.9, formality=0.1),
        )
        assert "density=1.000" in out
        assert "length=0.900" in out
        assert "formality=0.100" in out

    def test_rejects_non_sliders_argument(self) -> None:
        gen, state = _make_gen_with_axioms([("alice", "likes", "cat")])
        with pytest.raises(TypeError):
            gen.generate_controlled(state, sliders="not-sliders")  # type: ignore[arg-type]
