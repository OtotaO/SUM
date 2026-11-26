"""Tests for TomeSliders + controlled rendering."""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from sum_engine_internal.ensemble.slider_renderer import _axiom_key
from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
from sum_engine_internal.ensemble.tome_sliders import (
    FACT_PRESERVATION_REINFORCEMENT,
    SLIDER_BINS_PER_AXIS,
    TomeSliders,
    apply_density,
    build_system_prompt,
)


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


# ─── Cross-runtime pin: Python is the reference for the Worker twin ───
#
# worker/src/render/axis_prompts.ts::applyDensity must keep the same
# subset, in the same order, as apply_density does here. It sorted with
# ICU `localeCompare` instead of by codepoint, so on these keys the two
# runtimes kept DIFFERENT triples at the same density, and the surviving
# subset is what the deterministic tome (hence the signed tome_hash) is
# built from. These tests pin the kept key subset only, not tome parity
# with slider_renderer.render, whose tome keeps the input order and every
# copy of a kept key (see worker/src/render/axis_prompts.ts).
#
# This fixture is the shared source of truth. The Node side asserts
# against the same JSON in worker/test/density_smoke.mjs (`npm run
# test:density`, wired into quantum-ci.yml). These tests hold Python
# still: if they ever go red, the fixture must be regenerated with
# fixtures/density_sort/generate_fixture.py AND the Worker rechecked.

_FIXTURE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "fixtures"
    / "density_sort"
    / "apply_density_cross_runtime_v1.json"
)


def _load_fixture() -> dict:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


class TestDensityCrossRuntimeFixture:
    def test_fixture_keys_are_built_by_the_production_key_function(self) -> None:
        fx = _load_fixture()
        # The Worker's keyOf is `${s}||${p}||${o}`; the fixture's keys must
        # be exactly what the production Python key builder emits, or the
        # two runtimes are agreeing on the wrong string.
        all_keys = {k for case in fx["cases"] for k in case["expected_kept_keys"]}
        built = {_axiom_key((t[0], t[1], t[2])) for t in fx["triples"]}
        assert all_keys <= built
        assert built == {"a||p||o", "ab||p||o", "a b||p||o", "A||p||o"}

    def test_fixture_pins_a_case_where_icu_and_codepoint_disagree(self) -> None:
        # Guard the guard: if the triple set ever loses the property that
        # ICU collation and codepoint order disagree, the Node test stops
        # being able to catch a localeCompare regression.
        fx = _load_fixture()
        keys = [_axiom_key((t[0], t[1], t[2])) for t in fx["triples"]]
        icu_order = ["a b||p||o", "a||p||o", "A||p||o", "ab||p||o"]
        assert sorted(icu_order) != icu_order
        assert sorted(keys) != icu_order

    def test_apply_density_matches_every_fixture_case(self) -> None:
        fx = _load_fixture()
        keys = [_axiom_key((t[0], t[1], t[2])) for t in fx["triples"]]
        assert fx["cases"], "fixture carries no density cases"
        for case in fx["cases"]:
            assert apply_density(keys, case["density"]) == case["expected_kept_keys"], (
                f"density={case['density']}"
            )


# ─── FACT_PRESERVATION_REINFORCEMENT (v0.7 prompt hardening) ──────────
#
# build_system_prompt appends the clause when any of the four prompt axes is
# at or below 0.3 + 1e-6 (the comparison the code makes), whatever density is.

_PROMPT_AXES = ("length", "formality", "audience", "perspective")


class TestFactPreservationReinforcement:
    @pytest.mark.parametrize("axis", _PROMPT_AXES)
    @pytest.mark.parametrize(
        ("value", "present"),
        [
            (0.0, True),
            (0.3, True),
            # Above 0.3, but inside the 1e-6 tolerance of the comparison.
            (0.30000001, True),
            (0.35, False),
            (1.0, False),
        ],
    )
    def test_reinforcement_follows_one_axis(self, axis: str, value: float, present: bool) -> None:
        prompt = build_system_prompt(TomeSliders(**{axis: value}))
        assert prompt.count(FACT_PRESERVATION_REINFORCEMENT) == int(present)

    def test_reinforcement_absent_with_every_axis_at_half(self) -> None:
        sliders = TomeSliders(length=0.5, formality=0.5, audience=0.5, perspective=0.5)
        assert FACT_PRESERVATION_REINFORCEMENT not in build_system_prompt(sliders)

    def test_reinforcement_ignores_density(self) -> None:
        assert FACT_PRESERVATION_REINFORCEMENT not in build_system_prompt(TomeSliders(density=0.0))

    def test_reinforcement_appended_once_after_the_fragments(self) -> None:
        prompt = build_system_prompt(
            TomeSliders(length=0.0, formality=0.0, audience=0.0, perspective=0.0)
        )
        assert prompt.count(FACT_PRESERVATION_REINFORCEMENT) == 1
        assert prompt.endswith(" " + FACT_PRESERVATION_REINFORCEMENT)


# ─── Cross-runtime pin: the Worker's system prompt matches Python's ───
#
# worker/test/fixtures/system_prompt_digests.json holds a digest of
# build_system_prompt for every quantized slider combination, and the Node
# side checks the Worker's buildSystemPrompt against it in
# worker/test/prompt_smoke.mjs (`npm run test:prompts`, wired into
# quantum-ci.yml). This test holds the fixture to Python: if it goes red,
# regenerate with worker/test/generate_prompt_fixture.py AND rerun the Node
# check.

_PROMPT_FIXTURE_GENERATOR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "worker"
    / "test"
    / "generate_prompt_fixture.py"
)


def _prompt_fixture_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_prompt_fixture", _PROMPT_FIXTURE_GENERATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSystemPromptCrossRuntimeFixture:
    def test_committed_fixture_matches_python(self) -> None:
        gen = _prompt_fixture_generator()
        assert gen.serialize(gen.build()) == gen.OUT_PATH.read_text(encoding="utf-8"), (
            "regenerate with python worker/test/generate_prompt_fixture.py, "
            "then run npm run test:prompts in worker/"
        )

    def test_fixture_covers_every_quantized_combination(self) -> None:
        doc = _prompt_fixture_generator().build()
        assert doc["bin_centres"] == [0.1, 0.3, 0.5, 0.7, 0.9]
        assert len(doc["rows"]) == SLIDER_BINS_PER_AXIS ** len(doc["axes"]) == 625
        # Every combination gives a different prompt, so a wrong row cannot hide.
        assert len({row[-1] for row in doc["rows"]}) == 625


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
