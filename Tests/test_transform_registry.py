"""Registry + slider transform contract tests.

T1a coverage:
  1. Registry is populated with `slider` at import time.
  2. `get_transform(name)` and `list_transforms()` work.
  3. Slider canonical-path renders deterministically.
  4. Slider raises NotImplementedError on LLM-axis renders (until T1b).
  5. Slider's three `canonicalize_*` methods produce stable bytes.
  6. Slider parameters quantize to bin centres (sliders snap so two
     adjacent drag positions produce the same hash).
"""
from __future__ import annotations

import asyncio

import pytest

from sum_engine_internal.transforms import (
    Transform,
    TransformEnv,
    TransformResult,
    get_transform,
    list_transforms,
    register,
)
from sum_engine_internal.transforms.slider import (
    SLIDER_TRANSFORM,
    SliderTransform,
    _quantize,
    _requires_llm,
)


# ─── C1: registry shape ─────────────────────────────────────────────


def test_slider_is_registered():
    assert "slider" in list_transforms()
    assert get_transform("slider") is SLIDER_TRANSFORM


def test_slider_satisfies_transform_protocol():
    """The slider transform instance must satisfy the runtime-checkable
    Transform protocol."""
    assert isinstance(SLIDER_TRANSFORM, Transform)


def test_unknown_transform_raises_keyerror():
    with pytest.raises(KeyError, match="unknown transform"):
        get_transform("not-a-real-transform")


def test_double_register_with_same_instance_is_noop():
    """Idempotent on the exact same Transform instance. Re-registering
    the existing slider must NOT raise."""
    returned = register(SLIDER_TRANSFORM)
    assert returned is SLIDER_TRANSFORM


def test_double_register_with_different_instance_raises():
    """Two distinct objects sharing a name is a programming error;
    the registry refuses to silently overwrite."""
    impostor = SliderTransform()  # different instance, same name
    assert impostor is not SLIDER_TRANSFORM
    with pytest.raises(ValueError, match="already registered"):
        register(impostor)


# ─── C2: slider parameter quantization ──────────────────────────────


def test_slider_quantize_density_passes_through():
    """Density is the deterministic axis; not quantized."""
    q = _quantize({
        "density": 0.42, "length": 0.5, "formality": 0.5,
        "audience": 0.5, "perspective": 0.5,
    })
    assert q["density"] == 0.42


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0.0,  0.1),
        (0.05, 0.1),
        (0.18, 0.1),
        (0.20, 0.3),
        (0.30, 0.3),
        (0.49, 0.5),
        (0.50, 0.5),
        (0.69, 0.5),  # Per worker render.ts::snapToBin, idx = floor(0.69 * 5) = 3 → 0.7
        (0.70, 0.7),
        (0.90, 0.9),
        (1.00, 0.9),
    ],
)
def test_slider_quantize_llm_axis_to_bin_centres(raw, expected):
    """Length/formality/audience/perspective each snap to one of
    {0.1, 0.3, 0.5, 0.7, 0.9} — five bins, centres at (idx+0.5)/5."""
    q = _quantize({
        "density": 1.0, "length": raw, "formality": 0.5,
        "audience": 0.5, "perspective": 0.5,
    })
    # The test row's `expected` is approximate to the bin centre;
    # account for worker render.ts snap rule precisely.
    bins = 5
    idx = min(int(raw * bins), bins - 1)
    expected_exact = (idx + 0.5) / bins
    assert q["length"] == pytest.approx(expected_exact, abs=1e-9)


def test_slider_requires_llm_only_when_axis_off_centre():
    """LLM dispatch is required iff ANY of the four LLM axes is off
    its bin centre (0.5). Density doesn't trigger LLM dispatch."""
    centred = _quantize({"density": 1.0, "length": 0.5, "formality": 0.5,
                          "audience": 0.5, "perspective": 0.5})
    off = _quantize({"density": 1.0, "length": 0.9, "formality": 0.5,
                     "audience": 0.5, "perspective": 0.5})
    assert not _requires_llm(centred)
    assert _requires_llm(off)


# ─── C3: slider canonical-path apply ────────────────────────────────


def test_slider_canonical_path_produces_tome():
    """All LLM axes centred → deterministic prose composition;
    no LLM call required; provider is canonical-path."""
    result = asyncio.run(SLIDER_TRANSFORM.apply(
        input={"triples": [
            ("alice", "likes", "cats"),
            ("bob", "owns", "dog"),
        ]},
        parameters={
            "density": 1.0, "length": 0.5, "formality": 0.5,
            "audience": 0.5, "perspective": 0.5,
        },
        env=TransformEnv(),
    ))
    assert isinstance(result, TransformResult)
    assert result.provider == "canonical-path"
    assert result.model == "canonical-deterministic-v0"
    assert result.digital_source_type == "algorithmicMedia"
    assert result.llm_calls_made == 0
    assert isinstance(result.output, str)
    assert result.output  # non-empty tome
    # Both input triples should appear in the canonical-path tome
    assert "alice" in result.output and "cats" in result.output
    assert "bob" in result.output and "dog" in result.output


def test_slider_llm_axis_without_key_raises_value_error():
    """T1c-followup wired LLM-axis dispatch through the registry, but
    only when an OpenAI key is configured. Without one, the call
    raises a clear ValueError naming the operator-actionable fix —
    NOT a NotImplementedError (the legacy behaviour pre-T1c-followup).
    See Tests/test_transform_slider_llm_axis.py for the dispatch-
    succeeds-with-fake-LLM case."""
    with pytest.raises(ValueError, match="openai_api_key"):
        asyncio.run(SLIDER_TRANSFORM.apply(
            input={"triples": [("alice", "likes", "cats")]},
            parameters={
                "density": 1.0, "length": 0.9, "formality": 0.5,
                "audience": 0.5, "perspective": 0.5,
            },
            env=TransformEnv(),  # openai_api_key=None
        ))


# ─── C4: canonicalize_* byte stability ──────────────────────────────


def test_canonicalize_parameters_quantizes_before_hashing():
    """Two slider positions that quantize to the same bin produce
    byte-identical parameter canonicalisation. This is what lets
    the cache HIT on adjacent drag positions."""
    a = SLIDER_TRANSFORM.canonicalize_parameters({
        "density": 1.0, "length": 0.42, "formality": 0.5,
        "audience": 0.5, "perspective": 0.5,
    })
    b = SLIDER_TRANSFORM.canonicalize_parameters({
        "density": 1.0, "length": 0.48, "formality": 0.5,
        "audience": 0.5, "perspective": 0.5,
    })
    # 0.42 and 0.48 both quantize to bin 0.5
    assert a == b


def test_canonicalize_input_is_componentwise_sorted():
    """Input canonicalisation MUST sort triples component-wise so
    two callers with different insertion order produce the same
    input_hash."""
    a = SLIDER_TRANSFORM.canonicalize_input({"triples": [
        ("alice", "likes", "cats"),
        ("bob", "owns", "dog"),
    ]})
    b = SLIDER_TRANSFORM.canonicalize_input({"triples": [
        ("bob", "owns", "dog"),
        ("alice", "likes", "cats"),
    ]})
    assert a == b


def test_canonicalize_output_is_utf8_tome_bytes():
    """Output is a tome string; canonical bytes = UTF-8 encoding.
    Round-trip MUST recover the original string."""
    tome = "Alice likes cats. Bob owns a dog."
    out_bytes = SLIDER_TRANSFORM.canonicalize_output(tome)
    assert out_bytes.decode("utf-8") == tome


def test_canonicalize_input_rejects_bad_shape():
    """Slider input must be a dict with 'triples'; anything else
    raises ValueError at canonicalisation time (fails closed before
    any hashing happens)."""
    with pytest.raises(ValueError, match="triples"):
        SLIDER_TRANSFORM.canonicalize_input("just a string")
    with pytest.raises(ValueError, match="triples"):
        SLIDER_TRANSFORM.canonicalize_input({"axioms": []})


def test_canonicalize_parameters_rejects_missing_axis():
    """All five axes (density + 4 LLM) MUST be present; partial
    parameters fail fast."""
    with pytest.raises(ValueError, match="density"):
        SLIDER_TRANSFORM.canonicalize_parameters({})
    with pytest.raises(ValueError, match="length"):
        SLIDER_TRANSFORM.canonicalize_parameters({"density": 1.0})


def test_canonicalize_parameters_rejects_out_of_range():
    """[0, 1] is the valid range; violations fail fast."""
    with pytest.raises(ValueError, match=r"out of \[0, 1\]"):
        SLIDER_TRANSFORM.canonicalize_parameters({
            "density": 1.5, "length": 0.5, "formality": 0.5,
            "audience": 0.5, "perspective": 0.5,
        })
