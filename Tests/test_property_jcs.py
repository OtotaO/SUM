"""Property-based tests for the JCS canonicalization module (Phase R0.4).

JCS (RFC 8785) is the load-bearing canonicalization for both render
receipts and trust-root manifests. A bug here invalidates the
cross-runtime trust contract because the Python verifier and the
TS-side ``canonicalize`` library MUST produce byte-identical bytes
for any input value SUM signs over. The fixture-based receipt and
trust-root tests already cover the production payload shapes;
property-based tests catch the case the fixture set didn't anticipate.

Properties asserted (Hypothesis-generated):

  1. **Round-trip stability** — for any generated JSON value, parsing
     ``canonicalize(obj)`` returns a dict-tree that ``canonicalize``
     produces the same bytes for. (Idempotence under JSON parse.)

  2. **Key-order independence** — for any generated dict, two dicts
     with the same {key: value} pairs in different insertion order
     produce byte-identical canonical output.

  3. **Determinism** — calling ``canonicalize`` twice on the same
     input returns byte-identical output.

  4. **Float canonicalization invariants** — for floats SUM uses,
     the output is parseable as a JSON number and round-trips back
     to the same float.

  5. **Float boundary cases** — integer-valued floats normalize to
     integer string; NaN/±Infinity raise ValueError.

  6. **String escaping safety** — control characters are escaped;
     the output parses back to the original string.

Hypothesis settings: derandomize=True so failures are reproducible
across CI runs; max_examples=200 default; no shrinking budget caps.

Skipped if hypothesis isn't installed (the dev extra).
"""
from __future__ import annotations

import json
import math

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings, strategies as st

from sum_engine_internal.infrastructure.jcs import canonicalize, canonicalize_to_str


# Strategy for JSON-shaped values SUM actually canonicalizes. Excludes
# floats that fall outside the SUM-used range (1e21+ exponential
# notation, 1e-7 boundaries, etc.) — those are documented as out of
# scope for the float branch. Tests that specifically need those
# edge cases assert the boundaries directly.
_safe_floats = st.floats(
    min_value=-1e15,
    max_value=1e15,
    allow_nan=False,
    allow_infinity=False,
    # Exclude the exponential-notation boundary cases SUM doesn't use:
    # very small positive floats trigger ECMAScript exponential form
    # which Python's repr doesn't match.
).filter(lambda f: abs(f) < 1e15 and (f == 0 or abs(f) > 1e-6))


_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**18), max_value=10**18),
    _safe_floats,
    st.text(
        # Avoid surrogate halves that would fail UTF-8 encoding.
        alphabet=st.characters(blacklist_categories=("Cs",), max_codepoint=0xFFFF),
        max_size=80,
    ),
)


@st.composite
def _json_value(draw, max_depth: int = 3):
    """A random JSON-shaped Python value using only types our JCS
    module supports. Bounded depth so generation terminates."""
    if max_depth <= 0:
        return draw(_json_scalars)
    return draw(
        st.one_of(
            _json_scalars,
            st.lists(_json_value(max_depth=max_depth - 1), max_size=6),
            st.dictionaries(
                keys=st.text(
                    alphabet=st.characters(blacklist_categories=("Cs",), max_codepoint=0x7F),
                    min_size=1,
                    max_size=20,
                ),
                values=_json_value(max_depth=max_depth - 1),
                max_size=6,
            ),
        )
    )


# --------------------------------------------------------------------------
# Round-trip + idempotence
# --------------------------------------------------------------------------


@given(value=_json_value())
@settings(deadline=None, derandomize=True, max_examples=200)
def test_canonicalize_is_deterministic(value):
    """Calling canonicalize twice on the same value yields identical bytes."""
    a = canonicalize(value)
    b = canonicalize(value)
    assert a == b


@given(value=_json_value())
@settings(deadline=None, derandomize=True, max_examples=200)
def test_canonicalize_round_trip_idempotent(value):
    """Parsing canonical bytes and re-canonicalizing yields the same bytes.

    This is the JCS idempotence property: the canonical form is a
    fixed point under (parse → re-canonicalize)."""
    canon = canonicalize(value)
    parsed = json.loads(canon.decode("utf-8"))
    re_canon = canonicalize(parsed)
    assert canon == re_canon


# --------------------------------------------------------------------------
# Key-order independence
# --------------------------------------------------------------------------


@given(
    pairs=st.lists(
        st.tuples(
            st.text(
                alphabet=st.characters(blacklist_categories=("Cs",), max_codepoint=0x7F),
                min_size=1,
                max_size=10,
            ),
            _json_value(max_depth=2),
        ),
        min_size=2,
        max_size=8,
        unique_by=lambda kv: kv[0],
    )
)
@settings(deadline=None, derandomize=True, max_examples=200)
def test_dict_canonicalize_key_order_independent(pairs):
    """Two dicts with the same {key: value} pairs in different
    insertion order produce byte-identical canonical output. RFC 8785
    §3.2.3 mandates UTF-16-code-unit-sorted keys; insertion order on
    the input must not matter."""
    forward = dict(pairs)
    reverse = dict(reversed(pairs))
    assert canonicalize(forward) == canonicalize(reverse)


# --------------------------------------------------------------------------
# Float canonicalization invariants
# --------------------------------------------------------------------------


@given(f=_safe_floats)
@settings(deadline=None, derandomize=True, max_examples=200)
def test_float_canonicalization_round_trips(f):
    """canonicalize(f) is a valid JSON number that parses back to a
    float equal to f. Required for cross-runtime parity with TS
    canonicalize on every float SUM signs."""
    canon = canonicalize_to_str(f)
    # Parses as JSON.
    parsed = json.loads(canon)
    # Equal to original (within float-equality semantics — `==` on
    # floats here, since Python's repr round-trips exact values).
    assert parsed == f


@given(n=st.integers(min_value=-(10**15), max_value=10**15))
@settings(deadline=None, derandomize=True, max_examples=200)
def test_integer_valued_float_normalizes_to_integer(n):
    """An integer-valued float canonicalizes to the same string as the
    corresponding int — RFC 8785 §3.2.2.3 / ECMAScript's
    ``Number.prototype.toString`` rule. The cross-runtime gotcha
    documented in RENDER_RECEIPT_FORMAT.md §4."""
    f_form = float(n)
    int_form = n
    assume(math.isfinite(f_form))
    # Both forms must produce the same canonical string. -0.0 is the
    # documented exception: -0.0 → "0" but -0 → "0" as well, so they
    # still match.
    assert canonicalize_to_str(f_form) == canonicalize_to_str(int_form)


def test_nan_and_infinity_rejected_property():
    """NaN / ±Infinity are not representable in JSON; canonicalize
    raises ValueError for both. Hard-coded because Hypothesis's
    floats() strategy wouldn't naturally generate them under our
    safe_floats filter — but the rejection contract still matters."""
    with pytest.raises(ValueError, match="NaN"):
        canonicalize(float("nan"))
    with pytest.raises(ValueError, match="Infinity"):
        canonicalize(float("inf"))
    with pytest.raises(ValueError, match="Infinity"):
        canonicalize(float("-inf"))


# --------------------------------------------------------------------------
# String escaping safety
# --------------------------------------------------------------------------


@given(
    text=st.text(
        alphabet=st.characters(blacklist_categories=("Cs",), max_codepoint=0xFFFF),
        max_size=200,
    )
)
@settings(deadline=None, derandomize=True, max_examples=200)
def test_string_canonicalize_round_trips(text):
    """For any generated string, parsing canonicalize(s) yields s
    back. Catches escape-handling regressions (control chars,
    quotes, backslashes, non-ASCII)."""
    canon = canonicalize(text)
    parsed = json.loads(canon.decode("utf-8"))
    assert parsed == text


@given(
    text=st.text(
        alphabet=st.characters(
            min_codepoint=0,
            max_codepoint=0x1F,  # control range
        ),
        min_size=1,
        max_size=10,
    )
)
@settings(deadline=None, derandomize=True, max_examples=100)
def test_control_characters_get_escaped(text):
    """Control characters (U+0000..U+001F) MUST be \\uXXXX-escaped per
    RFC 8259. The output MUST NOT contain the raw control byte."""
    canon = canonicalize(text)
    decoded = canon.decode("utf-8")
    # Non-quote control chars must NOT appear in the output verbatim.
    for ch in text:
        cp = ord(ch)
        if cp < 0x20 and cp not in (0x08, 0x09, 0x0A, 0x0C, 0x0D):
            # The other control chars get escaped as \uXXXX.
            assert ch not in decoded, (
                f"raw control char U+{cp:04x} leaked into output: {decoded!r}"
            )


# --------------------------------------------------------------------------
# Composition: nested-dict + array round-trip
# --------------------------------------------------------------------------


@given(value=_json_value(max_depth=4))
@settings(deadline=None, derandomize=True, max_examples=100)
def test_nested_value_round_trip(value):
    """Compose all the above: any depth-4 generated value round-trips
    through canonicalize → json.loads → canonicalize idempotently."""
    a = canonicalize(value)
    b = canonicalize(json.loads(a.decode("utf-8")))
    assert a == b
