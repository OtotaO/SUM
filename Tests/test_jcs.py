"""Tests for sum_engine_internal.infrastructure.jcs — RFC 8785 JSON Canonicalization.

Covers:
- Key sort by UTF-16 code units (RFC 8785 §3.2.3)
- Primitive serialization (int, bool, null, str)
- Control-character escaping (§3.2.2)
- Array serialization
- Type-safety rejections (float, non-str keys, unsupported types)
- Idempotence (canonicalize ∘ canonicalize == canonicalize)
"""
from __future__ import annotations

import pytest

from sum_engine_internal.infrastructure.jcs import canonicalize, canonicalize_to_str


class TestPrimitives:
    def test_null(self) -> None:
        assert canonicalize_to_str(None) == "null"

    def test_true(self) -> None:
        assert canonicalize_to_str(True) == "true"

    def test_false(self) -> None:
        assert canonicalize_to_str(False) == "false"

    def test_positive_int(self) -> None:
        assert canonicalize_to_str(42) == "42"

    def test_negative_int(self) -> None:
        assert canonicalize_to_str(-7) == "-7"

    def test_zero(self) -> None:
        assert canonicalize_to_str(0) == "0"

    def test_large_int(self) -> None:
        assert canonicalize_to_str(2**63) == str(2**63)

    def test_float_integer_valued_normalized(self) -> None:
        # RFC 8785 §3.2.2.3: integer-valued floats normalize to integer
        # form. Cross-runtime contract with TS canonicalize@>=2.
        assert canonicalize_to_str(1.0) == "1"
        assert canonicalize_to_str(0.0) == "0"
        assert canonicalize_to_str(-0.0) == "0"
        assert canonicalize_to_str(42.0) == "42"

    def test_float_simple_decimals(self) -> None:
        # Slider-quantized values used by the render-receipt verifier.
        # Python repr matches ECMAScript Number.prototype.toString for
        # these; the positive-control fixture under fixtures/render_
        # receipts/ verifies cross-runtime byte equivalence empirically
        # via signature verification.
        assert canonicalize_to_str(0.1) == "0.1"
        assert canonicalize_to_str(0.3) == "0.3"
        assert canonicalize_to_str(0.5) == "0.5"
        assert canonicalize_to_str(0.7) == "0.7"
        assert canonicalize_to_str(0.9) == "0.9"
        assert canonicalize_to_str(1.5) == "1.5"

    def test_float_nan_and_inf_rejected(self) -> None:
        # NaN / ±Infinity are not representable in JSON; reject explicitly
        # rather than emit a non-canonical form.
        with pytest.raises(ValueError, match="NaN"):
            canonicalize(float("nan"))
        with pytest.raises(ValueError, match="Infinity"):
            canonicalize(float("inf"))
        with pytest.raises(ValueError, match="Infinity"):
            canonicalize(float("-inf"))


class TestStrings:
    def test_empty(self) -> None:
        assert canonicalize_to_str("") == '""'

    def test_ascii(self) -> None:
        assert canonicalize_to_str("hello") == '"hello"'

    def test_quote_escape(self) -> None:
        assert canonicalize_to_str('a"b') == '"a\\"b"'

    def test_backslash_escape(self) -> None:
        assert canonicalize_to_str("a\\b") == '"a\\\\b"'

    def test_solidus_not_escaped(self) -> None:
        # RFC 8259 §7 allows '/' to be escaped; RFC 8785 requires it NOT to be.
        assert canonicalize_to_str("a/b") == '"a/b"'

    def test_short_control_escapes(self) -> None:
        assert canonicalize_to_str("\b\t\n\f\r") == '"\\b\\t\\n\\f\\r"'

    def test_other_control_uses_hex(self) -> None:
        # \u0001 has no short escape; must be \u0001 (lowercase hex)
        assert canonicalize_to_str("\u0001") == '"\\u0001"'

    def test_unicode_passthrough(self) -> None:
        # BMP character outside control range: emit as UTF-8.
        s = "café"
        out = canonicalize(s).decode("utf-8")
        assert out == '"café"'


class TestObjects:
    def test_empty_object(self) -> None:
        assert canonicalize_to_str({}) == "{}"

    def test_single_pair(self) -> None:
        assert canonicalize_to_str({"a": 1}) == '{"a":1}'

    def test_keys_sorted(self) -> None:
        # Insertion order is {"b":1, "a":2}; JCS must emit a-first.
        obj = {"b": 1, "a": 2}
        assert canonicalize_to_str(obj) == '{"a":2,"b":1}'

    def test_nested(self) -> None:
        obj = {"outer": {"b": 1, "a": 2}}
        assert canonicalize_to_str(obj) == '{"outer":{"a":2,"b":1}}'

    def test_non_str_key_rejected(self) -> None:
        with pytest.raises(TypeError, match="object keys must be str"):
            canonicalize({1: "a"})

    def test_utf16_sort_for_supplementary(self) -> None:
        # Code points above 0xFFFF sort as their UTF-16 surrogate pair, NOT
        # as the raw code point. For a supplementary char (e.g. U+1F600,
        # encoded as 0xD83D 0xDE00), the sort key starts with 0xD83D which
        # is less than BMP code points in 0xE000-0xFFFF.
        bmp_char = "\uE000"          # comes first if sorting by code point
        supplementary = "\U0001F600"  # comes first if sorting by UTF-16
        obj = {bmp_char: 1, supplementary: 2}
        out = canonicalize_to_str(obj)
        # UTF-16 byte ordering: 0xD83D < 0xE000, so supplementary key is first
        assert out.startswith(f'{{"{supplementary}"')


class TestArrays:
    def test_empty_array(self) -> None:
        assert canonicalize_to_str([]) == "[]"

    def test_ints(self) -> None:
        assert canonicalize_to_str([1, 2, 3]) == "[1,2,3]"

    def test_mixed(self) -> None:
        assert canonicalize_to_str([1, "a", True, None]) == '[1,"a",true,null]'

    def test_tuple_coerced_to_array(self) -> None:
        assert canonicalize_to_str((1, 2)) == "[1,2]"

    def test_array_of_objects_preserves_element_order(self) -> None:
        # Arrays are positional — their element order is NOT sorted.
        obj = [{"b": 1}, {"a": 2}]
        assert canonicalize_to_str(obj) == '[{"b":1},{"a":2}]'


class TestIntegration:
    def test_idempotent(self) -> None:
        obj = {"z": [1, 2], "a": {"b": True, "a": None}}
        once = canonicalize(obj).decode("utf-8")
        # Parsing `once` back through json.loads and re-canonicalizing must
        # yield the same bytes (the whole point of canonicalization).
        import json

        twice = canonicalize(json.loads(once)).decode("utf-8")
        assert once == twice

    def test_unsupported_type_rejected(self) -> None:
        class X:
            pass

        with pytest.raises(TypeError, match="unsupported type"):
            canonicalize(X())

    def test_utf8_bytes_output(self) -> None:
        obj = {"msg": "café"}
        raw = canonicalize(obj)
        assert isinstance(raw, bytes)
        assert raw.decode("utf-8") == '{"msg":"café"}'
