"""RFC 8785 JSON Canonicalization Scheme (JCS) — pure-Python implementation.

JCS produces a deterministic UTF-8 byte sequence for a JSON value so that
cryptographic signatures remain stable across implementations. The output is
the byte string that downstream code hashes and signs — no framework, no
external canonicalizer, no C extension.

Scope. The Verifiable Credentials 2.0 + eddsa-jcs-2022 path used by SUM only
emits JSON values of these types:

    dict[str, ...]   object
    list             array
    tuple            array (JSON has no tuple; we coerce)
    str              string
    int / bool       number / boolean
    None             null

RFC 8785 additionally specifies rules for floats (IEEE 754 → ECMAScript
Number.prototype.toString). SUM never emits floats inside a VC credentialSubject
(we use integers and ISO-8601 datetime strings), so this implementation raises
``ValueError`` if a float is encountered rather than ship an untested
serialization path for a type we do not use.

The key-sort rule in RFC 8785 §3.2.3 is "code unit sequence of the UTF-16
representation". ``str.encode("utf-16-be")`` yields exactly that byte order
for sorting, which gives the correct answer for both BMP and supplementary
characters (Python's default string sort differs on supplementary code points).
"""
from __future__ import annotations

from typing import Any, Mapping

__all__ = ["canonicalize", "canonicalize_to_str"]


def canonicalize(obj: Any) -> bytes:
    """Return the RFC 8785 canonical UTF-8 bytes for ``obj``."""
    return _encode(obj).encode("utf-8")


def canonicalize_to_str(obj: Any) -> str:
    """Return the RFC 8785 canonical form as a ``str`` (not yet UTF-8 encoded).

    The JCS byte output is always UTF-8; the str form is equal to
    ``canonicalize(obj).decode('utf-8')``.
    """
    return _encode(obj)


def _encode(obj: Any) -> str:
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if obj is None:
        return "null"
    if isinstance(obj, str):
        return _encode_string(obj)
    if isinstance(obj, bool):  # pragma: no cover — unreachable; handled above
        return "true" if obj else "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        raise ValueError(
            "JCS: floating-point values are not supported by this implementation "
            "(use integer or ISO-8601 string representations instead)"
        )
    if isinstance(obj, Mapping):
        return _encode_object(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_encode(x) for x in obj) + "]"
    raise TypeError(f"JCS: unsupported type {type(obj).__name__}")


def _encode_object(obj: Mapping[Any, Any]) -> str:
    items = []
    for k in obj.keys():
        if not isinstance(k, str):
            raise TypeError(
                f"JCS: object keys must be str, got {type(k).__name__}"
            )
        items.append(k)
    # RFC 8785 §3.2.3: sort by UTF-16 code unit sequence.
    items.sort(key=lambda s: s.encode("utf-16-be"))
    encoded_pairs = [f"{_encode_string(k)}:{_encode(obj[k])}" for k in items]
    return "{" + ",".join(encoded_pairs) + "}"


def _encode_string(s: str) -> str:
    # RFC 8785 §3.2.2 / RFC 8259 §7. Short escapes for B/T/N/F/R; \uXXXX for
    # every other control character; only " and \ are backslash-escaped beyond
    # that. The solidus ('/') is NOT escaped. Hex digits are lowercase.
    out = ['"']
    for ch in s:
        cp = ord(ch)
        if cp == 0x22:
            out.append("\\\"")
        elif cp == 0x5C:
            out.append("\\\\")
        elif cp == 0x08:
            out.append("\\b")
        elif cp == 0x09:
            out.append("\\t")
        elif cp == 0x0A:
            out.append("\\n")
        elif cp == 0x0C:
            out.append("\\f")
        elif cp == 0x0D:
            out.append("\\r")
        elif cp < 0x20:
            out.append(f"\\u{cp:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)
