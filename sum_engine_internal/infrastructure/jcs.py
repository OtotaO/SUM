"""RFC 8785 JSON Canonicalization Scheme (JCS) — pure-Python implementation.

JCS produces a deterministic UTF-8 byte sequence for a JSON value so that
cryptographic signatures remain stable across implementations. The output is
the byte string that downstream code hashes and signs — no framework, no
external canonicalizer, no C extension.

Scope. SUM consumes this module on two surfaces:

  1. Verifiable Credentials 2.0 + eddsa-jcs-2022 (Phase E core, integer-only
     JSON values + ISO-8601 datetime strings — no floats).
  2. Render-receipt verification (Phase E.1 v0.9.C; the receipt payload's
     ``sliders_quantized`` field carries float values like ``0.5`` and
     ``1.0``, which JCS must canonicalize byte-identically to the
     TS-side ``canonicalize@>=2`` library that signed the receipt).

Supported types:

    dict[str, ...]   object
    list             array
    tuple            array (JSON has no tuple; we coerce)
    str              string
    int / bool       number / boolean
    float            number per RFC 8785 §3.2.2.3 (ECMAScript
                     Number.prototype.toString); see _encode_float.
    None             null

The float branch is the most cross-runtime-sensitive part of this
module. Cross-runtime byte-equivalence with the TS-side ``canonicalize@>=2``
library on every value the render receipt uses (currently slider bin
centres ``{0.1, 0.3, 0.5, 0.7, 0.9}`` plus ``1.0``) is the load-bearing
property — divergence here means a v0.9.C verifier rejects a correctly-
signed receipt because its reconstructed canonical bytes differ.
Empirically verified by the positive-control fixture under
``fixtures/render_receipts/``: signature verification only succeeds if
the canonical bytes match what the TS canonicalizer produced at signing
time. NaN / ±Infinity raise ``ValueError`` (not representable in JSON).

The key-sort rule in RFC 8785 §3.2.3 is "code unit sequence of the UTF-16
representation". ``str.encode("utf-16-be")`` yields exactly that byte order
for sorting, which gives the correct answer for both BMP and supplementary
characters (Python's default string sort differs on supplementary code points).
"""
from __future__ import annotations

import math
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
        return _encode_float(obj)
    if isinstance(obj, Mapping):
        return _encode_object(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_encode(x) for x in obj) + "]"
    raise TypeError(f"JCS: unsupported type {type(obj).__name__}")


def _encode_float(f: float) -> str:
    """RFC 8785 §3.2.2.3 float canonicalization (subset SUM uses).

    Cross-runtime contract: the string this returns MUST equal what
    ``canonicalize@>=2`` (Erdtman JS) and ECMAScript's
    ``Number.prototype.toString`` produce for the same float. The
    render-receipt verifier path depends on this being byte-stable
    across Python ↔ JS for every value that appears inside
    ``sliders_quantized``. Empirically verified by the positive-
    control fixture under ``fixtures/render_receipts/``: a
    correctly-signed receipt only verifies when canonical bytes
    match.
    """
    if math.isnan(f):
        raise ValueError("JCS: NaN is not representable in JSON")
    if math.isinf(f):
        raise ValueError("JCS: ±Infinity is not representable in JSON")
    # Integer-valued floats (incl. -0.0) normalize to integer per
    # Number.prototype.toString: 1.0 → "1", -0.0 → "0".
    if f == 0.0:
        return "0"
    if f == int(f) and -1e21 < f < 1e21:
        return str(int(f))
    # Non-integer floats: Python's repr matches ECMAScript's
    # Number.prototype.toString for simple decimals (0.5, 0.7, etc.).
    # Edge cases at exponential-notation boundaries (1e-7, 1e21+) are
    # out of scope for SUM's actual usage; if a future path needs
    # those, extend this branch with explicit ECMAScript
    # ToString-shortest-representation logic.
    return repr(f)


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
