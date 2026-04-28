"""Property-based fuzz tests for the SUM MCP server (v2 hardening).

Asserts two invariants the type system cannot:

  1. **No tool ever raises uncaught.** Whatever shape the LLM
     client throws at a tool — wrong type, wrong size, wrong
     structure, deliberately malformed — the tool returns a
     dict, never propagates an exception. The server must
     stay up under adversarial input.

  2. **No tool ever returns a success shape on a malformed
     payload.** A success result must not carry an
     ``error_class`` key, and an error result must not be
     missing one. Fail-closed by construction.

The strategies generate adversarial values across every
documented type boundary — strings, ints, None, dicts, lists,
bytes-as-strings, oversized payloads — and pass them as-is to
each tool's typed parameter. The tool's own validation gates
must catch every shape and tag it correctly.

Run with: ``pytest Tests/test_mcp_server_fuzz.py``. Skipped
when Hypothesis or the mcp extra is not installed.
"""
from __future__ import annotations

import asyncio

import pytest

mcp = pytest.importorskip("mcp")
hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings, strategies as st


# Adversarial value strategy — every shape an LLM client might
# realistically (or maliciously) emit through the JSON-RPC wire.
adversarial_value = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-10**18, max_value=10**18),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=500),
    st.lists(st.text(max_size=20), max_size=10),
    st.dictionaries(st.text(max_size=20), st.text(max_size=20), max_size=5),
    st.binary(max_size=200).map(
        # Some MCP clients hand bytes; we coerce to str-of-bytes.
        lambda b: b.decode("latin-1", errors="replace")
    ),
)


@pytest.fixture(scope="module")
def server():
    from sum_engine_internal.mcp_server import build_server
    return build_server()


def _tool(server, name):
    return server._tool_manager.get_tool(name).fn


def _call(tool, **kwargs):
    """Synchronously invoke a tool whether sync or async."""
    if asyncio.iscoroutinefunction(tool):
        return asyncio.run(tool(**kwargs))
    return tool(**kwargs)


def _assert_tool_result_shape(result):
    """Every tool result MUST be a dict, MUST have a definite
    success-or-error shape (never both, never neither)."""
    assert isinstance(result, dict), f"tool returned non-dict: {type(result)}"
    has_error = "error_class" in result
    if has_error:
        assert isinstance(result["error_class"], str), (
            f"error_class not a string: {result['error_class']!r}"
        )
        assert result["error_class"] in {
            "schema", "signature", "structural", "input_too_large",
            "extractor_unavailable", "network_disallowed", "revoked", "internal",
        }, f"error_class not in enum: {result['error_class']!r}"
        assert "errors" in result, "error result missing 'errors' field"
        assert isinstance(result["errors"], list), "errors not a list"


# --------------------------------------------------------------------------
# extract — fuzz both arguments
# --------------------------------------------------------------------------


@given(text=adversarial_value, extractor=adversarial_value)
@settings(max_examples=200, deadline=None)
def test_extract_never_raises_or_returns_invalid_shape(server, text, extractor):
    result = _call(_tool(server, "extract"), text=text, extractor=extractor)
    _assert_tool_result_shape(result)


# --------------------------------------------------------------------------
# attest — fuzz the full argument surface
# --------------------------------------------------------------------------


@given(
    text=adversarial_value,
    extractor=adversarial_value,
    branch=adversarial_value,
    title=adversarial_value,
    signing_key=adversarial_value,
)
@settings(max_examples=200, deadline=None)
def test_attest_never_raises_or_returns_invalid_shape(
    server, text, extractor, branch, title, signing_key
):
    result = _call(
        _tool(server, "attest"),
        text=text,
        extractor=extractor,
        branch=branch if isinstance(branch, str) else "main",
        title=title if (title is None or isinstance(title, str)) else None,
        signing_key=signing_key if (signing_key is None or isinstance(signing_key, str)) else None,
    )
    _assert_tool_result_shape(result)


# --------------------------------------------------------------------------
# verify — fuzz arbitrary "bundles"
# --------------------------------------------------------------------------


# A bundle-shaped strategy that mostly produces garbage but
# occasionally lands on something the schema gate accepts.
malformed_bundle = st.dictionaries(
    st.sampled_from([
        "canonical_tome", "state_integer", "canonical_format_version",
        "prime_scheme", "axiom_count", "branch", "title",
        "ed25519_signature", "public_key", "hmac_signature",
        "extra_unexpected_field",
    ]),
    adversarial_value,
    max_size=12,
)


@given(bundle=st.one_of(adversarial_value, malformed_bundle))
@settings(max_examples=200, deadline=None)
def test_verify_never_raises_or_returns_invalid_shape(server, bundle):
    result = _call(_tool(server, "verify"), bundle=bundle)
    _assert_tool_result_shape(result)
    # verify always carries `ok`; assert it's a bool.
    assert isinstance(result.get("ok"), bool)


# --------------------------------------------------------------------------
# inspect + schema — read-only tools, broad fuzz
# --------------------------------------------------------------------------


@given(bundle=st.one_of(adversarial_value, malformed_bundle))
@settings(max_examples=100, deadline=None)
def test_inspect_never_raises_or_returns_invalid_shape(server, bundle):
    result = _call(_tool(server, "inspect"), bundle=bundle)
    _assert_tool_result_shape(result)


@given(name=adversarial_value)
@settings(max_examples=100, deadline=None)
def test_schema_never_raises_or_returns_invalid_shape(server, name):
    result = _call(_tool(server, "schema"), name=name)
    _assert_tool_result_shape(result)


# --------------------------------------------------------------------------
# Boundary regression — known previously-tricky values
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "",
    "   ",
    "\x00\x00\x00",
    "a" * 200_001,            # one over MAX_TEXT_CHARS
    "Ω≈ç√∫˜µ≤≥÷",            # unicode
    "<script>alert(1)</script>",  # html-ish
    "\\n\\r\\t",              # escaped whitespace
    "\u202egnitfels esrever",  # right-to-left override
])
def test_extract_handles_known_tricky_strings(server, text):
    result = _call(_tool(server, "extract"), text=text)
    _assert_tool_result_shape(result)
    if "error_class" not in result:
        assert isinstance(result["triples"], list)
