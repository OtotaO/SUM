"""Tests for the SUM MCP server (v2 hardening).

Four layers of coverage:

  1. **Tool registration** — server boots, all five tools
     register, descriptions are non-empty.
  2. **Single-tool behaviour** — every input-validation gate,
     every error class, every success path covered explicitly.
  3. **Roundtrip** — extract → attest → verify is the primary
     integration path.
  4. **Cross-runtime byte-identity** — an MCP-attested bundle
     verifies via the *CLI verifier surface* unchanged. Locks
     the contract that the MCP path produces the same canonical
     bytes the CLI does, so the existing Python ↔ Node ↔ browser
     trust triangle extends to MCP-attested bundles.

The hardening fuzz tests live in
``Tests/test_mcp_server_fuzz.py`` (Hypothesis-based; require
the dev extra). They assert two invariants the type system
cannot: no tool ever raises uncaught, no tool ever returns a
success shape on a malformed payload.
"""
from __future__ import annotations

import pytest

# Skip the whole file if the optional `mcp` extra is not installed.
mcp = pytest.importorskip("mcp")

# Skip if spaCy / sieve extractor is not available.
pytest.importorskip("spacy")


@pytest.fixture(scope="module")
def server():
    from sum_engine_internal.mcp_server import build_server
    return build_server()


def _tool(server, name):
    """Reach the registered Python callable behind a FastMCP tool."""
    return server._tool_manager.get_tool(name).fn


# --------------------------------------------------------------------------
# Tool registration
# --------------------------------------------------------------------------


def test_server_boots_with_expected_tools(server):
    import asyncio
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {"extract", "attest", "verify", "inspect", "render", "schema"}


def test_every_tool_has_a_non_empty_description(server):
    import asyncio
    tools = asyncio.run(server.list_tools())
    for t in tools:
        assert t.description and len(t.description) > 20, (
            f"tool {t.name!r} description too short — MCP clients show "
            f"this to humans deciding whether to invoke"
        )


def test_error_class_enum_is_stable():
    """Pin the error-class enum. Adding new classes is fine; renaming
    or removing existing ones is a breaking change for callers
    pattern-matching on the string."""
    from sum_engine_internal.mcp_server.errors import ErrorClass
    expected = {
        "schema", "signature", "structural", "input_too_large",
        "extractor_unavailable", "network_disallowed", "revoked", "internal",
    }
    assert {c.value for c in ErrorClass} == expected


# --------------------------------------------------------------------------
# extract — input validation gates
# --------------------------------------------------------------------------


def test_extract_rejects_non_string_text(server):
    import asyncio
    result = asyncio.run(_tool(server, "extract")(text=42))
    assert result["error_class"] == "schema"


def test_extract_rejects_empty_text(server):
    import asyncio
    result = asyncio.run(_tool(server, "extract")(text="   "))
    assert result["error_class"] == "schema"


def test_extract_rejects_oversized_text(server):
    import asyncio
    from sum_engine_internal.mcp_server.server import MAX_TEXT_CHARS
    huge = "a" * (MAX_TEXT_CHARS + 1)
    result = asyncio.run(_tool(server, "extract")(text=huge))
    assert result["error_class"] == "input_too_large"
    assert str(MAX_TEXT_CHARS) in result["errors"][0]


def test_extract_rejects_unknown_extractor(server):
    import asyncio
    result = asyncio.run(_tool(server, "extract")(
        text="The press was invented.", extractor="bogus"
    ))
    assert result["error_class"] == "schema"


def test_extract_rejects_llm_without_network_optin(server, monkeypatch):
    """LLM extractor fails closed unless SUM_MCP_ALLOW_NETWORK=1
    was set when the server started. The MCP path deliberately
    does NOT fall through to the LLM auto-detect that the CLI
    uses, because a prompt-injected LLM client should not be
    able to spend the user's API tokens."""
    import asyncio
    from sum_engine_internal.mcp_server import server as srv_mod
    monkeypatch.setattr(srv_mod, "NETWORK_ALLOWED", False)
    result = asyncio.run(_tool(server, "extract")(
        text="The press was invented.", extractor="llm"
    ))
    assert result["error_class"] == "network_disallowed"


def test_extract_returns_triples_on_valid_input(server):
    import asyncio
    result = asyncio.run(_tool(server, "extract")(
        text="The printing press was invented by Johannes Gutenberg."
    ))
    if "error_class" in result:
        pytest.skip(f"extractor unavailable: {result['errors']}")
    assert result["count"] >= 1
    assert all(len(t) == 3 for t in result["triples"])
    assert result["extractor"] == "sieve"  # auto resolves to sieve in MCP


# --------------------------------------------------------------------------
# verify — input validation gates
# --------------------------------------------------------------------------


def test_verify_rejects_non_dict_bundle(server):
    result = _tool(server, "verify")(bundle="not a dict")
    assert result["error_class"] == "schema"
    assert result["ok"] is False


def test_verify_rejects_missing_canonical_tome(server):
    result = _tool(server, "verify")(bundle={
        "state_integer": "1", "canonical_format_version": "1.0.0"
    })
    assert result["error_class"] == "schema"
    assert "canonical_tome" in result["errors"][0]


def test_verify_rejects_unsupported_canonical_format(server):
    result = _tool(server, "verify")(bundle={
        "canonical_tome": "",
        "state_integer": "1",
        "canonical_format_version": "0.0.0-future",
    })
    assert result["error_class"] == "schema"


def test_verify_accepts_future_minor_version_under_1_x():
    """Forward-compat policy: 1.x versions accepted (additive
    fields tolerated); future major versions rejected."""
    from sum_engine_internal.mcp_server import build_server
    server = build_server()
    # 1.5.0 with the same scheme + matching state should pass
    # the schema gate (it'll fail later on state mismatch since
    # we put a fake state, which is fine — the test is only that
    # the schema gate accepts).
    result = _tool(server, "verify")(bundle={
        "canonical_tome": "",
        "state_integer": "1",
        "canonical_format_version": "1.5.0",
    })
    # No reject on the version itself; some downstream check fails
    # (axiom count is 0, claimed state is 1, reconstructed is 1, so
    # it actually passes!) — the point is no SCHEMA error from the version.
    assert result.get("error_class") != "schema" or "version" not in result["errors"][0]


def test_verify_rejects_oversized_tome(server):
    from sum_engine_internal.mcp_server.server import MAX_TOME_CHARS
    result = _tool(server, "verify")(bundle={
        "canonical_tome": "a" * (MAX_TOME_CHARS + 1),
        "state_integer": "1",
        "canonical_format_version": "1.0.0",
    })
    assert result["error_class"] == "input_too_large"


def test_verify_rejects_oversized_state_integer(server):
    from sum_engine_internal.mcp_server.server import MAX_STATE_INTEGER_DIGITS
    result = _tool(server, "verify")(bundle={
        "canonical_tome": "",
        "state_integer": "9" * (MAX_STATE_INTEGER_DIGITS + 1),
        "canonical_format_version": "1.0.0",
    })
    assert result["error_class"] == "input_too_large"


def test_verify_rejects_non_string_state_integer(server):
    result = _tool(server, "verify")(bundle={
        "canonical_tome": "",
        "state_integer": 12345,
        "canonical_format_version": "1.0.0",
    })
    assert result["error_class"] == "input_too_large"  # not-a-string trips the size guard's isinstance check


def test_verify_rejects_unsupported_prime_scheme(server):
    result = _tool(server, "verify")(bundle={
        "canonical_tome": "",
        "state_integer": "1",
        "canonical_format_version": "1.0.0",
        "prime_scheme": "sha256_128_v2",
    })
    assert result["error_class"] == "schema"


# --------------------------------------------------------------------------
# schema + inspect
# --------------------------------------------------------------------------


def test_schema_list_includes_known_schemas(server):
    result = _tool(server, "schema")(name="list")
    assert "sum.canonical_bundle.v1" in result["schemas"]
    assert "sum.render_receipt.v1" in result["schemas"]
    assert "sum.merkle_inclusion.v1" in result["schemas"]


def test_schema_unknown_returns_error_with_catalogue(server):
    result = _tool(server, "schema")(name="sum.does_not_exist.v9")
    assert result["error_class"] == "schema"
    assert "sum.canonical_bundle.v1" in result["known"]


def test_schema_named_returns_field_catalogue(server):
    result = _tool(server, "schema")(name="sum.canonical_bundle.v1")
    assert "error_class" not in result
    assert result["schema"] == "sum.canonical_bundle.v1"
    assert "fields" in result


def test_schema_rejects_non_string_name(server):
    result = _tool(server, "schema")(name=42)
    assert result["error_class"] == "schema"


def test_inspect_rejects_non_dict(server):
    result = _tool(server, "inspect")(bundle="oops")
    assert result["error_class"] == "schema"


def test_inspect_handles_empty_bundle(server):
    result = _tool(server, "inspect")(bundle={})
    assert result.get("axiom_count") is None
    assert result["signatures_present"] == {"ed25519": False, "hmac": False}


# --------------------------------------------------------------------------
# Roundtrip — extract → attest → verify
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def attested_bundle(server):
    import asyncio
    result = asyncio.run(_tool(server, "attest")(
        text="The printing press was invented by Johannes Gutenberg."
    ))
    if "error_class" in result:
        pytest.skip(f"attest unavailable: {result['errors']}")
    return result


def test_attest_produces_well_formed_bundle(attested_bundle):
    bundle = attested_bundle["bundle"]
    assert "canonical_tome" in bundle
    assert "state_integer" in bundle
    assert bundle["canonical_format_version"] == "1.0.0"
    assert bundle["sum_cli"]["produced_by"] == "mcp_server.v2"
    assert attested_bundle["axioms"] >= 1
    assert attested_bundle["source_uri"].startswith("sha256:")


def test_verify_accepts_self_attested_bundle(server, attested_bundle):
    bundle = attested_bundle["bundle"]
    result = _tool(server, "verify")(bundle=bundle)
    assert result["ok"] is True, f"errors: {result.get('errors')}"
    assert "error_class" not in result
    assert result["axioms"] == attested_bundle["axioms"]


def test_verify_rejects_tampered_canonical_tome(server, attested_bundle):
    import copy
    bundle = copy.deepcopy(attested_bundle["bundle"])
    bundle["canonical_tome"] = (
        bundle["canonical_tome"].rstrip("\n") + "\nThe earth orbits sun.\n"
    )
    result = _tool(server, "verify")(bundle=bundle)
    assert result["ok"] is False
    assert result["error_class"] == "structural"


def test_verify_rejects_tampered_state_integer(server, attested_bundle):
    import copy
    bundle = copy.deepcopy(attested_bundle["bundle"])
    bundle["state_integer"] = "999"
    result = _tool(server, "verify")(bundle=bundle)
    assert result["ok"] is False
    assert result["error_class"] == "structural"


def test_inspect_attested_bundle_reflects_attest_metadata(server, attested_bundle):
    bundle = attested_bundle["bundle"]
    summary = _tool(server, "inspect")(bundle=bundle)
    assert summary["axiom_count"] == attested_bundle["axioms"]
    assert summary["sum_cli"]["produced_by"] == "mcp_server.v2"


# --------------------------------------------------------------------------
# Cross-runtime byte-identity — MCP-attested bundle accepted by CLI
# --------------------------------------------------------------------------


def test_mcp_attested_bundle_verifies_via_cli_surface(attested_bundle, tmp_path):
    """The MCP server's attest path must produce bytes that the
    CLI's verify path accepts unchanged. This is what
    guarantees the cross-runtime triangle (Node, browser)
    extends to MCP-attested bundles."""
    import json
    import subprocess
    import sys

    bundle_path = tmp_path / "mcp_bundle.json"
    bundle_path.write_text(
        json.dumps(attested_bundle["bundle"]), encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, "-m", "sum_cli.main", "verify", "--input", str(bundle_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"CLI rejected an MCP-attested bundle:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    cli_payload = json.loads(result.stdout)
    assert cli_payload["ok"] is True
    assert cli_payload["axioms"] == attested_bundle["axioms"]


# --------------------------------------------------------------------------
# render — input validation gates
# --------------------------------------------------------------------------


def _attest_bundle_for_render(server, text: str = "Alice likes cats. Bob owns dogs."):
    """Helper: attest a small corpus and return the resulting bundle dict.
    Used as the input for render-tool round-trip tests."""
    import asyncio
    result = asyncio.run(_tool(server, "attest")(text=text))
    assert "error_class" not in result, f"attest helper failed: {result}"
    return result["bundle"]


def test_render_rejects_non_dict_bundle(server):
    import asyncio
    result = asyncio.run(_tool(server, "render")(bundle="not a dict"))
    assert result["error_class"] == "schema"


def test_render_rejects_missing_canonical_tome(server):
    import asyncio
    result = asyncio.run(_tool(server, "render")(
        bundle={"canonical_format_version": "1.0.0"}
    ))
    assert result["error_class"] == "schema"


def test_render_rejects_unsupported_canonical_format(server):
    import asyncio
    result = asyncio.run(_tool(server, "render")(
        bundle={"canonical_tome": "The a p o.", "canonical_format_version": "9.9.9"}
    ))
    assert result["error_class"] == "schema"


def test_render_accepts_future_minor_version_under_1_x(server):
    """Forward-compat: a future ``1.X.Y`` bundle should still render
    (additive minor bumps stay compatible)."""
    import asyncio
    bundle = _attest_bundle_for_render(server)
    bundle["canonical_format_version"] = "1.99.0"  # invent a future minor
    result = asyncio.run(_tool(server, "render")(bundle=bundle))
    assert "error_class" not in result, f"unexpected error: {result}"
    assert "tome" in result


def test_render_rejects_oversized_tome(server):
    import asyncio
    from sum_engine_internal.mcp_server.server import MAX_TOME_CHARS
    huge_tome = "a" * (MAX_TOME_CHARS + 1)
    result = asyncio.run(_tool(server, "render")(
        bundle={"canonical_tome": huge_tome, "canonical_format_version": "1.0.0"}
    ))
    assert result["error_class"] == "input_too_large"


def test_render_rejects_zero_axioms(server):
    """A valid bundle shape but with no parseable ``The S P O.`` lines
    must surface as STRUCTURAL — silently emitting an empty tome would
    deceive the caller."""
    import asyncio
    result = asyncio.run(_tool(server, "render")(
        bundle={
            "canonical_tome": "@canonical_version: 1.0.0\n# Title\n\n",
            "canonical_format_version": "1.0.0",
        }
    ))
    assert result["error_class"] == "structural"


def test_render_rejects_density_above_one(server):
    import asyncio
    bundle = _attest_bundle_for_render(server)
    result = asyncio.run(_tool(server, "render")(bundle=bundle, density=1.5))
    assert result["error_class"] == "schema"


def test_render_rejects_negative_length(server):
    import asyncio
    bundle = _attest_bundle_for_render(server)
    result = asyncio.run(_tool(server, "render")(bundle=bundle, length=-0.1))
    assert result["error_class"] == "schema"


def test_render_rejects_non_neutral_length_with_actionable_message(server):
    """Local-only path; the error message must point at the Worker so
    a caller knows where to go for LLM-conditioned rendering."""
    import asyncio
    bundle = _attest_bundle_for_render(server)
    result = asyncio.run(_tool(server, "render")(bundle=bundle, length=0.9))
    assert result["error_class"] == "schema"
    msg = " ".join(result["errors"])
    assert "length=0.9" in msg
    assert "/api/render" in msg or "Worker" in msg


def test_render_rejects_non_neutral_formality(server):
    import asyncio
    bundle = _attest_bundle_for_render(server)
    result = asyncio.run(_tool(server, "render")(bundle=bundle, formality=0.1))
    assert result["error_class"] == "schema"


# --------------------------------------------------------------------------
# render — happy path + round-trip integrity
# --------------------------------------------------------------------------


def test_render_returns_canonical_tome_at_default_sliders(server):
    import asyncio
    bundle = _attest_bundle_for_render(server)
    result = asyncio.run(_tool(server, "render")(bundle=bundle))
    assert "error_class" not in result
    assert result["mode"] == "local-deterministic"
    assert result["axiom_count_input"] == bundle["axiom_count"]
    assert result["sliders"] == {
        "density": 1.0, "length": 0.5, "formality": 0.5,
        "audience": 0.5, "perspective": 0.5,
    }
    tome = result["tome"]
    # Every input axiom should resurface as a canonical line.
    assert tome.count("The ") >= bundle["axiom_count"]


def test_render_round_trip_state_matches_source_at_full_density(server):
    """Load-bearing claim: at density=1.0 the rendered tome's axiom
    set re-mints to the source bundle's state integer, byte-for-byte.
    This is the MCP analogue of
    ``Tests/test_sum_cli_render.py::TestRoundTripFullDensity``."""
    import asyncio
    import math
    import re
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra

    bundle = _attest_bundle_for_render(server)
    result = asyncio.run(_tool(server, "render")(bundle=bundle))
    assert "error_class" not in result

    algebra = GodelStateAlgebra()
    line_re = re.compile(r"^The (\S+) (\S+) (.+)\.$")
    state = 1
    axioms = 0
    for line in result["tome"].splitlines():
        m = line_re.match(line.strip())
        if m:
            state = math.lcm(state, algebra.get_or_mint_prime(*m.groups()))
            axioms += 1
    assert axioms == bundle["axiom_count"]
    assert str(state) == bundle["state_integer"]


def test_render_density_zero_emits_no_axiom_lines(server):
    import asyncio
    import re
    bundle = _attest_bundle_for_render(server)
    result = asyncio.run(_tool(server, "render")(bundle=bundle, density=0.0))
    assert "error_class" not in result
    line_re = re.compile(r"^The (\S+) (\S+) (.+)\.$")
    kept = [
        ln for ln in result["tome"].splitlines()
        if line_re.match(ln.strip())
    ]
    assert kept == []


def test_render_density_keeps_lex_prefix(server):
    """density=0.5 with three lex-distinct subjects should keep
    floor(3 * 0.5) = 1 axiom — the lexicographically-first subject."""
    import asyncio
    import re
    bundle = _attest_bundle_for_render(
        server, text="Carol writes code. Alice likes cats. Bob owns dogs.",
    )
    result = asyncio.run(_tool(server, "render")(bundle=bundle, density=0.5))
    assert "error_class" not in result
    line_re = re.compile(r"^The (\S+) (\S+) (.+)\.$")
    kept = [
        line_re.match(ln.strip()).group(1)
        for ln in result["tome"].splitlines()
        if line_re.match(ln.strip())
    ]
    assert kept == ["alice"]


def test_render_emits_slider_header(server):
    import asyncio
    bundle = _attest_bundle_for_render(server)
    result = asyncio.run(_tool(server, "render")(bundle=bundle, density=0.7))
    assert "@sliders:" in result["tome"]
    assert "density=0.700" in result["tome"]
