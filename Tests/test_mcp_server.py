"""Tests for the SUM MCP server.

Three layers of coverage:

  1. **Tool registration** — the server boots, all five expected
     tools register, descriptions are non-empty.
  2. **Tool roundtrips** — extract → attest → verify is the
     primary integration path; every tool produces structured
     output that the next can consume.
  3. **Cross-runtime byte-identity** — a bundle attested through
     the MCP server verifies via the *CLI verifier surface* with
     an unchanged result. This is the load-bearing test: it
     proves the MCP path is byte-equivalent to the CLI path, so
     the existing cross-runtime triangle (Node /
     ``standalone_verifier``, browser ``single_file_demo``)
     extends to MCP-attested bundles transparently.

The tests reach the underlying tool callables directly via
``server._tool_manager.get_tool(name).fn`` rather than going over
the stdio transport — running an actual MCP wire roundtrip would
require a transport double or a subprocess, which buys nothing
on top of the structured-result contract these tests already
exercise. Wire-format coverage is the MCP SDK's responsibility
and is not duplicated here.
"""
from __future__ import annotations

import importlib

import pytest

# Skip the whole file if the optional `mcp` extra is not installed.
mcp = pytest.importorskip("mcp")

# Skip if spaCy / sieve extractor is not available — the CLI's
# extractor picker raises in that path before we can test the
# bundle round-trip. Sieve is the deterministic tier; the LLM
# tier requires a network key and is excluded from CI here.
pytest.importorskip("spacy")


@pytest.fixture(scope="module")
def server():
    from sum_engine_internal.mcp_server import build_server
    return build_server()


def _tool(server, name):
    """Reach the registered Python callable behind a FastMCP tool.

    Going through the wire transport is correct for an integration
    test against an external client, but for unit-level coverage
    of the server's own logic, the underlying callable is the
    right boundary."""
    return server._tool_manager.get_tool(name).fn


# --------------------------------------------------------------------------
# Tool registration
# --------------------------------------------------------------------------


def test_server_boots_with_expected_tools(server):
    import asyncio
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {"extract", "attest", "verify", "inspect", "schema"}


def test_every_tool_has_a_non_empty_description(server):
    import asyncio
    tools = asyncio.run(server.list_tools())
    for t in tools:
        assert t.description, f"tool {t.name!r} has no description"
        assert len(t.description) > 20, (
            f"tool {t.name!r} description suspiciously short — MCP "
            f"clients show this to humans deciding whether to invoke"
        )


# --------------------------------------------------------------------------
# Single-tool behaviour
# --------------------------------------------------------------------------


def test_extract_empty_input_returns_error(server):
    result = _tool(server, "extract")(text="   ")
    assert "error" in result
    assert "empty" in result["error"].lower()


def test_extract_returns_triples(server):
    extract = _tool(server, "extract")
    result = extract(text="The printing press was invented by Johannes Gutenberg.")
    if "error" in result:
        pytest.skip(f"extractor unavailable in this env: {result['error']}")
    assert result["count"] >= 1
    assert isinstance(result["triples"], list)
    assert all(len(t) == 3 for t in result["triples"])
    assert result["extractor"] in ("sieve", "llm")


def test_schema_list_includes_known_schemas(server):
    result = _tool(server, "schema")(name="list")
    assert "sum.canonical_bundle.v1" in result["schemas"]
    assert "sum.render_receipt.v1" in result["schemas"]
    assert "sum.merkle_inclusion.v1" in result["schemas"]


def test_schema_unknown_returns_error_with_catalogue(server):
    result = _tool(server, "schema")(name="sum.does_not_exist.v9")
    assert "error" in result
    assert "known" in result
    assert "sum.canonical_bundle.v1" in result["known"]


def test_schema_named_returns_field_catalogue(server):
    result = _tool(server, "schema")(name="sum.canonical_bundle.v1")
    assert result["schema"] == "sum.canonical_bundle.v1"
    assert "fields" in result
    assert "canonical_tome" in result["fields"]
    assert "state_integer" in result["fields"]


def test_inspect_handles_empty_bundle(server):
    result = _tool(server, "inspect")(bundle={})
    # Inspect is read-only; missing fields surface as None, not as errors.
    assert result["axiom_count"] is None
    assert result["state_integer_digits"] is None
    assert result["signatures_present"] == {"ed25519": False, "hmac": False}


def test_verify_rejects_bundle_missing_required_fields(server):
    result = _tool(server, "verify")(bundle={"branch": "main"})
    assert result["ok"] is False
    assert any("canonical_tome" in e for e in result["errors"])


def test_verify_rejects_unsupported_canonical_format(server):
    result = _tool(server, "verify")(bundle={
        "canonical_tome": "",
        "state_integer": "1",
        "canonical_format_version": "0.0.0-future",
    })
    assert result["ok"] is False
    assert any("canonical_format_version" in e for e in result["errors"])


# --------------------------------------------------------------------------
# Roundtrip — extract → attest → verify (the primary use case)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def attested_bundle(server):
    attest = _tool(server, "attest")
    result = attest(text="The printing press was invented by Johannes Gutenberg.")
    if "error" in result:
        pytest.skip(f"attest path unavailable: {result['error']}")
    return result


def test_attest_produces_well_formed_bundle(attested_bundle):
    bundle = attested_bundle["bundle"]
    assert "canonical_tome" in bundle
    assert "state_integer" in bundle
    assert bundle["canonical_format_version"] == "1.0.0"
    assert bundle["prime_scheme"] == "sha256_64_v1"
    assert bundle["sum_cli"]["produced_by"] == "mcp_server"
    assert attested_bundle["axioms"] >= 1
    assert attested_bundle["source_uri"].startswith("sha256:")


def test_inspect_attested_bundle_reflects_attest_metadata(server, attested_bundle):
    bundle = attested_bundle["bundle"]
    summary = _tool(server, "inspect")(bundle=bundle)
    assert summary["axiom_count"] == attested_bundle["axioms"]
    assert summary["bundle_version"] == bundle["bundle_version"]
    assert summary["sum_cli"]["produced_by"] == "mcp_server"


def test_verify_accepts_self_attested_bundle(server, attested_bundle):
    bundle = attested_bundle["bundle"]
    result = _tool(server, "verify")(bundle=bundle)
    assert result["ok"] is True, f"errors: {result.get('errors')}"
    assert result["axioms"] == attested_bundle["axioms"]
    # ed25519 may be absent (no key supplied) or present (codec
    # generated an ephemeral one) — both are valid; neither
    # should be invalid.
    assert result["signatures"]["ed25519"] in ("absent", "valid")
    assert result["signatures"]["hmac"] == "absent"  # no signing_key supplied


def test_verify_rejects_tampered_canonical_tome(server, attested_bundle):
    """Mutating any axiom in the tome breaks the
    state-integer reconstruction; the verifier rejects."""
    import copy
    bundle = copy.deepcopy(attested_bundle["bundle"])
    # Inject a fake axiom — changes state, bundle's claimed
    # state_integer no longer matches reconstruction.
    bundle["canonical_tome"] = (
        bundle["canonical_tome"].rstrip("\n")
        + "\nThe earth orbits sun.\n"
    )
    result = _tool(server, "verify")(bundle=bundle)
    assert result["ok"] is False
    assert any("state integer" in e or "axiom count" in e for e in result["errors"])


def test_verify_rejects_tampered_state_integer(server, attested_bundle):
    import copy
    bundle = copy.deepcopy(attested_bundle["bundle"])
    bundle["state_integer"] = "999"
    result = _tool(server, "verify")(bundle=bundle)
    assert result["ok"] is False


# --------------------------------------------------------------------------
# Cross-runtime byte-identity — MCP-attested bundle verifies via CLI surface
# --------------------------------------------------------------------------


def test_mcp_attested_bundle_verifies_via_cli_surface(server, attested_bundle, tmp_path):
    """The MCP server's attest path must produce bytes that the
    CLI's verify path accepts unchanged. This is what guarantees
    the cross-runtime triangle (Node, browser) extends to
    MCP-attested bundles — the canonical bytes are the contract,
    and the MCP server is just a different way to reach the
    canonical codec."""
    import json
    import subprocess
    import sys

    bundle_path = tmp_path / "mcp_bundle.json"
    bundle_path.write_text(
        json.dumps(attested_bundle["bundle"]), encoding="utf-8"
    )

    # Invoke the CLI verifier as a subprocess — same boundary a
    # downstream consumer of an MCP-attested bundle would cross.
    result = subprocess.run(
        [sys.executable, "-m", "sum_cli.main", "verify", "--input", str(bundle_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"CLI rejected an MCP-attested bundle:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    # The CLI emits a JSON success payload on stdout.
    cli_payload = json.loads(result.stdout)
    assert cli_payload["ok"] is True
    assert cli_payload["axioms"] == attested_bundle["axioms"]
