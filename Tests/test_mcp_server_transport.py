"""Transport-level contract test for the SUM MCP server's bind surface.

Complements ``Tests/test_mcp_server.py``, which exercises the same
tools in-process via ``server._tool_manager.get_tool(name).fn``.
This file goes through the actual stdio JSON-RPC transport: it
spawns ``python -m sum_engine_internal.mcp_server`` as a subprocess
and drives it through the official ``mcp`` Python client. The
goal is to lock the wire format and the bind-reference flow that
external MCP clients (Claude Code, other agents) actually depend
on — the in-process tests would not catch a regression caused by
the transport layer (FastMCP arg model, JSON-RPC framing,
StructuredContent vs TextContent, etc.).

Single integration test rather than one-per-tool: subprocess
startup is ~1 s per test, and the transport invariants compound
(if the wire is wrong for one tool it is wrong for all of them).
The in-process test file already covers per-tool behaviour
exhaustively; this file's job is the transport contract.
"""
from __future__ import annotations

import asyncio
import json
import sys

import pytest

# Skip the whole file if the optional `mcp` extra is not installed.
mcp = pytest.importorskip("mcp")
pytest.importorskip("mcp.client.stdio")
# Skip if spaCy / sieve extractor is not available — attest needs it.
pytest.importorskip("spacy")


def _parse(result):
    """Unwrap a CallToolResult to the underlying dict.

    FastMCP returns dict-shaped results both as ``StructuredContent``
    (preferred) and as JSON-encoded ``TextContent``. Single-dict
    returns are wrapped in ``{"result": ...}`` on the structured
    side; this function strips that wrapper so callers see the
    same shape they would over the in-process path.
    """
    if getattr(result, "structuredContent", None):
        sc = result.structuredContent
        if isinstance(sc, dict) and set(sc.keys()) == {"result"}:
            return sc["result"]
        return sc
    if result.content:
        return json.loads(result.content[0].text)
    return None


async def _run_session_smoke():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "sum_engine_internal.mcp_server"],
        env=None,
    )
    findings: dict = {}

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            findings["tool_names"] = sorted(t.name for t in tools.tools)

            async def call(name: str, args: dict):
                r = await session.call_tool(name, args)
                return _parse(r)

            findings["manifest_pre"] = await call("agent_surface_manifest", {})
            findings["attest"] = await call(
                "attest_bind",
                {"text": "Alice likes cats. Bob owns Rex. Carol writes code."},
            )
            bid = findings["attest"]["bind_id"]
            findings["verify"] = await call("verify_bind", {"bundle": bid})
            findings["render_ok"] = await call("render_bind", {"bundle": bid})
            findings["render_precondition"] = await call(
                "render_bind", {"bundle": bid, "length": 0.9},
            )
            findings["stale"] = await call(
                "verify_bind", {"bundle": "sha256:" + "0" * 64},
            )
            findings["inspect"] = await call("inspect_bind", {"bundle": bid})
            findings["manifest_post"] = await call("agent_surface_manifest", {})

    return findings


@pytest.fixture(scope="module")
def transport_findings():
    """Run the smoke session once; share the result across assertions.

    Subprocess startup dominates wall time (~1 s); doing it once and
    asserting many invariants over the same set of findings keeps
    the test cheap.
    """
    return asyncio.run(_run_session_smoke())


def test_bind_surface_advertised_over_transport(transport_findings):
    """Both surfaces should appear in list_tools over the wire — if
    they don't, an external MCP client cannot discover them at all,
    no matter what the in-process tests say."""
    names = set(transport_findings["tool_names"])
    legacy_inline = {"extract", "attest", "verify", "inspect", "render", "schema"}
    bind_surface = {
        "extract_bind", "attest_bind", "verify_bind", "render_bind",
        "inspect_bind", "agent_surface_manifest",
    }
    assert legacy_inline <= names, f"legacy regressed: {legacy_inline - names}"
    assert bind_surface <= names, f"bind missing: {bind_surface - names}"


def test_agent_surface_manifest_carries_schema_over_transport(transport_findings):
    m = transport_findings["manifest_pre"]
    assert m["schema"] == "sum.agent_surface_manifest.v1"
    assert m["verb"] == "bind"
    assert "registry_size" in m["runtime"]


def test_attest_bind_returns_bind_id_over_transport(transport_findings):
    r = transport_findings["attest"]
    assert "error_class" not in r, r
    assert r["bind_id"].startswith("sha256:")
    assert r["preview"]["axiom_count"] == 3


def test_verify_bind_resolves_reference_server_side(transport_findings):
    """The promise of the bind verb at the transport layer: the
    client passes only the sha256 string and the server resolves
    it from its in-process registry. If this regresses, every
    saving the bind verb gives is gone."""
    r = transport_findings["verify"]
    assert "error_class" not in r, r
    assert r["preview"]["ok"] is True
    assert r["preview"]["axioms"] == 3


def test_render_bind_resolves_reference_over_transport(transport_findings):
    r = transport_findings["render_ok"]
    assert "error_class" not in r, r
    assert r["preview"]["tome_chars"] > 0


def test_render_bind_typed_precondition_preserved_over_transport(
    transport_findings,
):
    """Errors must pass through unchanged — no bind_id wrapper around
    an error result. An agent client branches on `error_class`
    and that branching has to work over the wire."""
    r = transport_findings["render_precondition"]
    assert r["error_class"] == "schema"
    assert "non-neutral LLM-conditioned axes" in r["errors"][0]
    assert "bind_id" not in r


def test_stale_bind_reference_returns_typed_schema_error(transport_findings):
    r = transport_findings["stale"]
    assert r["error_class"] == "schema"
    assert "could not be resolved" in r["errors"][0]


def test_inspect_bind_resolves_reference_over_transport(transport_findings):
    r = transport_findings["inspect"]
    assert "error_class" not in r, r
    assert r["preview"]["axiom_count"] == 3


def test_server_side_registry_persists_across_calls(transport_findings):
    """Each successful bind tool call adds an entry. After one
    attest + one verify + one render + one inspect, the registry
    should hold at least four distinct values. The exact count
    is implementation detail (some calls bind their result, some
    don't), but it must monotonically increase across the session
    — the registry IS the shared state that makes bind references
    work."""
    pre = transport_findings["manifest_pre"]["runtime"]["registry_size"]
    post = transport_findings["manifest_post"]["runtime"]["registry_size"]
    assert post > pre, (
        f"registry did not grow ({pre} → {post}); the server is "
        f"discarding bound values between calls — the bind verb "
        f"depends on cross-call persistence within a session."
    )
