"""SUM MCP server (Model Context Protocol).

Exposes SUM's primary verbs as MCP tools so any MCP-aware client
(Claude Desktop, Claude Code, Cursor, Continue, custom agents)
can call SUM directly without shelling out to the ``sum`` CLI.

Public surface re-exported here:
    build_server()      — construct the FastMCP app (tools registered)
    main()              — stdio entry point (the binary `sum-mcp`)

Tools registered on the server:
    extract     — text → list of (subject, predicate, object) triples
    attest      — text → signed CanonicalBundle JSON
    verify      — bundle JSON → structured verification result
    inspect     — bundle JSON → human-readable summary fields
    schema      — return the canonical-bundle / receipt JSON schemas

The server is a thin façade over ``sum_engine_internal`` and
``sum_cli.main`` helpers. No new cryptography, no new canonical
codec — the MCP path produces the same bytes the CLI does, so a
bundle attested via MCP verifies identically via ``sum verify``
(and vice versa). The cross-runtime trust triangle holds.

Wire format: stdio (newline-delimited JSON-RPC 2.0) per the MCP
spec. Run via ``sum-mcp`` after ``pip install sum-engine[mcp]``,
or ``python -m sum_engine_internal.mcp_server``.
"""
from __future__ import annotations

from sum_engine_internal.mcp_server.server import build_server, main

__all__ = ["build_server", "main"]
