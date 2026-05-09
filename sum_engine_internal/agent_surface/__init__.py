"""Agent-surface layer: content-addressed bind registry and tool wrappers.

The MCP server (``sum_engine_internal.mcp_server.server``) exposes
typed tools with structured error classes. This module adds the
``bind`` verb on top of that surface — every tool's output gains a
content-addressed handle, every tool's input accepts either an inline
value OR a ``bind:`` reference, the runtime memoises bind_id → object.

This addresses the failure modes documented in
``docs/AGENT_SURFACE_FINDINGS.md`` (parse failures from full-bundle
round-trips, free-form-prose error interpretation). The substrate's
content-addressed identities (`prov_id`, `state_integer`,
``sha256(canonical_jcs(value))``) already exist; this module exposes
them as the access path.
"""
from sum_engine_internal.agent_surface.bind import (
    BindRegistry,
    BindNotFoundError,
    DEFAULT_REGISTRY,
)

__all__ = ["BindRegistry", "BindNotFoundError", "DEFAULT_REGISTRY"]
