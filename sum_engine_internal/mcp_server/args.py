"""Argument types shared by the MCP tools.

FastMCP (``mcp.server.fastmcp``, ``FuncMetadata.pre_parse_json``) runs
``json.loads`` on every string argument whose annotation is not exactly
``str``, and replaces the string when it parses to ``null``, an array or an
object. With ``str | None``, the string ``"null"`` therefore reached a tool
as ``None`` (a ``signing_key`` of ``"null"`` turned an HMAC check into a
keyless one), and ``"[1]"`` or ``"{}"`` failed argument validation.

``OptionalText`` is annotated as exactly ``str``, so FastMCP passes the text
through unchanged. The wrap validator still lets an explicit JSON ``null``
through as ``None``, and the published input schema stays "string or null",
the same schema ``str | None`` produced.
"""
from __future__ import annotations

from typing import Annotated, Any

from pydantic import WithJsonSchema, WrapValidator


def _none_passes(value: Any, handler: Any) -> Any:
    return None if value is None else handler(value)


OptionalText = Annotated[
    str,
    WrapValidator(_none_passes),
    WithJsonSchema({"anyOf": [{"type": "string"}, {"type": "null"}]}),
]
