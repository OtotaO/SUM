"""Regressions for immutable, typed, byte-bounded agent handles."""
import hashlib

import pytest

from sum_engine_internal.agent_surface.bind import BindRegistry, BindTooLargeError


def test_original_and_resolved_mutations_do_not_change_bound_snapshot():
    registry = BindRegistry()
    original = {"triples": [["alice", "like", "cats"]], "nested": {"count": 1}}
    handle = registry.bind(original)
    original["triples"][0][2] = "dogs"
    original["nested"]["count"] = 2
    resolved = registry.resolve(handle)
    assert resolved == {"triples": [["alice", "like", "cats"]], "nested": {"count": 1}}
    resolved["triples"].clear()
    resolved["nested"]["count"] = 3
    again = registry.resolve(handle)
    assert again == {"triples": [["alice", "like", "cats"]], "nested": {"count": 1}}
    assert registry.bind(again) == handle
    assert registry.bind(original) != handle


@pytest.mark.parametrize("values", [
    ('{"a":1}', b'{"a":1}', {"a": 1}),
    ("null", b"null", None),
    ("true", b"true", True),
    ("1", b"1", 1),
    ("[]", b"[]", []),
])
def test_equal_canonical_bytes_in_different_types_cannot_alias(values):
    for order in (values, values[::-1]):
        registry = BindRegistry()
        handles = [registry.bind(value) for value in order]
        assert len(set(handles)) == 3
        for handle, value in zip(handles, order):
            resolved = registry.resolve(handle)
            assert resolved == value
            assert type(resolved) is type(value)


def test_legacy_text_handles_are_stable_and_json_handles_are_versioned():
    registry = BindRegistry()
    assert registry.bind("hello") == "sha256:" + hashlib.sha256(b"hello").hexdigest()
    assert registry.bind({"a": 1}).startswith("sha256:v2:json:")
    assert registry.bind(b"hello").startswith("sha256:v2:bytes:")
    assert not registry.contains("sha256:" + hashlib.sha256(b'{"a":1}').hexdigest())


def test_canonical_json_equivalence_is_normalized_on_read():
    registry = BindRegistry()
    first = registry.bind({"b": (1.0, 2), "a": True})
    second = registry.bind({"a": True, "b": [1, 2]})
    assert first == second
    assert registry.resolve(first) == {"a": True, "b": [1, 2]}


def test_oversized_value_is_rejected_without_eviction_or_accounting_change():
    registry = BindRegistry(max_entries=2, max_total_bytes=10)
    first, second = registry.bind("aaaa"), registry.bind("bbbb")
    with pytest.raises(BindTooLargeError):
        registry.bind("x" * 11)
    assert registry.resolve(first) == "aaaa"
    assert registry.resolve(second) == "bbbb"
    assert registry.size() == 2
    assert registry._total_bytes == 8


def test_byte_budget_counts_utf8_and_keeps_immutable_accounting():
    registry = BindRegistry(max_total_bytes=10)
    with pytest.raises(BindTooLargeError):
        registry.bind("é" * 6)
    value = [1]
    handle = registry.bind(value)
    value.extend(range(1000))
    assert registry.resolve(handle) == [1]
    assert registry._total_bytes == 3
    registry.bind("1234567")
    assert registry._total_bytes == 10


def test_exact_budget_can_be_admitted_after_normal_lru_eviction():
    registry = BindRegistry(max_total_bytes=10)
    old = registry.bind("old")
    exact = registry.bind("x" * 10)
    assert not registry.contains(old)
    assert registry.resolve(exact) == "x" * 10
    assert registry._total_bytes == 10


def test_bind_wrapper_returns_size_error_instead_of_success_handle():
    from sum_engine_internal.agent_surface.mcp_bind import _wrap_result
    result = _wrap_result("extract", {"triples": [["alice", "like", "cat"]]}, BindRegistry(max_total_bytes=5))
    assert result["error_class"] == "input_too_large"
    assert "bind_id" not in result


@pytest.mark.asyncio
async def test_registered_bind_tool_audits_oversize_failure(capsys):
    pytest.importorskip("mcp")
    from mcp.server.fastmcp import FastMCP

    from sum_engine_internal.agent_surface.mcp_bind import register_bind_tools

    async def extract(text, extractor):
        return {"triples": [["alice", "like", "cat"]]}

    server = FastMCP("small registry test")
    register_bind_tools(server, BindRegistry(max_total_bytes=5), {"extract": extract})
    result = await server._tool_manager.get_tool("extract_bind").fn(text="Alice likes cats.")
    assert result["error_class"] == "input_too_large"
    assert '"result_class": "input_too_large"' in capsys.readouterr().err
