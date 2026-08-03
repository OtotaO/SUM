"""MCP verify_receipt replay-defense window (max_age_seconds).

verify_receipt forwards ``max_age_seconds`` to both ``sum_verify.verify`` and
``sum_verify.verify_chain_receipt``, but nothing tested it — so a refactor that
dropped or renamed the kwarg in the dispatch would compile, pass every other
MCP test, and silently turn the replay-defense window into a no-op for every
MCP agent (the classic fail-open regression on a security-adjacent option;
2026-07-31 review #27). These pin it on the committed goldens.

Kept in its own file so it composes cleanly with the meaning-tools test file
regardless of merge order.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")
pytest.importorskip("joserfc", reason="[receipt-verify] extra not installed")

_REPO = Path(__file__).resolve().parents[1]
_MEANING_FIX = _REPO / "fixtures" / "meaning_receipts_billsum"
_CHAIN_FIX = _REPO / "fixtures" / "chain_receipts_billsum"


def _load(base: Path, name: str):
    return json.loads((base / name).read_text("utf-8"))


@pytest.fixture(scope="module")
def verify_receipt():
    from sum_engine_internal.mcp_server import build_server
    server = build_server()
    return server._tool_manager.get_tool("verify_receipt").fn


@pytest.fixture(scope="module")
def billsum():
    return {
        "receipt": _load(_MEANING_FIX, "meaning_risk_receipt.billsum.golden.json"),
        "jwks": _load(_MEANING_FIX, "jwks.json"),
        "losses": _load(_MEANING_FIX, "losses_billsum.json")["losses"],
    }


@pytest.fixture(scope="module")
def chain():
    return {
        "chain": _load(_CHAIN_FIX, "chain_receipt.billsum.golden.json"),
        "jwks": _load(_CHAIN_FIX, "jwks.json"),
    }


def test_max_age_rejects_old_signed_at(verify_receipt, billsum):
    """A 1-second window on the 2026-06-08-signed golden must reject with a
    signature error — proving the kwarg reaches the replay-defense window."""
    out = asyncio.run(
        verify_receipt(billsum["receipt"], billsum["jwks"], max_age_seconds=1)
    )
    assert out["verified"] is False
    assert out["error_class"] == "signature"


def test_generous_max_age_still_verifies(verify_receipt, billsum):
    """A window wide enough to cover the golden's age must still verify — the
    kwarg gates, it does not break the happy path."""
    out = asyncio.run(
        verify_receipt(
            billsum["receipt"], billsum["jwks"],
            losses=billsum["losses"], max_age_seconds=10_000_000_000,  # ~317y
        )
    )
    assert "error_class" not in out
    assert out["verified"] is True and out["replayed"] is True


def test_max_age_plumbs_through_chain_path(verify_receipt, chain):
    """Same window plumbed through verify_chain_receipt."""
    out = asyncio.run(
        verify_receipt(chain["chain"], chain["jwks"], max_age_seconds=1)
    )
    assert out["verified"] is False
    assert out["error_class"] == "signature"
