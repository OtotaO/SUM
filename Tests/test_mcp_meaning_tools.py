"""Meaning-layer MCP tools — the receipt family over stdio for agent swarms.

Coverage layers, mirroring ``test_mcp_server.py``:

  1. Validation gates and error classes per tool (no model, no crypto).
  2. Real verification against the COMMITTED goldens (BillSum meaning-risk
     receipt + the certified chain) — the same artifacts CI already replays,
     so no torch and no network.
  3. Minting round-trips with a fresh in-test Ed25519 key (BYO-key path) —
     mint -> the tool's own self-verify -> re-verify via verify_receipt.
  4. THE SWARM ACCEPTANCE SMOKE: 16 parallel ``verify_receipt`` calls on the
     committed chain golden, all green (the plan's D1 acceptance).

Model-judge paths (meaning_diff / depth_frontier scoring) are exercised only
up to their validation gates here: the [judge] extra never runs in per-PR CI
(the monthly judge-smoke canary owns that), and these tests must stay green
without torch.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")
joserfc = pytest.importorskip(
    "joserfc", reason="[receipt-verify] extra not installed"
)
from joserfc.jwk import OKPKey  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_MEANING_FIX = _REPO / "fixtures" / "meaning_receipts_billsum"
_CHAIN_FIX = _REPO / "fixtures" / "chain_receipts_billsum"


def _load(base: Path, name: str):
    return json.loads((base / name).read_text("utf-8"))


@pytest.fixture(scope="module")
def server():
    from sum_engine_internal.mcp_server import build_server
    return build_server()


def _tool(server, name):
    return server._tool_manager.get_tool(name).fn


@pytest.fixture(scope="module")
def billsum_golden():
    return {
        "receipt": _load(_MEANING_FIX, "meaning_risk_receipt.billsum.golden.json"),
        "jwks": _load(_MEANING_FIX, "jwks.json"),
        "losses": _load(_MEANING_FIX, "losses_billsum.json")["losses"],
    }


@pytest.fixture(scope="module")
def chain_golden():
    return {
        "chain": _load(_CHAIN_FIX, "chain_receipt.billsum.golden.json"),
        "hop1": _load(_CHAIN_FIX, "hop1_summarize.golden.json"),
        "hop2": _load(_CHAIN_FIX, "hop2_extractive.golden.json"),
        "jwks": _load(_CHAIN_FIX, "jwks.json"),
        "losses_e2e": _load(_CHAIN_FIX, "losses_e2e.json")["losses"],
    }


@pytest.fixture(scope="module")
def keypair():
    key = OKPKey.generate_key("Ed25519")
    private = key.as_dict(private=True)
    private.update(kid="mcp-test-key-1", alg="EdDSA", use="sig")
    return private


# --------------------------------------------------------------------------
# verify_receipt — validation gates
# --------------------------------------------------------------------------


def test_verify_receipt_rejects_non_dict_inputs(server):
    fn = _tool(server, "verify_receipt")
    out = asyncio.run(fn("not a dict", {"keys": []}))
    assert out["error_class"] == "schema"
    assert out["verified"] is False


def test_verify_receipt_rejects_unknown_schema(server):
    fn = _tool(server, "verify_receipt")
    out = asyncio.run(fn({"schema": "sum.bogus.v9"}, {"keys": []}))
    assert out["error_class"] == "schema"
    assert "sum.meaning_risk_receipt.v1" in out["errors"][0]


def test_verify_receipt_rejects_bad_losses(server, billsum_golden):
    fn = _tool(server, "verify_receipt")
    out = asyncio.run(
        fn(billsum_golden["receipt"], billsum_golden["jwks"], losses=[0.5, 2.0])
    )
    assert out["error_class"] == "schema"


# --------------------------------------------------------------------------
# verify_receipt — the committed goldens (real crypto, no judge)
# --------------------------------------------------------------------------


def test_verify_receipt_replays_billsum_golden(server, billsum_golden):
    fn = _tool(server, "verify_receipt")
    out = asyncio.run(
        fn(
            billsum_golden["receipt"],
            billsum_golden["jwks"],
            losses=billsum_golden["losses"],
        )
    )
    assert "error_class" not in out
    assert out["verified"] is True
    assert out["replayed"] is True
    assert out["schema"] == "sum.meaning_risk_receipt.v1"
    assert out["risk_upper_bound"] == pytest.approx(0.645438)
    assert out["n"] == 64
    # The honesty layer rides the MCP verdict exactly as it rides the CLI's.
    assert "proxy_caveat" in out
    assert "not a substitute for human review" in out["proxy_caveat"].lower()
    assert "parallel" in out["concurrency"]


def test_verify_receipt_full_chain_replay(server, chain_golden):
    fn = _tool(server, "verify_receipt")
    out = asyncio.run(
        fn(
            chain_golden["chain"],
            chain_golden["jwks"],
            losses=chain_golden["losses_e2e"],
            hops=[chain_golden["hop1"], chain_golden["hop2"]],
        )
    )
    assert "error_class" not in out
    assert out["verified"] is True
    assert out["hops_replayed"] is True
    assert out["end_to_end_replayed"] is True
    assert out["n_hops"] == 2
    assert out["budget"] == pytest.approx(1.354628)
    assert out["joint_confidence"] == pytest.approx(0.90)
    # The no-triangle honesty line rides every chain verdict.
    assert "directed loss" in out["budget_scope"]


def test_verify_receipt_rejects_tampered_bound(server, billsum_golden):
    import copy
    fn = _tool(server, "verify_receipt")
    tampered = copy.deepcopy(billsum_golden["receipt"])
    tampered["payload"]["risk_upper_bound_micro"] -= 1
    out = asyncio.run(
        fn(tampered, billsum_golden["jwks"], losses=billsum_golden["losses"])
    )
    assert out["verified"] is False
    assert out["error_class"] in {"signature", "structural"}


def test_verify_receipt_rejects_reordered_hops(server, chain_golden):
    fn = _tool(server, "verify_receipt")
    out = asyncio.run(
        fn(
            chain_golden["chain"],
            chain_golden["jwks"],
            hops=[chain_golden["hop2"], chain_golden["hop1"]],  # order flipped
        )
    )
    assert out["verified"] is False
    assert out["error_class"] in {"signature", "structural"}


# --------------------------------------------------------------------------
# THE SWARM SMOKE — 16 parallel verifications, all green (D1 acceptance)
# --------------------------------------------------------------------------


def test_sixteen_parallel_chain_verifications_all_green(server, chain_golden):
    fn = _tool(server, "verify_receipt")

    async def _swarm():
        calls = [
            fn(
                chain_golden["chain"],
                chain_golden["jwks"],
                losses=chain_golden["losses_e2e"],
                hops=[chain_golden["hop1"], chain_golden["hop2"]],
            )
            for _ in range(16)
        ]
        return await asyncio.gather(*calls)

    results = asyncio.run(_swarm())
    assert len(results) == 16
    for out in results:
        assert "error_class" not in out, out
        assert out["verified"] is True
        assert out["hops_replayed"] is True
        assert out["budget"] == pytest.approx(1.354628)


# --------------------------------------------------------------------------
# meaning_diff / depth_frontier — validation gates (no model load)
# --------------------------------------------------------------------------


def test_meaning_diff_rejects_lexical(server):
    fn = _tool(server, "meaning_diff")
    out = asyncio.run(fn("a source.", "a rendering.", scorer="lexical"))
    assert out["error_class"] == "schema"
    assert "entailment" in out["errors"][0]


def test_meaning_diff_rejects_empty_source(server):
    fn = _tool(server, "meaning_diff")
    out = asyncio.run(fn("   ", "a rendering.", scorer="nli"))
    assert out["error_class"] == "schema"


def test_meaning_diff_rejects_oversized_input(server):
    from sum_engine_internal.mcp_server.meaning_tools import MAX_TEXT_CHARS
    fn = _tool(server, "meaning_diff")
    out = asyncio.run(fn("x" * (MAX_TEXT_CHARS + 1), "a rendering.", scorer="nli"))
    assert out["error_class"] == "input_too_large"


def test_depth_frontier_rejects_empty_versions(server):
    fn = _tool(server, "depth_frontier")
    out = asyncio.run(fn("a source.", [], scorer="nli"))
    assert out["error_class"] == "schema"


def test_depth_frontier_caps_version_count(server):
    from sum_engine_internal.mcp_server.meaning_tools import MAX_VERSIONS
    fn = _tool(server, "depth_frontier")
    out = asyncio.run(
        fn("a source.", ["v"] * (MAX_VERSIONS + 1), scorer="nli")
    )
    assert out["error_class"] == "input_too_large"


# --------------------------------------------------------------------------
# mint tools — key policy + BYO-losses round trip
# --------------------------------------------------------------------------


def test_mint_refuses_missing_private_key(server):
    fn = _tool(server, "mint_meaning_receipt")
    out = asyncio.run(
        fn(
            private_jwk={},  # no key material
            kid="k", corpus_id="c", transform="t", loss_definition="l",
            losses=[0.1, 0.2],
            scorer_name="test-scorer",
        )
    )
    assert out["error_class"] == "schema"
    assert "never generates or stores key material" in out["errors"][0]


def test_mint_refuses_public_only_jwk(server, keypair):
    fn = _tool(server, "mint_meaning_receipt")
    public_only = {k: v for k, v in keypair.items() if k != "d"}
    out = asyncio.run(
        fn(
            private_jwk=public_only,
            kid="k", corpus_id="c", transform="t", loss_definition="l",
            losses=[0.1, 0.2],
            scorer_name="test-scorer",
        )
    )
    assert out["error_class"] == "schema"


def test_mint_requires_exactly_one_evidence_mode(server, keypair):
    fn = _tool(server, "mint_meaning_receipt")
    out = asyncio.run(
        fn(
            private_jwk=keypair, kid="k", corpus_id="c", transform="t",
            loss_definition="l",  # neither losses nor pairs
        )
    )
    assert out["error_class"] == "schema"
    assert "exactly one" in out["errors"][0]


def test_mint_byo_losses_requires_scorer_name(server, keypair):
    fn = _tool(server, "mint_meaning_receipt")
    out = asyncio.run(
        fn(
            private_jwk=keypair, kid="k", corpus_id="c", transform="t",
            loss_definition="l", losses=[0.1, 0.2],
        )
    )
    assert out["error_class"] == "schema"
    assert "scorer_name" in out["errors"][0]


def test_mint_meaning_receipt_roundtrip_and_reverify(server, keypair):
    """BYO losses -> mint -> the tool's own self-verify -> independent
    re-verify through verify_receipt. The receipt and the small-n warning
    both come back; the private key never does."""
    mint = _tool(server, "mint_meaning_receipt")
    verify = _tool(server, "verify_receipt")
    losses = [0.1, 0.2, 0.15, 0.05, 0.3, 0.25, 0.1, 0.2]
    out = asyncio.run(
        mint(
            private_jwk=keypair,
            kid="mcp-test-key-1",
            corpus_id="mcp-test-corpus",
            transform="compress:test",
            loss_definition="1 - recall of source claims (test)",
            losses=losses,
            scorer_name="test-scorer",
            scorer_version="1",
            method="hoeffding",
        )
    )
    assert "error_class" not in out, out
    assert out["verdict"]["verified"] is True
    assert out["verdict"]["replayed"] is True
    assert out["verdict"]["n"] == 8
    assert any("n=8 is small" in w for w in out["warnings"])
    assert "proxy_caveat" in out["verdict"]
    # Key hygiene: no private material anywhere in the result.
    assert "d" not in out["public_jwks"]["keys"][0]
    assert keypair["d"] not in json.dumps(out["receipt"])
    assert keypair["d"] not in json.dumps(out["public_jwks"])
    # Independent re-verify through the other tool.
    out2 = asyncio.run(
        verify(out["receipt"], out["public_jwks"], losses=losses)
    )
    assert out2["verified"] is True and out2["replayed"] is True


def test_mint_chain_receipt_roundtrip(server, keypair):
    """Two minted hops -> mint_chain_receipt -> verified chain with the
    budget_scope honesty field, re-verified independently."""
    mint = _tool(server, "mint_meaning_receipt")
    mint_chain = _tool(server, "mint_chain_receipt")
    verify = _tool(server, "verify_receipt")

    def _hop(transform, losses):
        out = asyncio.run(
            mint(
                private_jwk=keypair, kid="mcp-test-key-1",
                corpus_id="mcp-test-corpus", transform=transform,
                loss_definition="1 - recall of source claims (test)",
                losses=losses, scorer_name="test-scorer", scorer_version="1",
                method="hoeffding",
            )
        )
        assert "error_class" not in out, out
        return out["receipt"], out["public_jwks"]

    hop1, jwks = _hop("compress:test", [0.1, 0.2, 0.15, 0.05] * 8)
    hop2, _ = _hop("translate:test", [0.05, 0.1, 0.1, 0.2] * 8)
    e2e = [0.2, 0.3, 0.25, 0.15] * 8

    out = asyncio.run(
        mint_chain(
            private_jwk=keypair, kid="mcp-test-key-1",
            hop_envelopes=[hop1, hop2],
            end_to_end_losses=e2e,
            scorer_name="test-scorer",
            loss_definition="direct source->final loss (test)",
        )
    )
    assert "error_class" not in out, out
    assert out["verdict"]["verified"] is True
    assert out["verdict"]["n_hops"] == 2
    assert "directed loss" in out["verdict"]["budget_scope"]
    assert keypair["d"] not in json.dumps(out["receipt"])
    assert keypair["d"] not in json.dumps(out["public_jwks"])

    out2 = asyncio.run(
        verify(
            out["receipt"], out["public_jwks"],
            losses=e2e, hops=[hop1, hop2],
        )
    )
    assert out2["verified"] is True
    assert out2["hops_replayed"] is True and out2["end_to_end_replayed"] is True


def test_mint_chain_requires_two_hops(server, keypair):
    fn = _tool(server, "mint_chain_receipt")
    out = asyncio.run(
        fn(private_jwk=keypair, kid="k", hop_envelopes=[{"schema": "x"}])
    )
    assert out["error_class"] == "schema"
