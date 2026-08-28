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
_PERSP_FIX = _REPO / "fixtures" / "perspective_receipts"


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


def test_mint_caps_pair_count(server, keypair):
    """The judged corpus is capped by COUNT. Fail closed, name the limit,
    never truncate: a receipt minted over a silently shortened corpus would
    certify a different corpus than the ``corpus_id`` it names."""
    from sum_engine_internal.mcp_server.meaning_tools import MAX_PAIRS
    fn = _tool(server, "mint_meaning_receipt")
    out = asyncio.run(
        fn(
            private_jwk=keypair, kid="k", corpus_id="c", transform="t",
            loss_definition="l",
            pairs=[{"source": "a source.", "rendering": "a rendering."}]
                  * (MAX_PAIRS + 1),
        )
    )
    assert out["error_class"] == "input_too_large"
    assert str(MAX_PAIRS) in out["errors"][0]
    # No receipt escapes on the refusal path.
    assert "receipt" not in out


def test_mint_caps_pair_total_chars(server, keypair):
    """Count alone is not enough: MAX_PAIRS entries each at the per-field
    prose cap would still be hundreds of MB. The aggregate cap fires on a
    pair list that is well under the count cap."""
    from sum_engine_internal.mcp_server.meaning_tools import (
        MAX_PAIRS,
        MAX_PAIRS_TOTAL_CHARS,
        MAX_TEXT_CHARS,
    )
    fn = _tool(server, "mint_meaning_receipt")
    n = MAX_PAIRS_TOTAL_CHARS // (2 * MAX_TEXT_CHARS) + 1  # 26 pairs
    assert n <= MAX_PAIRS  # the COUNT cap must not be what fires here
    big = {"source": "x" * MAX_TEXT_CHARS, "rendering": "y" * MAX_TEXT_CHARS}
    out = asyncio.run(
        fn(
            private_jwk=keypair, kid="k", corpus_id="c", transform="t",
            loss_definition="l", pairs=[dict(big) for _ in range(n)],
        )
    )
    assert out["error_class"] == "input_too_large"
    assert str(MAX_PAIRS_TOTAL_CHARS) in out["errors"][0]
    assert "receipt" not in out


def test_real_binding_gate_corpus_passes_the_pair_caps():
    """The largest corpus this project has ever certified — the committed
    BillSum binding-gate corpus, n=64 — sails through both caps untouched.
    This is what makes the numbers honest rather than arbitrary."""
    from sum_engine_internal.mcp_server.meaning_tools import _validate_pairs
    corpus = _load(_MEANING_FIX, "corpus_billsum_test_first64.json")["pairs"]
    assert len(corpus) == 64
    assert _validate_pairs(corpus) is None


def test_mint_normal_pair_count_is_not_capped(server, keypair):
    """A normal-sized pairs request passes the size gate and reaches the
    scorer gate. scorer='lexical' is used so the assertion is deterministic
    everywhere (no model load, no [judge] extra): the rejection that comes
    back is the scorer policy, NOT the cap."""
    fn = _tool(server, "mint_meaning_receipt")
    out = asyncio.run(
        fn(
            private_jwk=keypair, kid="k", corpus_id="c", transform="t",
            loss_definition="l",
            pairs=[
                {"source": f"source {i}.", "rendering": f"rendering {i}."}
                for i in range(3)
            ],
            scorer="lexical",
        )
    )
    assert out["error_class"] == "schema"
    assert "scorer must be" in out["errors"][0]


def test_verify_receipt_caps_jwks_key_count(server, billsum_golden):
    """An oversized caller JWKS is a work amplifier, not a key directory."""
    from sum_engine_internal.mcp_server.meaning_tools import MAX_JWKS_KEYS
    fn = _tool(server, "verify_receipt")
    key = billsum_golden["jwks"]["keys"][0]
    fat = {"keys": [dict(key, kid=f"k{i}") for i in range(MAX_JWKS_KEYS + 1)]}
    out = asyncio.run(fn(billsum_golden["receipt"], fat))
    assert out["error_class"] == "input_too_large"
    assert out["verified"] is False
    # The same receipt with its real (1-key) JWKS still verifies.
    ok = asyncio.run(
        fn(billsum_golden["receipt"], billsum_golden["jwks"])
    )
    assert ok["verified"] is True


def test_mint_chain_caps_hops_jwks_key_count(server, keypair, chain_golden):
    from sum_engine_internal.mcp_server.meaning_tools import MAX_JWKS_KEYS
    fn = _tool(server, "mint_chain_receipt")
    key = chain_golden["jwks"]["keys"][0]
    fat = {"keys": [dict(key, kid=f"k{i}") for i in range(MAX_JWKS_KEYS + 1)]}
    out = asyncio.run(
        fn(
            private_jwk=keypair, kid="k",
            hop_envelopes=[chain_golden["hop1"], chain_golden["hop2"]],
            hops_jwks=fat,
        )
    )
    assert out["error_class"] == "input_too_large"
    assert "receipt" not in out


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


# --------------------------------------------------------------------------
# 2026-07-31 review regressions (#7, #8, #12)
# --------------------------------------------------------------------------


def test_mint_chain_strips_private_d_from_merged_hops_jwks(server, keypair):
    """#7: a caller who passes the PRIVATE hop signing JWKS to hops_jwks must
    NOT get 'd' republished in the returned public_jwks (callers distribute
    that field)."""
    mint = _tool(server, "mint_meaning_receipt")
    mint_chain = _tool(server, "mint_chain_receipt")

    def _hop(transform, losses):
        out = asyncio.run(mint(
            private_jwk=keypair, kid="mcp-test-key-1",
            corpus_id="c", transform=transform,
            loss_definition="d", losses=losses,
            scorer_name="s", scorer_version="1", method="hoeffding",
        ))
        assert "error_class" not in out, out
        return out["receipt"]

    hop1 = _hop("compress:test", [0.1, 0.2, 0.15, 0.05] * 8)
    hop2 = _hop("translate:test", [0.05, 0.1, 0.1, 0.2] * 8)
    out = asyncio.run(mint_chain(
        private_jwk=keypair, kid="mcp-test-key-1",
        hop_envelopes=[hop1, hop2],
        hops_jwks={"keys": [dict(keypair)]},   # the PRIVATE jwk, by mistake
    ))
    assert "error_class" not in out, out
    for k in out["public_jwks"]["keys"]:
        assert "d" not in k, "private 'd' leaked into public_jwks via hops_jwks"
    assert keypair["d"] not in json.dumps(out["public_jwks"])


def test_mint_meaning_rejects_non_string_scorer_fields(server, keypair):
    """#8: scorer_name / scorer_version must be strings — a dict/list must not
    ride into the signed payload."""
    mint = _tool(server, "mint_meaning_receipt")
    losses = [0.1, 0.2, 0.15, 0.05, 0.3, 0.25, 0.1, 0.2]
    base = dict(
        private_jwk=keypair, kid="mcp-test-key-1", corpus_id="c",
        transform="t", loss_definition="d", losses=losses, method="hoeffding",
    )
    out = asyncio.run(mint(**base, scorer_name={"injected": "dict"}, scorer_version="1"))
    assert out["error_class"] == "schema"
    out = asyncio.run(mint(**base, scorer_name="s", scorer_version=["list", 1]))
    assert out["error_class"] == "schema"


def test_verify_receipt_rejects_perspective_with_named_schemas(server):
    """#12: the research-tier perspective schema is not in the offline SDK path;
    verify_receipt must reject it cleanly and name the four supported schemas
    (the contract the docstring now advertises)."""
    fn = _tool(server, "verify_receipt")
    persp = _PERSP_FIX / "perspective_risk_receipt.golden.json"
    if not persp.exists():
        pytest.skip("perspective golden fixture not present")
    receipt = json.loads(persp.read_text())
    out = asyncio.run(fn(receipt, {"keys": []}))
    assert out["verified"] is False
    assert out["error_class"] == "schema"
    assert "sum.meaning_risk_receipt.v1" in out["errors"][0]


# ---------------------------------------------------------------------------
# meaning_diff SUCCESS path — the layer this file's header said was uncovered.
#
# Regression pin for a bug that made the tool 100% broken on EVERY call since
# it shipped: the result builder read ``r.added_claims``, which MeaningReadout
# has never had (the field is ``unsupported_claims``). Every invocation paid
# the full judge cost, then died on AttributeError swallowed into a generic
# ``internal`` error. No test drove the success path because the [judge] extra
# never runs in per-PR CI -- so the judge is stubbed here and the REAL
# MeaningReadout dataclass is constructed, which keeps the test torch-free
# while still failing if the dataclass is renamed again.
# ---------------------------------------------------------------------------

def _stub_readout():
    from sum_engine_internal.research.meaning.meaning_loss import MeaningReadout
    return MeaningReadout(
        loss=0.25, preservation=0.75, recall=0.8, fidelity=1.0,
        source_claims=("a claim", "another claim"),
        preserved_claims=("a claim",),
        dropped_claims=("another claim",),
        transform_claims=("a claim", "an invented claim"),
        unsupported_claims=("an invented claim",),
        judge="stub-judge", judge_version="0",
    )


class _StubScorer:
    name = "stub-judge"
    version = "0"

    def explain(self, source, rendering):
        return _stub_readout()


@pytest.fixture
def stub_judge(monkeypatch):
    from sum_engine_internal.mcp_server import meaning_tools as mt
    monkeypatch.setattr(mt, "_need_research_extra", lambda: None)
    monkeypatch.setattr(mt, "_load_scorer", lambda scorer: (_StubScorer(), None))
    return mt


def test_meaning_diff_success_returns_the_documented_shape(server, stub_judge):
    """The docstring promises these keys. Before the fix, NONE were returned."""
    fn = _tool(server, "meaning_diff")
    out = asyncio.run(fn(source="a claim. another claim.", rendering="a claim."))

    assert "error_class" not in out, f"meaning_diff failed: {out}"
    for key in (
        "loss", "recall", "fidelity", "source_claims", "preserved_claims",
        "dropped_claims", "added_claims", "scorer", "scorer_version",
        "scope", "concurrency",
    ):
        assert key in out, f"documented key {key!r} missing from meaning_diff result"


def test_meaning_diff_added_claims_carries_the_unsupported_sentences(server, stub_judge):
    """``added_claims`` on the wire is MeaningReadout.unsupported_claims.

    This is the exact assertion the original bug failed: the builder reached
    for an attribute that does not exist on the readout.
    """
    fn = _tool(server, "meaning_diff")
    out = asyncio.run(fn(source="a claim. another claim.", rendering="a claim."))

    assert out["added_claims"] == ["an invented claim"]
    assert out["dropped_claims"] == ["another claim"]
    assert out["preserved_claims"] == ["a claim"]


def test_meaning_readout_has_no_added_claims_attribute():
    """Pins WHY the wire key is renamed, so nobody 'fixes' it back.

    If MeaningReadout ever grows a real ``added_claims`` field this test fails
    loudly and the rename in meaning_tools can be reconsidered deliberately.
    """
    assert not hasattr(_stub_readout(), "added_claims")


def test_meaning_diff_result_carries_the_scope_caveat(server, stub_judge):
    """A per-document MEASUREMENT must never be presentable as a bound."""
    fn = _tool(server, "meaning_diff")
    out = asyncio.run(fn(source="a claim. another claim.", rendering="a claim."))
    assert out["scope"]
    assert "bound" in out["scope"].lower() or "measurement" in out["scope"].lower()
