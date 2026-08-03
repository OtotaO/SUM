"""The BillSum CERTIFIED CHAIN golden — the first real ``sum.chain_receipt.v1``
over a real public-domain corpus (composed meaning-loss across two real hops).

Two real transforms of the same 32 US Congressional bills (BillSum, **CC0-1.0**):
  hop 1  bill -> reference summary          (the DATASET's own summarization;
                                             SUM did not perform it)
  hop 2  reference summary -> lead-N        (deterministic extractive compression,
         extractive compression             offline, llm_calls_made=0)
plus a DIRECT end-to-end leg (bill -> hop-2 output).

Honesty these tests pin, same discipline as the binding-gate golden:
  * The CERTIFICATE (each hop + the chain) replays offline over the committed
    integer-micro loss vectors — pure-Python, NO model, NO GPU, deterministic
    everywhere. That is what CI checks here (numpy + joserfc only, no torch).
  * The Bonferroni budget bounds the SUM of per-hop expected proxy losses; it
    does NOT bound the end-to-end loss (the proxy is a directed loss, not a
    metric — no triangle inequality). The budget can exceed 1.0.
  * Hop 1 (strict NLI on a full bill) is near-vacuous by design; hop 2 is
    gentler. The chain surfaces WHERE meaning is lost. We do not swap to a
    lenient judge to prettify it.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

joserfc = pytest.importorskip(
    "joserfc", reason="[receipt-verify] extra not installed"
)

from sum_engine_internal.research.meaning import (  # noqa: E402
    verify_chain_receipt,
    verify_meaning_risk_receipt,
)
from sum_engine_internal.research.meaning.chain_receipt import (  # noqa: E402
    BUDGET_SCOPE_STATEMENT,
)

_REPO = Path(__file__).resolve().parents[2]
_FIX = _REPO / "fixtures" / "chain_receipts_billsum"

# Pinned regression locks (from the committed golden; a regeneration that moves
# any of these is noticed). Filled from the generator's own output.
EXPECTED_CHAIN_ID = "9a8ab39f08522c50"
EXPECTED_BUDGET_MICRO = 1_354_628  # = 865_768 (hop1) + 488_860 (hop2)
EXPECTED_HOP1_UB_MICRO = 865_768  # abstractive summary, strict NLI, n=32
EXPECTED_HOP2_UB_MICRO = 488_860  # deterministic lead-N extractive, gentler
EXPECTED_E2E_UB_MICRO = 874_216  # direct end-to-end (< budget: no triangle)


def _load(name):
    return json.loads((_FIX / name).read_text("utf-8"))


@pytest.fixture(scope="module")
def chain():
    return _load("chain_receipt.billsum.golden.json")


@pytest.fixture(scope="module")
def hop1():
    return _load("hop1_summarize.golden.json")


@pytest.fixture(scope="module")
def hop2():
    return _load("hop2_extractive.golden.json")


@pytest.fixture(scope="module")
def jwks():
    return _load("jwks.json")


@pytest.fixture(scope="module")
def losses_hop1():
    return _load("losses_hop1.json")["losses"]


@pytest.fixture(scope="module")
def losses_hop2():
    return _load("losses_hop2.json")["losses"]


@pytest.fixture(scope="module")
def losses_e2e():
    return _load("losses_e2e.json")["losses"]


# ── the core claim: verify chain + full side-band replay ──────────────


def test_chain_golden_verifies_and_replays(chain, jwks, hop1, hop2, losses_e2e):
    """Headline: the committed chain receipt verifies and every leg replays
    over its committed integer-micro loss vector — pure-Python, no judge."""
    payload = verify_chain_receipt(
        chain, jwks, hop_envelopes=[hop1, hop2], end_to_end_losses=losses_e2e
    )
    assert payload["n_hops"] == 2
    assert payload["budget_micro"] == sum(
        h["risk_upper_bound_micro"] for h in payload["hops"]
    )
    assert payload["joint_delta_micro"] == 100_000  # 2 x 0.05
    assert payload["budget_scope"] == BUDGET_SCOPE_STATEMENT
    # regression locks
    assert payload["chain_id"] == EXPECTED_CHAIN_ID
    assert payload["budget_micro"] == EXPECTED_BUDGET_MICRO


def test_chain_verifies_without_side_band(chain, jwks):
    """Stage A + internal consistency only — no hops, no losses."""
    payload = verify_chain_receipt(chain, jwks)
    assert payload["chain_id"] == EXPECTED_CHAIN_ID


def test_hops_verify_and_replay(hop1, hop2, jwks, losses_hop1, losses_hop2):
    """Each hop is itself a replayable meaning-risk receipt over 32 bills."""
    p1 = verify_meaning_risk_receipt(hop1, jwks, losses=losses_hop1)
    assert p1["transform"] == "summarize:billsum-reference"
    assert p1["corpus_id"] == "billsum-test-first32-cc0"
    assert p1["n"] == 32
    assert p1["method"] == "hoeffding"
    assert p1["risk_upper_bound_micro"] == EXPECTED_HOP1_UB_MICRO

    p2 = verify_meaning_risk_receipt(hop2, jwks, losses=losses_hop2)
    assert p2["transform"] == "compress:lead-extractive-keep0.5"
    assert p2["n"] == 32
    assert p2["risk_upper_bound_micro"] == EXPECTED_HOP2_UB_MICRO


def test_chain_reports_honestly(chain):
    """Abstractive hop-1 loses more than deterministic extractive hop-2; the
    budget scope disclaims any end-to-end / triangle claim; the direct
    end-to-end leg is present and separately measured over 32 pairs."""
    pl = chain["payload"]
    ubs = [h["risk_upper_bound_micro"] for h in pl["hops"]]
    assert ubs[0] >= ubs[1]  # abstractive summary loses more than extractive lead-N
    assert ubs[0] == EXPECTED_HOP1_UB_MICRO
    scope = pl["budget_scope"].lower()
    assert "directed loss" in scope and "not bound the end-to-end" in scope
    assert pl["end_to_end"]["n"] == 32
    assert pl["end_to_end"]["risk_upper_bound_micro"] == EXPECTED_E2E_UB_MICRO


def test_chain_corpus_is_real_public_domain(losses_hop1):
    """The corpus is the real CC0 BillSum slice (not self-authored), n=32."""
    corpus = json.loads(
        (
            _REPO
            / "fixtures"
            / "meaning_receipts_billsum"
            / "corpus_billsum_test_first64.json"
        ).read_text("utf-8")
    )
    assert corpus["source_dataset"] == "FiscalNote/billsum"
    assert corpus["license"].startswith("CC0")
    assert len(losses_hop1) == 32
    assert all(0.0 <= x <= 1.0 for x in losses_hop1)


def test_hop2_finals_are_deterministic_lead_n(hop2):
    """The committed hop-2 outputs reproduce from the committed generator's
    lead-N function — deterministic, offline, no model."""
    finals = _load("finals_lead_extractive.json")
    gen = _load_generator()
    corpus = json.loads(
        (
            _REPO
            / "fixtures"
            / "meaning_receipts_billsum"
            / "corpus_billsum_test_first64.json"
        ).read_text("utf-8")
    )
    # Pin the fixture shape so a truncated/regenerated-at-wrong-keep finals
    # file cannot pass silently (2026-07-31 review #26): the committed hop-2
    # receipt is transform="compress:lead-extractive-keep0.5" over 32 pairs, so
    # keep MUST be 0.5 and there must be exactly 32 aligned entries. Without
    # these, a finals file regenerated at keep=0.3 stays self-consistent and
    # green while desyncing from the signed receipt's evidence.
    assert finals["keep"] == 0.5
    assert len(finals["finals"]) == 32
    for entry, pair in zip(finals["finals"], corpus["pairs"][:32], strict=True):
        assert entry["id"] == pair["id"]
        assert entry["final"] == gen.lead_extractive(pair["rendering"], finals["keep"])


# ── byte-stable regeneration (judge-free: reads the committed losses) ──


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "_chain_gen", _FIX / "generate_a2_chain_fixture.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_chain_golden_is_byte_stable(chain, hop1, hop2, jwks):
    """Re-running the generator reproduces the committed fixture exactly.
    Deterministic + judge-free, because build() reads the committed loss
    vectors rather than re-running the model."""
    gen = _load_generator()
    r = gen.build()
    assert r["hop1"] == hop1
    assert r["hop2"] == hop2
    assert r["chain"] == chain
    assert r["jwks"] == jwks
