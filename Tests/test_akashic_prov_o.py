"""Tests for AkashicLedger.to_prov_jsonld — ledger → PROV-O JSON-LD integration."""
from __future__ import annotations

import json

import pytest

from sum_engine_internal.infrastructure.akashic_ledger import AkashicLedger


@pytest.mark.asyncio
async def test_empty_branch_emits_empty_graph(tmp_path):
    ledger = AkashicLedger(str(tmp_path / "empty.db"))
    out = await ledger.to_prov_jsonld("main")
    parsed = json.loads(out)
    assert parsed["@graph"] == []
    assert parsed["@id"] == "urn:sum:audit"


@pytest.mark.asyncio
async def test_single_mint_event_yields_activity_and_entity(tmp_path):
    ledger = AkashicLedger(str(tmp_path / "single.db"))
    await ledger.append_event(
        "MINT",
        prime=7,
        axiom_key="alice||likes||cat",
        branch="main",
        source_url="https://example.org",
        confidence=0.9,
        ingested_at="2026-04-17T20:00:00+00:00",
    )
    out = await ledger.to_prov_jsonld("main")
    parsed = json.loads(out)
    types = {n.get("@type") for n in parsed["@graph"]}
    assert "prov:Activity" in types
    assert "prov:Entity" in types


@pytest.mark.asyncio
async def test_multiple_events_chain_via_prev_seq_id(tmp_path):
    ledger = AkashicLedger(str(tmp_path / "chain.db"))
    for i, axiom in enumerate([
        "alice||likes||cat",
        "bob||owns||dog",
        "carol||plays||piano",
    ]):
        await ledger.append_event(
            "MINT", prime=2 * i + 3, axiom_key=axiom, branch="main"
        )
    out = await ledger.to_prov_jsonld("main")
    parsed = json.loads(out)
    activities = [
        n for n in parsed["@graph"] if n.get("@type") == "prov:Activity"
    ]
    assert len(activities) == 3
    # The second and third activities should link back via prov:wasInformedBy
    assert "prov:wasInformedBy" in activities[1]
    assert "prov:wasInformedBy" in activities[2]


@pytest.mark.asyncio
async def test_branch_isolation(tmp_path):
    ledger = AkashicLedger(str(tmp_path / "branches.db"))
    await ledger.append_event(
        "MINT", prime=3, axiom_key="a||b||c", branch="main"
    )
    await ledger.append_event(
        "MINT", prime=5, axiom_key="d||e||f", branch="experimental"
    )
    main_graph = json.loads(await ledger.to_prov_jsonld("main"))
    exp_graph = json.loads(await ledger.to_prov_jsonld("experimental"))
    # Each branch contains exactly its own events + entities
    assert len([n for n in main_graph["@graph"] if n.get("@type") == "prov:Activity"]) == 1
    assert len([n for n in exp_graph["@graph"] if n.get("@type") == "prov:Activity"]) == 1
    # Entities are distinct
    main_keys = {
        n.get("https://sum.ototao.dev/ns/v1#axiomKey")
        for n in main_graph["@graph"]
        if n.get("@type") == "prov:Entity"
    }
    assert "a||b||c" in main_keys
    assert "d||e||f" not in main_keys
