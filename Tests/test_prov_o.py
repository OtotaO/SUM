"""Tests for sum_engine_internal.infrastructure.prov_o — PROV-O JSON-LD emission."""
from __future__ import annotations

import json

from sum_engine_internal.infrastructure.prov_o import (
    SUM_VOCAB,
    _axiom_iri,
    _event_iri,
    _source_iri,
    dump_prov_jsonld,
    event_to_prov_o,
    events_to_prov_graph,
)


# ─── IRI helpers ──────────────────────────────────────────────────────


class TestIris:
    def test_axiom_iri_percent_encodes(self) -> None:
        iri = _axiom_iri("alice||age||30")
        assert iri.startswith("urn:sum:axiom:")
        assert "alice" in iri
        assert "%7C%7C" in iri  # '||' encoded

    def test_event_iri(self) -> None:
        assert _event_iri(42) == "urn:sum:event:42"

    def test_source_iri_http_passthrough(self) -> None:
        assert _source_iri("https://nasa.gov/data") == "https://nasa.gov/data"

    def test_source_iri_non_http_wraps(self) -> None:
        iri = _source_iri("paper.pdf")
        assert iri.startswith("urn:sum:source:")

    def test_source_iri_empty(self) -> None:
        assert _source_iri("") == ""


# ─── Single event emission ────────────────────────────────────────────


class TestEventToProvO:
    def test_mint_event_emits_activity_and_entity(self) -> None:
        event = {
            "seq_id": 1,
            "operation": "MINT",
            "prime": "7",
            "axiom_key": "alice||age||30",
            "branch": "main",
            "source_url": "https://nasa.gov/alice",
            "confidence": 0.85,
            "ingested_at": "2026-04-17T20:00:00+00:00",
            "prev_hash": "abc123",
        }
        nodes = event_to_prov_o(event)
        assert len(nodes) == 2

        activity = next(n for n in nodes if n["@type"] == "prov:Activity")
        assert activity[f"{SUM_VOCAB}operation"] == "MINT"
        assert activity[f"{SUM_VOCAB}branch"] == "main"
        assert activity[f"{SUM_VOCAB}prime"] == "7"
        assert activity["prov:generatedAtTime"] == "2026-04-17T20:00:00+00:00"
        assert activity[f"{SUM_VOCAB}prevHash"] == "abc123"
        assert activity["prov:used"]["@id"] == _axiom_iri("alice||age||30")

        entity = next(n for n in nodes if n["@type"] == "prov:Entity")
        assert entity[f"{SUM_VOCAB}axiomKey"] == "alice||age||30"
        assert entity["prov:hadPrimarySource"]["@id"] == "https://nasa.gov/alice"
        assert entity[f"{SUM_VOCAB}confidence"] == 0.85

    def test_event_without_axiom_emits_only_activity(self) -> None:
        event = {
            "seq_id": 5,
            "operation": "SYNC",
            "prime": "1",
            "branch": "main",
        }
        nodes = event_to_prov_o(event)
        assert len(nodes) == 1
        assert nodes[0]["@type"] == "prov:Activity"

    def test_event_with_prev_seq_id_links_activities(self) -> None:
        event = {
            "seq_id": 3,
            "operation": "MUL",
            "prime": "11",
            "axiom_key": "bob||likes||jazz",
            "branch": "main",
            "prev_seq_id": 2,
        }
        nodes = event_to_prov_o(event)
        activity = next(n for n in nodes if n["@type"] == "prov:Activity")
        assert activity["prov:wasInformedBy"]["@id"] == _event_iri(2)

    def test_event_without_source_url_omits_primary_source(self) -> None:
        event = {
            "seq_id": 1,
            "operation": "MINT",
            "prime": "7",
            "axiom_key": "x||y||z",
        }
        nodes = event_to_prov_o(event)
        entity = next(n for n in nodes if n["@type"] == "prov:Entity")
        assert "prov:hadPrimarySource" not in entity


# ─── Graph composition ────────────────────────────────────────────────


class TestEventsToProvGraph:
    def test_single_event_graph(self) -> None:
        events = [{
            "seq_id": 1,
            "operation": "MINT",
            "prime": "7",
            "axiom_key": "alice||likes||cat",
        }]
        g = events_to_prov_graph(events)
        assert "@context" in g
        assert "@graph" in g
        assert len(g["@graph"]) == 2  # 1 activity + 1 entity

    def test_entity_deduplication(self) -> None:
        events = [
            {
                "seq_id": 1,
                "operation": "MINT",
                "prime": "7",
                "axiom_key": "alice||likes||cat",
                "source_url": "https://nasa.gov",
            },
            {
                "seq_id": 2,
                "operation": "MUL",
                "prime": "7",
                "axiom_key": "alice||likes||cat",
            },
        ]
        g = events_to_prov_graph(events)
        # 2 activities + 1 deduplicated entity
        assert len(g["@graph"]) == 3
        entities = [n for n in g["@graph"] if n["@type"] == "prov:Entity"]
        assert len(entities) == 1

    def test_context_carries_prov_namespace(self) -> None:
        g = events_to_prov_graph([])
        assert g["@context"]["prov"] == "http://www.w3.org/ns/prov#"
        assert g["@context"]["sum"].startswith("https://")

    def test_graph_iri_customizable(self) -> None:
        g = events_to_prov_graph([], graph_iri="urn:sum:test-graph")
        assert g["@id"] == "urn:sum:test-graph"


# ─── JSON-LD serialization ────────────────────────────────────────────


class TestDumpProvJsonld:
    def test_roundtrips_via_json(self) -> None:
        events = [{
            "seq_id": 1,
            "operation": "MINT",
            "prime": "7",
            "axiom_key": "alice||likes||cat",
            "source_url": "https://example.org",
            "ingested_at": "2026-04-17T20:00:00+00:00",
        }]
        text = dump_prov_jsonld(events)
        parsed = json.loads(text)
        assert parsed["@id"] == "urn:sum:audit"
        assert any(
            n["@type"] == "prov:Activity" for n in parsed["@graph"]
        )

    def test_empty_events_produces_empty_graph(self) -> None:
        text = dump_prov_jsonld([])
        parsed = json.loads(text)
        assert parsed["@graph"] == []

    def test_indent_none_produces_compact_output(self) -> None:
        text = dump_prov_jsonld([], indent=None)
        assert "\n" not in text or text.count("\n") <= 1
