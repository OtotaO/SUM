"""
PROV-O Provenance Adapter — W3C-Standards Emission for Akashic Ledger Events

Serializes semantic_events rows into W3C PROV-O (JSON-LD) so any PROV-compliant
consumer can validate SUM's provenance chain without SUM-specific tooling.

Each Akashic event maps to:
    prov:Activity   — the operation (MINT / MUL / DIV / SYNC / DEDUCED)
    prov:Entity     — the axiom affected by the operation
    prov:used       — Activity → Entity link
    prov:wasInformedBy — Activity → prior Activity (Merkle chain)
    prov:generatedAtTime — ingested_at timestamp
    prov:hadPrimarySource — source_url when present

SUM-specific attributes (prime, branch, prev_hash, confidence) are attached
as namespaced properties under the "sum:" prefix. They are optional for
strict PROV-O consumers but preserved for round-trip fidelity.

This is Polytaxis Bucket A §4: PROV-O / PROV-STAR alignment. Namespace
mapping only — no data reorganization of the underlying ledger.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence
from urllib.parse import quote

PROV_CONTEXT = "https://www.w3.org/ns/prov.jsonld"
SUM_NS = "urn:sum:"
SUM_VOCAB = "https://sum.ototao.dev/ns/v1#"


def _axiom_iri(axiom_key: str) -> str:
    """Deterministic IRI for an axiom: urn:sum:axiom:<percent-encoded-key>."""
    return f"{SUM_NS}axiom:{quote(axiom_key, safe='')}"


def _event_iri(seq_id: int) -> str:
    """Deterministic IRI for a ledger event by sequence id."""
    return f"{SUM_NS}event:{seq_id}"


def _source_iri(url: str) -> str:
    """If the source looks like a URL, use it verbatim; otherwise wrap as urn."""
    stripped = url.strip()
    if not stripped:
        return ""
    if stripped.startswith("http://") or stripped.startswith("https://"):
        return stripped
    return f"{SUM_NS}source:{quote(stripped, safe='')}"


def event_to_prov_o(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert one ledger event row to PROV-O-in-JSON-LD node list.

    Accepts a mapping with keys: seq_id, operation, prime, axiom_key,
    branch, source_url, confidence, ingested_at, prev_hash, (prev_seq_id).

    Returns a list of PROV-O nodes suitable for inclusion in a named graph.
    Always emits one prov:Activity and, when axiom_key is non-empty, one
    prov:Entity.
    """
    seq_id = int(event["seq_id"])
    operation = str(event["operation"])
    axiom_key = str(event.get("axiom_key", ""))
    source_url = str(event.get("source_url", ""))
    ingested_at = str(event.get("ingested_at", ""))
    branch = str(event.get("branch", "main"))
    prev_hash = str(event.get("prev_hash", ""))
    prev_seq_id = event.get("prev_seq_id")
    prime = str(event.get("prime", ""))
    confidence = event.get("confidence", None)

    activity: dict[str, Any] = {
        "@id": _event_iri(seq_id),
        "@type": "prov:Activity",
        f"{SUM_VOCAB}operation": operation,
        f"{SUM_VOCAB}branch": branch,
    }
    if prime:
        activity[f"{SUM_VOCAB}prime"] = prime
    if ingested_at:
        activity["prov:generatedAtTime"] = ingested_at
    if prev_hash:
        activity[f"{SUM_VOCAB}prevHash"] = prev_hash
    if prev_seq_id is not None:
        activity["prov:wasInformedBy"] = {"@id": _event_iri(int(prev_seq_id))}

    nodes: list[dict[str, Any]] = [activity]

    if axiom_key:
        entity_iri = _axiom_iri(axiom_key)
        activity["prov:used"] = {"@id": entity_iri}
        entity: dict[str, Any] = {
            "@id": entity_iri,
            "@type": "prov:Entity",
            f"{SUM_VOCAB}axiomKey": axiom_key,
        }
        src_iri = _source_iri(source_url)
        if src_iri:
            entity["prov:hadPrimarySource"] = {"@id": src_iri}
        if confidence is not None:
            entity[f"{SUM_VOCAB}confidence"] = float(confidence)
        nodes.append(entity)

    return nodes


def events_to_prov_graph(
    events: Sequence[Mapping[str, Any]],
    graph_iri: str = "urn:sum:audit",
) -> dict[str, Any]:
    """Convert a sequence of ledger events to a PROV-O JSON-LD graph.

    De-duplicates entity nodes by IRI so a single axiom referenced by
    multiple events appears once with aggregated provenance edges.
    """
    nodes: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for event in events:
        for node in event_to_prov_o(event):
            iri = node["@id"]
            if iri in nodes:
                existing = nodes[iri]
                for key, value in node.items():
                    if key in ("@id", "@type"):
                        continue
                    if key not in existing:
                        existing[key] = value
            else:
                nodes[iri] = dict(node)
                order.append(iri)

    return {
        "@context": {
            "prov": "http://www.w3.org/ns/prov#",
            "sum": SUM_VOCAB,
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        },
        "@id": graph_iri,
        "@graph": [nodes[iri] for iri in order],
    }


def dump_prov_jsonld(
    events: Sequence[Mapping[str, Any]],
    graph_iri: str = "urn:sum:audit",
    indent: int | None = 2,
) -> str:
    """Serialize ledger events as a PROV-O JSON-LD document string."""
    return json.dumps(
        events_to_prov_graph(events, graph_iri=graph_iri),
        indent=indent,
        sort_keys=False,
        ensure_ascii=False,
    )
