"""Evidence-chain layer over CanonicalBundle.

Pins the structural-claims surface (layer 2 above the wires arc):

  - EvidenceLink.to_dict shape
  - verify_chain_well_formed: canonical claim shape, no duplicate
    (claim, byte_range) 4-tuples, distinct byte ranges for the
    same claim are OK
  - verify_chain_covers_axioms: every bundle axiom has at least
    one supporting link
  - compose_bundle_with_evidence: round-trip import succeeds, the
    chain field is present + well-formed, signature unaffected
    (OUTSIDE the signed payload — same discipline as wires #1-#6)
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sum_engine_internal.evidence import (
    EvidenceLink, EvidenceChainError,
    compose_bundle_with_evidence,
    verify_chain_well_formed, verify_chain_covers_axioms,
)
from sum_engine_internal.infrastructure.provenance import ProvenanceRecord


def _prov(uri="sha256:" + "ab" * 32, byte_start=0, byte_end=10, excerpt="x"):
    return ProvenanceRecord(
        source_uri=uri,
        byte_start=byte_start,
        byte_end=byte_end,
        extractor_id="sum.test:v1",
        timestamp=datetime.now(timezone.utc).isoformat(),
        text_excerpt=excerpt,
    )


# -- EvidenceLink.to_dict ---------------------------------------------


def test_evidence_link_to_dict_shape():
    link = EvidenceLink(claim="alice||likes||cats", provenance=_prov())
    d = link.to_dict()
    assert d["claim"] == "alice||likes||cats"
    assert isinstance(d["provenance"], dict)
    for k in ("source_uri", "byte_start", "byte_end", "extractor_id",
              "timestamp", "text_excerpt", "schema_version"):
        assert k in d["provenance"]


# -- verify_chain_well_formed -----------------------------------------


def test_well_formed_accepts_valid_chain():
    links = [
        EvidenceLink(claim="alice||likes||cats", provenance=_prov(byte_start=0, byte_end=10)),
        EvidenceLink(claim="bob||owns||dog", provenance=_prov(byte_start=10, byte_end=20)),
    ]
    verify_chain_well_formed(links)  # no raise


def test_well_formed_rejects_non_canonical_claim_shape():
    links = [
        EvidenceLink(claim="alice likes cats", provenance=_prov()),
    ]
    with pytest.raises(EvidenceChainError, match="canonical"):
        verify_chain_well_formed(links)


def test_well_formed_rejects_empty_components():
    links = [
        EvidenceLink(claim="alice||||cats", provenance=_prov()),
    ]
    with pytest.raises(EvidenceChainError):
        verify_chain_well_formed(links)


def test_well_formed_rejects_duplicate_claim_and_byte_range():
    links = [
        EvidenceLink(claim="alice||likes||cats", provenance=_prov(byte_start=0, byte_end=10)),
        EvidenceLink(claim="alice||likes||cats", provenance=_prov(byte_start=0, byte_end=10)),
    ]
    with pytest.raises(EvidenceChainError, match="duplicate"):
        verify_chain_well_formed(links)


def test_well_formed_accepts_same_claim_at_different_byte_ranges():
    """Same fact extracted from two different sentences is two
    pieces of independent evidence — distinct byte ranges, so
    NOT a duplicate."""
    links = [
        EvidenceLink(claim="alice||likes||cats", provenance=_prov(byte_start=0, byte_end=10)),
        EvidenceLink(claim="alice||likes||cats", provenance=_prov(byte_start=20, byte_end=35)),
    ]
    verify_chain_well_formed(links)  # no raise


# -- verify_chain_covers_axioms ---------------------------------------


def test_covers_axioms_passes_when_every_axiom_supported():
    links = [
        EvidenceLink(claim="alice||likes||cats", provenance=_prov()),
        EvidenceLink(claim="bob||owns||dog", provenance=_prov(byte_start=10, byte_end=20)),
    ]
    axioms = ["alice||likes||cats", "bob||owns||dog"]
    verify_chain_covers_axioms(links, axioms)  # no raise


def test_covers_axioms_rejects_uncovered_axiom():
    links = [
        EvidenceLink(claim="alice||likes||cats", provenance=_prov()),
    ]
    axioms = ["alice||likes||cats", "bob||owns||dog"]
    with pytest.raises(EvidenceChainError, match="not covered"):
        verify_chain_covers_axioms(links, axioms)


def test_covers_axioms_allows_more_links_than_axioms():
    """A single axiom can have multiple supporting links (same fact
    from multiple source sentences); coverage just needs ≥1."""
    links = [
        EvidenceLink(claim="alice||likes||cats", provenance=_prov(byte_start=0, byte_end=10)),
        EvidenceLink(claim="alice||likes||cats", provenance=_prov(byte_start=20, byte_end=35)),
    ]
    axioms = ["alice||likes||cats"]
    verify_chain_covers_axioms(links, axioms)


# -- compose_bundle_with_evidence (integration) ------------------------


@pytest.fixture
def codec():
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec
    algebra = GodelStateAlgebra()
    gen = AutoregressiveTomeGenerator(algebra)
    return CanonicalCodec(algebra, gen, signing_key="evidence_test_key")


_TEXT = (
    "Alice likes cats. Bob owns a dog. Carol writes books. "
    "Dave teaches mathematics."
)


def test_composed_bundle_includes_evidence_chain_field(codec):
    bundle = compose_bundle_with_evidence(codec, _TEXT, branch="t")
    assert "axiom_evidence_chain" in bundle
    chain = bundle["axiom_evidence_chain"]
    assert isinstance(chain, list)
    assert len(chain) >= 1
    for link in chain:
        assert "claim" in link
        assert "provenance" in link
        assert "byte_start" in link["provenance"]
        assert "source_uri" in link["provenance"]


def test_composed_bundle_chain_covers_every_axiom(codec):
    """The composer must verify coverage before returning — if it
    didn't, the chain field would be present but soundness-broken."""
    bundle = compose_bundle_with_evidence(codec, _TEXT, branch="t")
    chain_claims = {link["claim"] for link in bundle["axiom_evidence_chain"]}
    # Every axiom in the bundle must appear in chain claims
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    state = int(bundle["state_integer"])
    axioms = codec.tome_generator.extract_active_axioms(state)
    for ax in axioms:
        assert ax in chain_claims, f"axiom {ax!r} uncovered"


def test_composed_bundle_round_trips(codec):
    """The bundle must remain structurally valid for the codec's
    standard import path (chain field is OUTSIDE the signed
    payload — must not affect signature verification)."""
    bundle = compose_bundle_with_evidence(codec, _TEXT, branch="t")
    state_int = int(bundle["state_integer"])
    assert codec.import_bundle(bundle) == state_int


def test_composed_bundle_signature_unaffected_by_chain(codec):
    """Outside-signed-payload invariant — signing payload is
    canonical_tome|state_integer|timestamp; chain field must not
    enter it."""
    bundle = compose_bundle_with_evidence(codec, _TEXT, branch="t")
    sig_recompute = codec._sign(
        bundle["canonical_tome"],
        bundle["state_integer"],
        bundle["timestamp"],
    )
    assert sig_recompute == bundle["signature"]


def test_composed_bundle_chain_well_formed(codec):
    """Re-deserialise the dict-encoded chain and verify well-
    formedness on the wire."""
    bundle = compose_bundle_with_evidence(codec, _TEXT, branch="t")
    # Reconstruct EvidenceLink objects from the wire-encoded dicts
    links = [
        EvidenceLink(
            claim=raw["claim"],
            provenance=ProvenanceRecord(**{
                k: v for k, v in raw["provenance"].items()
            }),
        )
        for raw in bundle["axiom_evidence_chain"]
    ]
    verify_chain_well_formed(links)  # no raise
