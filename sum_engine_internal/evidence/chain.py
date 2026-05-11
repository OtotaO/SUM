"""Evidence-chain primitives + composer.

EvidenceLink ties one bundle axiom claim to a ``ProvenanceRecord``
(source_uri + byte_range + extractor_id + timestamp + excerpt).
verify_chain_* check the chain is internally consistent and
covers the bundle's axioms. compose_bundle_with_evidence is the
batteries-included path: sieve a text, encode it, export the
bundle, attach the chain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from sum_engine_internal.infrastructure.provenance import ProvenanceRecord


class EvidenceChainError(ValueError):
    """Raised when an evidence chain fails a well-formedness or
    coverage invariant. Distinct from InvalidProvenanceError —
    that's per-record; this is per-chain."""


@dataclass(frozen=True, slots=True)
class EvidenceLink:
    """One claim in the evidence chain.

    ``claim`` is the canonical axiom string ``"s||p||o"`` (matching
    the substrate's internal axiom serialisation).
    ``provenance`` is the source-text ProvenanceRecord this axiom
    was extracted from.

    Future fields (slot reserved):
      - ``derived_from: tuple[str, ...]`` — claim IDs of supports
        when this claim is non-leaf (implied by other claims).
      - ``derivation_rule: str | None`` — id of the inference rule
        used. ``None`` for leaf claims.
    """

    claim: str
    provenance: ProvenanceRecord

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "provenance": self.provenance.to_dict(),
        }


def verify_chain_well_formed(links: Iterable[EvidenceLink]) -> None:
    """Check chain-level invariants.

    - Every ``claim`` must have the canonical ``"s||p||o"`` shape
      (3 non-empty components after split on ``"||"``).
    - No duplicate ``(claim, source_uri, byte_start, byte_end)``
      4-tuple — that would be a redundant attestation that adds
      no information. Distinct byte ranges for the same claim are
      OK (the same fact extracted from different sentences).

    Raises EvidenceChainError on the first violation; succeeds
    silently otherwise. ProvenanceRecord-level invariants are
    enforced at record construction, so they're trusted here."""
    seen: set[tuple[str, str, int, int]] = set()
    for i, link in enumerate(links):
        parts = link.claim.split("||")
        if len(parts) != 3 or not all(p.strip() for p in parts):
            raise EvidenceChainError(
                f"link {i}: claim {link.claim!r} is not canonical "
                f"'s||p||o' shape (got {len(parts)} parts)"
            )
        key = (
            link.claim,
            link.provenance.source_uri,
            link.provenance.byte_start,
            link.provenance.byte_end,
        )
        if key in seen:
            raise EvidenceChainError(
                f"link {i}: duplicate (claim, source_uri, byte_range) "
                f"4-tuple: {key!r}"
            )
        seen.add(key)


def verify_chain_covers_axioms(
    links: Iterable[EvidenceLink], axioms: Iterable[str],
) -> None:
    """Check every axiom has at least one supporting link.

    The chain may have MORE links than there are axioms (a single
    axiom can be supported by evidence from multiple source
    sentences), but every axiom must appear as ``link.claim`` for
    at least one link. Raises EvidenceChainError on the first
    uncovered axiom."""
    covered: set[str] = {link.claim for link in links}
    for axiom in axioms:
        if axiom not in covered:
            raise EvidenceChainError(
                f"axiom {axiom!r} is not covered by any evidence link"
            )


def compose_bundle_with_evidence(
    codec,
    text: str,
    *,
    branch: str = "main",
    source_uri: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> dict:
    """Sieve ``text`` → encode state → export bundle → attach chain.

    Runs ``DeterministicSieve.extract_with_provenance`` to get
    (triple, ProvenanceRecord) pairs, encodes them via the codec's
    algebra, exports the standard bundle, then builds + attaches
    the evidence chain as ``axiom_evidence_chain``.

    The returned bundle dict satisfies BOTH:
      - ``codec.import_bundle(bundle)`` round-trips to the same
        state integer (bundle is structurally valid).
      - ``verify_chain_well_formed`` + ``verify_chain_covers_axioms``
        succeed against the bundle's axioms (chain is sound).

    Raises EvidenceChainError if the chain doesn't cover every
    axiom in the bundle (would mean the codec dropped or added
    axioms relative to what the sieve produced — a load-bearing
    invariant of the substrate).
    """
    from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve

    sieve = DeterministicSieve()
    pairs = sieve.extract_with_provenance(
        text, source_uri=source_uri, timestamp=timestamp,
    )
    triples = [p[0] for p in pairs]
    state = codec.algebra.encode_chunk_state(triples)
    bundle = codec.export_bundle(state, branch=branch)

    links = [
        EvidenceLink(
            claim=f"{s}||{p}||{o}".lower(),
            provenance=record,
        )
        for (s, p, o), record in pairs
    ]
    verify_chain_well_formed(links)

    bundle_axioms = _bundle_active_axioms(codec, state)
    verify_chain_covers_axioms(links, bundle_axioms)

    bundle["axiom_evidence_chain"] = [link.to_dict() for link in links]
    return bundle


def _bundle_active_axioms(codec, state) -> list[str]:
    """Internal: read the active axioms a bundle would carry for
    a given state. Mirrors what canonical_codec does internally
    so we can verify the chain covers everything that lands in
    the signed payload."""
    return codec.tome_generator.extract_active_axioms(state)
