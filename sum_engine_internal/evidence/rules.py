"""Inference rules for non-leaf evidence chains.

Where ``chain.EvidenceLink`` records leaf evidence (axiom →
source-text provenance), this module derives NEW claims from
existing ones via operator-curated rules. Each derived claim
gets a chain link with ``derivation_rule`` and ``derived_from``
populated — non-leaf evidence the substrate can verify
without trusting the producer.

v0 ships ONE well-defined rule:

  - **TransitiveClosureRule** — for any predicate ``p`` declared
    ``TRANSITIVE`` in ``SUBSTRATE_PREDICATE_LIBRARY`` (currently:
    ``contain``), derive ``(X, p, Z)`` from ``(X, p, Y) ∧
    (Y, p, Z)``. Soundness validated by the operator's curation
    discipline (transitive declarations are conservative — only
    rigorous spatial/containment relations).

The rule is applied to fixpoint: derived claims feed back into
the next iteration until no new claims are produced. The fixpoint
is guaranteed to terminate because the universe of subjects /
objects is fixed (the rule doesn't introduce new entities) and
the predicate is fixed per-rule application.

Future rules (slot reserved):
  - SymmetricClosureRule (no symmetric predicates declared yet).
  - InverseRule (would need an `INVERSE_OF` declaration in the
    predicate library).
  - Lean-4 entailment certs attach to each non-leaf link via the
    ``derivation_rule`` id.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional, Protocol

from sum_engine_internal.evidence.chain import EvidenceLink
from sum_engine_internal.infrastructure.provenance import ProvenanceRecord


def _claim_id(s: str, p: str, o: str) -> str:
    """Canonical claim id matching the substrate's axiom format."""
    return f"{s}||{p}||{o}".lower()


def _split_claim(claim: str) -> tuple[str, str, str]:
    parts = claim.split("||")
    if len(parts) != 3:
        raise ValueError(f"claim {claim!r} is not canonical 's||p||o' shape")
    return parts[0], parts[1], parts[2]


class InferenceRule(Protocol):
    """One inference rule. ``id`` names it for the chain link;
    ``derive_from`` runs one pass over the input claims and
    returns newly-derived (claim_id, supporting_claim_ids)
    pairs. The composer drives the fixpoint by re-invoking
    until ``derive_from`` returns empty."""

    id: str

    def derive_from(
        self, claims: frozenset[str],
    ) -> Iterable[tuple[str, tuple[str, ...]]]:
        ...


@dataclass(frozen=True, slots=True)
class TransitiveClosureRule:
    """Derive (X, p, Z) from (X, p, Y) ∧ (Y, p, Z) for any p
    declared TRANSITIVE in the predicate library.

    ``transitive_predicates`` is the set of predicates this rule
    is permitted to close over. Operator-supplied; defaults to
    pulling from ``SUBSTRATE_PREDICATE_LIBRARY`` at compose time
    (lazy import to avoid load-time coupling)."""

    id: str = "transitive_closure"
    transitive_predicates: frozenset[str] = frozenset()

    @classmethod
    def from_substrate_library(cls) -> "TransitiveClosureRule":
        """Build with the default substrate-curated transitive set."""
        from sum_engine_internal.research.smt_consistency import (
            SUBSTRATE_PREDICATE_LIBRARY,
        )
        from sum_engine_internal.research.smt_consistency.consistency import (
            PredicateProperty as P,
        )
        transitive = frozenset(
            pred for pred, props in SUBSTRATE_PREDICATE_LIBRARY.items()
            if P.TRANSITIVE in props
        )
        return cls(transitive_predicates=transitive)

    def derive_from(
        self, claims: frozenset[str],
    ) -> Iterable[tuple[str, tuple[str, ...]]]:
        # Index by predicate then subject for the inner-loop join.
        # Format: by_pred[p][subject] = list[(object, claim_id)]
        by_pred: dict[str, dict[str, list[tuple[str, str]]]] = {}
        for c in claims:
            try:
                s, p, o = _split_claim(c)
            except ValueError:
                continue
            if p not in self.transitive_predicates:
                continue
            by_pred.setdefault(p, {}).setdefault(s, []).append((o, c))
        for p, by_subj in by_pred.items():
            for s, s_succ in by_subj.items():
                for y, parent_claim in s_succ:
                    # y is the intermediate node — look up its successors
                    for z, child_claim in by_pred[p].get(y, []):
                        if z == s:
                            # Skip self-loop closures (irreflexive +
                            # transitive predicates would flag a
                            # cycle as inconsistent; not our job here)
                            continue
                        derived = _claim_id(s, p, z)
                        if derived in claims:
                            continue
                        yield derived, (parent_claim, child_claim)


def _derived_provenance(
    parent_provenance: ProvenanceRecord,
    rule_id: str,
) -> ProvenanceRecord:
    """For derived claims, the ProvenanceRecord points at the
    FIRST supporting claim's source — the derivation has no source
    text of its own, but per-record invariants still need a valid
    source. We tag the extractor_id to make derivation explicit.

    text_excerpt is replaced with a short "derived" marker so
    consumers can cheaply distinguish leaf from non-leaf even if
    they don't know about the derivation_rule field."""
    return ProvenanceRecord(
        source_uri=parent_provenance.source_uri,
        byte_start=parent_provenance.byte_start,
        byte_end=parent_provenance.byte_end,
        extractor_id=f"sum.evidence:rule={rule_id}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        text_excerpt=f"[derived: {rule_id}]",
    )


def derive_non_leaf_links(
    leaf_links: list[EvidenceLink],
    rules: list[InferenceRule],
    *,
    max_iterations: int = 16,
) -> list[EvidenceLink]:
    """Apply ``rules`` to fixpoint over the leaf-link claim set,
    returning the new (non-leaf) links discovered.

    Each derived link's ``provenance`` is cloned from the FIRST
    supporting claim's record (with extractor_id retagged to
    name the rule). The actual logical derivation is captured by
    ``derivation_rule`` + ``derived_from``.

    Hard cap at ``max_iterations`` to bound the fixpoint —
    transitive closure on a finite set converges in <= O(n) but
    a malformed rule could blow up; the cap is a safety net.

    The leaf-link claim set is treated as a frozenset (claims, not
    links) so the rule engine doesn't accidentally derive the same
    claim from two different leaf links — the chain stays minimal.
    """
    leaf_provenance: dict[str, ProvenanceRecord] = {
        link.claim: link.provenance for link in leaf_links
    }
    all_claims: set[str] = set(leaf_provenance.keys())
    derived: dict[str, EvidenceLink] = {}

    for _ in range(max_iterations):
        new_this_pass: list[tuple[str, tuple[str, ...], str]] = []
        for rule in rules:
            for derived_claim, supports in rule.derive_from(
                frozenset(all_claims),
            ):
                if derived_claim in all_claims:
                    continue
                new_this_pass.append((derived_claim, supports, rule.id))
        if not new_this_pass:
            break
        for derived_claim, supports, rule_id in new_this_pass:
            if derived_claim in all_claims:
                continue
            # Provenance: borrow from the FIRST support's source.
            # If a support is itself derived, follow back to a leaf
            # — derived provenance pointing at derived provenance
            # would obscure the actual source.
            anchor = supports[0]
            visited: set[str] = set()
            while anchor not in leaf_provenance:
                if anchor in visited:
                    break  # cycle safety; shouldn't happen
                visited.add(anchor)
                anchor_link = derived.get(anchor)
                if anchor_link is None or not anchor_link.derived_from:
                    break
                anchor = anchor_link.derived_from[0]
            base_prov = leaf_provenance.get(
                anchor, leaf_links[0].provenance,
            )
            derived[derived_claim] = EvidenceLink(
                claim=derived_claim,
                provenance=_derived_provenance(base_prov, rule_id),
                derived_from=tuple(supports),
                derivation_rule=rule_id,
            )
            all_claims.add(derived_claim)

    return list(derived.values())
