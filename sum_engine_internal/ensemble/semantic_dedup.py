"""
Semantic Deduplicator — Near-Duplicate Axiom Detection

Phase 25: Prevents state bloat from semantically equivalent axioms
that differ only in surface text (e.g., "orbits" vs "revolves_around",
"New York" vs "new york").

Uses zero-cost string similarity:
    1. Normalization — lowercase, strip, collapse whitespace
    2. Jaccard token overlap — set intersection / union of words
    3. Levenshtein ratio — edit distance / max length

No LLM or embedding calls required.

Usage:
    dedup = SemanticDeduplicator()
    result = dedup.deduplicate(
        "earth||revolves_around||sun",
        existing_axioms=["earth||orbits||sun", "mars||has||moons"],
    )
    # result = DedupResult(canonical="earth||orbits||sun", is_duplicate=True, ...)

Author: ototao
License: Apache License 2.0
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


# ─── Predicate Synonym Groups ────────────────────────────────────────
# Common predicates that are semantically equivalent.
# Each group maps to its canonical form (first element).

PREDICATE_SYNONYMS = {
    "orbits": "orbits",
    "revolves_around": "orbits",
    "circles": "orbits",
    "goes_around": "orbits",
    "is_a": "is_a",
    "is_an": "is_a",
    "is_type": "is_a",
    "type_of": "is_a",
    "kind_of": "is_a",
    "has": "has",
    "has_a": "has",
    "possesses": "has",
    "owns": "has",
    "contains": "contains",
    "includes": "contains",
    "has_part": "contains",
    "located_in": "located_in",
    "lives_in": "located_in",
    "resides_in": "located_in",
    "is_in": "located_in",
    "situated_in": "located_in",
    "created_by": "created_by",
    "made_by": "created_by",
    "authored_by": "created_by",
    "written_by": "created_by",
    "invented_by": "created_by",
    "causes": "causes",
    "leads_to": "causes",
    "results_in": "causes",
    "produces": "causes",
    "part_of": "part_of",
    "belongs_to": "part_of",
    "member_of": "part_of",
    "component_of": "part_of",
}

DEFAULT_THRESHOLD = 0.80


@dataclass
class DedupResult:
    """Result of a deduplication check."""
    canonical_key: str       # The resolved canonical axiom key
    is_duplicate: bool       # True if a near-duplicate was found
    duplicate_of: str        # The existing axiom it duplicates ('' if none)
    similarity: float        # Similarity score (0.0 if not duplicate)
    method: str              # Which signal triggered: 'exact'|'predicate'|'fuzzy'|'none'


class SemanticDeduplicator:
    """Zero-cost semantic deduplication engine.

    Detects near-duplicate axioms using layered string similarity:
    1. Exact match (prime identity — already handled by algebra)
    2. Predicate synonym canonicalization
    3. Fuzzy matching via Jaccard + Levenshtein
    """

    def normalize(self, axiom_key: str) -> str:
        """Normalize axiom key to canonical form.

        - Lowercase
        - Strip whitespace
        - Collapse multiple underscores/spaces
        - Canonicalize predicate via synonym table
        """
        parts = axiom_key.split("||")
        if len(parts) != 3:
            return axiom_key.strip().lower()

        s, p, o = [x.strip().lower() for x in parts]

        # Collapse whitespace and underscores
        s = re.sub(r"[\s_]+", "_", s).strip("_")
        p = re.sub(r"[\s_]+", "_", p).strip("_")
        o = re.sub(r"[\s_]+", "_", o).strip("_")

        # Canonicalize predicate
        p = PREDICATE_SYNONYMS.get(p, p)

        return f"{s}||{p}||{o}"

    @staticmethod
    def _tokenize(s: str) -> set:
        """Split a string into word tokens."""
        return set(re.split(r"[_\|\s]+", s.lower()))

    @staticmethod
    def jaccard_similarity(a: str, b: str) -> float:
        """Jaccard token-overlap coefficient.

        |A ∩ B| / |A ∪ B|
        """
        tokens_a = SemanticDeduplicator._tokenize(a)
        tokens_b = SemanticDeduplicator._tokenize(b)

        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    @staticmethod
    def levenshtein_ratio(a: str, b: str) -> float:
        """Edit distance ratio: 1.0 = identical, 0.0 = completely different.

        Uses dynamic programming Levenshtein distance.
        """
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0

        m, n = len(a), len(b)

        # Optimize: only keep two rows
        prev = list(range(n + 1))
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            curr[0] = i
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,      # deletion
                    curr[j - 1] + 1,   # insertion
                    prev[j - 1] + cost  # substitution
                )
            prev, curr = curr, [0] * (n + 1)

        distance = prev[n]
        return 1.0 - (distance / max(m, n))

    def combined_similarity(self, a: str, b: str) -> float:
        """Weighted combination of Jaccard and Levenshtein.

        Weights: 60% Jaccard (semantic), 40% Levenshtein (syntactic).
        """
        j = self.jaccard_similarity(a, b)
        l = self.levenshtein_ratio(a, b)
        return 0.6 * j + 0.4 * l

    def find_near_duplicates(
        self,
        axiom_key: str,
        existing_axioms: List[str],
        threshold: float = DEFAULT_THRESHOLD,
    ) -> List[tuple]:
        """Find axioms similar to axiom_key above the threshold.

        Returns list of (existing_axiom, similarity, method) sorted by
        similarity descending.
        """
        normalized = self.normalize(axiom_key)
        results = []

        for existing in existing_axioms:
            norm_existing = self.normalize(existing)

            # Layer 1: Exact after normalization
            if normalized == norm_existing:
                results.append((existing, 1.0, "exact"))
                continue

            # Layer 2: Check predicate synonym match
            parts_new = normalized.split("||")
            parts_ex = norm_existing.split("||")
            if (len(parts_new) == 3 and len(parts_ex) == 3
                    and parts_new[0] == parts_ex[0]
                    and parts_new[2] == parts_ex[2]
                    and parts_new[1] == parts_ex[1]):
                # After canonicalization they match
                results.append((existing, 1.0, "predicate"))
                continue

            # Layer 3: Fuzzy similarity
            sim = self.combined_similarity(normalized, norm_existing)
            if sim >= threshold:
                results.append((existing, sim, "fuzzy"))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def deduplicate(
        self,
        axiom_key: str,
        existing_axioms: List[str],
        threshold: float = DEFAULT_THRESHOLD,
    ) -> DedupResult:
        """Check if axiom_key is a near-duplicate of any existing axiom.

        Returns a DedupResult with the canonical key and match info.
        """
        normalized = self.normalize(axiom_key)
        duplicates = self.find_near_duplicates(
            axiom_key, existing_axioms, threshold
        )

        if duplicates:
            best_match, sim, method = duplicates[0]
            logger.info(
                "Dedup: '%s' → duplicate of '%s' (%.2f, %s)",
                axiom_key, best_match, sim, method,
            )
            return DedupResult(
                canonical_key=self.normalize(best_match),
                is_duplicate=True,
                duplicate_of=best_match,
                similarity=sim,
                method=method,
            )

        return DedupResult(
            canonical_key=normalized,
            is_duplicate=False,
            duplicate_of="",
            similarity=0.0,
            method="none",
        )
