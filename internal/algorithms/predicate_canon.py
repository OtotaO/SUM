"""
Predicate Canonicalizer — Controlled Vocabulary for Semantic Primes

Maps free-form LLM predicates to a normalized vocabulary so that
semantically identical relationships produce the same Gödel prime.

Without this:
    "sun||causes||warmth"    → prime₁
    "sun||leads_to||warmth"  → prime₂    (different prime, same meaning!)

With this:
    "sun||causes||warmth"    → prime₁
    "sun||leads_to||warmth"  → prime₁    (same prime via canonicalization)

The canonical vocabulary is intentionally small — it maps the long tail
of LLM-generated predicates into the set that CausalDiscovery's
TRANSITIVE_PREDICATES and INVERSE_PREDICATES can reason over.

Author: ototao
License: Apache License 2.0
"""


# ─── Canonical Vocabulary ─────────────────────────────────────────────

CANONICAL_MAP: dict[str, str] = {
    # → causes family
    "leads_to":     "causes",
    "triggers":     "causes",
    "results_in":   "causes",
    "produces":     "causes",
    "generates":    "causes",
    "creates":      "causes",
    "induces":      "causes",
    "drives":       "causes",
    "provokes":     "causes",
    "elicits":      "causes",
    "yields":       "causes",
    "brings_about": "causes",

    # → inhibits family
    "prevents":     "inhibits",
    "blocks":       "inhibits",
    "stops":        "inhibits",
    "suppresses":   "inhibits",
    "reduces":      "inhibits",
    "hinders":      "inhibits",
    "impedes":      "inhibits",
    "decreases":    "inhibits",
    "diminishes":   "inhibits",
    "constrains":   "inhibits",

    # → implies family
    "suggests":     "implies",
    "indicates":    "implies",
    "means":        "implies",
    "entails":      "implies",
    "demonstrates": "implies",
    "shows":        "implies",
    "proves":       "implies",
    "evidences":    "implies",

    # → requires family
    "needs":        "requires",
    "depends_on":   "requires",
    "relies_on":    "requires",
    "necessitates": "requires",

    # → is_a family (taxonomy)
    "is_type_of":   "is_a",
    "is_kind_of":   "is_a",
    "belongs_to":   "is_a",
    "is_part_of":   "has_part",
    "contains":     "has_part",
    "includes":     "has_part",
    "comprises":    "has_part",

    # → has_property family
    "has":          "has_property",
    "possesses":    "has_property",
    "exhibits":     "has_property",
    "displays":     "has_property",
    "features":     "has_property",
    "characterized_by": "has_property",

    # → treats family (inverse of inhibits in CausalDiscovery)
    "cures":        "treats",
    "heals":        "treats",
    "remedies":     "treats",
    "alleviates":   "treats",
    "mitigates":    "treats",

    # → enables family
    "allows":       "enables",
    "permits":      "enables",
    "facilitates":  "enables",
    "supports":     "enables",
    "empowers":     "enables",

    # → uses family
    "utilizes":     "uses",
    "employs":      "uses",
    "applies":      "uses",
    "leverages":    "uses",

    # → located_in family
    "found_in":     "located_in",
    "exists_in":    "located_in",
    "resides_in":   "located_in",
    "occurs_in":    "located_in",
}

# The set of canonical predicates (roots) — these are never remapped
CANONICAL_PREDICATES: frozenset[str] = frozenset({
    "causes", "inhibits", "implies", "requires", "is_a",
    "has_part", "has_property", "treats", "enables", "uses",
    "located_in", "leads_to",
    # Keep CausalDiscovery's originals
    "solves",
})


def canonicalize(predicate: str) -> str:
    """
    Maps a free-form predicate string to its canonical form.

    If the predicate is already canonical or has no mapping,
    it is returned unchanged (the system remains open-world).

    Args:
        predicate: The raw predicate string (lowercased).

    Returns:
        The canonical predicate.
    """
    normalized = predicate.strip().lower().replace(" ", "_")
    return CANONICAL_MAP.get(normalized, normalized)
