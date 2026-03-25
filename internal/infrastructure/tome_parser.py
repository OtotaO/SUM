"""
Canonical Tome Parser — Centralized Tome Parsing Subsystem

Extracts axiom keys from the canonical tome text representation.
The canonical tome format is:
    - subject||predicate||object [prime: NNN]

This module centralizes parsing logic that was previously inline
in quantum_router.py, ensuring a single source of truth for the
canonical tome format contract.

Author: ototao
License: Apache License 2.0
"""

from typing import List, Tuple, Optional


def parse_axiom_key(axiom_key: str) -> Optional[Tuple[str, str, str]]:
    """Parse a '||'-delimited axiom key into (subject, predicate, object).

    Returns None if the key doesn't have exactly 3 parts.
    """
    parts = axiom_key.split("||")
    if len(parts) == 3:
        return parts[0].strip(), parts[1].strip(), parts[2].strip()
    return None


def parse_canonical_tome(tome_text: str) -> List[Tuple[str, str, str]]:
    """Parse a canonical tome string into a list of (subject, predicate, object) tuples.

    Canonical format per line:
        - subject||predicate||object [prime: NNN]

    The optional [prime: NNN] annotation is stripped before parsing.
    Lines not matching the canonical format are silently skipped.

    Returns:
        List of (subject, predicate, object) tuples, lowercased and stripped.
    """
    axioms = []
    for line in tome_text.splitlines():
        line = line.strip()
        # Canonical format: "- subject||predicate||object [prime: NNN]"
        if line.startswith("- ") and "||" in line:
            axiom_part = line[2:].strip()
            # Strip optional prime annotation
            if " [prime:" in axiom_part:
                axiom_part = axiom_part[:axiom_part.index(" [prime:")].strip()
            parsed = parse_axiom_key(axiom_part)
            if parsed:
                s, p, o = parsed
                axioms.append((s.lower(), p.lower(), o.lower()))
    return axioms
