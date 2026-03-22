#!/usr/bin/env python3
"""
Forensic Diff — Debug tool for Gödel State Integer divergence.

Given two state integers (or bundles), computes:
  - Missing axioms: in state_a but not state_b
  - Extra axioms: in state_b but not state_a
  - Shared axioms: in both

Usage:
    python scripts/forensic_diff.py <state_a> <state_b>
    python scripts/forensic_diff.py --bundles <bundle_a.json> <bundle_b.json>
    python scripts/forensic_diff.py --bundle <bundle.json> --state <integer>

Part of the Verifiability Fortress — Workstream 8 (Proof Artifacts).

Author: ototao
License: Apache License 2.0
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.ensemble.tome_generator import AutoregressiveTomeGenerator


def factorize_state(algebra: GodelStateAlgebra, state: int) -> dict:
    """
    Factorize a state integer into its component axiom primes.

    Returns:
        dict mapping axiom_key -> prime for all axioms dividing the state.
    """
    found = {}
    for axiom_key, prime in algebra.axiom_to_prime.items():
        if state > 1 and state % prime == 0:
            found[axiom_key] = prime
    return found


def diff_states(
    algebra: GodelStateAlgebra, state_a: int, state_b: int
) -> dict:
    """
    Compute the semantic diff between two Gödel State Integers.

    Returns:
        {
            "shared": {axiom_key: prime, ...},
            "only_a": {axiom_key: prime, ...},
            "only_b": {axiom_key: prime, ...},
            "gcd": int,
            "delta_a_to_b": int,
            "delta_b_to_a": int,
        }
    """
    gcd = math.gcd(state_a, state_b)

    axioms_a = factorize_state(algebra, state_a)
    axioms_b = factorize_state(algebra, state_b)

    shared = {k: v for k, v in axioms_a.items() if k in axioms_b}
    only_a = {k: v for k, v in axioms_a.items() if k not in axioms_b}
    only_b = {k: v for k, v in axioms_b.items() if k not in axioms_a}

    return {
        "shared": shared,
        "only_a": only_a,
        "only_b": only_b,
        "gcd": gcd,
        "delta_a_to_b": state_b // gcd if gcd > 0 else 0,
        "delta_b_to_a": state_a // gcd if gcd > 0 else 0,
    }


def load_state_from_bundle(path: str) -> int:
    """Load a state integer from a bundle JSON file."""
    with open(path) as f:
        bundle = json.load(f)
    return int(bundle["state_integer"])


def format_axiom_key(key: str) -> str:
    """Format axiom key for display: 'alice||likes||cats' → 'alice likes cats'."""
    return key.replace("||", " ")


def print_diff(result: dict, label_a: str = "A", label_b: str = "B"):
    """Pretty-print the forensic diff result."""
    print(f"\n{'='*60}")
    print(f"  FORENSIC DIFF: {label_a} ↔ {label_b}")
    print(f"{'='*60}")

    print(f"\n  GCD (shared integer): {result['gcd']}")
    print(f"  Delta {label_a}→{label_b}: {result['delta_a_to_b']}")
    print(f"  Delta {label_b}→{label_a}: {result['delta_b_to_a']}")

    shared = result["shared"]
    only_a = result["only_a"]
    only_b = result["only_b"]

    print(f"\n  Shared axioms ({len(shared)}):")
    if shared:
        for key in sorted(shared):
            print(f"    ✓ {format_axiom_key(key)}")
    else:
        print("    (none)")

    print(f"\n  Only in {label_a} ({len(only_a)}):")
    if only_a:
        for key in sorted(only_a):
            print(f"    − {format_axiom_key(key)}")
    else:
        print("    (none)")

    print(f"\n  Only in {label_b} ({len(only_b)}):")
    if only_b:
        for key in sorted(only_b):
            print(f"    + {format_axiom_key(key)}")
    else:
        print("    (none)")

    # Verdict
    if not only_a and not only_b:
        print(f"\n  ✅ States are IDENTICAL")
    elif not only_a:
        print(f"\n  ⊂ {label_a} is a SUBSET of {label_b}")
    elif not only_b:
        print(f"\n  ⊃ {label_a} is a SUPERSET of {label_b}")
    else:
        print(f"\n  ≠ States DIVERGE: {len(only_a)} removed, {len(only_b)} added")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Forensic Diff — debug Gödel State Integer divergence"
    )
    parser.add_argument(
        "--bundles", nargs=2, metavar=("A.json", "B.json"),
        help="Compare two bundle JSON files",
    )
    parser.add_argument(
        "--states", nargs=2, metavar=("INT_A", "INT_B"),
        help="Compare two state integers directly",
    )
    parser.add_argument(
        "--bundle", metavar="BUNDLE.json",
        help="Single bundle file (use with --state)",
    )
    parser.add_argument(
        "--state", metavar="INTEGER",
        help="Single state integer (use with --bundle)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON instead of human-readable",
    )

    args = parser.parse_args()

    # Initialize algebra (needed to know axiom→prime mapping)
    algebra = GodelStateAlgebra()

    if args.bundles:
        state_a = load_state_from_bundle(args.bundles[0])
        state_b = load_state_from_bundle(args.bundles[1])
        label_a = Path(args.bundles[0]).stem
        label_b = Path(args.bundles[1]).stem

        # Extract axioms from bundle tomes to seed the algebra
        for path in args.bundles:
            _seed_algebra_from_bundle(algebra, path)

    elif args.states:
        state_a = int(args.states[0])
        state_b = int(args.states[1])
        label_a = "State_A"
        label_b = "State_B"

    elif args.bundle and args.state:
        state_a = load_state_from_bundle(args.bundle)
        state_b = int(args.state)
        label_a = Path(args.bundle).stem
        label_b = "Given_State"
        _seed_algebra_from_bundle(algebra, args.bundle)

    else:
        parser.print_help()
        sys.exit(1)

    result = diff_states(algebra, state_a, state_b)

    if args.json:
        output = {
            "state_a": str(state_a),
            "state_b": str(state_b),
            "gcd": str(result["gcd"]),
            "delta_a_to_b": str(result["delta_a_to_b"]),
            "delta_b_to_a": str(result["delta_b_to_a"]),
            "shared_count": len(result["shared"]),
            "only_a_count": len(result["only_a"]),
            "only_b_count": len(result["only_b"]),
            "shared": list(result["shared"].keys()),
            "only_a": list(result["only_a"].keys()),
            "only_b": list(result["only_b"].keys()),
        }
        print(json.dumps(output, indent=2))
    else:
        print_diff(result, label_a, label_b)


def _seed_algebra_from_bundle(algebra: GodelStateAlgebra, path: str):
    """Extract axiom keys from a bundle's canonical tome and seed the algebra."""
    import re
    with open(path) as f:
        bundle = json.load(f)
    tome = bundle.get("canonical_tome", "")
    pattern = re.compile(r"^The\s+(\S+)\s+(\S+)\s+(\S+)\.$", re.MULTILINE)
    for match in pattern.finditer(tome):
        s, p, o = match.group(1), match.group(2), match.group(3)
        algebra.get_or_mint_prime(s.lower(), p.lower(), o.lower())


if __name__ == "__main__":
    main()
