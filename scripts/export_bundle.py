#!/usr/bin/env python3
"""
Export a canonical bundle to a JSON file.

Usage:
    python scripts/export_bundle.py [output_path] [--axioms "s,p,o" ...]

If no axioms are given, exports a demo 3-axiom state.
If no output path is given, writes to stdout.

Example:
    python scripts/export_bundle.py /tmp/bundle.json
    python scripts/export_bundle.py /tmp/bundle.json --axioms "alice,likes,cats" "bob,knows,python"

Then verify with:
    node standalone_verifier/verify.js /tmp/bundle.json

Phase 16: Independent Semantic Witness.
"""

import json
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.ensemble.tome_generator import AutoregressiveTomeGenerator
from internal.infrastructure.canonical_codec import CanonicalCodec


def main():
    args = sys.argv[1:]

    # Parse output path
    output_path = None
    axiom_strs = []
    i = 0
    while i < len(args):
        if args[i] == "--axioms":
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                axiom_strs.append(args[i])
                i += 1
        else:
            output_path = args[i]
            i += 1

    # Default demo axioms
    if not axiom_strs:
        axiom_strs = ["alice,likes,cats", "bob,knows,python", "earth,orbits,sun"]

    # Build state
    algebra = GodelStateAlgebra()
    tome_gen = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(algebra, tome_gen, signing_key="export-demo-key")

    state = 1
    for axiom_str in axiom_strs:
        parts = axiom_str.split(",")
        if len(parts) != 3:
            print(f"Warning: Skipping invalid axiom format: {axiom_str}", file=sys.stderr)
            continue
        s, p, o = [x.strip() for x in parts]
        prime = algebra.get_or_mint_prime(s, p, o)
        state = math.lcm(state, prime)

    # Export
    bundle = codec.export_bundle(state, branch="export-cli", title="CLI Export Bundle")

    bundle_json = json.dumps(bundle, indent=2)

    if output_path:
        with open(output_path, "w") as f:
            f.write(bundle_json)
        print(f"Bundle exported to {output_path}")
        print(f"  Axioms: {bundle['axiom_count']}")
        print(f"  State digits: {len(bundle['state_integer'])}")
        print(f"\nVerify with:")
        print(f"  node standalone_verifier/verify.js {output_path}")
    else:
        print(bundle_json)


if __name__ == "__main__":
    main()
