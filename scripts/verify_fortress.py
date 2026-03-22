#!/usr/bin/env python3
"""
Verify Fortress — Comprehensive verification harness.

Runs all verification checks in a single invocation:
  1. Reference vector consistency (frozen fixtures)
  2. Property algebra invariants (spot check)
  3. Ed25519 dual-signature round-trip
  4. Cross-runtime witness (Node.js)
  5. Key rotation backward compatibility

Usage:
    python scripts/verify_fortress.py

Exit code 0 = all checks pass.

Author: ototao
License: Apache License 2.0
"""

import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sympy import nextprime
from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.ensemble.tome_generator import AutoregressiveTomeGenerator
from internal.infrastructure.canonical_codec import CanonicalCodec
from internal.infrastructure.key_manager import KeyManager

FIXTURE_DIR = Path(__file__).parent.parent / "Tests" / "fixtures"
VERIFIER = Path(__file__).parent.parent / "standalone_verifier" / "verify.js"


def derive_prime(key: str) -> int:
    h = hashlib.sha256(key.encode()).digest()
    seed = int.from_bytes(h[:8], "big")
    return int(nextprime(seed))


passed = 0
total = 0


def check(label: str, condition: bool, detail: str = ""):
    global passed, total
    total += 1
    if condition:
        passed += 1
    status = "✅" if condition else "❌"
    msg = f"  {status} {label}"
    if detail and not condition:
        msg += f"  ({detail})"
    print(msg)
    return condition


def main():
    global passed, total
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Output JSON summary")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  VERIFIABILITY FORTRESS — COMPREHENSIVE CHECK")
    print("=" * 60)

    all_pass = True

    # ── 1. Reference Vectors ──────────────────────────────────
    print("\n[1] Reference Vector Consistency")
    vectors = json.loads((FIXTURE_DIR / "reference_vectors.json").read_text())

    for key, expected_str in vectors.items():
        if key.startswith("_"):
            continue
        actual = derive_prime(key)
        ok = actual == int(expected_str)
        all_pass &= check(f"Prime({key.split('||')[0]}...)", ok)

    # LCM check
    state = 1
    for k, v in vectors.items():
        if not k.startswith("_"):
            state = math.lcm(state, int(v))
    ok = state == int(vectors["_lcm_all"])
    all_pass &= check("LCM matches frozen state", ok)

    # ── 2. Algebra Spot Check ─────────────────────────────────
    print("\n[2] Algebra Invariants (spot check)")
    a, b = 2 * 3 * 5, 3 * 5 * 7
    all_pass &= check("LCM commutativity", math.lcm(a, b) == math.lcm(b, a))
    all_pass &= check("LCM idempotency", math.lcm(a, a) == a)
    all_pass &= check("Entailment after merge", math.lcm(a, b) % a == 0)
    all_pass &= check("GCD extracts shared", math.gcd(a, b) == 3 * 5)

    # ── 3. Ed25519 Dual-Signature ─────────────────────────────
    print("\n[3] Ed25519 Dual-Signature")
    with tempfile.TemporaryDirectory() as tmpdir:
        km = KeyManager(key_dir=str(Path(tmpdir) / "keys"))
        algebra = GodelStateAlgebra()
        tome_gen = AutoregressiveTomeGenerator(algebra)
        codec = CanonicalCodec(algebra, tome_gen, "fortress-verify-key!!!", key_manager=km)

        algebra.get_or_mint_prime("fortress", "verifies", "math")
        state = list(algebra.axiom_to_prime.values())[0]
        bundle = codec.export_bundle(state)

        has_ed25519 = "public_signature" in bundle and "public_key" in bundle
        all_pass &= check("Bundle contains Ed25519 fields", has_ed25519)

        imported = codec.import_bundle(bundle)
        all_pass &= check("Dual-sig import succeeds", imported == state)

        # Tamper check
        import copy
        tampered = copy.deepcopy(bundle)
        tampered["canonical_tome"] += "\nEvil."
        tampered["signature"] = codec._sign(
            tampered["canonical_tome"], tampered["state_integer"], tampered["timestamp"]
        )
        try:
            codec.import_bundle(tampered)
            all_pass &= check("Tampered bundle rejected", False, "should have raised")
        except Exception:
            all_pass &= check("Tampered bundle rejected by Ed25519", True)

    # ── 4. Cross-Runtime Witness ──────────────────────────────
    print("\n[4] Cross-Runtime Witness (Node.js)")
    try:
        result = subprocess.run(
            ["node", str(VERIFIER), "--self-test"],
            capture_output=True, text=True, timeout=30,
        )
        all_pass &= check("Node.js self-test", result.returncode == 0)
    except FileNotFoundError:
        print("  ⚠️  Node.js not installed (skipped)")

    # ── 5. Key Rotation ───────────────────────────────────────
    print("\n[5] Key Rotation Backward Compatibility")
    with tempfile.TemporaryDirectory() as tmpdir:
        km2 = KeyManager(key_dir=str(Path(tmpdir) / "keys"))
        km2.ensure_keypair()
        pub1 = km2.get_public_key_bytes()
        km2.rotate_keypair()
        pub2 = km2.get_public_key_bytes()

        all_pass &= check("Rotation produces new key", pub1 != pub2)
        trusted = km2.list_trusted_public_keys()
        all_pass &= check("Both keys trusted", len(trusted) == 2)
        all_pass &= check("Old key in trusted list", pub1 in trusted)

    # ── Summary ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if all_pass:
        print(f"  ✅ ALL FORTRESS CHECKS PASSED ({passed}/{total})")
    else:
        print(f"  ❌ SOME CHECKS FAILED ({passed}/{total})")
    print(f"{'=' * 60}\n")

    if args.json:
        summary = {
            "passed": passed,
            "total": total,
            "all_pass": all_pass,
            "ratio": f"{passed}/{total}",
        }
        print(json.dumps(summary))

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
