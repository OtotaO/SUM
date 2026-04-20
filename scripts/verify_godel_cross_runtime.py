"""Cross-runtime Gödel algebra byte-identity harness.

Asserts Python and JavaScript produce the SAME primes for every axiom
key, and the SAME state integer for every triple list. If Python's
``GodelStateAlgebra.get_or_mint_prime`` and JS's ``mintPrime`` ever
disagree, every cross-runtime bundle verification breaks silently —
Python-minted state integers cannot be reproduced in the browser.

This is the load-bearing M1c contract for the single-file demo.

Two layers of check:
  K1: for each fixture axiom key, mint a prime in both runtimes and
      assert they equal. Catches prime-derivation drift.
  K2: for each fixture triple list, encode the full state integer in
      both runtimes (product of LCMs) and assert they equal. Catches
      accumulation-order or LCM-semantic drift.

Exit codes:
    0  all fixtures agree
    1  one or more divergences
    3  JS CLI missing
    4  node not on PATH
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra

REPO_ROOT = Path(__file__).resolve().parent.parent
GODEL_CLI = REPO_ROOT / "single_file_demo" / "godel_cli.js"

AXIOM_KEYS: list[str] = [
    "alice||like||cat",
    "bob||own||dog",
    "paris||be||city",
    "einstein||propose||relativity",
    "water||contain||hydrogen",
    "marie_curie||win||nobel prizes",
    "marie_curie||be||physicist",
    "shakespeare||write||hamlet",
    "shakespeare||write||macbeth",
    "dolphin||be||mammal",
    # Unicode in the axiom key (if an extractor ever emits one)
    "café||like||cat",
    # Long multi-word object
    "gravity||attract||object toward earth",
]

TRIPLE_LISTS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("single triple", [("alice", "like", "cat")]),
    ("two triples",   [("alice", "like", "cat"), ("bob", "own", "dog")]),
    ("five triples",  [
        ("alice", "like", "cat"),
        ("bob", "own", "dog"),
        ("paris", "be", "city"),
        ("einstein", "propose", "relativity"),
        ("dolphin", "be", "mammal"),
    ]),
    ("repeated triple", [
        # LCM semantics: encoding the same triple twice produces the
        # same state as encoding it once. Verify both sides agree.
        ("alice", "like", "cat"),
        ("alice", "like", "cat"),
    ]),
    ("order permutation A", [
        ("alice", "like", "cat"),
        ("bob", "own", "dog"),
        ("paris", "be", "city"),
    ]),
    ("order permutation B", [
        ("paris", "be", "city"),
        ("alice", "like", "cat"),
        ("bob", "own", "dog"),
    ]),
]


def _js_mint(axiom_key: str) -> int:
    r = subprocess.run(
        ["node", str(GODEL_CLI)],
        input=json.dumps({"op": "mint", "axiom_key": axiom_key}).encode("utf-8"),
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"godel_cli mint failed ({r.returncode}): "
            f"{r.stderr.decode('utf-8', errors='replace')}"
        )
    return int(r.stdout.decode("utf-8"))


def _js_encode(triples: list[tuple[str, str, str]]) -> int:
    r = subprocess.run(
        ["node", str(GODEL_CLI)],
        input=json.dumps(
            {"op": "encode", "triples": [list(t) for t in triples]},
            ensure_ascii=False,
        ).encode("utf-8"),
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"godel_cli encode failed ({r.returncode}): "
            f"{r.stderr.decode('utf-8', errors='replace')}"
        )
    return int(r.stdout.decode("utf-8"))


def main() -> int:
    if not GODEL_CLI.exists():
        print(f"[harness] {GODEL_CLI} missing", file=sys.stderr)
        return 3
    try:
        v = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if v.returncode != 0:
            return 4
        print(f"[harness] node {v.stdout.strip()}")
    except FileNotFoundError:
        print("[harness] node not on PATH", file=sys.stderr)
        return 4

    algebra = GodelStateAlgebra()
    failures: list[str] = []

    print("\n── K1: prime-derivation parity ──")
    for key in AXIOM_KEYS:
        s, p, o = key.split("||")
        py_prime = algebra.get_or_mint_prime(s, p, o)
        try:
            js_prime = _js_mint(key)
        except Exception as e:
            failures.append(f"  mint[{key!r}]: js threw {e!r}")
            continue
        if py_prime == js_prime:
            print(f"  [OK] {key}  → {py_prime}")
        else:
            failures.append(
                f"  mint[{key!r}]: py={py_prime} js={js_prime}"
            )

    print("\n── K2: state-encoding parity ──")
    for label, triples in TRIPLE_LISTS:
        algebra2 = GodelStateAlgebra()  # fresh algebra per test
        py_state = algebra2.encode_chunk_state(list(triples))
        try:
            js_state = _js_encode(triples)
        except Exception as e:
            failures.append(f"  encode[{label}]: js threw {e!r}")
            continue
        if py_state == js_state:
            print(f"  [OK] {label}  → state.bit_length={py_state.bit_length()}")
        else:
            failures.append(
                f"  encode[{label}]: py={py_state} js={js_state}"
            )

    if failures:
        print("\nGÖDEL CROSS-RUNTIME: REGRESSION", file=sys.stderr)
        for f in failures:
            print(f, file=sys.stderr)
        return 1
    print("\nGÖDEL CROSS-RUNTIME: ALL FIXTURES AGREE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
