"""Cross-runtime Gödel byte-identity harness for `sha256_128_v2`.

Companion to `verify_godel_cross_runtime.py` (the v1 harness). This
script asserts Python and Node produce **byte-identical primes and
state integers** under the `sha256_128_v2` prime scheme:

  * SHA-256 → first 16 bytes big-endian → u128 seed
  * nextprime(seed) via BPSW (`sympy.nextprime`)

Why a separate harness, not just an `--scheme=v2` flag on the v1
script: keeping them as parallel scripts gives a clean
"v1-byte-identity" gate AND a "v2-byte-identity" gate in CI, each
green or red on its own. If a future regression breaks v2 only, the
v1 gate stays green and the failing diagnostic points unambiguously
at the v2 codepath.

Two layers of check:

  K1-v2: for each fixture axiom key, mint a v2 prime in both runtimes
         and assert they equal. Catches v2 prime-derivation drift.
  K2-v2: for each fixture triple list, encode the full state integer
         under v2 in both runtimes (LCM of v2 primes) and assert they
         equal. Catches accumulation-order or LCM-semantic drift.

Exit codes:
    0  all fixtures agree
    1  one or more divergences
    3  godel_cli.js missing
    4  node not on PATH

This harness does NOT switch the active scheme. It calls v2 directly
on both sides via the explicit `scheme` field on the Node CLI and
the explicit `_deterministic_prime_v2` helper on the Python side.
The default scheme stays `sha256_64_v1`; flipping it is an
operator decision separate from this byte-identity proof.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import sympy

REPO_ROOT = Path(__file__).resolve().parent.parent
GODEL_CLI = REPO_ROOT / "single_file_demo" / "godel_cli.js"
SCHEME = "sha256_128_v2"

# Same fixtures as the v1 harness. If v1 and v2 ever drift on the
# same fixture set, the diff is the diagnostic.
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
    "café||like||cat",
    "gravity||attract||object toward earth",
]

TRIPLE_LISTS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("single triple", [("alice", "like", "cat")]),
    ("two triples", [("alice", "like", "cat"), ("bob", "own", "dog")]),
    ("five triples", [
        ("alice", "like", "cat"),
        ("bob", "own", "dog"),
        ("paris", "be", "city"),
        ("einstein", "propose", "relativity"),
        ("dolphin", "be", "mammal"),
    ]),
    ("repeated triple", [
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


def _py_mint_v2(axiom_key: str) -> int:
    """Python-side v2 prime: SHA-256 → first 16 bytes → nextprime.

    Calls sympy.nextprime directly so this script is self-contained
    and does NOT depend on the SUM_PRIME_SCHEME env var.
    """
    h = hashlib.sha256(axiom_key.encode("utf-8")).digest()
    seed = int.from_bytes(h[:16], byteorder="big")
    return sympy.nextprime(seed)


def _py_encode_v2(triples: list[tuple[str, str, str]]) -> int:
    """Python-side v2 state encoding: LCM of v2 primes."""
    state = 1
    for s, p, o in triples:
        key = f"{s.strip().lower()}||{p.strip().lower()}||{o.strip().lower()}"
        state = math.lcm(state, _py_mint_v2(key))
    return state


def _js_call(payload: dict) -> int:
    """Invoke godel_cli.js with the given payload + return BigInt."""
    r = subprocess.run(
        ["node", str(GODEL_CLI)],
        input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        cwd=str(REPO_ROOT),
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"godel_cli failed ({r.returncode}): "
            f"{r.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return int(r.stdout.decode("utf-8"))


def _js_mint_v2(axiom_key: str) -> int:
    return _js_call({"op": "mint", "axiom_key": axiom_key, "scheme": SCHEME})


def _js_encode_v2(triples: list[tuple[str, str, str]]) -> int:
    return _js_call({
        "op": "encode",
        "triples": [list(t) for t in triples],
        "scheme": SCHEME,
    })


def main() -> int:
    if not GODEL_CLI.exists():
        print(f"[harness-v2] {GODEL_CLI} missing", file=sys.stderr)
        return 3
    try:
        v = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if v.returncode != 0:
            return 4
        print(f"[harness-v2] node {v.stdout.strip()}; scheme={SCHEME}")
    except FileNotFoundError:
        print("[harness-v2] node not on PATH", file=sys.stderr)
        return 4

    failures: list[str] = []

    print("\n── K1-v2: v2 prime-derivation parity ──")
    for key in AXIOM_KEYS:
        py_prime = _py_mint_v2(key)
        try:
            js_prime = _js_mint_v2(key)
        except Exception as e:
            failures.append(f"  mint-v2[{key!r}]: js threw {e!r}")
            continue
        if py_prime == js_prime:
            print(f"  [OK] {key}  → {py_prime}")
        else:
            failures.append(
                f"  mint-v2[{key!r}]: py={py_prime} js={js_prime}"
            )

    print("\n── K2-v2: v2 state-encoding parity ──")
    for label, triples in TRIPLE_LISTS:
        py_state = _py_encode_v2(triples)
        try:
            js_state = _js_encode_v2(triples)
        except Exception as e:
            failures.append(f"  encode-v2[{label}]: js threw {e!r}")
            continue
        if py_state == js_state:
            print(f"  [OK] {label}  → state.bit_length={py_state.bit_length()}")
        else:
            failures.append(
                f"  encode-v2[{label}]: py={py_state} js={js_state}"
            )

    if failures:
        print(f"\nGÖDEL CROSS-RUNTIME ({SCHEME}): REGRESSION", file=sys.stderr)
        for f in failures:
            print(f, file=sys.stderr)
        return 1
    print(f"\nGÖDEL CROSS-RUNTIME ({SCHEME}): ALL FIXTURES AGREE")
    print(f"K1-v2 + K2-v2 byte-identity locked: {len(AXIOM_KEYS)} keys + {len(TRIPLE_LISTS)} state encodings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
