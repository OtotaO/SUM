"""
Prime Scheme Registry — Stage 2 of Carmack Hardening

Defines immutable scheme IDs that pin the entire prime derivation
pipeline.  Any change to the hash function, seed width, primality
test, or nextprime algorithm constitutes a new scheme.

Current scheme:
    sha256_64_v1  —  SHA-256 → first 8 bytes big-endian → nextprime(seed)

Future (gated behind Stage 3):
    sha256_128_v2 —  SHA-256 → first 16 bytes → BPSW-nextprime(seed)

The scheme ID is:
    • Embedded in every exported bundle (``prime_scheme`` field)
    • Exposed in /state and /branches API responses
    • Exchanged in P2P handshake
    • Validated on import: mismatched scheme → rejection

Author: ototao
License: Apache License 2.0
"""

from dataclasses import dataclass
from typing import Optional
import os

# ─── The default scheme ──────────────────────────────────────────────

_DEFAULT_SCHEME = "sha256_64_v1"
_VALID_SCHEMES = {"sha256_64_v1", "sha256_128_v2"}


def _resolve_scheme() -> str:
    """Resolve the active scheme from env var SUM_PRIME_SCHEME.

    Rules:
      1. If SUM_PRIME_SCHEME is unset or empty → default to sha256_64_v1
      2. If set to a known scheme → use it
      3. If set to an unknown value → crash immediately (fail-closed)

    This runs once at import time. To switch scheme, restart the process
    with the new env var value.
    """
    env = os.environ.get("SUM_PRIME_SCHEME", "").strip()
    if not env:
        return _DEFAULT_SCHEME
    if env not in _VALID_SCHEMES:
        raise RuntimeError(
            f"FATAL: SUM_PRIME_SCHEME={env!r} is not a known scheme. "
            f"Valid schemes: {sorted(_VALID_SCHEMES)}. "
            f"Fix the environment variable or remove it to use the default ({_DEFAULT_SCHEME})."
        )
    return env


CURRENT_SCHEME = _resolve_scheme()


@dataclass(frozen=True)
class PrimeScheme:
    """Immutable description of a prime derivation pipeline."""
    scheme_id: str
    hash_algorithm: str
    seed_bytes: int
    primality_test: str
    description: str
    deprecated: bool = False


# ─── Scheme catalog ──────────────────────────────────────────────────

SCHEMES: dict[str, PrimeScheme] = {
    "sha256_64_v1": PrimeScheme(
        scheme_id="sha256_64_v1",
        hash_algorithm="sha256",
        seed_bytes=8,
        primality_test="deterministic_miller_rabin_12_witnesses",
        description=(
            "SHA-256 → first 8 bytes big-endian → u64 seed → "
            "nextprime(seed) via deterministic Miller-Rabin with "
            "witnesses {2,3,5,7,11,13,17,19,23,29,31,37}. "
            "Proven correct for all n < 3.3×10²⁴."
        ),
    ),
    # Stage 3 placeholder — NOT active until explicitly gated
    "sha256_128_v2": PrimeScheme(
        scheme_id="sha256_128_v2",
        hash_algorithm="sha256",
        seed_bytes=16,
        primality_test="bpsw",
        description=(
            "SHA-256 → first 16 bytes big-endian → u128 seed → "
            "nextprime(seed) via BPSW (Baillie-PSW). "
            "No known BPSW pseudoprimes exist."
        ),
        deprecated=False,  # Not deprecated, just not yet active
    ),
}


def get_scheme(scheme_id: str) -> PrimeScheme:
    """Look up a scheme by ID. Raises ValueError if unknown."""
    if scheme_id not in SCHEMES:
        raise ValueError(
            f"Unknown prime scheme: {scheme_id!r}. "
            f"Known schemes: {list(SCHEMES.keys())}"
        )
    return SCHEMES[scheme_id]


def get_current_scheme() -> PrimeScheme:
    """Return the active scheme."""
    return SCHEMES[CURRENT_SCHEME]


def is_compatible(scheme_id: str) -> bool:
    """Check if a scheme is compatible with the current node.

    Currently only sha256_64_v1 is active. A node running v1
    MUST reject state from a v2 node because the primes are
    derived differently.
    """
    return scheme_id == CURRENT_SCHEME


def validate_scheme_or_raise(scheme_id: str, context: str = "operation") -> None:
    """Validate that a scheme is compatible. Raises ValueError if not."""
    if not is_compatible(scheme_id):
        current = get_current_scheme()
        raise ValueError(
            f"Incompatible prime scheme for {context}: "
            f"received {scheme_id!r}, this node runs {current.scheme_id!r}. "
            f"Mixed-scheme operations are forbidden — primes are derived "
            f"differently and would corrupt state."
        )
