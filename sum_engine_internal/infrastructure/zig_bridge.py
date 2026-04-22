"""
Horizon III: Bare-Metal FFI Bridge (Zig → Python)

Routes heavy mathematical operations to the natively compiled Zig core
(``libsum_core``) via Python's ``ctypes``.  If the compiled binary is
not present, the bridge gracefully degrades — ``zig_engine`` will be
``None`` and callers fall back to the pure-Python implementation.

The Strangler Fig Pattern:
    Phase 17:  ``sum_get_deterministic_prime`` (SHA-256 → Miller-Rabin)
    Phase 17b: ``sum_bigint_lcm/gcd/mod/divisible_by_u64`` (state arithmetic)

Author: ototao
License: Apache License 2.0
"""

import ctypes
import logging
import os
import platform

logger = logging.getLogger(__name__)

# Maximum buffer size for BigInt results (64 KB — supports integers
# with ~150,000+ decimal digits, far beyond near-term needs)
_BIGINT_BUF_CAP = 65536


class ZigMathEngine:
    """
    Horizon III: Bare-Metal FFI Bridge.

    Loads the compiled Zig shared library and exposes typed Python
    wrappers around its C-ABI exports.  If the library is not found
    or fails to load, ``self.lib`` remains ``None`` and all methods
    return ``None`` (signalling callers to use the Python fallback).
    """

    def __init__(self):
        self.lib = None
        self._load_library()

    def _load_library(self):
        """Attempt to discover and load ``libsum_core`` for the current platform."""
        base_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "core-zig", "zig-out", "lib"
            )
        )

        system = platform.system().lower()
        if system == "darwin":
            lib_name = "libsum_core.dylib"
        elif system == "windows":
            lib_name = "sum_core.dll"
        else:
            lib_name = "libsum_core.so"

        lib_path = os.path.join(base_dir, lib_name)

        if os.path.exists(lib_path):
            try:
                self.lib = ctypes.CDLL(lib_path)
                self._bind_phase17()
                self._bind_phase17_v2()
                self._bind_phase17b()
                logger.info("⚡ BARE-METAL ZIG CORE ENGAGED ⚡")
            except Exception as exc:
                logger.warning("Failed to bind Zig Core C-ABI: %s", exc)
                self.lib = None
        else:
            logger.debug(
                "Zig core not found at %s — using Python fallback.", lib_path
            )

    def _bind_phase17(self):
        """Bind Phase 17 exports: deterministic prime derivation."""
        self.lib.sum_get_deterministic_prime.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t,
        ]
        self.lib.sum_get_deterministic_prime.restype = ctypes.c_uint64

    def _bind_phase17_v2(self):
        """Bind Stage 3 export: v2 deterministic prime (128-bit BPSW)."""
        self.lib.sum_get_deterministic_prime_v2.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t,
            ctypes.c_char_p,  # 16-byte output buffer
        ]
        self.lib.sum_get_deterministic_prime_v2.restype = ctypes.c_int32

    def _bind_phase17b(self):
        """Bind Phase 17b exports: BigInt arithmetic (LCM, GCD, mod, divisibility)."""
        # sum_bigint_gcd / sum_bigint_lcm / sum_bigint_mod share the same signature
        for name in ("sum_bigint_gcd", "sum_bigint_lcm", "sum_bigint_mod"):
            fn = getattr(self.lib, name)
            fn.argtypes = [
                ctypes.c_char_p, ctypes.c_size_t,  # a_ptr, a_len
                ctypes.c_char_p, ctypes.c_size_t,  # b_ptr, b_len
                ctypes.c_char_p, ctypes.c_size_t,  # out_ptr, out_cap
                ctypes.POINTER(ctypes.c_size_t),    # out_len
            ]
            fn.restype = ctypes.c_int32

        self.lib.sum_bigint_divisible_by_u64.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t,  # a_ptr, a_len
            ctypes.c_uint64,                    # prime
        ]
        self.lib.sum_bigint_divisible_by_u64.restype = ctypes.c_int32

        self.lib.sum_batch_mint_primes.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t,   # axioms_ptr, axioms_len
            ctypes.POINTER(ctypes.c_uint64),     # out_primes
            ctypes.c_size_t,                     # out_cap
        ]
        self.lib.sum_batch_mint_primes.restype = ctypes.c_int32

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Return ``True`` if the Zig shared library is loaded and ready."""
        return self.lib is not None

    # ------------------------------------------------------------------
    # Phase 17: Deterministic Prime
    # ------------------------------------------------------------------

    def get_deterministic_prime(self, axiom: str) -> int | None:
        """SHA-256(axiom) → 8-byte seed → next prime (via Zig bare-metal)."""
        if self.lib is None:
            return None
        encoded = axiom.encode("utf-8")
        return self.lib.sum_get_deterministic_prime(encoded, len(encoded))

    def get_deterministic_prime_v2(self, axiom: str) -> int | None:
        """SHA-256(axiom) → 16-byte seed → next prime via BPSW (Zig bare-metal)."""
        if self.lib is None:
            return None
        encoded = axiom.encode("utf-8")
        out_buf = ctypes.create_string_buffer(16)
        rc = self.lib.sum_get_deterministic_prime_v2(encoded, len(encoded), out_buf)
        if rc != 0:
            return None
        return int.from_bytes(out_buf.raw, byteorder="big")

    # ------------------------------------------------------------------
    # Phase 17b: BigInt Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _int_to_bytes(n: int) -> bytes:
        """Convert a Python int to big-endian bytes."""
        if n == 0:
            return b"\x00"
        byte_len = (n.bit_length() + 7) // 8
        return n.to_bytes(byte_len, byteorder="big")

    @staticmethod
    def _bytes_to_int(data: bytes, length: int) -> int:
        """Convert big-endian bytes to a Python int."""
        return int.from_bytes(data[:length], byteorder="big")

    def _call_bigint_binary(self, fn_name: str, a: int, b: int) -> int | None:
        """Generic caller for bigint binary ops (GCD, LCM, mod)."""
        if self.lib is None:
            return None

        a_bytes = self._int_to_bytes(a)
        b_bytes = self._int_to_bytes(b)
        out_buf = ctypes.create_string_buffer(_BIGINT_BUF_CAP)
        out_len = ctypes.c_size_t(0)

        fn = getattr(self.lib, fn_name)
        rc = fn(
            a_bytes, len(a_bytes),
            b_bytes, len(b_bytes),
            out_buf, _BIGINT_BUF_CAP,
            ctypes.byref(out_len),
        )

        if rc != 0:
            logger.warning("Zig %s returned error code %d", fn_name, rc)
            return None

        return self._bytes_to_int(out_buf.raw, out_len.value)

    # ------------------------------------------------------------------
    # Phase 17b: Public API
    # ------------------------------------------------------------------

    def bigint_lcm(self, a: int, b: int) -> int | None:
        """Compute LCM(a, b) via Zig bare-metal."""
        return self._call_bigint_binary("sum_bigint_lcm", a, b)

    def bigint_gcd(self, a: int, b: int) -> int | None:
        """Compute GCD(a, b) via Zig bare-metal."""
        return self._call_bigint_binary("sum_bigint_gcd", a, b)

    def bigint_mod(self, a: int, b: int) -> int | None:
        """Compute a % b via Zig bare-metal."""
        return self._call_bigint_binary("sum_bigint_mod", a, b)

    def is_divisible_by(self, state: int, prime: int) -> bool | None:
        """
        Check if ``state % prime == 0`` via optimized streaming modular
        arithmetic (no BigInt construction needed for the prime).
        """
        if self.lib is None:
            return None

        state_bytes = self._int_to_bytes(state)
        rc = self.lib.sum_bigint_divisible_by_u64(
            state_bytes, len(state_bytes), prime
        )

        if rc == -2:
            return None
        return rc == 1


    def batch_mint_primes(self, axiom_keys: list) -> list:
        """
        Batch-mint deterministic primes from a list of axiom key strings.
        Amortizes FFI overhead across the batch.

        Returns list of u64 primes, or None if Zig is unavailable.
        """
        if not self.lib:
            return None
        try:
            # Concatenate with null separators
            joined = b'\x00'.join(k.encode('utf-8') for k in axiom_keys) + b'\x00'
            out_cap = len(axiom_keys)
            out_primes = (ctypes.c_uint64 * out_cap)()
            count = self.lib.sum_batch_mint_primes(
                joined, len(joined), out_primes, out_cap
            )
            if count < 0:
                return None
            return [out_primes[i] for i in range(count)]
        except Exception:
            return None


# ─── Module-level singleton ──────────────────────────────────────────
# Importers access ``zig_engine`` directly.  If the library isn't compiled,
# this is simply ``None`` and the Python fallback takes over seamlessly.

try:
    zig_engine = ZigMathEngine()
    if not zig_engine.available:
        zig_engine = None
except Exception:
    zig_engine = None
