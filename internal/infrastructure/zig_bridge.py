"""
Horizon III: Bare-Metal FFI Bridge (Zig → Python)

Routes heavy cryptographic operations to the natively compiled Zig core
(``libsum_core``) via Python's ``ctypes``.  If the compiled binary is
not present, the bridge gracefully degrades — ``zig_engine`` will be
``None`` and callers fall back to the pure-Python implementation.

The Strangler Fig Pattern:
    This module is the first tendril.  It wraps a single C-ABI export
    (``sum_get_deterministic_prime``) and will gradually expand to cover
    LCM, GCD, and state arithmetic as the Zig core matures.

Author: ototao
License: Apache License 2.0
"""

import ctypes
import logging
import os
import platform

logger = logging.getLogger(__name__)


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
                self.lib.sum_get_deterministic_prime.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_size_t,
                ]
                self.lib.sum_get_deterministic_prime.restype = ctypes.c_uint64
                logger.info("⚡ BARE-METAL ZIG CORE ENGAGED ⚡")
            except Exception as exc:
                logger.warning("Failed to bind Zig Core C-ABI: %s", exc)
                self.lib = None
        else:
            logger.debug(
                "Zig core not found at %s — using Python fallback.", lib_path
            )

    @property
    def available(self) -> bool:
        """Return ``True`` if the Zig shared library is loaded and ready."""
        return self.lib is not None

    def get_deterministic_prime(self, axiom: str) -> int | None:
        """
        SHA-256(axiom) → 8-byte seed → next prime (via Zig bare-metal).

        Returns:
            The deterministic prime, or ``None`` if the Zig core is unavailable.
        """
        if self.lib is None:
            return None
        encoded = axiom.encode("utf-8")
        return self.lib.sum_get_deterministic_prime(encoded, len(encoded))


# ─── Module-level singleton ──────────────────────────────────────────
# Importers access ``zig_engine`` directly.  If the library isn't compiled,
# this is simply ``None`` and the Python fallback takes over seamlessly.

try:
    zig_engine = ZigMathEngine()
    if not zig_engine.available:
        zig_engine = None
except Exception:
    zig_engine = None
