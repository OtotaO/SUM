"""
State Encoding — Dual-Format State Transport

Stage 1 of the Carmack-directed hardening program.

Provides hex↔decimal conversion for Gödel State Integers with
backward-compatible detection and validation.

Transport surfaces use decimal strings today (``str(state)``).
This module adds hex representation as a companion format:
    decimal:  "1898585074409907150524167558344558620554613878579045806247"
    hex:      "0x597462b17bc9a1c247d9a7ba1b3c4d5e6f7a8b9c0d1e2f37"

The ``to_dual()`` helper emits both in one call. The ``parse_state()``
helper auto-detects hex (``0x`` prefix) or decimal input.

Author: ototao
License: Apache License 2.0
"""

import logging

logger = logging.getLogger(__name__)


def to_hex(state: int) -> str:
    """Convert a Gödel state integer to hex string with 0x prefix."""
    if state < 0:
        raise ValueError("State integer must be non-negative")
    return hex(state)


def from_hex(hex_str: str) -> int:
    """Parse a hex string (with or without 0x prefix) to integer."""
    hex_str = hex_str.strip()
    if not hex_str.startswith("0x") and not hex_str.startswith("0X"):
        hex_str = "0x" + hex_str
    return int(hex_str, 16)


def parse_state(value: str) -> int:
    """Auto-detect hex or decimal and parse to integer.

    Accepts:
        "0x1a2b3c"  → hex
        "1234567"   → decimal

    Returns:
        The parsed integer.

    Raises:
        ValueError: If the string is not a valid integer in either format.
    """
    value = value.strip()
    if value.startswith("0x") or value.startswith("0X"):
        return int(value, 16)
    return int(value)


def to_dual(state: int) -> dict:
    """Return both decimal and hex representations for API responses.

    Usage in endpoint handlers:
        return {
            **to_dual(state),
            "other_field": ...,
        }

    Produces:
        {
            "state_decimal": "123456789...",
            "state_hex": "0x75bcd15...",
        }
    """
    return {
        "state_decimal": str(state),
        "state_hex": to_hex(state),
    }


def dual_field(field_name: str, state: int) -> dict:
    """Return decimal + hex for a named field.

    Example:
        dual_field("new_global_state", 42)
        → {"new_global_state": "42", "new_global_state_hex": "0x2a"}

    Preserves backward compatibility: the original decimal field
    stays unchanged, and a new ``_hex`` companion is added.
    """
    return {
        field_name: str(state),
        f"{field_name}_hex": to_hex(state),
    }
