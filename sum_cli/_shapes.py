"""Input-shape checks for sum_cli commands.

Valid JSON of the wrong type is malformed input (exit 2, as each
command's help says), not a verification failure (exit 1). Kept out of
main.py, whose size is frozen (Tests/test_main_py_size_frozen.py).
"""
from __future__ import annotations

import sys
from typing import Any


def shape_error(obj: Any, kind: str) -> int:
    """Print why ``obj`` is not a usable ``kind`` and return 2, else 0.

    kind: "bundle" (verify: object with string canonical_tome and
    state_integer when present), "inspect" (object, string tome when
    present) or "receipt" (object).
    """
    if not isinstance(obj, dict):
        msg = f"{'receipt' if kind == 'receipt' else 'bundle'} must be a JSON object"
    elif kind != "receipt" and not isinstance(obj.get("canonical_tome", ""), str):
        msg = "canonical_tome must be a string"
    elif kind == "bundle" and not isinstance(obj.get("state_integer", ""), str):
        msg = "state_integer must be a string"
    else:
        return 0
    print(f"sum: {msg}", file=sys.stderr)
    return 2
