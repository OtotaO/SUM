"""Regenerate the cross-runtime density-sort fixture from the Python reference.

Python is the reference implementation for density subsetting:
`sum_engine_internal.ensemble.tome_sliders.apply_density` sorts axiom keys
with the builtin `sorted()`, i.e. by Unicode codepoint. The Worker twin
(`worker/src/render/axis_prompts.ts::applyDensity`) must reproduce that
byte-for-byte, because the surviving subset is what the deterministic tome
is built from and the tome hash is a signed receipt field.

The triple set below is chosen so that ICU collation and codepoint order
disagree on which triples survive:

    subject "a"    lowercase
    subject "ab"   a proper prefix extension of "a"
    subject "a b"  space-bearing, sorts before "ab" by codepoint (U+0020)
    subject "A"    uppercase, sorts before every lowercase by codepoint

Run from the repository root:

    python fixtures/density_sort/generate_fixture.py
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sum_engine_internal.ensemble.slider_renderer import _axiom_key  # noqa: E402
from sum_engine_internal.ensemble.tome_sliders import apply_density  # noqa: E402

OUT_PATH = REPO_ROOT / "fixtures" / "density_sort" / "apply_density_cross_runtime_v1.json"

TRIPLES: list[list[str]] = [
    ["a", "p", "o"],
    ["ab", "p", "o"],
    ["a b", "p", "o"],
    ["A", "p", "o"],
]

DENSITIES = [0.5, 0.75, 1.0]


def main() -> None:
    keys = [_axiom_key((t[0], t[1], t[2])) for t in TRIPLES]
    cases = [
        {"density": d, "expected_kept_keys": apply_density(keys, d)}
        for d in DENSITIES
    ]
    doc = {
        "schema": "sum.density_sort_cross_runtime.v1",
        "generated_by": "fixtures/density_sort/generate_fixture.py",
        "reference_runtime": "python",
        "key_format": (
            "s||p||o, built by "
            "sum_engine_internal.ensemble.slider_renderer._axiom_key"
        ),
        "note": (
            "Python sorted() orders these keys by Unicode codepoint. "
            "JavaScript String.prototype.localeCompare orders them by ICU "
            "collation, which is a different order, which changed which "
            "triples survive density < 1 and therefore changed the signed "
            "tome_hash. Every runtime must reproduce expected_kept_keys "
            "exactly, in this order."
        ),
        "triples": TRIPLES,
        "cases": cases,
    }
    OUT_PATH.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    for case in cases:
        print(f"  density={case['density']}: {case['expected_kept_keys']}")


if __name__ == "__main__":
    main()
