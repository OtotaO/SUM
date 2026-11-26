"""Write the cross-runtime fixture for the LLM system prompt.

Python is the reference: `build_system_prompt` in
sum_engine_internal/ensemble/tome_sliders.py. Its Worker twin,
`buildSystemPrompt` in worker/src/render/axis_prompts.ts, must send the
model the same prompt byte for byte. The fixture holds one row for every
quantized combination of the four prompt axes (5 bins each, so 625 rows):
the four slider values and the first 16 hex characters of sha256 over the
UTF-8 prompt. The prompts themselves are not stored, which keeps the file
small.

worker/test/prompt_smoke.mjs (`npm run test:prompts`) recomputes every
digest from the Worker code. Tests/test_tome_sliders.py rebuilds the fixture
in memory with this module and compares it with the committed bytes, so a
prompt change on either side fails a test until the fixture is regenerated
and the other side matches.

Run from the repository root:

    python worker/test/generate_prompt_fixture.py
"""

from __future__ import annotations

import hashlib
import itertools
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sum_engine_internal.ensemble.tome_sliders import (  # noqa: E402
    SLIDER_BINS_PER_AXIS,
    TomeSliders,
    build_system_prompt,
    quantize,
    snap_to_bin,
)

OUT_PATH = pathlib.Path(__file__).resolve().parent / "fixtures" / "system_prompt_digests.json"
AXES = ("length", "formality", "audience", "perspective")
DIGEST_HEX_CHARS = 16


def bin_centres() -> list[float]:
    """The values quantize() gives a prompt axis: snap_to_bin of each bin's midpoint."""
    return [snap_to_bin((i + 0.5) / SLIDER_BINS_PER_AXIS) for i in range(SLIDER_BINS_PER_AXIS)]


def prompt_digest(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:DIGEST_HEX_CHARS]


def build() -> dict:
    """The fixture as a dict. Pure: reads no files and writes none."""
    rows = []
    for values in itertools.product(bin_centres(), repeat=len(AXES)):
        # The renderer quantizes before it builds the prompt
        # (slider_renderer.render); bin centres map to themselves.
        sliders = quantize(TomeSliders(**dict(zip(AXES, values))))
        rows.append([*(getattr(sliders, axis) for axis in AXES), prompt_digest(build_system_prompt(sliders))])
    return {
        "schema": "sum.system_prompt_cross_runtime.v1",
        "generated_by": "worker/test/generate_prompt_fixture.py",
        "reference_runtime": "python",
        "reference": "sum_engine_internal.ensemble.tome_sliders.build_system_prompt",
        "digest": f"first {DIGEST_HEX_CHARS} hex characters of sha256 over the UTF-8 system prompt",
        "axes": list(AXES),
        "bin_centres": bin_centres(),
        "density": TomeSliders().density,
        "rows": rows,
    }


def serialize(doc: dict) -> str:
    """JSON text with one row per line, so the file stays small and diffs stay readable."""
    items = list(doc.items())
    lines = ["{"]
    for index, (key, value) in enumerate(items):
        comma = "," if index < len(items) - 1 else ""
        if key == "rows":
            body = ",\n".join(f"    {json.dumps(row)}" for row in value)
            lines.append(f"  {json.dumps(key)}: [\n{body}\n  ]{comma}")
        else:
            lines.append(f"  {json.dumps(key)}: {json.dumps(value)}{comma}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_PATH.write_text(serialize(build()), encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
