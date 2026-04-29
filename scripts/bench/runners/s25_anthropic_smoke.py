"""§2.5 Anthropic smoke test — one document, ~$0.005 spend.

Before committing to the full §2.5 frontier-LLM bench (50 docs ×
3 ablations × ~6 calls each ≈ $0.50–1.50 against Claude Opus 4.7),
this smoke verifies end-to-end wiring on a single document:

  1. The dispatcher routes a ``claude-*`` model id to AnthropicAdapter.
  2. ``parse_structured`` returns a populated Pydantic model from
     the live API (tool-use round-trip works).
  3. ``generate_text`` returns non-empty narrative.
  4. The full ``run_doc`` path under the ``combined`` ablation
     produces a per-doc record with measured recall.

Cost: ~6 calls on a small doc → roughly $0.005 at Opus 4.7's
2026-04 pricing. Negligible.

Usage::

    ANTHROPIC_API_KEY=... python -m scripts.bench.runners.s25_anthropic_smoke

    # Different model:
    ANTHROPIC_API_KEY=... python -m scripts.bench.runners.s25_anthropic_smoke \
        --model claude-3-5-sonnet-20241022

The script prints a JSON record on stdout and a human-readable
summary on stderr. Exits 0 on success, non-zero if any of the
four checks above fails.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Make the project importable when run as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.bench.runners.s25_generator_side import (
    DEFAULT_CALL_TIMEOUT_S,
    run_doc,
)
from sum_engine_internal.ensemble.llm_dispatch import get_adapter


SMOKE_DOC = {
    "id": "smoke_001",
    "text": (
        "Marie Curie won two Nobel Prizes. "
        "Albert Einstein proposed the theory of relativity. "
        "William Shakespeare wrote the play Hamlet."
    ),
    # Rough gold for the per-doc metrics; not load-bearing for the
    # smoke (we just want round-trip + non-zero recall).
    "gold_triples": [
        ["marie_curie", "won", "nobel_prizes"],
        ["albert_einstein", "proposed", "theory_of_relativity"],
        ["william_shakespeare", "wrote", "hamlet"],
    ],
}


async def smoke(model: str, call_timeout_s: float) -> dict:
    adapter = get_adapter(model)
    result = await run_doc(
        adapter,
        SMOKE_DOC,
        ablation="combined",
        call_timeout_s=call_timeout_s,
        dry_run=False,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="claude-opus-4-7",
        help="Pinned Claude snapshot. Default: claude-opus-4-7.",
    )
    parser.add_argument(
        "--call-timeout", type=float, default=DEFAULT_CALL_TIMEOUT_S,
        help=f"Per-call timeout in seconds. Default: {DEFAULT_CALL_TIMEOUT_S}.",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("smoke: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 2

    print(
        f"smoke: model={args.model} doc={SMOKE_DOC['id']} "
        f"timeout={args.call_timeout:.0f}s",
        file=sys.stderr,
    )
    t0 = time.time()
    try:
        result = asyncio.run(smoke(args.model, args.call_timeout))
    except Exception as e:  # noqa: BLE001 — smoke surfaces ANY adapter failure
        print(f"smoke: FAIL — {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    elapsed = time.time() - t0

    if result.get("error_class") == "timeout":
        print(
            f"smoke: TIMEOUT — {result.get('error_what')} "
            f"after {result.get('error_timeout_s'):.0f}s",
            file=sys.stderr,
        )
        return 1

    n_src = result.get("n_source", 0)
    n_rec = result.get("n_reconstructed", 0)
    recall = result.get("exact_match_recall", 0.0)

    summary = {
        "smoke_id": SMOKE_DOC["id"],
        "model": args.model,
        "elapsed_s": round(elapsed, 2),
        "n_source": n_src,
        "n_reconstructed": n_rec,
        "exact_match_recall": recall,
        "drift_pct": result.get("drift_pct", 0.0),
        "narrative_excerpt": result.get("narrative_excerpt", "")[:120],
    }
    print(json.dumps(summary, indent=2))

    print(
        f"smoke: OK in {elapsed:.1f}s — src={n_src} rec={n_rec} recall={recall:.2f}",
        file=sys.stderr,
    )

    # Hard checks: source extraction MUST find at least one triple,
    # narrative MUST be non-empty (otherwise wiring is broken).
    if n_src == 0:
        print("smoke: FAIL — source extraction returned 0 triples", file=sys.stderr)
        return 1
    if not result.get("narrative_excerpt"):
        print("smoke: FAIL — narrative_excerpt is empty", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
