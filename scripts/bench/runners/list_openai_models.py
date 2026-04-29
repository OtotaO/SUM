"""List the frontier OpenAI models available to the active key.

Used before the §2.5 GPT-5.5 leg to confirm which snapshot id to
target. The OpenAI API returns model listings sorted by created
time; this script filters to recent frontier-class families
(``gpt-5*``, ``gpt-4.5*``, ``o4-*``, ``o3-*``) and prints them with
their pinned snapshot ids and creation dates.

Usage::

    OPENAI_API_KEY=... python -m scripts.bench.runners.list_openai_models

    # Or filter to a specific family:
    OPENAI_API_KEY=... python -m scripts.bench.runners.list_openai_models --prefix gpt-5

No spend; ``models.list`` is free.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
import sys


# Frontier-class prefixes worth surfacing for §2.5 work.
_DEFAULT_PREFIXES = ("gpt-5", "gpt-4.5", "o4-", "o3-")


async def list_models(prefixes: tuple[str, ...]) -> list[dict]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    page = await client.models.list()
    rows: list[dict] = []
    for m in page.data:
        if not any(m.id.startswith(p) for p in prefixes):
            continue
        rows.append(
            {
                "id": m.id,
                "created": dt.datetime.utcfromtimestamp(m.created).strftime("%Y-%m-%d"),
                "owned_by": getattr(m, "owned_by", "?"),
            }
        )
    rows.sort(key=lambda r: r["id"])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix", action="append", default=None,
        help="Only show models whose id starts with this prefix. "
             "Repeatable. Default: gpt-5 / gpt-4.5 / o4- / o3-.",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set", file=sys.stderr)
        return 2

    prefixes = tuple(args.prefix) if args.prefix else _DEFAULT_PREFIXES
    rows = asyncio.run(list_models(prefixes))

    if not rows:
        print(
            f"No models matching prefixes {prefixes} on this account.",
            file=sys.stderr,
        )
        print(
            "Re-run with --prefix gpt-4 (or any other family) to broaden the search.",
            file=sys.stderr,
        )
        return 1

    print(f"{'id':<40} {'created':<12} owned_by")
    print("-" * 70)
    for r in rows:
        print(f"{r['id']:<40} {r['created']:<12} {r['owned_by']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
