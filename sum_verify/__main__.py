"""``python -m sum_verify <receipt.json> --jwks <jwks.json> [--losses <losses.json>]``

A minimal command-line on-ramp for the verify SDK — the same verdict the
library returns, for a quick offline check without writing code. Exit 0 =
verified, 1 = rejected, 2 = usage error. For richer output (perspective
cohorts, the layered ``--explain``) use the full ``sum verify-meaning``
CLI; this is deliberately tiny and dependency-light.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from sum_verify import SUPPORTED_SCHEMAS, __version__, verify


def _read_json(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _unwrap_loss_vector(obj: Any) -> Any:
    # Accept both a bare ``[..]`` vector and the metadata-wrapped shape the
    # committed fixtures use (``{"losses": [..], "judge": .., "note": ..}``)
    # so ``--losses <committed losses file>`` replays the golden out of the
    # box — the same contract as ``sum verify-meaning``.
    if isinstance(obj, dict):
        for key in ("losses", "values"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sum_verify",
        description=(
            "Verify a SUM receipt (meaning-risk / render / transform) "
            "with no heavy dependencies."
        ),
    )
    parser.add_argument("receipt", help="receipt envelope JSON ('-' for stdin)")
    parser.add_argument(
        "--jwks", required=True, help="issuer JWKS JSON (/.well-known/jwks.json)"
    )
    parser.add_argument(
        "--losses",
        help=(
            "per-pair losses (bare list or {'losses': [...]}) to replay a "
            "meaning-risk receipt's bound offline"
        ),
    )
    parser.add_argument(
        "--max-age-seconds",
        type=int,
        default=None,
        help="reject receipts whose signed_at is older than this many seconds",
    )
    parser.add_argument("--version", action="version", version=f"sum_verify {__version__}")
    args = parser.parse_args(argv)

    try:
        receipt = _read_json(args.receipt)
        jwks = _read_json(args.jwks)
        losses = _unwrap_loss_vector(_read_json(args.losses)) if args.losses else None
    except (OSError, json.JSONDecodeError) as e:
        print(f"sum_verify: cannot read input: {e}", file=sys.stderr)
        return 2

    schema = receipt.get("schema") if isinstance(receipt, dict) else None
    if schema not in SUPPORTED_SCHEMAS:
        print(
            f"sum_verify: unsupported schema {schema!r}; handles "
            f"{', '.join(SUPPORTED_SCHEMAS)}",
            file=sys.stderr,
        )
        return 2

    try:
        result = verify(receipt, jwks, losses=losses, max_age_seconds=args.max_age_seconds)
    except Exception as e:  # noqa: BLE001 — surface every failure as rc=1
        print(
            json.dumps({"verified": False, "error": type(e).__name__, "detail": str(e)})
        )
        return 1

    payload = result if isinstance(result, dict) else getattr(result, "payload", {})
    verdict = {
        "verified": True,
        "schema": schema,
        "replayed": losses is not None and schema == "sum.meaning_risk_receipt.v1",
    }
    if isinstance(payload, dict):
        if "scorer" in payload:
            verdict["scorer"] = payload.get("scorer")
        if "not_covered" in payload:
            verdict["not_covered"] = payload.get("not_covered")
    if schema == "sum.meaning_risk_receipt.v1":
        # Credibility hygiene (unsigned, corpus-agnostic): a clean PASS here is
        # a CRYPTOGRAPHIC fact (valid signature + a bound the committed losses
        # replay to) — NOT evidence that meaning was preserved. The bound is
        # over a NAMED PROXY; where that proxy has been measured against human
        # faithfulness judgments (SummEval) it correlated only modestly
        # (Spearman rho ~= 0.27-0.33). Directionally valid, not a substitute
        # for human review. We deliberately do NOT bake a number into a signed
        # field (the SummEval rho was measured on a different corpus+judge than
        # any given receipt's). See docs/PROOF_BOUNDARY.md.
        verdict["proxy_caveat"] = (
            "verified=true is a cryptographic fact (signature + replayed "
            "bound), not evidence meaning was preserved. The bound is over a "
            "named proxy; vs human judgments (SummEval) the proxy correlated "
            "only modestly (Spearman rho ~0.27-0.33). Not a substitute for "
            "human review."
        )
    print(json.dumps(verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
