"""Phase E.1 v0.9.D — receipt-aware bench audit runner.

Hits ``/api/render`` against the live Worker for a small set of cells,
captures each render receipt, runs the v0.9.C Python verifier
(``sum_engine_internal.render_receipt.verify_receipt``) on it, emits an
NDJSON audit trail file. Same audit trail can be replayed offline via
``--replay`` to re-verify every receipt without re-hitting the Worker.

The audit-trail property: a bench run becomes its own verifiable
attestation. Anyone with the saved NDJSON + the JWKS at the kid's
issuance time can re-verify that the engine produced the claimed
renders, without re-paying the LLM cost or trusting the original
runner's word.

Distinct from the existing slider drift bench (Tests/benchmarks/
slider_drift_bench.py) which uses the local Python adapter for
fact-preservation measurement and doesn't touch render receipts.
This runner is opt-in, narrow, and only exercises the receipt-audit
contract — it is NOT a replacement for the drift bench.

Usage:
    # Live audit (hits the worker, costs LLM tokens):
    python -m scripts.bench.runners.receipt_audit \\
        --worker https://sum-demo.ototao.workers.dev \\
        --cells 8 \\
        --out audit_trail.ndjson

    # Offline replay (no network, no LLM cost):
    python -m scripts.bench.runners.receipt_audit \\
        --replay audit_trail.ndjson

Cells are deterministic by default: same triples, varying slider
positions across the bin centres (audience / formality / length /
perspective at {0.1, 0.3, 0.5, 0.7, 0.9}). Density at 1.0 + four
neutral LLM axes hits the canonical (deterministic) path; non-neutral
positions exercise the LLM path.

The runner requires sum-engine[receipt-verify] (joserfc) for the
Python verifier. It does NOT require any LLM SDK — the Worker handles
the LLM call server-side; the runner only reads the response.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Cell generator
# ---------------------------------------------------------------------------


def generate_cells(n: int) -> list[dict[str, Any]]:
    """Deterministic cell generator: same triples, varying sliders.

    The first cell always exercises the canonical (deterministic) path:
    density=1.0, all four LLM axes at neutral 0.5. Subsequent cells
    move one axis at a time off neutral so the runner exercises both
    the canonical and the LLM render paths within a small N.
    """
    triples = [
        ["alice", "graduated", "2012"],
        ["alice", "born", "1990"],
    ]

    cells = []

    # Cell 0: canonical path (all neutral).
    cells.append(
        {
            "name": "canonical_neutral",
            "triples": triples,
            "slider_position": {
                "density": 1.0,
                "length": 0.5,
                "formality": 0.5,
                "audience": 0.5,
                "perspective": 0.5,
            },
            "expected_path": "canonical",
        }
    )

    # Cells 1+: LLM path. One axis at a time off neutral.
    axis_positions = [
        ("formality", 0.7),
        ("audience", 0.3),
        ("length", 0.7),
        ("perspective", 0.3),
        ("formality", 0.9),
        ("audience", 0.1),
        ("length", 0.3),
    ]
    for axis, value in axis_positions[: max(0, n - 1)]:
        slider = {
            "density": 1.0,
            "length": 0.5,
            "formality": 0.5,
            "audience": 0.5,
            "perspective": 0.5,
        }
        slider[axis] = value
        cells.append(
            {
                "name": f"llm_{axis}_{value}",
                "triples": triples,
                "slider_position": slider,
                "expected_path": "llm",
            }
        )

    return cells[:n]


# ---------------------------------------------------------------------------
# Live audit
# ---------------------------------------------------------------------------


# Cloudflare-fronted endpoints reject the default Python-urllib User-
# Agent with 403; pretend to be a normal client. The Worker's actual
# auth is OIDC for /api/render (none for read endpoints); the UA is
# only for getting past the CDN's bot-defense layer.
_HTTP_UA = "sum-receipt-audit/0.1 (+https://github.com/OtotaO/SUM)"


def fetch_jwks(worker_url: str) -> dict[str, Any]:
    url = f"{worker_url.rstrip('/')}/.well-known/jwks.json"
    req = urllib.request.Request(url, headers={"user-agent": _HTTP_UA})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_render(worker_url: str, cell: dict[str, Any]) -> dict[str, Any]:
    url = f"{worker_url.rstrip('/')}/api/render"
    body = json.dumps(
        {
            "triples": cell["triples"],
            "slider_position": cell["slider_position"],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json", "user-agent": _HTTP_UA},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def classify_path(receipt_payload: dict[str, Any]) -> str:
    """canonical-path receipts use model=canonical-deterministic-v0;
    LLM-path receipts carry the actual model id."""
    model = receipt_payload.get("model", "")
    if model.startswith("canonical-deterministic"):
        return "canonical"
    return "llm"


@dataclasses.dataclass
class AuditEntry:
    name: str
    expected_path: str
    actual_path: str
    receipt_kid: str
    receipt_model: str
    receipt_provider: str
    receipt_signed_at: str
    verify_outcome: str  # "verified" | "rejected"
    verify_error_class: str | None
    verify_error_message: str | None
    wall_clock_ms: float
    render_id: str
    response: dict[str, Any]  # full /api/render response (preserved)
    jwks: dict[str, Any]      # JWKS at audit time (preserved)


def audit_cell(
    cell: dict[str, Any], worker_url: str, jwks: dict[str, Any]
) -> AuditEntry:
    from sum_engine_internal.render_receipt import VerifyError, verify_receipt

    t0 = time.perf_counter()
    response = call_render(worker_url, cell)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    receipt = response.get("render_receipt") or {}
    payload = receipt.get("payload") or {}
    actual_path = classify_path(payload)

    verify_outcome = "verified"
    error_class: str | None = None
    error_message: str | None = None
    try:
        verify_receipt(receipt, jwks)
    except VerifyError as e:
        verify_outcome = "rejected"
        error_class = e.error_class
        error_message = str(e)

    return AuditEntry(
        name=cell["name"],
        expected_path=cell["expected_path"],
        actual_path=actual_path,
        receipt_kid=receipt.get("kid", ""),
        receipt_model=payload.get("model", ""),
        receipt_provider=payload.get("provider", ""),
        receipt_signed_at=payload.get("signed_at", ""),
        verify_outcome=verify_outcome,
        verify_error_class=error_class,
        verify_error_message=error_message,
        wall_clock_ms=elapsed_ms,
        render_id=response.get("render_id", ""),
        response=response,
        jwks=jwks,
    )


def run_live(args: argparse.Namespace) -> int:
    cells = generate_cells(args.cells)
    print(f"Receipt audit: {len(cells)} cell(s) against {args.worker}")
    print("Fetching JWKS…", flush=True)
    jwks = fetch_jwks(args.worker)
    kids = [k.get("kid") for k in jwks.get("keys", [])]
    print(f"  JWKS kids: {kids}\n")

    entries: list[AuditEntry] = []
    for i, cell in enumerate(cells):
        print(f"  [{i + 1}/{len(cells)}] {cell['name']} (expected: {cell['expected_path']}-path)…", flush=True)
        try:
            entry = audit_cell(cell, args.worker, jwks)
        except urllib.error.HTTPError as e:
            print(f"    HTTP error: {e.code} {e.reason}")
            return 1
        except urllib.error.URLError as e:
            print(f"    network error: {e.reason}")
            return 1
        entries.append(entry)
        if entry.verify_outcome == "verified":
            print(
                f"    ✓ verified (path={entry.actual_path}, "
                f"model={entry.receipt_model[:32]}, {entry.wall_clock_ms:.0f} ms)"
            )
        else:
            print(
                f"    ✗ {entry.verify_outcome} ({entry.verify_error_class}): "
                f"{entry.verify_error_message}"
            )

    write_audit_trail(args.out, entries, jwks)
    return summarise(entries)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Audit-trail I/O + summary
# ---------------------------------------------------------------------------


AUDIT_SCHEMA = "sum.receipt_audit_trail.v1"


def write_audit_trail(
    out: str | None, entries: list[AuditEntry], jwks: dict[str, Any]
) -> None:
    if not out:
        return
    out_path = Path(out)
    with out_path.open("w", encoding="utf-8") as f:
        # Header line: schema + JWKS snapshot. Replay uses this to
        # find the keys; live audit captures kid + JWKS together so
        # rotation after the audit doesn't invalidate replay.
        header = {
            "schema": AUDIT_SCHEMA,
            "kind": "header",
            "issued_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "jwks": jwks,
            "n_cells": len(entries),
        }
        f.write(json.dumps(header) + "\n")
        for e in entries:
            f.write(json.dumps({"kind": "entry", **dataclasses.asdict(e)}) + "\n")
    print(f"\nAudit trail written: {out_path} ({len(entries)} entries)")


def _read_audit_trail(path: Path) -> tuple[dict[str, Any], list[AuditEntry]]:
    jwks: dict[str, Any] = {}
    entries: list[AuditEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("kind") == "header":
                jwks = d.get("jwks", {})
            elif d.get("kind") == "entry":
                fields = {fld.name for fld in dataclasses.fields(AuditEntry)}
                payload = {k: v for k, v in d.items() if k in fields}
                if "jwks" not in payload:
                    payload["jwks"] = jwks
                entries.append(AuditEntry(**payload))
    return jwks, entries


def summarise(entries: list[AuditEntry]) -> int:
    n = len(entries)
    n_verified = sum(1 for e in entries if e.verify_outcome == "verified")
    n_rejected = n - n_verified
    n_canonical = sum(1 for e in entries if e.actual_path == "canonical")
    n_llm = sum(1 for e in entries if e.actual_path == "llm")
    n_path_match = sum(1 for e in entries if e.actual_path == e.expected_path)
    models = sorted({e.receipt_model for e in entries if e.receipt_model})
    print()
    print(f"Audit summary:")
    print(f"  cells:                    {n}")
    print(f"  receipts verified:        {n_verified}/{n} ({100 * n_verified / n:.0f}%)")
    print(f"  receipts rejected:        {n_rejected}")
    print(f"  canonical-path receipts:  {n_canonical}")
    print(f"  LLM-path receipts:        {n_llm}")
    print(f"  expected/actual path match: {n_path_match}/{n}")
    print(f"  models seen:              {models}")
    return 0 if n_rejected == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--worker",
        default="https://sum-demo.ototao.workers.dev",
        help="Worker base URL (default: %(default)s)",
    )
    mode.add_argument(
        "--replay",
        metavar="FILE",
        help="Re-verify a saved audit trail without hitting the Worker.",
    )
    parser.add_argument(
        "--cells",
        type=int,
        default=8,
        help="Number of cells (default: %(default)s; max %(default)s).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path to save the NDJSON audit trail (live mode only).",
    )
    args = parser.parse_args()

    if args.replay:
        # Replace the placeholder _decode_entry path with the
        # cleaner _read_audit_trail function.
        from sum_engine_internal.render_receipt import VerifyError, verify_receipt
        path = Path(args.replay)
        print(f"Receipt audit replay: {path}")
        jwks, entries = _read_audit_trail(path)
        n_verified = 0
        n_rejected = 0
        for i, entry in enumerate(entries):
            receipt = entry.response.get("render_receipt") or {}
            try:
                verify_receipt(receipt, jwks)
                print(
                    f"  [{i + 1}/{len(entries)}] {entry.name}: replay ✓ verified "
                    f"(path={entry.actual_path})"
                )
                n_verified += 1
            except VerifyError as e:
                print(
                    f"  [{i + 1}/{len(entries)}] {entry.name}: replay ✗ rejected "
                    f"({e.error_class}): {e}"
                )
                n_rejected += 1
        print(
            f"\nReplay summary: {n_verified}/{len(entries)} verified, "
            f"{n_rejected} rejected"
        )
        return 0 if n_rejected == 0 else 1

    return run_live(args)


if __name__ == "__main__":
    sys.exit(main())
