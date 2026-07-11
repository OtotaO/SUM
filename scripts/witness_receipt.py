"""Witness log — append-only, hash-chained public record of issued receipts.

Closes (the first rung of) the witness gap: a self-signed receipt proves
the issuer's key signed it, but nothing about WHEN it existed — the issuer
could re-sign a doctored payload tomorrow and claim it was always so. This
log makes each receipt's existence a public, ordered, tamper-evident event:

- Each entry commits the receipt's canonical hash (RFC 8785 JCS bytes of
  the full envelope) plus the hash of the previous entry — an in-file
  hash chain, so editing history breaks every later entry.
- The log lives in the public repo; every append rides a commit, so the
  claim "this receipt existed, unaltered, by <date>" is checkable against
  git history and the public host's records.

Trust model, honestly (docs/TRANSPARENCY_LOG.md is the full statement):
this v1 is a WEAK witness — a public git host is not a verifiable-
timestamp authority, and history is force-pushable (detectably, not
provably). It upgrades "trust my key" to "trust my key + the public
record of when I published", which is what a disputant checks first.
The next rung, when one external relying party exists, is mirroring
entries to an external transparency log (e.g. Sigstore Rekor), which
makes the timestamp third-party-attested. Do not present this log as
that rung.

Usage:
  python scripts/witness_receipt.py append <receipt.json> [--note TEXT]
  python scripts/witness_receipt.py verify [--no-files]

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sum_engine_internal.infrastructure.jcs import canonicalize  # noqa: E402

LOG_PATH = REPO / "transparency" / "log.jsonl"


def canonical_receipt_hash(envelope: dict) -> str:
    """``sha256-<hex>`` over the RFC 8785 canonical bytes of the full
    envelope (schema + kid + payload + jws) — the same hash shape every
    other ``*_hash`` field in the receipt family uses."""
    return "sha256-" + hashlib.sha256(canonicalize(envelope)).hexdigest()


def _entry_hash(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "entry_hash"}
    return "sha256-" + hashlib.sha256(canonicalize(body)).hexdigest()


def read_log(log_path: Path = LOG_PATH) -> list[dict]:
    if not log_path.exists():
        return []
    entries = []
    for i, line in enumerate(log_path.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"log line {i + 1} is not valid JSON: {e}") from e
    return entries


def append_entry(
    receipt_path: Path,
    *,
    log_path: Path = LOG_PATH,
    note: "str | None" = None,
    now: "str | None" = None,
) -> dict:
    envelope = json.loads(receipt_path.read_text())
    if not isinstance(envelope, dict) or "schema" not in envelope:
        raise ValueError(
            f"{receipt_path} is not a receipt envelope (no 'schema' field)"
        )
    entries = read_log(log_path)
    receipt_hash = canonical_receipt_hash(envelope)
    for e in entries:
        if e["receipt_hash"] == receipt_hash:
            raise ValueError(
                f"receipt already witnessed at seq {e['seq']} "
                f"({e['witnessed_at']}) — the log is append-once per receipt"
            )
    try:
        source = str(receipt_path.resolve().relative_to(REPO))
    except ValueError:
        source = receipt_path.name  # out-of-repo receipt: record basename
    entry = {
        "seq": len(entries) + 1,
        "witnessed_at": now
        or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema": envelope["schema"],
        "kid": envelope.get("kid"),
        "receipt_hash": receipt_hash,
        "source_path": source,
        "prev_entry_hash": entries[-1]["entry_hash"] if entries else None,
    }
    if note:
        entry["note"] = note
    entry["entry_hash"] = _entry_hash(entry)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    return entry


def verify_log(
    log_path: Path = LOG_PATH, *, check_files: bool = True
) -> dict:
    """Recompute the whole chain. Returns a report; ``ok`` is False on any
    break (bad seq, broken prev-chain, entry-hash mismatch, or — when
    ``check_files`` — a witnessed receipt file present in the repo whose
    canonical hash no longer matches its entry)."""
    entries = read_log(log_path)
    problems = []
    prev_hash = None
    for e in entries:
        i = e.get("seq")
        if i != (entries.index(e) + 1):
            problems.append(f"seq {i}: non-monotonic sequence")
        if e.get("prev_entry_hash") != prev_hash:
            problems.append(f"seq {i}: prev_entry_hash chain broken")
        if _entry_hash(e) != e.get("entry_hash"):
            problems.append(f"seq {i}: entry_hash mismatch (entry edited)")
        if check_files:
            src = REPO / e.get("source_path", "")
            if src.is_file():
                actual = canonical_receipt_hash(json.loads(src.read_text()))
                if actual != e["receipt_hash"]:
                    problems.append(
                        f"seq {i}: {e['source_path']} no longer hashes to "
                        f"its witnessed receipt_hash"
                    )
        # Chain on the RECOMPUTED hash, not the stored one: an attacker who
        # edits an entry AND re-stamps its entry_hash would otherwise leave
        # a chain that still "links"; recomputing makes any edit cascade
        # into every later entry's prev check.
        prev_hash = _entry_hash(e)
    return {"ok": not problems, "n_entries": len(entries), "problems": problems}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_append = sub.add_parser("append", help="witness a receipt file")
    p_append.add_argument("receipt", type=Path)
    p_append.add_argument("--note", default=None)
    p_append.add_argument("--log", type=Path, default=LOG_PATH)
    p_verify = sub.add_parser("verify", help="verify the whole chain")
    p_verify.add_argument("--log", type=Path, default=LOG_PATH)
    p_verify.add_argument(
        "--no-files", action="store_true",
        help="skip re-hashing witnessed receipt files present in the repo",
    )
    args = ap.parse_args()

    if args.cmd == "append":
        entry = append_entry(args.receipt, log_path=args.log, note=args.note)
        print(json.dumps(entry, indent=1))
        print(
            "witnessed — commit transparency/log.jsonl to make it public",
            file=sys.stderr,
        )
        return 0
    report = verify_log(args.log, check_files=not args.no_files)
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
