"""
Single-command verifier for every receipt cited in the preprint.

Reads each `fixtures/bench_receipts/<file>.json` referenced from
`docs/arxiv/sheaf-detector-note-v0.md`, asserts the `bench_digest`
field matches the digest cited in the preprint prose (when one is
quoted), and reports a per-receipt OK / DRIFT / MISSING line.

This is the "one-line answer to *can I reproduce this?*" — a
reviewer (or a fresh-checkout CI run) gets a deterministic pass/fail
without having to re-run any LLM-mediated capture. The receipts are
already byte-stable per the §4.8 cross-machine matrix; this script
only verifies the *citation chain* between preprint prose and
on-disk receipts.

What it does NOT do:
  - Re-run the benches (those have their own pinned tests; see
    Tests/research/*).
  - Verify Phase 1 (LLM-capture) snapshots — those are already
    committed and the digest pins of the cited receipts are
    computed against them.

Output: a `sum.preprint_receipt_audit.v1` summary, byte-stable
across runs (it only reads on-disk JSON; no compute).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PREPRINT = REPO / "docs" / "arxiv" / "sheaf-detector-note-v0.md"
RECEIPTS_DIR = REPO / "fixtures" / "bench_receipts"


_RECEIPT_REF_RE = re.compile(
    r"fixtures/bench_receipts/([A-Za-z0-9_.-]+\.json)"
)
# Prefix-matching-prefix: digest cite of the form `7b364fc6…cc4b75e`
# (8 leading hex chars, ellipsis, 5-7 trailing hex chars). The preprint
# uses this contracted form throughout.
_DIGEST_CITE_RE = re.compile(
    r"`([0-9a-f]{8})…([0-9a-f]{4,8})`"
)


def _find_referenced_receipts(preprint_text: str) -> set[str]:
    """Receipt filenames referenced from the preprint."""
    return set(_RECEIPT_REF_RE.findall(preprint_text))


def _find_digest_cites_for_receipt(
    preprint_text: str, receipt_filename: str,
) -> list[tuple[str, str]]:
    """Return [(prefix, suffix)] for any digest cited *near* a
    reference to this receipt (within 600 chars of the reference)."""
    cites: list[tuple[str, str]] = []
    for m in re.finditer(re.escape(receipt_filename), preprint_text):
        start = max(0, m.start() - 600)
        end = min(len(preprint_text), m.end() + 600)
        nearby = preprint_text[start:end]
        for d in _DIGEST_CITE_RE.finditer(nearby):
            cites.append((d.group(1), d.group(2)))
    # de-dupe
    return list(set(cites))


def _walk_for_digests(obj, path: str = "") -> dict[str, str]:
    """Walk a parsed JSON object; collect every `bench_digest` field
    that appears anywhere in the tree, keyed by jq-style path."""
    found: dict[str, str] = {}
    if isinstance(obj, dict):
        if "bench_digest" in obj and isinstance(obj["bench_digest"], str):
            found[path or "$"] = obj["bench_digest"]
        for k, v in obj.items():
            found.update(_walk_for_digests(v, f"{path}.{k}" if path else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.update(_walk_for_digests(v, f"{path}[{i}]"))
    return found


def _digest_matches_cite(
    digests_in_receipt: dict[str, str], prefix: str, suffix: str,
) -> bool:
    for full in digests_in_receipt.values():
        if full.startswith(prefix) and full.endswith(suffix):
            return True
    return False


def main() -> int:
    if not PREPRINT.exists():
        print(f"ERROR: preprint not found at {PREPRINT}", file=sys.stderr)
        return 2
    text = PREPRINT.read_text()

    referenced = sorted(_find_referenced_receipts(text))
    print(f"Preprint references {len(referenced)} receipt(s).\n")

    n_ok = 0
    n_drift = 0
    n_missing = 0
    rows: list[dict[str, object]] = []

    for fname in referenced:
        path = RECEIPTS_DIR / fname
        if not path.exists():
            print(f"  [MISSING] {fname}")
            n_missing += 1
            rows.append({"file": fname, "status": "MISSING"})
            continue

        try:
            obj = json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  [PARSE-FAIL] {fname}: {e}")
            n_missing += 1
            rows.append({"file": fname, "status": "PARSE_FAIL", "error": str(e)})
            continue

        digests = _walk_for_digests(obj)
        cites = _find_digest_cites_for_receipt(text, fname)

        if not cites:
            # Receipt referenced but no nearby digest cite — that's fine,
            # not every reference is a digest cite.
            print(f"  [REF-OK]   {fname}  ({len(digests)} digest field(s); no nearby cite)")
            n_ok += 1
            rows.append({
                "file": fname, "status": "REF_OK_NO_CITE",
                "n_digests_in_receipt": len(digests),
            })
            continue

        # All cited prefix/suffix pairs must be matched by some digest
        # in the receipt.
        unmatched = [
            (p, s) for (p, s) in cites
            if not _digest_matches_cite(digests, p, s)
        ]
        if not unmatched:
            print(f"  [OK]       {fname}  ({len(cites)} cite(s) verified)")
            n_ok += 1
            rows.append({
                "file": fname, "status": "OK",
                "n_cites_verified": len(cites),
            })
        else:
            print(f"  [DRIFT]    {fname}  unmatched cite(s): {unmatched}")
            print(f"             receipt's actual digest(s): "
                  f"{list(digests.values())[:3]}")
            n_drift += 1
            rows.append({
                "file": fname, "status": "DRIFT",
                "unmatched_cites": [list(c) for c in unmatched],
                "actual_digests_sample": list(digests.values())[:3],
            })

    print()
    print(f"Summary: {n_ok} ok, {n_drift} drift, {n_missing} missing.")

    summary: dict[str, object] = {
        "schema": "sum.preprint_receipt_audit.v1",
        "preprint": str(PREPRINT.relative_to(REPO)),
        "n_referenced": len(referenced),
        "n_ok": n_ok,
        "n_drift": n_drift,
        "n_missing": n_missing,
        "rows": rows,
    }
    print()
    print(json.dumps(summary, indent=2))

    if n_drift or n_missing:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
