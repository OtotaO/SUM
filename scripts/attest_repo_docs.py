"""Self-attestation: run SUM on its own canonical docs.

The thesis statement of this project — verifiable bidirectional
knowledge distillation — is most credible when applied to the
project itself. This script feeds the repo's load-bearing
documentation files through the omni-format pivot + the
``DeterministicSieve`` extractor + the canonical-bundle codec, and
emits one ``CanonicalBundle`` per document plus a summary index.

Output artifacts (committed to the repo, refreshed by CI):

  * ``meta/self_attestation.jsonl`` — JSONL of bundles (one
    compact JSON per line, in canonical doc order). Each line is
    a complete CanonicalBundle, structurally identical to what
    ``sum attest --input <doc>`` would produce. Fully verifiable
    via ``sum verify`` (state-integer reconstruction) without any
    secret.

  * ``meta/self_attestation.summary.json`` — small index pointing
    at the JSONL: per-doc paths, source URIs (sha256: of doc
    bytes), markdown_sha256 (of the markdown intermediate), state-
    integer digit counts, axiom counts, generated_at. Lets
    downstream consumers route at the path/URI level without
    parsing every bundle.

Drift gate (``--check``): re-runs the attestation in memory,
strips wall-clock fields, and compares to the on-disk artifacts.
Exits non-zero if any doc's content has changed without the self-
attestation being refreshed. Same mechanism as
``scripts/repo_manifest.py``.

Usage::

    # Refresh both artifacts
    python -m scripts.attest_repo_docs

    # CI gate: fails if either artifact is stale
    python -m scripts.attest_repo_docs --check

This is the first deliberate "use the system to make more
systems" move: SUM's verifiable substrate applied to SUM's own
docs. Every claim in README / CHANGELOG / PROOF_BOUNDARY /
FEATURE_CATALOG / RENDER_RECEIPT_FORMAT is now bound to a
content-addressable receipt anyone can replay.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "sum.self_attestation.v1"

# Canonical doc list. Adding a doc here is a deliberate scope
# decision: the doc is then load-bearing for the self-attestation
# claim, and a bytes-level edit to it requires a refresh.
CANONICAL_DOCS: tuple[str, ...] = (
    "README.md",
    "CHANGELOG.md",
    "docs/PROOF_BOUNDARY.md",
    "docs/FEATURE_CATALOG.md",
    "docs/RENDER_RECEIPT_FORMAT.md",
)

OUT_JSONL = REPO_ROOT / "meta" / "self_attestation.jsonl"
OUT_SUMMARY = REPO_ROOT / "meta" / "self_attestation.summary.json"


# ---------------------------------------------------------------------
# Per-doc attestation (in-process, no subprocess churn)
# ---------------------------------------------------------------------


def _attest_one_doc(rel_path: str) -> dict:
    """Run the same pipeline ``sum attest --input <rel_path>`` would
    run, in-process. Returns the bundle dict."""
    abs_path = REPO_ROOT / rel_path
    if not abs_path.exists():
        raise FileNotFoundError(f"self-attest: missing canonical doc {rel_path}")

    # Lazy imports — keep --check fast when we're just diffing JSON.
    from sum_engine_internal.adapters.format_pivot import convert_to_markdown
    from sum_engine_internal.algorithms.chunked_corpus import state_for_corpus
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec

    converted = convert_to_markdown(abs_path)
    text = converted.markdown.strip()
    if not text:
        raise RuntimeError(f"self-attest: doc {rel_path} converted to empty markdown")

    algebra = GodelStateAlgebra()  # type: ignore[no-untyped-call]
    state, triples = state_for_corpus(text, algebra)
    if not triples:
        raise RuntimeError(
            f"self-attest: doc {rel_path} extracted zero triples "
            f"(input may be too short or all-prose-no-claims)"
        )

    tome_generator = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(algebra, tome_generator)
    bundle = codec.export_bundle(state, branch="main", title=f"SUM self-attestation: {rel_path}")

    bundle["sum_cli"] = {
        "extractor": "sieve",
        "source_uri": converted.source_uri,
        "source_path": rel_path,
        "input_format": converted.input_format,
        "converter": converted.converter,
        "source_bytes_len": converted.source_bytes_len,
        "markdown_sha256": converted.markdown_sha256,
        "axiom_count": len(triples),
        "self_attestation": True,
    }
    return bundle


# ---------------------------------------------------------------------
# Stable view (drift comparison)
# ---------------------------------------------------------------------


def _stable_bundle_view(bundle: dict) -> dict:
    """Strip wall-clock fields so --check only fails on substantive
    drift (state_integer, source_uri, markdown_sha256)."""
    b = json.loads(json.dumps(bundle))  # deep copy
    # CanonicalBundle's wall-clock field
    b.pop("timestamp", None)
    sidecar = b.get("sum_cli")
    if isinstance(sidecar, dict):
        sidecar.pop("generated_at", None)
        sidecar.pop("cli_version", None)
    return b


def _stable_jsonl(jsonl_text: str) -> list[dict]:
    return [_stable_bundle_view(json.loads(ln)) for ln in jsonl_text.splitlines() if ln.strip()]


# ---------------------------------------------------------------------
# Build / write artifacts
# ---------------------------------------------------------------------


def build_self_attestation() -> tuple[list[dict], dict]:
    """Build (bundles, summary). Pure compute — does not write to disk."""
    bundles: list[dict] = []
    summary_entries: list[dict] = []
    for rel_path in CANONICAL_DOCS:
        bundle = _attest_one_doc(rel_path)
        bundles.append(bundle)
        sidecar = bundle["sum_cli"]
        summary_entries.append(
            {
                "path": rel_path,
                "source_uri": sidecar["source_uri"],
                "input_format": sidecar["input_format"],
                "converter": sidecar["converter"],
                "markdown_sha256": sidecar["markdown_sha256"],
                "axiom_count": sidecar["axiom_count"],
                "state_integer_digits": len(bundle["state_integer"]),
            }
        )

    summary = {
        "schema": SCHEMA,
        "issued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "doc_count": len(bundles),
        "docs": summary_entries,
        "jsonl_path": str(OUT_JSONL.relative_to(REPO_ROOT)),
        "verifier_recipe": (
            "Each line of meta/self_attestation.jsonl is a CanonicalBundle. "
            "Replay: pip install sum-engine[sieve,omni-format] && "
            "for each line, write to bundle.json and run `sum verify --input bundle.json`. "
            "State integer is reconstructible from the canonical_tome alone "
            "(no secret needed); a passing verify proves the bundle's claims "
            "match the doc bytes whose sha256 is in sum_cli.source_uri."
        ),
    }
    return bundles, summary


def _write_jsonl(bundles: list[dict]) -> None:
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for b in bundles:
            json.dump(b, f, separators=(",", ":"))
            f.write("\n")


def _write_summary(summary: dict) -> None:
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Drift-gate mode: rebuild self-attestation in memory, "
            "compare to on-disk artifacts (stable view), exit non-zero "
            "on substantive drift. CI runs this on every PR."
        ),
    )
    args = parser.parse_args()

    if args.check:
        if not OUT_JSONL.exists() or not OUT_SUMMARY.exists():
            print(
                "self-attest: meta/self_attestation.{jsonl,summary.json} missing. "
                "Refresh: python -m scripts.attest_repo_docs",
                file=sys.stderr,
            )
            return 2

        # Load on-disk
        on_disk_jsonl = OUT_JSONL.read_text(encoding="utf-8")
        on_disk_summary = json.loads(OUT_SUMMARY.read_text(encoding="utf-8"))

        # Rebuild
        fresh_bundles, fresh_summary = build_self_attestation()

        # Compare bundles (stable view)
        on_disk_stable = _stable_jsonl(on_disk_jsonl)
        fresh_stable = [_stable_bundle_view(b) for b in fresh_bundles]
        if on_disk_stable != fresh_stable:
            print(
                "SELF-ATTESTATION DRIFT: bundles changed without refresh.\n"
                "Refresh: python -m scripts.attest_repo_docs",
                file=sys.stderr,
            )
            # Surface which doc(s) drifted
            for old, new in zip(on_disk_stable, fresh_stable):
                if old != new:
                    path = (new.get("sum_cli") or {}).get("source_path", "?")
                    old_state = old.get("state_integer", "?")[:16] + "…"
                    new_state = new.get("state_integer", "?")[:16] + "…"
                    print(
                        f"  {path}: state {old_state} → {new_state}",
                        file=sys.stderr,
                    )
            return 1

        # Compare summary (excluding issued_at)
        on_disk_summary_stable = {k: v for k, v in on_disk_summary.items() if k != "issued_at"}
        fresh_summary_stable = {k: v for k, v in fresh_summary.items() if k != "issued_at"}
        if on_disk_summary_stable != fresh_summary_stable:
            print(
                "SELF-ATTESTATION DRIFT: summary index changed without refresh.\n"
                "Refresh: python -m scripts.attest_repo_docs",
                file=sys.stderr,
            )
            return 1

        print(
            f"self-attest: meta/self_attestation.* current "
            f"({len(fresh_bundles)} docs, stable-fields match)",
            file=sys.stderr,
        )
        return 0

    # Default: refresh
    bundles, summary = build_self_attestation()
    _write_jsonl(bundles)
    _write_summary(summary)
    print(
        f"self-attest: wrote {len(bundles)} bundles → "
        f"{OUT_JSONL.relative_to(REPO_ROOT)} + summary",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
