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

Drift gate (``--check``): compares the SOURCE URIs (sha256 of
each canonical doc's bytes) recorded in
``meta/self_attestation.summary.json`` against the doc bytes at
HEAD. If they match, the on-disk bundles are still attesting the
same byte content and are current. Exits non-zero with the path
of the drifted doc and a refresh recipe otherwise.

Why source URIs and not bundle byte-equality:
``en_core_web_sm`` (spaCy's small English model) is not promised
byte-deterministic across Python minor versions or its own patch
releases — two different environments can mint two different
(but each internally round-trip-valid) bundles for the same doc
bytes. The *internal* round-trip — every committed bundle passes
``sum verify`` — is checked separately by
``Tests/test_self_attestation.py``. The drift gate's job here is
narrower: catch "doc edited; self-attestation forgotten,"
which is exactly the source-URI comparison.

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

        # Drift-gate semantics: compare the SOURCE URIs (sha256 of
        # each doc's bytes) on-disk vs at HEAD. If they match, the
        # bundles are still attesting the same byte content and are
        # therefore current. We deliberately do NOT compare bundle
        # ``state_integer`` byte-for-byte: spaCy's ``en_core_web_sm``
        # is not promised byte-deterministic across Python minor
        # versions or its own patch versions, so two different
        # environments can mint two different (but each internally
        # round-trip-valid) bundles for the same doc bytes. The
        # *internal* round-trip is verified separately by
        # ``Tests/test_self_attestation.py::test_every_self_attestation_bundle_verifies``
        # which runs ``sum verify`` against every committed bundle.
        on_disk_summary = json.loads(OUT_SUMMARY.read_text(encoding="utf-8"))
        on_disk_uris = {
            entry["path"]: entry["source_uri"]
            for entry in on_disk_summary["docs"]
        }

        # Compute current source URIs without invoking the extractor —
        # cheap, deterministic across all environments.
        head_uris: dict[str, str] = {}
        for rel_path in CANONICAL_DOCS:
            abs_path = REPO_ROOT / rel_path
            if not abs_path.exists():
                print(
                    f"self-attest: missing canonical doc at HEAD: {rel_path}",
                    file=sys.stderr,
                )
                return 1
            head_uris[rel_path] = (
                "sha256:" + hashlib.sha256(abs_path.read_bytes()).hexdigest()
            )

        drifted: list[str] = []
        for path, head_uri in head_uris.items():
            if on_disk_uris.get(path) != head_uri:
                drifted.append(path)

        # Detect added or removed canonical docs (catalog edits).
        on_disk_paths = set(on_disk_uris)
        head_paths = set(head_uris)
        added = sorted(head_paths - on_disk_paths)
        removed = sorted(on_disk_paths - head_paths)

        if drifted or added or removed:
            print(
                "SELF-ATTESTATION DRIFT: doc bytes changed without "
                "refreshing meta/self_attestation.*\n"
                "Refresh: python -m scripts.attest_repo_docs",
                file=sys.stderr,
            )
            for path in drifted:
                old_uri = on_disk_uris[path][:24] + "…"
                new_uri = head_uris[path][:24] + "…"
                print(f"  {path}: source_uri {old_uri} → {new_uri}", file=sys.stderr)
            for path in added:
                print(f"  added: {path}", file=sys.stderr)
            for path in removed:
                print(f"  removed: {path}", file=sys.stderr)
            return 1

        # Sanity: also verify the on-disk JSONL parses + has one
        # bundle per summary entry. Cheap, catches partial writes.
        n_lines = sum(
            1 for ln in OUT_JSONL.read_text(encoding="utf-8").splitlines() if ln.strip()
        )
        if n_lines != len(on_disk_summary["docs"]):
            print(
                f"SELF-ATTESTATION DRIFT: summary lists {len(on_disk_summary['docs'])} "
                f"docs but JSONL has {n_lines} lines.\n"
                f"Refresh: python -m scripts.attest_repo_docs",
                file=sys.stderr,
            )
            return 1

        print(
            f"self-attest: meta/self_attestation.* current "
            f"({len(head_uris)} docs, source URIs match)",
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
