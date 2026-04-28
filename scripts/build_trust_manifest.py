#!/usr/bin/env python3
"""Build an UNSIGNED trust-root manifest from local release facts (Phase R0.2).

Gathers what's available in the local release context — git commit,
release tag, dist/ artifact hashes, JWKS contents — and emits the
``sum.trust_root.v1`` payload as JSON. The output is ready to be
signed with ``scripts/sign_trust_manifest.py``.

This script does NOT sign. It does NOT need any private key material.
It can be run on any machine with read access to the repo + dist/ +
the JWKS endpoint.

Usage:
    python scripts/build_trust_manifest.py \\
        --release v0.3.1 \\
        --commit "$(git rev-parse HEAD)" \\
        --dist-dir dist \\
        --jwks-url https://sum-demo.ototao.workers.dev/.well-known/jwks.json \\
        --algorithm-current sha256_64_v1 \\
        --algorithm-next sha256_128_v2 \\
        --out unsigned_manifest.json

See ``docs/TRUST_ROOT_FORMAT.md`` for the schema. Cross-references
``scripts/hash_dist.py`` for the artifact-hashing primitive (this
script reuses ``sha256_file`` and ``read_project_version``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


_HTTP_UA = "sum-build-trust-manifest/0.1 (+https://github.com/OtotaO/SUM)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch_jwks_bytes(url: str) -> tuple[bytes, dict[str, Any]]:
    """Fetch JWKS endpoint; return (raw_bytes, parsed_dict).

    The raw bytes are what the consumer hashes (the hash signs the
    actual served bytes, not a re-canonicalisation that might differ).
    """
    req = urllib.request.Request(url, headers={"user-agent": _HTTP_UA})
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
    return raw, json.loads(raw.decode("utf-8"))


def classify_dist_artifact(filename: str) -> str | None:
    """Infer artifact `kind` from filename. Returns None for files we
    don't know how to classify (caller skips them)."""
    if filename.endswith(".whl"):
        return "pypi-wheel"
    if filename.endswith(".tar.gz"):
        return "pypi-sdist"
    return None


def gather_dist_artifacts(dist_dir: Path) -> list[dict[str, Any]]:
    if not dist_dir.is_dir():
        raise SystemExit(f"dist directory not found: {dist_dir}")
    artifacts: list[dict[str, Any]] = []
    for p in sorted(dist_dir.iterdir()):
        if not p.is_file():
            continue
        kind = classify_dist_artifact(p.name)
        if kind is None:
            continue
        artifacts.append(
            {
                "name": p.name,
                "kind": kind,
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
                # Attestation status: this script can't know whether
                # PEP 740 / GitHub Artifact Attestation / Cosign were
                # applied — that's a CI-side fact. The signer (or
                # release CI) should override these via --pypi-
                # provenance / --github-attestation / --cosign-bundle
                # flags before signing. Default to "absent" so the
                # signed manifest is honest about what's known.
                "pypi_provenance": "absent",
                "github_attestation": "absent",
                "cosign_bundle": "absent",
            }
        )
    if not artifacts:
        raise SystemExit(
            f"no .whl or .tar.gz files found in {dist_dir}; "
            f"build with `python -m build --sdist --wheel --outdir {dist_dir}` first"
        )
    return artifacts


def build_manifest(
    *,
    release: str,
    commit: str,
    artifacts: list[dict[str, Any]],
    jwks_raw: bytes,
    jwks_parsed: dict[str, Any],
    algorithm_current: str,
    algorithm_next: str | None,
    revoked_kids: list[str],
) -> dict[str, Any]:
    """Build the unsigned manifest payload (everything inside
    ``payload`` of the signed envelope). The signer wraps this with
    schema + kid + jws."""
    current_kids = sorted(
        k.get("kid") for k in jwks_parsed.get("keys", []) if k.get("kid")
    )
    return {
        "issued_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "repo": "OtotaO/SUM",
        "commit": commit,
        "release": release,
        "artifacts": artifacts,
        "render_receipt_jwks": {
            "current_kids": list(current_kids),
            "revoked_kids": list(revoked_kids),
            "jwks_sha256": sha256_bytes(jwks_raw),
        },
        "algorithm_registry": {
            "prime_scheme_current": algorithm_current,
            "prime_scheme_next": algorithm_next,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, help="Release tag (e.g. v0.3.1).")
    parser.add_argument(
        "--commit",
        required=True,
        help="Git commit SHA the release is built from (40-char hex).",
    )
    parser.add_argument(
        "--dist-dir",
        default="dist",
        help="Directory containing the release artifacts (default: %(default)s).",
    )
    parser.add_argument(
        "--jwks-url",
        default="https://sum-demo.ototao.workers.dev/.well-known/jwks.json",
        help="Render-receipt JWKS endpoint to snapshot (default: %(default)s).",
    )
    parser.add_argument(
        "--algorithm-current",
        default="sha256_64_v1",
        help="Currently-active prime scheme (default: %(default)s).",
    )
    parser.add_argument(
        "--algorithm-next",
        default="sha256_128_v2",
        help='Next-planned prime scheme (default: %(default)s; pass "" for null).',
    )
    parser.add_argument(
        "--revoked-kid",
        action="append",
        default=[],
        help="Kid to mark as revoked (repeatable; default: empty list).",
    )
    parser.add_argument(
        "--pypi-provenance",
        default=None,
        help='Override pypi_provenance for all artifacts (e.g. "present").',
    )
    parser.add_argument(
        "--github-attestation",
        default=None,
        help='Override github_attestation for all artifacts (e.g. "present").',
    )
    parser.add_argument(
        "--cosign-bundle",
        default=None,
        help='Override cosign_bundle for all artifacts (e.g. "present").',
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: stdout).",
    )
    args = parser.parse_args()

    if len(args.commit) != 40 or not all(c in "0123456789abcdef" for c in args.commit):
        raise SystemExit(
            f"commit must be a 40-char lowercase hex SHA, got {args.commit!r}"
        )

    artifacts = gather_dist_artifacts(Path(args.dist_dir))
    for a in artifacts:
        if args.pypi_provenance is not None:
            a["pypi_provenance"] = args.pypi_provenance
        if args.github_attestation is not None:
            a["github_attestation"] = args.github_attestation
        if args.cosign_bundle is not None:
            a["cosign_bundle"] = args.cosign_bundle

    jwks_raw, jwks_parsed = fetch_jwks_bytes(args.jwks_url)

    payload = build_manifest(
        release=args.release,
        commit=args.commit,
        artifacts=artifacts,
        jwks_raw=jwks_raw,
        jwks_parsed=jwks_parsed,
        algorithm_current=args.algorithm_current,
        algorithm_next=args.algorithm_next or None,
        revoked_kids=args.revoked_kid,
    )

    out_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(out_text, encoding="utf-8")
        print(f"unsigned manifest payload written: {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(out_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
