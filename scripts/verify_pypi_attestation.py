#!/usr/bin/env python3
"""Verify PEP 740 attestations on a published PyPI artifact.

Three checks, all must pass:
  1. The artifact downloaded from the index has a SHA-256 matching the
     sha recorded in the local `sum.dist_hashes.v1` manifest. (Same-bytes
     check — closes the "PyPI silently substituted a different artifact"
     gap.)
  2. The PEP 740 attestation cryptographically verifies (via the
     `pypi-attestations` CLI invoked through this script).
  3. The Trusted Publisher identity recorded in the attestation matches
     the expected GitHub repo + workflow.

This script is invoked twice per release:
  - After TestPyPI publish, with `--index test` — this is the load-bearing
    fail-closed PRE-PROMOTION gate. Failure here means do NOT upload to
    production PyPI.
  - After production PyPI publish, with `--index prod` — this is
    POST-PUBLISH DETECTION. The production attestation cannot exist before
    the production upload, so this run is an alarm-not-gate; it catches
    the case where TestPyPI and production diverge unexpectedly. Failure
    here means yank/revoke/announce, per docs/INCIDENT_RESPONSE.md (R0.1).

Usage:
    python scripts/verify_pypi_attestation.py \
        --index {test|prod} \
        --project sum-engine \
        --version 0.3.1 \
        --repository https://github.com/OtotaO/SUM \
        --workflow .github/workflows/publish-pypi.yml \
        --dist-hashes dist_hashes.json

The `pypi-attestations` upstream CLI labels itself experimental, so the
release pipeline pins its version (`pip install pypi-attestations==X.Y.Z`)
rather than tracking the latest tag. This script wraps the CLI invocation
so a future CLI-surface change is contained to one file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


PYPI_BASES = {
    "test": "https://test.pypi.org",
    "prod": "https://pypi.org",
}


def fetch_artifact(index_base: str, project: str, version: str, filename: str, dest: Path) -> None:
    """Download an artifact from the index by filename."""
    # PyPI/TestPyPI both expose /simple/<project>/ with anchor URLs;
    # the file is also at /packages/.../filename via the JSON API.
    json_url = f"{index_base}/pypi/{project}/{version}/json"
    with urllib.request.urlopen(json_url) as resp:
        meta = json.loads(resp.read().decode("utf-8"))
    urls = meta.get("urls") or []
    target = next((u for u in urls if u.get("filename") == filename), None)
    if target is None:
        raise SystemExit(
            f"filename {filename!r} not found in {json_url} "
            f"(available: {[u.get('filename') for u in urls]})"
        )
    download_url = target["url"]
    with urllib.request.urlopen(download_url) as src, dest.open("wb") as out:
        shutil.copyfileobj(src, out)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_pypi_attestations_verify(
    artifact_path: Path,
    index_base: str,
    project: str,
    version: str,
    repository: str,
    workflow: str,
) -> tuple[int, str]:
    """Invoke `pypi-attestations verify pypi <artifact>`.

    Returns (returncode, stdout+stderr). The CLI's `verify pypi` subcommand
    fetches the attestation from the index and verifies it; the identity
    check is enforced by passing --repository / --workflow flags.

    The exact flag surface is pinned by the version installed in CI; this
    function exists so a future CLI-surface change is contained.
    """
    cmd = [
        "pypi-attestations",
        "verify",
        "pypi",
        "--repository",
        repository,
        "--workflow",
        workflow,
    ]
    # Some CLI versions accept --index, others infer from the artifact URL.
    # Pass the project + version positionally so the CLI fetches the
    # correct attestation alongside the artifact.
    cmd += [f"{project}=={version}"]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={"PYPI_INDEX_URL": f"{index_base}/simple/", **__import__("os").environ},
    )
    return proc.returncode, (proc.stdout + proc.stderr)


def load_dist_hashes(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != "sum.dist_hashes.v1":
        raise SystemExit(
            f"unexpected schema in {path}: "
            f"expected sum.dist_hashes.v1, got {raw.get('schema')!r}"
        )
    return {a["filename"]: a["sha256"] for a in raw["artifacts"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", choices=("test", "prod"), required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--repository", required=True, help="https://github.com/OWNER/REPO")
    parser.add_argument("--workflow", required=True, help=".github/workflows/FILE.yml")
    parser.add_argument(
        "--dist-hashes",
        required=True,
        help="path to local sum.dist_hashes.v1 JSON",
    )
    args = parser.parse_args()

    index_base = PYPI_BASES[args.index]
    expected = load_dist_hashes(Path(args.dist_hashes))
    failures: list[str] = []

    print(f"verifying {args.project}=={args.version} on {index_base}")
    print(f"  expected repository: {args.repository}")
    print(f"  expected workflow:   {args.workflow}")
    print(f"  local sha256s:       {len(expected)} artifact(s)")

    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, expected_sha in sorted(expected.items()):
            print(f"\n--- {filename} ---")
            dest = Path(tmpdir) / filename
            try:
                fetch_artifact(index_base, args.project, args.version, filename, dest)
            except Exception as e:  # noqa: BLE001 — surface the real error
                failures.append(f"{filename}: download failed: {e}")
                print(f"  FAIL download: {e}")
                continue

            actual_sha = sha256_file(dest)
            if actual_sha != expected_sha:
                failures.append(
                    f"{filename}: sha256 mismatch — "
                    f"local {expected_sha} vs index {actual_sha}"
                )
                print(f"  FAIL sha: local {expected_sha} != index {actual_sha}")
                continue
            print(f"  OK   sha:  {actual_sha}")

        # Run the attestation/identity verify once per release (the CLI
        # iterates artifacts internally given a project==version target).
        print("\n--- attestation verify ---")
        rc, out = run_pypi_attestations_verify(
            Path(tmpdir),
            index_base,
            args.project,
            args.version,
            args.repository,
            args.workflow,
        )
        if rc != 0:
            failures.append(f"pypi-attestations verify failed (rc={rc}):\n{out}")
            print(f"  FAIL: pypi-attestations rc={rc}")
            print(out)
        else:
            print("  OK   attestation + identity verified")

    if failures:
        sys.stderr.write(f"\nFAIL: {len(failures)} verification check(s) failed.\n")
        for f in failures:
            sys.stderr.write(f"  - {f}\n")
        sys.stderr.write(
            "\nIf this is the --index test pre-promotion gate: do NOT upload to\n"
            "production PyPI. Investigate the divergence first.\n"
            "If this is the --index prod post-publish detection: follow the\n"
            "PyPI release-compromise runbook in docs/INCIDENT_RESPONSE.md\n"
            "(yank/revoke/announce), do not ignore.\n"
        )
        return 1

    print(f"\nOK: {args.project}=={args.version} verified on {index_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
