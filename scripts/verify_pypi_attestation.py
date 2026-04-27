#!/usr/bin/env python3
"""Verify PEP 740 attestations on a published PyPI artifact.

FOUR checks, all must pass:
  1. Same-bytes — the artifact downloaded from the index has a SHA-256
     matching the local sum.dist_hashes.v1 manifest. Closes the
     "PyPI silently substituted a different artifact" gap.
  2. Cryptographic verify — the PEP 740 attestation cryptographically
     verifies, via the upstream `pypi-attestations` CLI's documented
     `verify pypi --repository <url> <distribution>` invocation. The
     CLI takes a distribution file or URL (NOT project==version) and
     does NOT accept --workflow; this script does the workflow check
     separately, in step 4.
  3. Repository identity — the Trusted Publisher claim in the
     attestation matches the expected `--repository` URL. Enforced by
     the CLI's `--repository` flag.
  4. Workflow identity — the publication workflow path AND ref pattern
     in the provenance match the expected values. Fetched directly from
     PyPI's Integrity API (`/integrity/<project>/<version>/<filename>/
     provenance`) and parsed; the upstream CLI does not accept a
     --workflow flag, so this script does the parse itself.

The four-check structure exists because PyPI's documented `pypi-
attestations verify pypi --repository ...` flow only enforces
repository-level identity — "some Trusted Publisher for this repo
signed these bytes." That's weaker than what SUM needs, which is
"this exact repo + this exact release workflow + this exact tag
signed these bytes." Steps 1 and 4 close the gap that step 2/3 don't.

Invoked twice per release:
  - After TestPyPI publish, with `--index test`. Load-bearing fail-
    closed PRE-PROMOTION gate. Failure here MUST block publish-pypi.
  - After production PyPI publish, with `--index prod`. POST-PUBLISH
    DETECTION (alarm, not gate; the production attestation cannot
    exist before production upload). Failure here means yank/revoke/
    announce per docs/INCIDENT_RESPONSE.md once R0 lands.

Usage:
    python scripts/verify_pypi_attestation.py \
        --index {test|prod} \
        --project sum-engine \
        --version 0.3.1 \
        --repository https://github.com/OtotaO/SUM \
        --workflow .github/workflows/publish-pypi.yml \
        --ref-prefix refs/tags/v \
        --dist-hashes dist_hashes.json

Pin `pypi-attestations` at a known-good version in CI (the upstream
CLI labels itself experimental). Current target: 0.0.29 (Dec 2025).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


PYPI_BASES = {
    "test": "https://test.pypi.org",
    "prod": "https://pypi.org",
}


def fetch_artifact_url(index_base: str, project: str, version: str, filename: str) -> str:
    """Look up a specific filename's download URL via the JSON API."""
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
    return target["url"]


def download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as src, dest.open("wb") as out:
        shutil.copyfileobj(src, out)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_provenance(index_base: str, project: str, version: str, filename: str) -> dict[str, Any]:
    """Fetch the PEP 740 provenance JSON via the PyPI Integrity API.

    Endpoint shape:
        {index_base}/integrity/{project}/{version}/{filename}/provenance
    Response is JSON. The exact schema evolves; this caller is defensive
    and uses a recursive walker (see find_publisher_claims) to locate
    the workflow + repository + ref claims wherever they are.
    """
    url = f"{index_base}/integrity/{project}/{version}/{filename}/provenance"
    try:
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 — surface the real error
        raise SystemExit(f"failed to fetch provenance from {url}: {e}") from e


def find_publisher_claims(obj: Any, claims: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Recursively walk a JSON-shaped structure collecting publisher claims.

    Returns a dict mapping claim-name → list of values seen. Claim names
    looked for: `repository`, `workflow`, `ref` (case-insensitive top-
    level keys). Resilient to schema drift across PyPI provenance
    formats — looks for the field name wherever it appears in the tree.

    Also captures keys that look like SLSA Provenance v1 / GitHub
    Actions identity fields: `path` (workflow path inside the repo),
    `repositoryUri`, `workflowRef`.
    """
    if claims is None:
        claims = {"repository": [], "workflow": [], "ref": []}

    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = k.lower()
            if kl == "repository" or kl == "repositoryuri":
                if isinstance(v, str):
                    claims["repository"].append(v)
            elif kl == "workflow":
                if isinstance(v, str):
                    claims["workflow"].append(v)
                elif isinstance(v, dict):
                    # Nested workflow object: pull `path` + `ref`.
                    p = v.get("path")
                    if isinstance(p, str):
                        claims["workflow"].append(p)
                    r = v.get("ref")
                    if isinstance(r, str):
                        claims["ref"].append(r)
            elif kl == "workflowref":
                if isinstance(v, str):
                    claims["workflow"].append(v)
            elif kl == "path" and isinstance(v, str) and ".github/workflows/" in v:
                # SLSA Provenance v1 buildDefinition.externalParameters.workflow.path
                claims["workflow"].append(v)
            elif kl == "ref":
                if isinstance(v, str):
                    claims["ref"].append(v)
            find_publisher_claims(v, claims)
    elif isinstance(obj, list):
        for item in obj:
            find_publisher_claims(item, claims)

    return claims


def assert_identity(
    claims: dict[str, list[str]],
    expected_repo: str,
    expected_workflow: str,
    expected_ref_prefix: str,
) -> list[str]:
    """Return a list of failure messages; empty list = pass."""
    failures: list[str] = []

    # Normalise: GitHub provenance can encode the repository as either
    # the URL ("https://github.com/OWNER/REPO") or just "OWNER/REPO".
    expected_repo_norm = expected_repo.rstrip("/")
    expected_repo_short = expected_repo_norm.removeprefix("https://github.com/")

    repo_ok = any(
        v == expected_repo_norm or v == expected_repo_short or v == expected_repo_norm + ".git"
        for v in claims["repository"]
    )
    if not repo_ok:
        failures.append(
            f"repository identity mismatch: expected {expected_repo_norm} "
            f"(or {expected_repo_short}), saw {claims['repository']}"
        )

    workflow_ok = any(expected_workflow in v for v in claims["workflow"])
    if not workflow_ok:
        failures.append(
            f"workflow identity mismatch: expected path containing "
            f"{expected_workflow!r}, saw {claims['workflow']}"
        )

    ref_ok = any(v.startswith(expected_ref_prefix) for v in claims["ref"])
    if not ref_ok:
        failures.append(
            f"ref pattern mismatch: expected prefix {expected_ref_prefix!r}, "
            f"saw {claims['ref']}"
        )

    return failures


def run_pypi_attestations_verify(
    repository: str,
    distribution: str,
) -> tuple[int, str]:
    """Invoke `pypi-attestations verify pypi --repository <url> <dist>`.

    Documented CLI shape per pypi-attestations >= 0.0.29:
        pypi-attestations verify pypi --repository <repo-url> <dist-or-url>

    `<dist-or-url>` is either a local distribution file path or a URL
    pointing to one. The CLI fetches the .publish.attestation file
    alongside (from the index, when given a URL).

    Returns (returncode, stdout+stderr).
    """
    cmd = [
        "pypi-attestations",
        "verify",
        "pypi",
        "--repository",
        repository,
        distribution,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
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
    parser.add_argument(
        "--repository",
        required=True,
        help="https://github.com/OWNER/REPO — the expected Trusted Publisher",
    )
    parser.add_argument(
        "--workflow",
        required=True,
        help=".github/workflows/FILE.yml — the expected workflow path",
    )
    parser.add_argument(
        "--ref-prefix",
        default="refs/tags/v",
        help="expected git ref prefix (default: refs/tags/v)",
    )
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
    print(f"  expected ref prefix: {args.ref_prefix}")
    print(f"  local sha256s:       {len(expected)} artifact(s)")

    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, expected_sha in sorted(expected.items()):
            print(f"\n--- {filename} ---")
            try:
                url = fetch_artifact_url(index_base, args.project, args.version, filename)
            except Exception as e:  # noqa: BLE001
                failures.append(f"{filename}: URL lookup failed: {e}")
                print(f"  FAIL url lookup: {e}")
                continue

            # CHECK 1: same-bytes (sha256 vs local manifest)
            dest = Path(tmpdir) / filename
            try:
                download(url, dest)
            except Exception as e:  # noqa: BLE001
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
            print(f"  OK   sha:  {actual_sha[:24]}…")

            # CHECK 2 + 3: pypi-attestations verify pypi (crypto + repo identity)
            rc, out = run_pypi_attestations_verify(args.repository, url)
            if rc != 0:
                failures.append(
                    f"{filename}: pypi-attestations verify failed (rc={rc}):\n{out}"
                )
                print(f"  FAIL crypto/repo: rc={rc}")
                print(f"  {out.strip()}")
                continue
            print(f"  OK   crypto + repo identity (pypi-attestations rc=0)")

            # CHECK 4: workflow identity + ref (parsed from Integrity API)
            try:
                provenance = fetch_provenance(
                    index_base, args.project, args.version, filename
                )
            except SystemExit as e:
                failures.append(f"{filename}: provenance fetch failed: {e}")
                print(f"  FAIL provenance fetch: {e}")
                continue

            claims = find_publisher_claims(provenance)
            wf_failures = assert_identity(
                claims,
                args.repository,
                args.workflow,
                args.ref_prefix,
            )
            if wf_failures:
                for msg in wf_failures:
                    failures.append(f"{filename}: {msg}")
                    print(f"  FAIL identity: {msg}")
                continue
            print(
                f"  OK   workflow identity + ref "
                f"(workflow={claims['workflow'][:1]}, ref={claims['ref'][:1]})"
            )

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
    print(f"     same-bytes ✓  crypto ✓  repo identity ✓  workflow identity ✓  ref ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
