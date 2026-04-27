#!/usr/bin/env python3
"""Verify PEP 740 attestations on a published PyPI artifact.

FOUR checks, all must pass:
  1. Same-bytes — the artifact downloaded from the index has a SHA-256
     matching the local sum.dist_hashes.v1 manifest. Closes the
     "PyPI silently substituted a different artifact" gap.
  2. Cryptographic + repository identity — `pypi-attestations verify
     pypi --provenance-file <prov> --repository <url> <dist>` performs
     the Sigstore-backed signature check and asserts the Trusted
     Publisher claim matches the expected --repository. The
     --provenance-file mode is used (not URL-inferred) so TestPyPI vs
     production sourcing is unambiguous.
  3. Coarse publisher identity (advisory) — parse the Integrity-API
     JSON's `publisher` object and check repository + workflow
     basename + environment match expected values. Per PyPI's
     documented Integrity API shape, `publisher.workflow` is the
     basename ("publish-pypi.yml") not the full path; this check
     accepts either form.
  4. Authoritative workflow + tag identity (SAN) — extract the
     Sigstore signing certificate(s) from the provenance bundle,
     parse the Subject Alternative Name URI, and assert it matches
     the expected GitHub Actions OIDC identity:
         https://github.com/OWNER/REPO/<workflow-path>@<ref>
     This is the form Sigstore documents and is the load-bearing
     identity check — the JSON `publisher` object alone is too coarse
     (no exact workflow path, often no ref). If the SAN cannot be
     extracted the script fails closed with a precise message
     ("cannot verify tag ref from provenance; do not promote") so a
     verifier-shape limitation is visible as such, not silently
     conflated with a malicious-artifact reading.

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
        --environment testpypi \
        --dist-hashes dist_hashes.json

Pin `pypi-attestations` at a known-good version in CI (the upstream
CLI labels itself experimental). Current target: 0.0.29 (Dec 2025).
The `cryptography` library is pulled in transitively via pypi-
attestations → sigstore-python.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
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


# ---------------------------------------------------------------------------
# Network + hashing helpers
# ---------------------------------------------------------------------------


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

    Endpoint: {index_base}/integrity/{project}/{version}/{filename}/provenance
    """
    url = f"{index_base}/integrity/{project}/{version}/{filename}/provenance"
    try:
        with urllib.request.urlopen(url) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 — surface the real error
        raise SystemExit(f"failed to fetch provenance from {url}: {e}") from e


def load_dist_hashes(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != "sum.dist_hashes.v1":
        raise SystemExit(
            f"unexpected schema in {path}: "
            f"expected sum.dist_hashes.v1, got {raw.get('schema')!r}"
        )
    return {a["filename"]: a["sha256"] for a in raw["artifacts"]}


# ---------------------------------------------------------------------------
# Coarse JSON publisher check (advisory)
# ---------------------------------------------------------------------------


def find_publisher(obj: Any) -> dict[str, Any] | None:
    """Find the first `publisher` object in a provenance JSON tree."""
    if isinstance(obj, dict):
        pub = obj.get("publisher")
        if isinstance(pub, dict):
            return pub
        for v in obj.values():
            result = find_publisher(v)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = find_publisher(item)
            if result is not None:
                return result
    return None


def check_publisher_coarse(
    publisher: dict[str, Any] | None,
    expected_repo: str,
    expected_workflow: str,
    expected_environment: str | None,
) -> list[str]:
    """Advisory check on the Integrity API's `publisher` object.

    Per PyPI's documented Integrity API:
        publisher.repository = "OWNER/REPO"   (NOT a URL)
        publisher.workflow   = "FILE.yml"     (basename, NOT full path)
        publisher.environment = "<env>" or "" (may be empty)
        publisher.kind        = "GitHub"

    This check is advisory because (a) the basename-only workflow form
    cannot uniquely pin a workflow and (b) the environment may be empty
    if Trusted Publishing was configured without an environment. The
    SAN check (assert_san_identity) is the load-bearing identity check.
    Mismatches here are still surfaced — they indicate an unexpected
    publisher even when the SAN matches, which is worth knowing.
    """
    failures: list[str] = []
    if publisher is None:
        failures.append(
            "no `publisher` object found in provenance JSON (advisory; "
            "SAN check will still run)"
        )
        return failures

    expected_repo_short = expected_repo.removeprefix("https://github.com/").rstrip("/")
    expected_workflow_basename = expected_workflow.rsplit("/", 1)[-1]

    repo_value = publisher.get("repository", "")
    if repo_value not in (expected_repo_short, expected_repo, expected_repo + ".git"):
        failures.append(
            f"publisher.repository = {repo_value!r}, "
            f"expected {expected_repo_short!r} or {expected_repo!r}"
        )

    workflow_value = publisher.get("workflow", "")
    if workflow_value not in (expected_workflow_basename, expected_workflow):
        failures.append(
            f"publisher.workflow = {workflow_value!r}, expected basename "
            f"{expected_workflow_basename!r} or full path {expected_workflow!r}"
        )

    if expected_environment:
        env_value = publisher.get("environment", "")
        if env_value and env_value != expected_environment:
            failures.append(
                f"publisher.environment = {env_value!r}, "
                f"expected {expected_environment!r}"
            )

    return failures


# ---------------------------------------------------------------------------
# Authoritative SAN identity check (cryptographic)
# ---------------------------------------------------------------------------


def extract_raw_byte_strings(obj: Any, out: list[str] | None = None) -> list[str]:
    """Walk a JSON tree collecting every value at a `*[Rr]aw[Bb]ytes` key.

    Resilient to:
      - PyPI Integrity API snake_case (`raw_bytes`)
      - Sigstore Bundle JSON camelCase (`rawBytes`)
      - Either `certificate.*` or `x509CertificateChain.certificates[]` paths
      - Nested envelopes / future schema drift
    """
    if out is None:
        out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and k.replace("_", "").lower() == "rawbytes":
                out.append(v)
            else:
                extract_raw_byte_strings(v, out)
    elif isinstance(obj, list):
        for item in obj:
            extract_raw_byte_strings(item, out)
    return out


def parse_certificates(provenance: dict[str, Any]) -> list[Any]:
    """Try to parse every base64'd value found at a *raw_bytes/rawBytes key.

    Skips values that don't decode as DER X.509 certificates. Imports the
    `cryptography` library lazily so unit tests that don't exercise this
    function don't require it on PYTHONPATH.
    """
    from cryptography import x509  # lazy import

    candidates = extract_raw_byte_strings(provenance)
    certs = []
    for c in candidates:
        try:
            der = base64.b64decode(c, validate=True)
        except Exception:  # noqa: BLE001
            continue
        try:
            certs.append(x509.load_der_x509_certificate(der))
        except Exception:  # noqa: BLE001
            continue
    return certs


def extract_san_uris(cert: Any) -> list[str]:
    from cryptography import x509  # lazy import

    try:
        ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    except x509.ExtensionNotFound:
        return []
    return [
        u.value
        for u in ext.value
        if isinstance(u, x509.UniformResourceIdentifier)
    ]


def expected_san_prefix(repository: str, workflow: str, ref_prefix: str) -> str:
    """The Sigstore documented SAN URI form for GitHub Actions OIDC:

        https://github.com/OWNER/REPO/<workflow-path>@<ref>

    `workflow` is the in-repo path (e.g. `.github/workflows/publish-pypi.yml`).
    `ref_prefix` is the expected git ref prefix (e.g. `refs/tags/v`); the
    full SAN includes the specific tag (e.g. `refs/tags/v0.3.1`), so this
    function returns the prefix and the caller does a startswith match.
    """
    repo_short = repository.removeprefix("https://github.com/").rstrip("/")
    workflow_path = workflow.lstrip("/")
    return f"https://github.com/{repo_short}/{workflow_path}@{ref_prefix}"


def assert_san_identity(
    certs: list[Any],
    repository: str,
    workflow: str,
    ref_prefix: str,
) -> tuple[list[str], list[str]]:
    """Returns (failures, observed_uris).

    Empty `failures` means the SAN matched. If no certs are extractable
    or no SAN URI matches, fail closed with a precise message that
    distinguishes verifier-shape limitations from artifact-tamper
    readings ("cannot verify tag ref from provenance; do not promote").
    """
    if not certs:
        return (
            [
                "no Sigstore certificates extractable from provenance — "
                "cannot verify tag ref from provenance; do not promote. "
                "(Verifier limitation, not an artifact-tamper signal: the "
                "provenance JSON shape may have evolved.)"
            ],
            [],
        )

    expected_prefix = expected_san_prefix(repository, workflow, ref_prefix)
    all_uris: list[str] = []
    for cert in certs:
        all_uris.extend(extract_san_uris(cert))

    if not all_uris:
        return (
            [
                f"Sigstore certificate(s) found ({len(certs)}) but no SAN "
                f"URIs present — cannot verify tag ref from provenance; "
                f"do not promote."
            ],
            [],
        )

    matches = [u for u in all_uris if u.startswith(expected_prefix)]
    if not matches:
        return (
            [
                f"no SAN URI matches expected prefix {expected_prefix!r}; "
                f"saw {all_uris}"
            ],
            all_uris,
        )

    return ([], matches)


# ---------------------------------------------------------------------------
# pypi-attestations CLI (crypto + repo identity)
# ---------------------------------------------------------------------------


def write_provenance_to_file(provenance: dict[str, Any], dest: Path) -> None:
    dest.write_text(json.dumps(provenance, separators=(",", ":")), encoding="utf-8")


def run_pypi_attestations_verify(
    repository: str,
    distribution_path: Path,
    provenance_file: Path,
) -> tuple[int, str]:
    """Invoke `pypi-attestations verify pypi` in --provenance-file mode.

    Documented CLI shape per pypi-attestations >= 0.0.29:
        pypi-attestations verify pypi \
            --repository <https-github-url> \
            --provenance-file <local-prov.json> \
            <local-distribution-file>

    --provenance-file is preferred over URL-inference so the CLI's
    TestPyPI vs production sourcing is unambiguous and the artifact
    has already been integrity-checked locally.
    """
    cmd = [
        "pypi-attestations",
        "verify",
        "pypi",
        "--repository",
        repository,
        "--provenance-file",
        str(provenance_file),
        str(distribution_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
        "--environment",
        default=None,
        help="expected GitHub Actions environment (advisory; default: skip)",
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
    advisories: list[str] = []

    print(f"verifying {args.project}=={args.version} on {index_base}")
    print(f"  expected repository: {args.repository}")
    print(f"  expected workflow:   {args.workflow}")
    print(f"  expected ref prefix: {args.ref_prefix}")
    print(
        f"  expected environment: "
        f"{args.environment if args.environment else '(advisory: skip)'}"
    )
    print(f"  local sha256s:       {len(expected)} artifact(s)")

    with tempfile.TemporaryDirectory() as tmpdir:
        for filename, expected_sha in sorted(expected.items()):
            print(f"\n--- {filename} ---")

            # CHECK 1: same-bytes
            try:
                url = fetch_artifact_url(
                    index_base, args.project, args.version, filename
                )
            except Exception as e:  # noqa: BLE001
                failures.append(f"{filename}: URL lookup failed: {e}")
                print(f"  FAIL url lookup: {e}")
                continue

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

            # Fetch provenance once; reused for checks 2, 3, 4.
            try:
                provenance = fetch_provenance(
                    index_base, args.project, args.version, filename
                )
            except SystemExit as e:
                failures.append(f"{filename}: provenance fetch failed: {e}")
                print(f"  FAIL provenance fetch: {e}")
                continue

            prov_path = Path(tmpdir) / f"{filename}.provenance.json"
            write_provenance_to_file(provenance, prov_path)

            # CHECK 2: pypi-attestations verify pypi (crypto + repo identity)
            rc, out = run_pypi_attestations_verify(args.repository, dest, prov_path)
            if rc != 0:
                failures.append(
                    f"{filename}: pypi-attestations verify failed (rc={rc}):\n{out}"
                )
                print(f"  FAIL crypto/repo: rc={rc}")
                print(f"  {out.strip()}")
                continue
            print(f"  OK   crypto + repo identity (pypi-attestations rc=0)")

            # CHECK 3: coarse publisher (advisory)
            publisher = find_publisher(provenance)
            advisory_msgs = check_publisher_coarse(
                publisher,
                args.repository,
                args.workflow,
                args.environment,
            )
            if advisory_msgs:
                for m in advisory_msgs:
                    advisories.append(f"{filename}: {m}")
                    print(f"  ADVISORY publisher: {m}")
            else:
                print(f"  OK   publisher (coarse): "
                      f"repo + workflow basename"
                      f"{' + environment' if args.environment else ''} match")

            # CHECK 4: authoritative SAN identity
            certs = parse_certificates(provenance)
            san_failures, observed = assert_san_identity(
                certs, args.repository, args.workflow, args.ref_prefix
            )
            if san_failures:
                for m in san_failures:
                    failures.append(f"{filename}: {m}")
                    print(f"  FAIL SAN: {m}")
                continue
            print(
                f"  OK   SAN identity (workflow + ref): "
                f"{observed[0] if observed else '(no URI captured)'}"
            )

    if advisories and not failures:
        sys.stderr.write(
            f"\nWARNING: {len(advisories)} advisory mismatch(es) on the "
            f"coarse JSON publisher check, but the authoritative SAN check "
            f"passed. The publisher JSON shape may have drifted; re-read "
            f"PyPI's Integrity API docs and update the coarse-check "
            f"expected values if appropriate.\n"
        )
        for a in advisories:
            sys.stderr.write(f"  - {a}\n")

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
    print(f"     same-bytes ✓  crypto ✓  repo identity ✓  publisher (advisory) ✓  SAN identity ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
