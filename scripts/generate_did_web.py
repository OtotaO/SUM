"""Generate an Ed25519 issuer keypair and the DID document to host at
``https://{domain}/.well-known/did.json``.

This is the one-time bootstrap for running SUM as a VC 2.0 issuer on a
public domain (Cloudflare Pages, GitHub Pages, any static host). Run it
once per deployment; commit the DID document, protect the private key.

Usage:

    python -m scripts.generate_did_web --domain sum-demo.pages.dev \\
        --out-dir single_file_demo/.well-known \\
        --private-key-out keys/did_web_issuer.pem

The script emits three artifacts:

  1. {out-dir}/did.json               — the DID document. COMMIT this.
  2. {private-key-out}                — PEM-encoded Ed25519 private key.
                                         DO NOT COMMIT. Add to .gitignore
                                         and rotate when deploying to
                                         production.
  3. {out-dir}/did-key.txt            — a sibling `did:key:z6Mk...` URI
                                         that can be used as `alsoKnownAs`
                                         so verifiers can self-resolve
                                         when the domain is unreachable.

A verifier resolving `did:web:{domain}` fetches `/.well-known/did.json`
and extracts `publicKeyMultibase` from the `verificationMethod` entry.
Every W3C-conformant verifier (DIF Universal Resolver, Veramo, Spruce
ssi, Digital Bazaar's stack) can then verify any SUM-issued VC whose
`proof.verificationMethod` points at this DID.

The private key is what SUM's `sign_credential` needs. Keep it out of
git; mount it at runtime from a secret store (Cloudflare Pages
environment variable, Vercel secret, local keychain).

Exit codes:
    0  all artefacts written
    1  argument error
    2  I/O failure (permission, disk full, overwrite refused)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from internal.infrastructure.verifiable_credential import (
    build_did_web_document,
    ed25519_to_did_key,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="generate_did_web",
        description=(
            "Mint an Ed25519 keypair and emit a did:web DID document "
            "for a hosted SUM deployment."
        ),
    )
    p.add_argument(
        "--domain", required=True,
        help="Hosted domain for the issuer (e.g. 'sum-demo.pages.dev'). "
             "No scheme, no trailing slash.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("single_file_demo/.well-known"),
        help="Directory to write did.json + did-key.txt into. Default: "
             "single_file_demo/.well-known (matches the Cloudflare Pages "
             "deploy layout documented in README).",
    )
    p.add_argument(
        "--private-key-out", type=Path, default=Path("keys/did_web_issuer.pem"),
        help="Path to write the PEM-encoded Ed25519 private key. Default: "
             "keys/did_web_issuer.pem. ENSURE THIS PATH IS IN .gitignore.",
    )
    p.add_argument(
        "--key-id", default="key-1",
        help="Fragment used in verificationMethod ids. Default: 'key-1'.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files. Default: refuse if present.",
    )
    return p.parse_args()


def _write_file(path: Path, content: bytes, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise SystemExit(
            f"refusing to overwrite existing file {path} "
            f"(re-run with --force to overwrite)"
        )
    path.write_bytes(content)


def main() -> int:
    args = _parse_args()

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    did_key_uri = ed25519_to_did_key(public_key)

    did_document = build_did_web_document(
        domain=args.domain,
        public_key=public_key,
        key_id=args.key_id,
        also_known_as=[did_key_uri],
    )

    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    did_doc_path = args.out_dir / "did.json"
    did_key_path = args.out_dir / "did-key.txt"

    try:
        _write_file(
            did_doc_path,
            json.dumps(did_document, indent=2).encode("utf-8") + b"\n",
            args.force,
        )
        _write_file(did_key_path, (did_key_uri + "\n").encode("utf-8"), args.force)
        _write_file(args.private_key_out, pem, args.force)
    except SystemExit:
        raise
    except OSError as e:
        print(f"error writing output: {e}", file=sys.stderr)
        return 2

    print("did:web issuer bootstrap complete.")
    print(f"  DID:              did:web:{args.domain}")
    print(f"  did:key (AKA):    {did_key_uri}")
    print(f"  DID document:     {did_doc_path}")
    print(f"  did-key sidecar:  {did_key_path}")
    print(f"  private key:      {args.private_key_out}  (DO NOT COMMIT)")
    print()
    print("Next steps:")
    print(f"  1. Deploy the contents of {args.out_dir.parent}/ to "
          f"https://{args.domain}/ so /.well-known/did.json is fetchable.")
    print(f"  2. Add {args.private_key_out} to .gitignore (see docs/DID_SETUP.md).")
    print(f"  3. In the bundle-minting path, set verificationMethod to "
          f"'did:web:{args.domain}#{args.key_id}' and pass the PEM as the "
          f"Ed25519PrivateKey.")
    print(f"  4. Verify the setup: curl -sL https://{args.domain}/.well-known/did.json | jq .")
    return 0


if __name__ == "__main__":
    sys.exit(main())
