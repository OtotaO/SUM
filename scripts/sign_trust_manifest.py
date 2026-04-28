#!/usr/bin/env python3
"""Sign a trust-root manifest payload with an Ed25519 OKP JWK (Phase R0.2).

Wraps an unsigned payload (from ``scripts/build_trust_manifest.py``)
with a JWS signature and emits the full ``sum.trust_root.v1`` envelope.

Usage:
    python scripts/sign_trust_manifest.py \\
        --in unsigned_payload.json \\
        --signing-jwk /path/to/trust_root_private.jwk \\
        --kid sum-trust-root-2026-04-27-1 \\
        --out sum.trust_root.v1.json

The signing JWK must be an Ed25519 OKP private key (``kty=OKP``,
``crv=Ed25519``, with the private scalar ``d`` present). Generate
one with:

    python -c "
    from joserfc.jwk import OKPKey
    import json, sys
    k = OKPKey.generate_key('Ed25519')
    sys.stdout.write(json.dumps(k.as_dict(private=True), indent=2))
    " > trust_root_private.jwk

Keep ``trust_root_private.jwk`` off the disk after the signing
operation. Treat it like the render-receipt private key: secret-store
in production, ``shred -u`` after release-time use.

The kid value should follow the rotation pattern documented in
``docs/RENDER_RECEIPT_FORMAT.md`` §6 (date-stamped, monotonic
suffix). Trust-root kids are distinct from render-receipt kids by
design.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add repo root to sys.path so the script can run as
# `python scripts/sign_trust_manifest.py` without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sum_engine_internal.infrastructure.jose_envelope import sign_jose_envelope  # noqa: E402
from sum_engine_internal.trust_root import SUPPORTED_SCHEMA  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        dest="input_path",
        required=True,
        help="Path to the unsigned payload JSON (output of build_trust_manifest.py).",
    )
    parser.add_argument(
        "--signing-jwk",
        required=True,
        help="Path to the Ed25519 OKP private JWK (sensitive — see header doc).",
    )
    parser.add_argument(
        "--kid",
        required=True,
        help="The kid to embed in the signed envelope (e.g. sum-trust-root-2026-04-27-1).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for the signed manifest (default: stdout).",
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.input_path).read_text(encoding="utf-8"))
    private_jwk = json.loads(Path(args.signing_jwk).read_text(encoding="utf-8"))

    if private_jwk.get("kty") != "OKP" or private_jwk.get("crv") != "Ed25519":
        raise SystemExit(
            f"signing JWK must be Ed25519 OKP, got "
            f"kty={private_jwk.get('kty')} crv={private_jwk.get('crv')}"
        )
    if "d" not in private_jwk:
        raise SystemExit(
            "signing JWK is missing the private scalar `d`; "
            "you supplied a public JWK by mistake"
        )

    envelope = sign_jose_envelope(
        payload,
        private_jwk=private_jwk,
        kid=args.kid,
    )
    envelope["schema"] = SUPPORTED_SCHEMA

    # Emit with sorted keys at the top level; the inner JCS
    # canonicalization at sign time is what the signature actually
    # binds, so the outer envelope's pretty-printing is presentation,
    # not security.
    out_text = json.dumps(envelope, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(out_text, encoding="utf-8")
        print(f"signed manifest written: {args.out}", file=sys.stderr)
        print(f"  kid:    {envelope['kid']}", file=sys.stderr)
        print(f"  schema: {envelope['schema']}", file=sys.stderr)
    else:
        sys.stdout.write(out_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
