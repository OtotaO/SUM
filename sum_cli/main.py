"""SUM CLI entry point.

Subcommands:
    sum attest   — stdin prose → CanonicalBundle JSON on stdout (optionally
                   signed with HMAC via --signing-key and/or Ed25519 via
                   --ed25519-key; unsigned by default)
    sum verify   — stdin/file bundle → exit 0 on match, 1 on mismatch
    sum resolve  — prov_id → ProvenanceRecord JSON (local ledger lookup)
    sum version  — print version string
    sum --help   — auto-generated usage

Design notes (read once, explained here so the code stays terse):

  - argparse only. No click, no typer. Stdlib keeps the install minimal and
    cold-start fast (`python -c "import sum_cli"` must stay under 100 ms).
  - Heavy imports (spacy, openai, akashic ledger) are lazy. Loading the
    module to call --version or --help must NOT drag the whole
    internal.algorithms.* tree into memory.
  - Stdin / stdout / stderr contract is Unix-shaped: one thing in, one
    thing out, diagnostics on stderr, exit code signals success/failure.
    Agents can pipe sum into jq, into file > bundle.json, into
    curl -d @bundle.json, etc. — standard tool composition.
  - Every command prints machine-readable output on stdout (JSON) and
    human-readable context on stderr. Quiet by default unless --verbose.
  - No surprises. `sum attest` exits with non-zero if extraction yielded
    zero triples (nothing to attest is a failure, not a success).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from sum_cli import __version__


# ─── Extractor selection (lazy, dependency-aware) ────────────────────

def _pick_extractor(override: Optional[str] = None) -> str:
    """Pick the extractor at runtime. Honors --extractor override; otherwise
    adapts to whichever dependency is installed.

    Rules:
      1. If --extractor is given, honor it (fail fast if deps missing).
      2. If spaCy + en_core_web_sm are importable, prefer 'sieve' (offline).
      3. If OPENAI_API_KEY is set, use 'llm' (network-dependent).
      4. Otherwise, fail with a helpful install hint.
    """
    if override:
        return override
    try:
        import spacy  # noqa: F401 — availability check only
        spacy.load("en_core_web_sm")
        return "sieve"
    except Exception:
        pass
    if os.environ.get("OPENAI_API_KEY"):
        return "llm"
    raise SystemExit(
        "sum: no extractor available. Install one of:\n"
        "    pip install 'sum-engine[sieve]' && python -m spacy download en_core_web_sm\n"
        "    pip install 'sum-engine[llm]'   && export OPENAI_API_KEY=...\n"
        "Or pass --extractor explicitly."
    )


def _extract_sieve(text: str) -> list[tuple[str, str, str]]:
    from internal.algorithms.syntactic_sieve import DeterministicSieve
    sieve = DeterministicSieve()  # type: ignore[no-untyped-call]
    return sieve.extract_triplets(text)


def _extract_llm(text: str, model: str) -> list[tuple[str, str, str]]:
    import asyncio
    from internal.ensemble.live_llm_adapter import LiveLLMAdapter
    adapter = LiveLLMAdapter(model=model)
    return asyncio.run(adapter.extract_triplets(text))


def _extract(text: str, extractor: str, model: Optional[str]) -> list[tuple[str, str, str]]:
    if extractor == "sieve":
        return _extract_sieve(text)
    if extractor == "llm":
        return _extract_llm(text, model or "gpt-4o-mini-2024-07-18")
    raise SystemExit(f"sum: unknown extractor {extractor!r}")


# ─── attest ──────────────────────────────────────────────────────────

def _read_input(path: Optional[str]) -> str:
    if path is None or path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class _PemFileKeyManager:
    """Single-file Ed25519 KeyManager adapter.

    Satisfies the surface CanonicalCodec actually uses — ``ensure_keypair``
    and ``get_public_key_bytes`` — by loading a PKCS8 Ed25519 PEM file
    produced by ``python -m scripts.generate_did_web``. No rotation, no
    archive: a bundle mint is a single point-in-time act, and key
    management lives in the generator script, not in this CLI.
    """

    def __init__(self, pem_path):
        from pathlib import Path

        self._path = Path(pem_path)
        self._private_key = None
        self._public_key = None

    def ensure_keypair(self):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        if self._private_key is not None and self._public_key is not None:
            return self._private_key, self._public_key
        sk = serialization.load_pem_private_key(self._path.read_bytes(), password=None)
        if not isinstance(sk, Ed25519PrivateKey):
            raise SystemExit(
                f"sum: {self._path} is not an Ed25519 private key "
                f"(CanonicalCodec's public-key attestation is Ed25519-only). "
                f"Regenerate with `python -m scripts.generate_did_web`."
            )
        self._private_key = sk
        self._public_key = sk.public_key()
        return self._private_key, self._public_key

    def get_public_key_bytes(self) -> bytes:
        from cryptography.hazmat.primitives import serialization

        _, pub = self.ensure_keypair()
        return pub.public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )


def _load_ed25519_key(pem_path: str) -> _PemFileKeyManager:
    from pathlib import Path

    path = Path(pem_path)
    if not path.exists():
        raise SystemExit(
            f"sum: Ed25519 key not found at {path}. "
            f"Generate one with: python -m scripts.generate_did_web "
            f"--domain YOUR.DOMAIN --private-key-out {path} "
            f"(see docs/DID_SETUP.md)."
        )
    return _PemFileKeyManager(path)


def cmd_attest(args: argparse.Namespace) -> int:
    text = _read_input(args.input).strip()
    if not text:
        print("sum: empty input", file=sys.stderr)
        return 2

    # Source URI: either user-supplied or derived from a SHA-256 of the input
    # bytes (content-addressable by default — matches provenance.py's
    # sha256_uri_for_text helper).
    source_uri = args.source or (
        "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    )

    extractor = _pick_extractor(args.extractor)
    if args.verbose:
        print(f"sum: extractor={extractor} source={source_uri}", file=sys.stderr)

    triples = _extract(text, extractor, args.model)
    if not triples:
        print(
            "sum: extractor returned zero triples. "
            "Input may be too short, negated, or hedged — "
            "see docs/FEATURE_CATALOG.md entries 6-9 for the suppression rules.",
            file=sys.stderr,
        )
        return 3

    # Build the CanonicalBundle via the existing codec path — no
    # reimplementation of Ed25519/HMAC, no reimplementation of canonical
    # tome generation. The CLI is a thin wrapper.
    from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from internal.infrastructure.canonical_codec import CanonicalCodec

    key_manager = _load_ed25519_key(args.ed25519_key) if args.ed25519_key else None

    algebra = GodelStateAlgebra()  # type: ignore[no-untyped-call]
    tome_generator = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(
        algebra,
        tome_generator,
        signing_key=args.signing_key,
        key_manager=key_manager,
    )
    state = algebra.encode_chunk_state(list(triples))
    bundle = codec.export_bundle(
        state,
        branch=args.branch,
        title=args.title,
    )

    # Optional: attach a lightweight sidecar naming the extractor + source
    # URI so downstream consumers can trace provenance without the full
    # AkashicLedger. This is additive — the CanonicalBundle schema
    # ignores unknown keys, so adding them is forward-compatible.
    bundle["sum_cli"] = {
        "extractor": extractor,
        "source_uri": source_uri,
        "cli_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    json.dump(bundle, sys.stdout, indent=2 if args.pretty else None)
    sys.stdout.write("\n")

    if args.verbose:
        print(
            f"sum: minted {len(triples)} axiom(s), state_integer has "
            f"{len(bundle['state_integer'])} digits",
            file=sys.stderr,
        )
    return 0


# ─── verify ──────────────────────────────────────────────────────────

_SUPPORTED_CANONICAL_FORMAT = "1.0.0"
_SUPPORTED_PRIME_SCHEME = "sha256_64_v1"


def _verify_ed25519_bundle(bundle: dict) -> str:
    """Verify the embedded Ed25519 signature over the SUM payload line.

    Returns one of:
      * "absent"    — no Ed25519 fields present; nothing to verify.
      * "verified"  — signature validates against the embedded pubkey.
      * "invalid"   — fields are present but verification failed.

    Self-contained — no key lookup, no network. The bundle embeds the
    Ed25519 public key alongside its signature; a verifier needs only
    the public key's hex and the standard Ed25519 primitive.
    """
    pub_sig = bundle.get("public_signature")
    pub_key = bundle.get("public_key")
    if not pub_sig or not pub_key:
        return "absent"
    try:
        import base64

        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        sig_b64 = pub_sig.split(":", 1)[1]
        pub_b64 = pub_key.split(":", 1)[1]
        sig_bytes = base64.b64decode(sig_b64)
        pub_bytes = base64.b64decode(pub_b64)
        key = Ed25519PublicKey.from_public_bytes(pub_bytes)
        payload = (
            f"{bundle['canonical_tome']}|"
            f"{bundle['state_integer']}|"
            f"{bundle['timestamp']}"
        ).encode("utf-8")
        key.verify(sig_bytes, payload)
        return "verified"
    except (InvalidSignature, Exception):
        return "invalid"


def _verify_hmac_bundle(bundle: dict, signing_key: Optional[str]) -> str:
    """Verify the HMAC-SHA256 signature over the SUM payload line.

    Returns one of:
      * "absent"    — no signature field present.
      * "skipped"   — field present but no --signing-key supplied.
      * "verified"  — field present and verifies under the supplied key.
      * "invalid"   — field present but does not verify.

    Truthful: "skipped" is not a pass. In --strict mode the caller
    must treat it as a failure.
    """
    sig = bundle.get("signature")
    if not sig:
        return "absent"
    if signing_key is None:
        return "skipped"
    import hashlib
    import hmac as _hmac

    payload = (
        f"{bundle['canonical_tome']}|"
        f"{bundle['state_integer']}|"
        f"{bundle['timestamp']}"
    ).encode("utf-8")
    expected = (
        "hmac-sha256:"
        + _hmac.new(signing_key.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    )
    return "verified" if _hmac.compare_digest(expected, sig) else "invalid"


def cmd_verify(args: argparse.Namespace) -> int:
    raw = _read_input(args.input)
    try:
        bundle = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"sum: bundle is not valid JSON: {e}", file=sys.stderr)
        return 2

    # Required fields per the Phase 16 ABI.
    for field in ("canonical_tome", "state_integer", "canonical_format_version"):
        if field not in bundle:
            print(f"sum: bundle missing required field: {field}", file=sys.stderr)
            return 2

    if bundle["canonical_format_version"] != _SUPPORTED_CANONICAL_FORMAT:
        print(
            f"sum: unsupported canonical_format_version "
            f"{bundle['canonical_format_version']!r} "
            f"(this CLI speaks {_SUPPORTED_CANONICAL_FORMAT})",
            file=sys.stderr,
        )
        return 2

    # Prime-scheme gate: the reconstruction below assumes sha256_64_v1. A
    # bundle under any other scheme would factor differently and silently
    # produce a state-integer mismatch. Reject up front with a clear
    # pointer rather than let the error surface as "state integer mismatch".
    declared_scheme = bundle.get("prime_scheme", _SUPPORTED_PRIME_SCHEME)
    if declared_scheme != _SUPPORTED_PRIME_SCHEME:
        print(
            f"sum: unsupported prime_scheme {declared_scheme!r} "
            f"(this CLI speaks {_SUPPORTED_PRIME_SCHEME}). "
            f"See docs/PROOF_BOUNDARY.md for the active scheme registry.",
            file=sys.stderr,
        )
        return 2

    # Signature verification runs BEFORE reconstruction — a forged bundle
    # with a valid-looking tome but mismatched signatures should fail on
    # the cryptographic gate first, before we spend CPU on factoring.
    ed25519_status = _verify_ed25519_bundle(bundle)
    hmac_status = _verify_hmac_bundle(bundle, args.signing_key)

    if ed25519_status == "invalid":
        print(
            "sum: ✗ Ed25519 signature invalid — "
            "bundle content does not match the embedded public key",
            file=sys.stderr,
        )
        return 1
    if hmac_status == "invalid":
        print(
            "sum: ✗ HMAC signature invalid — "
            "bundle tampered or signed with a different key",
            file=sys.stderr,
        )
        return 1
    if args.strict:
        if ed25519_status == "absent" and hmac_status == "absent":
            print(
                "sum: ✗ --strict: no signatures present to verify",
                file=sys.stderr,
            )
            return 1
        if hmac_status == "skipped":
            print(
                "sum: ✗ --strict: HMAC signature present but no --signing-key supplied",
                file=sys.stderr,
            )
            return 1

    # Reconstruct the state integer from the canonical tome lines — same
    # logic verify.js runs in Node, same bytes Python produces.
    import math
    import re
    from internal.algorithms.semantic_arithmetic import GodelStateAlgebra

    algebra = GodelStateAlgebra()  # type: ignore[no-untyped-call]
    pattern = re.compile(r"^The (\S+) (\S+) (.+)\.$")
    state = 1
    axioms = 0
    for line in bundle["canonical_tome"].splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        subj, pred, obj = match.group(1), match.group(2), match.group(3)
        prime = algebra.get_or_mint_prime(subj, pred, obj)
        state = math.lcm(state, prime)
        axioms += 1

    claimed_state_str = bundle["state_integer"]
    try:
        claimed_state = int(claimed_state_str)
    except ValueError:
        print(f"sum: state_integer is not an integer: {claimed_state_str!r}", file=sys.stderr)
        return 2

    expected_count = bundle.get("axiom_count")
    if expected_count is not None and axioms != expected_count:
        print(
            f"sum: ✗ axiom count mismatch (parsed {axioms}, claimed {expected_count})",
            file=sys.stderr,
        )
        return 1

    if state != claimed_state:
        short_got = str(state)[:32]
        short_claim = claimed_state_str[:32]
        print(
            f"sum: ✗ state integer mismatch: "
            f"claimed {short_claim}…, reconstructed {short_got}…",
            file=sys.stderr,
        )
        return 1

    # Machine-readable success payload on stdout; human message on stderr.
    result = {
        "ok": True,
        "axioms": axioms,
        "state_integer_digits": len(claimed_state_str),
        "branch": bundle.get("branch", "main"),
        "bundle_version": bundle.get("bundle_version", "unknown"),
        "signatures": {"hmac": hmac_status, "ed25519": ed25519_status},
    }
    json.dump(result, sys.stdout, indent=2 if args.pretty else None)
    sys.stdout.write("\n")
    marks = f"hmac={hmac_status}, ed25519={ed25519_status}"
    print(
        f"sum: ✓ verified {axioms} axiom(s), state integer matches ({marks})",
        file=sys.stderr,
    )
    return 0


# ─── resolve ─────────────────────────────────────────────────────────

def cmd_resolve(args: argparse.Namespace) -> int:
    import asyncio
    from internal.infrastructure.akashic_ledger import AkashicLedger

    ledger = AkashicLedger(db_path=args.db)
    record = asyncio.run(ledger.get_provenance_record(args.prov_id))
    if record is None:
        print(f"sum: prov_id {args.prov_id!r} not found in {args.db}", file=sys.stderr)
        return 1

    payload = record.to_dict() if hasattr(record, "to_dict") else dict(record.__dict__)
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


# ─── Argparse wiring ─────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sum",
        description=(
            "SUM — bidirectional knowledge distillation with optional "
            "cryptographic attestation. Pipe prose, get a CanonicalBundle "
            "whose state integer anyone can re-derive, verify anywhere."
        ),
        epilog=(
            "Examples:\n"
            "  echo 'Alice likes cats.' | sum attest > bundle.json\n"
            "  sum verify < bundle.json                 # structural only\n"
            "  sum attest --ed25519-key keys/issuer.pem | sum verify --strict\n"
            "  sum resolve prov:abc123... --db akashic.db\n"
            "\n"
            "Attestation layers (all optional, compose freely):\n"
            "  state integer   — content-addressed integrity (always present)\n"
            "  --signing-key   — HMAC-SHA256 for shared-secret peers\n"
            "  --ed25519-key   — Ed25519 public-key attestation (W3C VC 2.0)\n"
            "\n"
            "For the full feature catalog, see "
            "https://github.com/OtotaO/SUM/blob/main/docs/FEATURE_CATALOG.md"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"sum {__version__}"
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True, metavar="<command>")

    # attest
    p_attest = subparsers.add_parser(
        "attest",
        help="Extract facts from stdin prose and emit a CanonicalBundle (optionally signed).",
        description=(
            "Reads prose from stdin (or --input), extracts (subject, predicate, "
            "object) triples, mints a prime per triple via sha256_64_v1, LCMs "
            "them into a Gödel state integer, and emits a CanonicalBundle as "
            "JSON on stdout. Signatures are OPT-IN: add --signing-key for HMAC "
            "or --ed25519-key for Ed25519 public-key attestation (the two "
            "compose). Without either flag the bundle is unsigned — the state "
            "integer still content-addresses the axiom set, so structural "
            "integrity is verifiable by anyone. Exit codes: 0 on success, 2 "
            "on malformed input, 3 when extraction yields zero triples (with "
            "diagnostic on stderr; stdout stays empty)."
        ),
    )
    p_attest.add_argument("--input", "-i", help="Read from this path instead of stdin ('-' for stdin).")
    p_attest.add_argument(
        "--extractor", choices=["sieve", "llm"],
        help="Override extractor selection. Default: auto-detect (sieve if spaCy installed, llm if OPENAI_API_KEY set).",
    )
    p_attest.add_argument(
        "--model",
        help="LLM model ID (pinned snapshot) for --extractor=llm. Default: gpt-4o-mini-2024-07-18.",
    )
    p_attest.add_argument("--source", help="Source URI for the bundle. Default: sha256: of input bytes.")
    p_attest.add_argument("--branch", default="main", help="Branch name for the bundle. Default: main.")
    p_attest.add_argument("--title", default="Attested Tome", help="Tome title. Default: 'Attested Tome'.")
    p_attest.add_argument(
        "--signing-key", default=None,
        help=(
            "HMAC signing key for the bundle signature. Default: unset — "
            "no HMAC is emitted. Set this only for shared-secret peers; "
            "for public-domain transport, rely on Ed25519 + did:web / "
            "did:key (see docs/DID_SETUP.md)."
        ),
    )
    p_attest.add_argument(
        "--ed25519-key", default=None, metavar="PEM",
        help=(
            "PATH to an Ed25519 PEM private key. When set, the emitted "
            "bundle carries public_signature + public_key, verifiable by "
            "`sum verify` or any W3C VC 2.0 verifier with no shared "
            "secret. Generate a key with: python -m scripts.generate_did_web "
            "(see docs/DID_SETUP.md)."
        ),
    )
    p_attest.add_argument("--pretty", action="store_true", help="Pretty-print output JSON.")
    p_attest.add_argument("--verbose", "-v", action="store_true", help="Emit diagnostics on stderr.")
    p_attest.set_defaults(func=cmd_attest)

    # verify
    p_verify = subparsers.add_parser(
        "verify",
        help="Verify a CanonicalBundle: signatures + state-integer reconstruction.",
        description=(
            "Reads a CanonicalBundle JSON from stdin (or --input/arg) and "
            "verifies, in order: "
            "(1) Ed25519 signature over the payload line, if present. "
            "Self-contained — the public key is embedded in the bundle. "
            "(2) HMAC-SHA256 signature, if --signing-key is supplied. "
            "Without the key, a present HMAC is reported as 'skipped' "
            "(not a pass). "
            "(3) Canonical tome reconstruction — re-derive primes via "
            "sha256_64_v1, LCM them, compare to the claimed state_integer. "
            "Exits 0 on match, 1 on signature or state mismatch, 2 on "
            "malformed input. The JSON result on stdout reports which "
            "signatures were verified vs absent vs skipped."
        ),
    )
    p_verify.add_argument("--input", "-i", help="Read from this path instead of stdin ('-' for stdin).")
    p_verify.add_argument(
        "--signing-key", default=None,
        help=(
            "HMAC key to verify the bundle's 'signature' field against. "
            "Omit for Ed25519-only or unsigned bundles."
        ),
    )
    p_verify.add_argument(
        "--strict", action="store_true",
        help=(
            "Require at least one verifiable signature. Fail if a "
            "signature field is present but not verifiable "
            "(e.g. HMAC present without --signing-key), or if the bundle "
            "has no signatures at all."
        ),
    )
    p_verify.add_argument("--pretty", action="store_true", help="Pretty-print result JSON on stdout.")
    p_verify.set_defaults(func=cmd_verify)

    # resolve
    p_resolve = subparsers.add_parser(
        "resolve",
        help="Look up a ProvenanceRecord by prov_id in a local Akashic ledger.",
        description=(
            "Opens the SQLite ledger at --db (default: ./akashic.db) and "
            "returns the ProvenanceRecord matching the given prov_id. Exits "
            "0 if found, 1 if not."
        ),
    )
    p_resolve.add_argument("prov_id", help="Content-addressable provenance id (e.g. 'prov:abc123...').")
    p_resolve.add_argument("--db", default="akashic.db", help="Path to the SQLite ledger. Default: ./akashic.db.")
    p_resolve.set_defaults(func=cmd_resolve)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("sum: interrupted", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 — CLI is the top-level boundary.
        print(f"sum: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
