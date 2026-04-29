"""SUM CLI entry point.

Subcommands:
    sum attest     — stdin prose → CanonicalBundle JSON on stdout
                     (optionally HMAC + Ed25519 + per-triple ledger)
    sum verify     — bundle → exit 0 on match, 1 on signature/state mismatch
    sum resolve    — prov_id → ProvenanceRecord JSON (ledger lookup)
    sum ledger     — introspect a ledger: list | stats | head
    sum inspect    — structural read of a bundle (no crypto, no reconstruction)
    sum schema     — JSON Schema for bundle | provenance | credential
    sum --version  — print version string
    sum --help     — auto-generated usage

Design notes (read once, explained here so the code stays terse):

  - argparse only. No click, no typer. Stdlib keeps the install minimal and
    cold-start fast (`python -c "import sum_cli"` must stay under 100 ms).
  - Heavy imports (spacy, openai, akashic ledger) are lazy. Loading the
    module to call --version or --help must NOT drag the whole
    sum_engine_internal.algorithms.* tree into memory.
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
    from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve
    sieve = DeterministicSieve()  # type: ignore[no-untyped-call]
    return sieve.extract_triplets(text)


def _extract_sieve_with_provenance(text: str, source_uri: str):
    """Sieve extraction that also returns per-triple ProvenanceRecords.

    Used by the --ledger path so every triple comes with its
    originating byte range in the source. Returns a pair:

        (triples, [(triple, ProvenanceRecord), …])

    Triples are the same list _extract_sieve would return; the paired
    form carries the extra evidence the ledger needs.
    """
    from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve

    sieve = DeterministicSieve()  # type: ignore[no-untyped-call]
    pairs = sieve.extract_with_provenance(text, source_uri=source_uri)
    triples = [triple for triple, _ in pairs]
    return triples, pairs


def _extract_llm(text: str, model: str) -> list[tuple[str, str, str]]:
    import asyncio
    from sum_engine_internal.ensemble.live_llm_adapter import LiveLLMAdapter
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

    # Build the CanonicalBundle via the existing codec path — no
    # reimplementation of Ed25519/HMAC, no reimplementation of canonical
    # tome generation. The CLI is a thin wrapper.
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec

    key_manager = _load_ed25519_key(args.ed25519_key) if args.ed25519_key else None

    algebra = GodelStateAlgebra()  # type: ignore[no-untyped-call]

    # Three extraction paths, all converging on (state, triples):
    #
    #   --ledger             → provenance-recording sieve (per-triple
    #                          byte ranges; cannot use chunked path
    #                          because the byte-range extractor is
    #                          one-shot).
    #
    #   --extractor=sieve    → chunked path via state_for_corpus.
    #                          Single-chunk fast path for inputs ≤
    #                          chunk_chars (current behaviour); for
    #                          larger inputs the spaCy sentencizer
    #                          chunks on sentence boundaries and the
    #                          state remains byte-identical to the
    #                          unchunked encoding (Tests/test_chunked_
    #                          state_composition.py asserts this).
    #
    #   --extractor=llm      → unchunked LLM extraction. The chunked
    #                          equivalence does NOT hold for LLM
    #                          extractors (they resolve coreference
    #                          across chunk boundaries) — see
    #                          compose_chunk_states docstring. Future
    #                          work could add a chunked LLM path with
    #                          a different equivalence claim, but it
    #                          is NOT this PR.
    prov_records = None
    if args.ledger:
        if extractor != "sieve":
            raise SystemExit(
                f"sum: --ledger requires --extractor=sieve (today's only "
                f"provenance-producing extractor). Got extractor={extractor!r}. "
                f"Re-run with --extractor=sieve, or omit --ledger."
            )
        triples, prov_records = _extract_sieve_with_provenance(text, source_uri)
        state = algebra.encode_chunk_state(list(triples))
    elif extractor == "sieve":
        from sum_engine_internal.algorithms.chunked_corpus import state_for_corpus
        state, triples = state_for_corpus(text, algebra)
    else:
        triples = _extract(text, extractor, args.model)
        state = algebra.encode_chunk_state(list(triples))

    if not triples:
        print(
            "sum: extractor returned zero triples. "
            "Input may be too short, negated, or hedged — "
            "see docs/FEATURE_CATALOG.md entries 6-9 for the suppression rules.",
            file=sys.stderr,
        )
        return 3

    tome_generator = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(
        algebra,
        tome_generator,
        signing_key=args.signing_key,
        key_manager=key_manager,
    )
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

    # Ledger path: record byte-level provenance for every extracted
    # triple and attach the returned prov_ids to the bundle so downstream
    # tools (sum resolve, other AkashicLedger clients) can walk from an
    # axiom back to its source byte range and original sentence excerpt.
    if args.ledger and prov_records is not None:
        import asyncio
        from sum_engine_internal.infrastructure.akashic_ledger import AkashicLedger

        ledger = AkashicLedger(db_path=args.ledger)

        async def _record_all():
            ids = []
            for (s, p, o), rec in prov_records:
                axiom_key = f"{s.lower()}||{p.lower()}||{o.lower()}"
                pid = await ledger.record_provenance(rec, axiom_key=axiom_key)
                ids.append(pid)
            return ids

        prov_ids = asyncio.run(_record_all())
        bundle["sum_cli"]["prov_ids"] = prov_ids
        bundle["sum_cli"]["ledger"] = args.ledger
        if args.verbose:
            print(
                f"sum: recorded {len(prov_ids)} provenance entries → {args.ledger}",
                file=sys.stderr,
            )

    json.dump(bundle, sys.stdout, indent=2 if args.pretty else None)
    sys.stdout.write("\n")

    if args.verbose:
        print(
            f"sum: minted {len(triples)} axiom(s), state_integer has "
            f"{len(bundle['state_integer'])} digits",
            file=sys.stderr,
        )
    return 0


# ─── attest-batch ────────────────────────────────────────────────────


def cmd_attest_batch(args: argparse.Namespace) -> int:
    """Attest each input file independently and emit one bundle per
    line of stdout (JSONL).

    Per-file independence: a parse failure on one file does not abort
    the batch; the file is reported on stderr and the run continues.
    Exit code is 0 if every file produced a bundle, 1 if any file
    failed.

    Constraints (deliberately narrower than ``sum attest``):
      - ``--ledger`` is unsupported here. Batch ledger writes need a
        per-file db routing decision; out of scope for this surface.
        Use ``sum attest --ledger`` per file when provenance is
        required.
      - ``--ed25519-key`` and ``--signing-key`` apply to every bundle
        in the batch (one signing identity for the whole run).
      - ``--source`` is auto-derived per file (sha256: of the file's
        bytes); no override (would collide across files).

    Output shape: one bundle JSON per line on stdout, in input
    order. Stable for piping to ``jq -c`` or further filters.
    """
    files: list[str] = list(getattr(args, "files", []) or [])
    if not files:
        print(
            "sum: attest-batch requires at least one input file. "
            "Use `sum attest` for a single file from stdin.",
            file=sys.stderr,
        )
        return 2

    extractor = _pick_extractor(args.extractor)
    if args.verbose:
        print(
            f"sum: attest-batch extractor={extractor} files={len(files)}",
            file=sys.stderr,
        )

    # Build a single algebra + codec for the whole batch. The algebra's
    # axiom→prime cache is process-local but every prime is deterministic,
    # so re-encoding the same triple in two files produces the same
    # prime — no cross-file state pollution to worry about.
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
    from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec

    key_manager = _load_ed25519_key(args.ed25519_key) if args.ed25519_key else None
    algebra = GodelStateAlgebra()  # type: ignore[no-untyped-call]
    tome_generator = AutoregressiveTomeGenerator(algebra)
    codec = CanonicalCodec(
        algebra,
        tome_generator,
        signing_key=args.signing_key,
        key_manager=key_manager,
    )

    failed = 0
    succeeded = 0
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except OSError as e:
            print(
                f"sum: file={path} error=read_failed: {e}",
                file=sys.stderr,
            )
            failed += 1
            continue

        if not text:
            print(f"sum: file={path} error=empty_input", file=sys.stderr)
            failed += 1
            continue

        source_uri = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

        try:
            if extractor == "sieve":
                from sum_engine_internal.algorithms.chunked_corpus import (
                    state_for_corpus,
                )
                state, triples = state_for_corpus(text, algebra)
            else:
                triples = _extract(text, extractor, args.model)
                state = algebra.encode_chunk_state(list(triples))
        except Exception as e:  # noqa: BLE001 — surface ANY extraction failure
            print(
                f"sum: file={path} error=extraction_failed: {e!r}",
                file=sys.stderr,
            )
            failed += 1
            continue

        if not triples:
            print(
                f"sum: file={path} error=zero_triples "
                f"(input may be too short, negated, or hedged)",
                file=sys.stderr,
            )
            failed += 1
            continue

        bundle = codec.export_bundle(
            state,
            branch=args.branch,
            title=args.title,
        )
        bundle["sum_cli"] = {
            "extractor": extractor,
            "source_uri": source_uri,
            "source_path": path,
            "cli_version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "batch": True,
        }
        # JSONL: compact, one line per record, no pretty-print.
        json.dump(bundle, sys.stdout)
        sys.stdout.write("\n")
        succeeded += 1
        if args.verbose:
            print(
                f"sum: file={path} ok triples={len(triples)} "
                f"digits={len(bundle['state_integer'])}",
                file=sys.stderr,
            )

    if args.verbose or failed:
        print(
            f"sum: attest-batch done — {succeeded} succeeded, {failed} failed",
            file=sys.stderr,
        )
    return 0 if failed == 0 else 1


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
    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra

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

    # Extraction provenance — closes the THREAT_MODEL.md §3.3
    # "signed ≠ true" visibility gap. The signature proves the
    # canonical tome maps to the state integer; it does NOT prove
    # the axioms are factually correct, and it does NOT prove
    # re-extraction would be reproducible. Surfacing the extractor
    # as a first-class verify-output field lets a downstream
    # consumer branch on `extraction.verifiable` without parsing
    # the sum_cli sidecar by hand.
    sidecar = bundle.get("sum_cli") if isinstance(bundle.get("sum_cli"), dict) else None
    extractor = sidecar.get("extractor") if sidecar else None
    extraction = {
        "extractor": extractor,                      # "sieve" | "llm" | None
        "verifiable": extractor == "sieve",          # deterministic re-extraction
        "source": "sum_cli sidecar" if sidecar else "absent",
    }

    # Machine-readable success payload on stdout; human message on stderr.
    result = {
        "ok": True,
        "axioms": axioms,
        "state_integer_digits": len(claimed_state_str),
        "branch": bundle.get("branch", "main"),
        "bundle_version": bundle.get("bundle_version", "unknown"),
        "signatures": {"hmac": hmac_status, "ed25519": ed25519_status},
        "extraction": extraction,
    }
    json.dump(result, sys.stdout, indent=2 if args.pretty else None)
    sys.stdout.write("\n")
    marks = f"hmac={hmac_status}, ed25519={ed25519_status}"
    if extractor:
        verif = "verifiable" if extraction["verifiable"] else "advisory"
        marks += f", extractor={extractor} ({verif})"
    print(
        f"sum: ✓ verified {axioms} axiom(s), state integer matches ({marks})",
        file=sys.stderr,
    )
    return 0


# ─── resolve ─────────────────────────────────────────────────────────

def cmd_resolve(args: argparse.Namespace) -> int:
    import asyncio
    from sum_engine_internal.infrastructure.akashic_ledger import AkashicLedger

    ledger = AkashicLedger(db_path=args.db)
    record = asyncio.run(ledger.get_provenance_record(args.prov_id))
    if record is None:
        print(f"sum: prov_id {args.prov_id!r} not found in {args.db}", file=sys.stderr)
        return 1

    payload = record.to_dict() if hasattr(record, "to_dict") else dict(record.__dict__)
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


# ─── ledger ──────────────────────────────────────────────────────────
#
# Agentic-first introspection for an AkashicLedger. An agent that wants to
# know "what's in this ledger?" without already having a prov_id in hand
# previously had no answer — resolve looked up by id, and no other command
# enumerated. These three subcommands fill that gap: list (NDJSON rows),
# stats (one-shot summary), head (current state integer per branch).
#
# Output discipline: one JSON object per line (NDJSON) for list, a single
# pretty-printed JSON object for stats and head. Agents can pipe list
# into jq / jsonl tooling; humans can read stats / head directly.

def _open_ledger_or_exit(db_path: str):
    """Sanity-check the ledger path. Exit 2 with a clear message if the
    file does not exist — the ledger auto-creates tables on __init__,
    so "opening" a nonexistent path would silently make a new empty
    database, which is almost never what an agent introspecting a
    ledger wants."""
    from pathlib import Path

    from sum_engine_internal.infrastructure.akashic_ledger import AkashicLedger

    if not Path(db_path).exists():
        print(
            f"sum: ledger not found at {db_path}. "
            f"Mint one with: sum attest --ledger {db_path} < prose.txt",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return AkashicLedger(db_path=db_path)


def cmd_ledger_list(args: argparse.Namespace) -> int:
    """Enumerate prov_ids with their linked axiom_keys and evidence spans.

    Output: NDJSON (one record per line). --limit caps output size;
    --axiom filters to a single axiom_key; --since takes an ISO 8601
    timestamp and emits only records at-or-after. Combined filters AND.
    """
    import sqlite3

    _open_ledger_or_exit(args.db)
    # Direct SQL — the ledger does not expose a list-all method and
    # adding one would force an async surface we do not need here.
    # Two-table join: axiom_provenance (axiom_key → prov_id) +
    # provenance_records (prov_id → record_json).
    clauses = []
    params: list[object] = []
    if args.axiom:
        clauses.append("ap.axiom_key = ?")
        params.append(args.axiom)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    limit_clause = f" LIMIT {int(args.limit)}" if args.limit else ""

    emitted = 0
    with sqlite3.connect(args.db) as conn:
        rows = conn.execute(
            f"SELECT ap.prov_id, ap.axiom_key, pr.record_json "
            f"FROM axiom_provenance AS ap "
            f"JOIN provenance_records AS pr ON pr.prov_id = ap.prov_id"
            f"{where}"
            f" ORDER BY ap.prov_id{limit_clause}",
            params,
        ).fetchall()

        for prov_id, axiom_key, record_json in rows:
            rec = json.loads(record_json)
            if args.since and rec.get("timestamp", "") < args.since:
                continue
            out = {
                "prov_id": prov_id,
                "axiom_key": axiom_key,
                "source_uri": rec.get("source_uri"),
                "byte_start": rec.get("byte_start"),
                "byte_end": rec.get("byte_end"),
                "timestamp": rec.get("timestamp"),
                "extractor_id": rec.get("extractor_id"),
            }
            json.dump(out, sys.stdout)
            sys.stdout.write("\n")
            emitted += 1
    if args.verbose:
        print(f"sum: emitted {emitted} record(s) from {args.db}", file=sys.stderr)
    return 0


def cmd_ledger_stats(args: argparse.Namespace) -> int:
    """Emit a one-shot summary of the ledger's state."""
    import asyncio
    import sqlite3

    ledger = _open_ledger_or_exit(args.db)

    with sqlite3.connect(args.db) as conn:
        total_prov, = conn.execute(
            "SELECT COUNT(*) FROM provenance_records"
        ).fetchone()
        distinct_axioms, = conn.execute(
            "SELECT COUNT(DISTINCT axiom_key) FROM axiom_provenance"
        ).fetchone()
        # Pull min/max timestamp across the stored JSON; SQLite cannot
        # index inside JSON blobs without a generated column, so we
        # accept the scan — ledger introspection is not a hot path.
        ts_rows = conn.execute(
            "SELECT record_json FROM provenance_records"
        ).fetchall()

    timestamps = [
        json.loads(r[0]).get("timestamp", "") for r in ts_rows
        if r[0]
    ]
    timestamps = [t for t in timestamps if t]

    chain_tip = asyncio.run(ledger.get_chain_tip())
    branch_heads = asyncio.run(ledger.load_branch_heads())
    # Size in digits, not the integer itself (branch heads get huge and
    # an agent parsing JSON does not want a megabyte integer inline).
    branches = {
        name: {"state_integer_digits": len(str(state_int))}
        for name, state_int in branch_heads.items()
    }

    out = {
        "db_path": args.db,
        "provenance_records_total": total_prov,
        "distinct_axiom_keys": distinct_axioms,
        "earliest_timestamp": min(timestamps) if timestamps else None,
        "latest_timestamp": max(timestamps) if timestamps else None,
        "chain_tip_hash": chain_tip,
        "branches": branches,
    }
    json.dump(out, sys.stdout, indent=2 if args.pretty else None)
    sys.stdout.write("\n")
    return 0


def cmd_ledger_head(args: argparse.Namespace) -> int:
    """Emit current state integer for one branch (if --branch) or all."""
    import asyncio

    ledger = _open_ledger_or_exit(args.db)
    heads = asyncio.run(ledger.load_branch_heads())
    if args.branch:
        state = heads.get(args.branch)
        if state is None:
            print(
                f"sum: branch {args.branch!r} not found in {args.db}. "
                f"Known branches: {sorted(heads)}",
                file=sys.stderr,
            )
            return 1
        out = {"branch": args.branch, "state_integer": str(state)}
    else:
        # String-encode every branch's state integer to avoid JSON
        # int-precision loss for agents whose parsers use doubles.
        out = {
            "branches": {
                name: {"state_integer": str(state)}
                for name, state in heads.items()
            }
        }
    json.dump(out, sys.stdout, indent=2 if args.pretty else None)
    sys.stdout.write("\n")
    return 0


# ─── inspect ─────────────────────────────────────────────────────────
#
# Fast, crypto-free read of a bundle's structural shape. Answers "what
# does this bundle contain?" without running Ed25519 verification,
# re-deriving primes, or reconstructing the state integer — useful when
# an agent wants to route based on bundle attributes before deciding
# whether to pay the full verify cost.

def cmd_inspect(args: argparse.Namespace) -> int:
    raw = _read_input(args.input)
    try:
        b = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"sum: bundle is not valid JSON: {e}", file=sys.stderr)
        return 2

    # Count axioms by parsing the canonical tome's line structure — same
    # regex verify uses, but we do not build primes or check the state.
    import re
    line_re = re.compile(r"^The (\S+) (\S+) (.+)\.$")
    tome = b.get("canonical_tome", "")
    axiom_lines = [
        line for line in tome.splitlines()
        if line_re.match(line.strip())
    ]

    state_str = b.get("state_integer", "")
    signatures = {
        "hmac_present": bool(b.get("signature")),
        "ed25519_present": bool(b.get("public_signature") and b.get("public_key")),
    }

    out = {
        "bundle_version": b.get("bundle_version"),
        "canonical_format_version": b.get("canonical_format_version"),
        "prime_scheme": b.get("prime_scheme", "sha256_64_v1"),
        "branch": b.get("branch"),
        "timestamp": b.get("timestamp"),
        "is_delta": bool(b.get("is_delta", False)),
        "axiom_count_claimed": b.get("axiom_count"),
        "axiom_count_parsed": len(axiom_lines),
        "state_integer_digits": len(state_str),
        "signatures": signatures,
        # Surface sum_cli sidecar if present — agents want to know
        # whether a bundle carries provenance refs without digging.
        "sum_cli": b.get("sum_cli"),
    }
    json.dump(out, sys.stdout, indent=2 if args.pretty else None)
    sys.stdout.write("\n")
    return 0


# ─── schema ──────────────────────────────────────────────────────────
#
# Print JSON Schema for each output shape. Agents validating SUM output
# programmatically previously had to reverse-engineer the shape from
# prose docs; these schemas are the ground truth.

_BUNDLE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/OtotaO/SUM/schemas/canonical-bundle.json",
    "title": "CanonicalBundle",
    "description": "A self-contained, optionally-signed SUM knowledge transport unit.",
    "type": "object",
    "required": [
        "bundle_version", "canonical_format_version", "branch",
        "axiom_count", "canonical_tome", "state_integer", "timestamp",
    ],
    "properties": {
        "bundle_version": {"type": "string", "examples": ["1.1.0"]},
        "canonical_format_version": {"type": "string", "examples": ["1.0.0"]},
        "branch": {"type": "string"},
        "axiom_count": {"type": "integer", "minimum": 0},
        "canonical_tome": {
            "type": "string",
            "description": "Markdown-ish rendering with `The S P O.` lines, one axiom per line.",
        },
        "state_integer": {
            "type": "string",
            "description": "Decimal-encoded Gödel state integer (string, not number, to preserve precision).",
            "pattern": "^[0-9]+$",
        },
        "state_integer_hex": {"type": "string", "pattern": "^0x[0-9a-f]+$"},
        "timestamp": {"type": "string", "format": "date-time"},
        "prime_scheme": {"type": "string", "enum": ["sha256_64_v1", "sha256_128_v2"]},
        "is_delta": {"type": "boolean"},
        "signature": {
            "type": "string",
            "pattern": "^hmac-sha256:[0-9a-f]{64}$",
            "description": "Optional HMAC-SHA256 over {tome|state|timestamp}.",
        },
        "public_signature": {
            "type": "string",
            "pattern": "^ed25519:.+$",
            "description": "Optional Ed25519 signature (base64) over {tome|state|timestamp}.",
        },
        "public_key": {
            "type": "string",
            "pattern": "^ed25519:.+$",
            "description": "Optional embedded Ed25519 public key (base64).",
        },
        "sum_cli": {
            "type": "object",
            "description": "Non-normative sidecar from the sum CLI: extractor, source_uri, prov_ids, cli_version.",
            "additionalProperties": True,
        },
    },
    "additionalProperties": True,
}

_PROVENANCE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/OtotaO/SUM/schemas/provenance-record.json",
    "title": "ProvenanceRecord",
    "description": "Byte-level evidence that a given axiom was extracted from a specific source span by a specific extractor.",
    "type": "object",
    "required": ["source_uri", "byte_start", "byte_end", "extractor_id", "timestamp", "text_excerpt"],
    "properties": {
        "source_uri": {
            "type": "string",
            "description": "sha256:<64-hex>, doi:, https://, or urn:sum:source:.",
        },
        "byte_start": {"type": "integer", "minimum": 0},
        "byte_end": {"type": "integer", "minimum": 1},
        "extractor_id": {"type": "string", "examples": ["sum.sieve:deterministic_v1"]},
        "timestamp": {"type": "string", "format": "date-time"},
        "text_excerpt": {"type": "string", "maxLength": 1024},
        "schema_version": {"type": "string", "examples": ["1.0.0"]},
    },
    "additionalProperties": False,
}

_CREDENTIAL_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/OtotaO/SUM/schemas/verifiable-credential.json",
    "title": "VerifiableCredential 2.0 (eddsa-jcs-2022)",
    "description": "W3C VC 2.0 credential shape SUM emits via sign_credential. Not a complete VC 2.0 schema — just the subset SUM produces.",
    "type": "object",
    "required": ["@context", "type", "issuer", "credentialSubject", "proof"],
    "properties": {
        "@context": {"type": "array", "items": {"type": "string"}},
        "type": {"type": "array", "items": {"type": "string"}},
        "issuer": {"type": "string", "description": "did:key:, did:web:, or https://"},
        "validFrom": {"type": "string", "format": "date-time"},
        "credentialSubject": {"type": "object"},
        "proof": {
            "type": "object",
            "required": ["type", "cryptosuite", "verificationMethod", "proofPurpose", "proofValue"],
            "properties": {
                "type": {"const": "DataIntegrityProof"},
                "cryptosuite": {"const": "eddsa-jcs-2022"},
                "verificationMethod": {"type": "string"},
                "proofPurpose": {"const": "assertionMethod"},
                "proofValue": {"type": "string", "description": "Multibase base58btc-encoded Ed25519 signature."},
                "created": {"type": "string", "format": "date-time"},
            },
        },
    },
    "additionalProperties": True,
}

_SCHEMA_BY_NAME = {
    "bundle": _BUNDLE_SCHEMA,
    "provenance": _PROVENANCE_SCHEMA,
    "credential": _CREDENTIAL_SCHEMA,
}


def cmd_schema(args: argparse.Namespace) -> int:
    schema = _SCHEMA_BY_NAME.get(args.shape)
    if schema is None:
        print(
            f"sum: unknown schema {args.shape!r}. Known: "
            f"{sorted(_SCHEMA_BY_NAME)}",
            file=sys.stderr,
        )
        return 2
    json.dump(schema, sys.stdout, indent=2)
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
            "  sum verify < bundle.json                  # structural only\n"
            "  sum attest --ed25519-key keys/issuer.pem | sum verify --strict\n"
            "  sum attest --ledger akashic.db < prose.txt\n"
            "  sum resolve prov:abc123... --db akashic.db\n"
            "  sum ledger list --db akashic.db           # NDJSON of prov_ids\n"
            "  sum ledger stats --db akashic.db --pretty\n"
            "  sum inspect bundle.json --pretty          # no-crypto read\n"
            "  sum schema bundle                         # JSON Schema stdout\n"
            "\n"
            "Attestation layers (all optional, compose freely):\n"
            "  state integer   — content-addressed integrity (always present)\n"
            "  --signing-key   — HMAC-SHA256 for shared-secret peers\n"
            "  --ed25519-key   — Ed25519 public-key attestation (W3C VC 2.0)\n"
            "  --ledger        — per-triple byte-level ProvenanceRecords\n"
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
    p_attest.add_argument(
        "--ledger", default=None, metavar="DB",
        help=(
            "PATH to an AkashicLedger SQLite file. When set, record per-"
            "triple byte-level ProvenanceRecords (source URI, byte range, "
            "sentence excerpt, extractor ID) and attach the resulting "
            "prov_ids to bundle.sum_cli.prov_ids. Later: `sum resolve "
            "<prov_id> --db <DB>` retrieves the evidence span. Requires "
            "--extractor=sieve (today's only provenance-producing extractor)."
        ),
    )
    p_attest.add_argument("--pretty", action="store_true", help="Pretty-print output JSON.")
    p_attest.add_argument("--verbose", "-v", action="store_true", help="Emit diagnostics on stderr.")
    p_attest.set_defaults(func=cmd_attest)

    # attest-batch
    p_attest_batch = subparsers.add_parser(
        "attest-batch",
        help="Attest each input file independently; emit one bundle per line on stdout (JSONL).",
        description=(
            "Per-file batch attestation. Reads N file paths as positional "
            "args, extracts triples from each via the sieve, mints one "
            "CanonicalBundle per file, and emits the bundles as JSONL on "
            "stdout (one compact JSON per line, in input order). Per-file "
            "failures (read errors, zero triples, extraction errors) are "
            "reported on stderr in `sum: file=<path> error=<reason>` format "
            "and the run continues. Exit code is 0 if every file produced "
            "a bundle, 1 if any failed. Signatures (--ed25519-key, "
            "--signing-key) apply to every bundle in the batch. --ledger "
            "is unsupported here (use `sum attest --ledger` per file when "
            "byte-level provenance is required)."
        ),
    )
    p_attest_batch.add_argument("files", nargs="+", help="One or more input file paths.")
    p_attest_batch.add_argument(
        "--extractor", choices=["sieve", "llm"],
        help="Override extractor selection. Default: auto-detect.",
    )
    p_attest_batch.add_argument(
        "--model",
        help="LLM model ID (pinned snapshot) for --extractor=llm. Default: gpt-4o-mini-2024-07-18.",
    )
    p_attest_batch.add_argument("--branch", default="main", help="Branch name. Default: main.")
    p_attest_batch.add_argument("--title", default="Attested Tome", help="Tome title. Default: 'Attested Tome'.")
    p_attest_batch.add_argument(
        "--signing-key", default=None,
        help="HMAC signing key applied to every bundle in the batch. Default: unset.",
    )
    p_attest_batch.add_argument(
        "--ed25519-key", default=None, metavar="PEM",
        help="PATH to an Ed25519 PEM private key. Applied to every bundle in the batch.",
    )
    p_attest_batch.add_argument("--verbose", "-v", action="store_true", help="Emit diagnostics on stderr.")
    p_attest_batch.set_defaults(func=cmd_attest_batch)

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

    # ledger — introspect an AkashicLedger without a prov_id in hand.
    p_ledger = subparsers.add_parser(
        "ledger",
        help="Introspect an AkashicLedger (list prov_ids, stats, branch heads).",
        description=(
            "Agentic introspection for an AkashicLedger. Three subcommands: "
            "list (NDJSON rows, one per prov_id), stats (one-shot summary), "
            "head (current state integer per branch)."
        ),
    )
    p_ledger_sub = p_ledger.add_subparsers(dest="ledger_cmd", required=True, metavar="<ledger-cmd>")

    p_list = p_ledger_sub.add_parser(
        "list",
        help="Enumerate prov_ids in the ledger (NDJSON on stdout).",
        description=(
            "Emits one JSON object per line (NDJSON): {prov_id, axiom_key, "
            "source_uri, byte_start, byte_end, timestamp, extractor_id}. "
            "Filters compose with AND: --axiom (exact axiom_key match), "
            "--since (ISO 8601, record timestamp >= since), --limit (max rows)."
        ),
    )
    p_list.add_argument("--db", default="akashic.db", help="SQLite ledger path. Default: ./akashic.db.")
    p_list.add_argument("--axiom", default=None, help="Filter to this exact axiom_key (e.g. 'alice||like||cat').")
    p_list.add_argument("--since", default=None, help="Only records with timestamp >= this ISO 8601 string.")
    p_list.add_argument("--limit", type=int, default=0, help="Max rows to emit (0 = unlimited).")
    p_list.add_argument("--verbose", "-v", action="store_true", help="Print row count on stderr.")
    p_list.set_defaults(func=cmd_ledger_list)

    p_stats = p_ledger_sub.add_parser(
        "stats",
        help="One-shot summary of ledger state: totals, timestamp range, chain tip, branches.",
    )
    p_stats.add_argument("--db", default="akashic.db", help="SQLite ledger path. Default: ./akashic.db.")
    p_stats.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    p_stats.set_defaults(func=cmd_ledger_stats)

    p_head = p_ledger_sub.add_parser(
        "head",
        help="Current state integer for one branch (--branch) or all branches.",
    )
    p_head.add_argument("--db", default="akashic.db", help="SQLite ledger path. Default: ./akashic.db.")
    p_head.add_argument("--branch", default=None, help="Specific branch name. Default: all branches.")
    p_head.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    p_head.set_defaults(func=cmd_ledger_head)

    # inspect — structural read of a bundle, no crypto, no reconstruction.
    p_inspect = subparsers.add_parser(
        "inspect",
        help="Read bundle metadata without running verification (fast, offline).",
        description=(
            "Reads a CanonicalBundle JSON and emits structural metadata: "
            "axiom counts (claimed + parsed), state integer size in digits, "
            "signature fields present, bundle + format versions, timestamp, "
            "branch, sum_cli sidecar if present. Does NOT verify signatures "
            "or reconstruct the state integer — use `sum verify` for that. "
            "Useful when an agent wants to route a bundle by shape before "
            "paying the full verify cost."
        ),
    )
    p_inspect.add_argument("--input", "-i", help="Read from this path instead of stdin ('-' for stdin).")
    p_inspect.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    p_inspect.set_defaults(func=cmd_inspect)

    # schema — JSON Schema for each shape SUM emits.
    p_schema = subparsers.add_parser(
        "schema",
        help="Emit JSON Schema for one of SUM's output shapes.",
        description=(
            "Prints a JSON Schema (Draft 2020-12) for one of SUM's output "
            "shapes. Use this to validate agent output before trusting it. "
            "Shapes: bundle (CanonicalBundle), provenance (ProvenanceRecord), "
            "credential (W3C VC 2.0 eddsa-jcs-2022)."
        ),
    )
    p_schema.add_argument(
        "shape",
        choices=["bundle", "provenance", "credential"],
        help="Which output shape to emit the schema for.",
    )
    p_schema.set_defaults(func=cmd_schema)

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
