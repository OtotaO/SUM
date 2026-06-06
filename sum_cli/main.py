"""SUM CLI entry point.

Subcommands:
    sum attest     — stdin prose → CanonicalBundle JSON on stdout
                     (optionally HMAC + Ed25519 + per-triple ledger)
    sum verify     — bundle → exit 0 on match, 1 on signature/state mismatch
    sum render     — bundle → tome text under 5-axis slider control
                     (inverse of attest; --use-worker for LLM axes + signed receipt)
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
import warnings
from datetime import datetime, timezone
from typing import Any, Optional

# Suppress transformers' `FutureWarning: torch.utils._pytree._register_pytree_node
# is deprecated`. The library prints this warning to stdout on import (not
# stderr), which poisons `sum attest > bundle.json` — the resulting file
# starts with the warning text before the JSON envelope, making it
# unparseable. Suppressing the warning at CLI-entry time is the smallest
# fix that doesn't require upstream-library patching. Surfaced as F1 in
# docs/DOGFOOD_FINDINGS_2026-05-17.md.
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
# Some transformer paths emit through torch directly; cover both.
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

from sum_cli import __version__


# ─── Extractor selection (lazy, dependency-aware) ────────────────────

def _pick_extractor(override: Optional[str] = None) -> str:
    """Pick the extractor at runtime. Honors --extractor override; otherwise
    adapts to whichever dependency is installed.

    Rules:
      1. If --extractor is given, honor it (fail fast if deps missing).
      2. If spaCy is importable, prefer 'sieve' (offline). The probe
         constructs ``DeterministicSieve()`` which auto-downloads
         ``en_core_web_sm`` on first run if missing — surfaces the
         download as a one-line stderr announcement, then proceeds.
         Direct ``spacy.load`` would NOT trigger the auto-download
         and would silently fall through, which broke cold-install
         onboarding (PyPI install → first ``sum attest`` → "no
         extractor available" even though [sieve] was just installed).
      3. If OPENAI_API_KEY is set, use 'llm' (network-dependent).
      4. Otherwise, fail with a helpful install hint.
    """
    if override:
        return override
    try:
        import spacy  # noqa: F401 — availability check only
        # Construct the sieve so its OSError-catching auto-downloader
        # fires when en_core_web_sm is absent. The sieve instance is
        # discarded; ``cmd_attest`` will reconstruct it. Cheap because
        # spaCy caches the loaded model in-process.
        from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve
        DeterministicSieve()
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


def _ingest_path(path: str, *, fmt: str = "auto") -> tuple[str, str, dict]:
    """Read input from *path* and return (text, source_uri, sidecar).

    ``fmt`` controls the omni-format pivot:

      - ``auto`` (default): route by file extension. Plaintext and
        markdown are read directly; PDF/HTML/DOCX/EPUB/etc. are
        converted through markitdown (requires
        ``pip install 'sum-engine[omni-format]'``).
      - ``raw``: read bytes verbatim, decode as UTF-8 (CRLF→LF
        normalised). No conversion, no markitdown dep needed.
      - ``markdown``: same as auto for already-markdown files;
        for other formats, force-route through markitdown.

    The returned ``sidecar`` is a dict of conversion metadata to
    attach to the bundle's ``sum_cli`` block so verifiers can
    replay the conversion: input_format, converter id, markdown
    sha256, original-bytes length and URI.

    The source URI is anchored to the **original input bytes**,
    not the markdown — a receipt for a PDF binds to the PDF, not
    to the markdown intermediate.
    """
    from sum_engine_internal.adapters.format_pivot import (
        ConvertedDocument,
        convert_to_markdown,
    )

    if fmt == "raw":
        text = _read_input(path)
        # Read the bytes ourselves to anchor source_uri; _read_input
        # already decoded with UTF-8 strict mode but for source URI
        # determinism we hash the on-disk bytes.
        from pathlib import Path
        raw_bytes = Path(path).read_bytes() if path and path != "-" else text.encode("utf-8")
        source_uri = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()
        return text, source_uri, {
            "input_format": "raw",
            "converter": "raw-readthrough",
            "source_bytes_len": len(raw_bytes),
        }

    if path is None or path == "-":
        # stdin can't be omni-converted (no extension to route by).
        text = sys.stdin.read()
        source_uri = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        return text, source_uri, {
            "input_format": "plaintext",
            "converter": "stdin-readthrough",
            "source_bytes_len": len(text.encode("utf-8")),
        }

    converted: ConvertedDocument = convert_to_markdown(path)
    sidecar = {
        "input_format": converted.input_format,
        "converter": converted.converter,
        "source_bytes_len": converted.source_bytes_len,
        "markdown_sha256": converted.markdown_sha256,
    }
    return converted.markdown, converted.source_uri, sidecar


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
    fmt = getattr(args, "format", "auto") or "auto"
    try:
        if args.input and args.input != "-":
            text, derived_uri, format_sidecar = _ingest_path(args.input, fmt=fmt)
            text = text.strip()
        else:
            # stdin: no extension to route by; treat as plaintext.
            text = _read_input(args.input).strip()
            derived_uri = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
            format_sidecar = {
                "input_format": "plaintext",
                "converter": "stdin-readthrough",
                "source_bytes_len": len(text.encode("utf-8")),
            }
    except RuntimeError as e:
        # markitdown missing for an omni-format input → actionable error
        print(f"sum: {e}", file=sys.stderr)
        return 4
    except FileNotFoundError as e:
        print(f"sum: {e}", file=sys.stderr)
        return 2

    if not text:
        print("sum: empty input", file=sys.stderr)
        return 2

    # Source URI: caller override > original-bytes hash. The original-
    # bytes hash anchors the bundle to the artifact-as-shipped (the PDF
    # itself, not the markdown), preserving the verifiable claim.
    source_uri = args.source or derived_uri

    extractor = _pick_extractor(args.extractor)
    if args.verbose:
        print(
            f"sum: extractor={extractor} format={format_sidecar.get('input_format')} "
            f"source={source_uri}",
            file=sys.stderr,
        )

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

    # Surface the extracted axioms on the bundle so downstream transforms
    # (`sum transform apply compose`, slider input shape) can consume the
    # attest output directly without re-parsing canonical_tome. The data
    # exists internally as ``triples``; before this it was dropped at
    # serialization. Additive — the signature covers
    # ``canonical_tome|state_integer|timestamp``, not the bundle JSON, so
    # writing a new top-level key does not invalidate any existing
    # signature. Format mirrors what compose._bundle_triples expects:
    # list of {subject, predicate, object} dicts.
    bundle["axioms"] = [
        {"subject": s, "predicate": p, "object": o}
        for (s, p, o) in triples
    ]

    # Optional: attach a lightweight sidecar naming the extractor + source
    # URI so downstream consumers can trace provenance without the full
    # AkashicLedger. This is additive — the CanonicalBundle schema
    # ignores unknown keys, so adding them is forward-compatible.
    bundle["sum_cli"] = {
        "extractor": extractor,
        "source_uri": source_uri,
        "cli_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # Omni-format pivot record. ``input_format`` and ``converter``
        # let a verifier replay the conversion: re-fetch the bytes
        # whose sha256 matches source_uri, run the named converter
        # version, hash the markdown, compare to ``markdown_sha256``.
        # Drift in any of those surfaces immediately.
        **format_sidecar,
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

    from sum_cli.audit_log import emit_audit_event
    emit_audit_event("attest", {
        "source_uri": source_uri,
        "axiom_count": len(triples),
        "state_integer_digits": len(bundle["state_integer"]),
        "extractor": extractor,
        "branch": args.branch,
        "signed": "public_signature" in bundle,
        "hmac": "signature" in bundle,
        "input_format": format_sidecar.get("input_format"),
    })
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

    fmt = getattr(args, "format", "auto") or "auto"

    # Optional MinHash-based dedup. When --dedup-threshold is set,
    # each file's text is sketched with a 128-permutation MinHash
    # over word 3-shingles; files whose Jaccard estimate against any
    # earlier-accepted file equals or exceeds the threshold are
    # skipped (reported on stderr in `sum: file=<path> dedup_skipped`
    # form). Threshold semantics: 1.0 = byte-identical only, 0.0 = nuke
    # the batch. Empirically, 0.85 catches near-duplicates without
    # confusing "two docs about the same topic" with "the same doc."
    # Disabled when --dedup-threshold is None or absent.
    dedup_threshold: Optional[float] = getattr(args, "dedup_threshold", None)
    accepted_signatures: list[tuple[str, "object"]] = []  # (path, MinHash)
    if dedup_threshold is not None and not (0.0 < dedup_threshold <= 1.0):
        print(
            f"sum: --dedup-threshold must be in (0.0, 1.0], got {dedup_threshold}",
            file=sys.stderr,
        )
        return 2

    failed = 0
    succeeded = 0
    deduplicated = 0
    for path in files:
        try:
            text, source_uri, format_sidecar = _ingest_path(path, fmt=fmt)
            text = text.strip()
        except FileNotFoundError as e:
            print(f"sum: file={path} error=read_failed: {e}", file=sys.stderr)
            failed += 1
            continue
        except RuntimeError as e:
            # Conversion failure (markitdown missing or upstream error)
            print(f"sum: file={path} error=conversion_failed: {e}", file=sys.stderr)
            failed += 1
            continue
        except OSError as e:
            print(f"sum: file={path} error=read_failed: {e}", file=sys.stderr)
            failed += 1
            continue

        if not text:
            print(f"sum: file={path} error=empty_input", file=sys.stderr)
            failed += 1
            continue

        # Dedup pre-check: hash before extracting. Skipping files at
        # this point saves the expensive sieve / LLM call.
        if dedup_threshold is not None:
            from sum_engine_internal.algorithms.minhash import signature_for_text
            sig = signature_for_text(text)
            duplicate_of: Optional[str] = None
            best_jaccard = 0.0
            for prior_path, prior_sig in accepted_signatures:
                j = sig.jaccard(prior_sig)
                if j > best_jaccard:
                    best_jaccard = j
                    if j >= dedup_threshold:
                        duplicate_of = prior_path
            if duplicate_of is not None:
                print(
                    f"sum: file={path} dedup_skipped "
                    f"jaccard={best_jaccard:.3f} duplicate_of={duplicate_of}",
                    file=sys.stderr,
                )
                deduplicated += 1
                continue

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

        # Record the signature only AFTER successful extraction so
        # files that failed earlier guards (empty input, conversion
        # error) don't poison the dedup set.
        if dedup_threshold is not None:
            accepted_signatures.append((path, sig))

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
            # Omni-format pivot metadata (per file): see cmd_attest
            # for the verifier-replay contract.
            **format_sidecar,
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

    if args.verbose or failed or deduplicated:
        msg = f"sum: attest-batch done — {succeeded} succeeded, {failed} failed"
        if deduplicated:
            msg += f", {deduplicated} dedup_skipped"
        print(msg, file=sys.stderr)
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


def _build_verify_explanation(
    *,
    bundle: dict,
    axioms: int,
    ed25519_status: str,
    hmac_status: str,
    extraction: dict,
) -> dict:
    """Produce a ``sum.verify_explained.v1`` layered report.

    Companion to the destination design in
    ``docs/ZENITH_FRAMING_2026-05-16.md`` §5. Each verification
    dimension carries a status, a human-readable detail string, and
    an epistemic-status tag per ``docs/PROOF_BOUNDARY.md`` §5
    (``provable`` / ``certified`` / ``empirical-benchmark`` /
    ``not-asserted``).

    v1 surfaces what the existing verifier already checks; the
    forward-compat levers (source-evidence-coverage,
    semantic-preservation) are present with ``not-measured`` status
    when not applicable, so consumers can branch on them today and
    the doc stays stable when the underlying signals arrive.
    """
    # Cryptographic integrity — provable layer.
    if ed25519_status == "verified":
        crypto_status = "pass"
        crypto_detail = "Ed25519 signature verifies against embedded public key"
    elif ed25519_status == "absent" and hmac_status == "verified":
        crypto_status = "pass"
        crypto_detail = "HMAC signature verifies against caller-supplied key"
    elif ed25519_status == "absent" and hmac_status == "absent":
        crypto_status = "absent"
        crypto_detail = "no signatures present — integrity not cryptographically attested"
    else:
        crypto_status = "advisory"
        crypto_detail = f"ed25519={ed25519_status}, hmac={hmac_status}"

    # Canonical reconstruction — provable layer.
    canonical_status = "pass"  # we wouldn't reach this code on mismatch
    canonical_detail = (
        f"reconstruct(parse(canonical_tome)) = state_integer "
        f"({axioms} axioms parsed, state matches)"
    )

    # Axiom consistency (z3) — provable when present.
    consistency_check = bundle.get("axiom_consistency_check") or {}
    if consistency_check.get("consistent") is True:
        consistency_status = "pass"
        consistency_detail = (
            f"z3 consistent, n_predicates_checked="
            f"{consistency_check.get('n_predicates_checked', 0)}"
        )
    elif consistency_check.get("consistent") is False:
        consistency_status = "advisory"
        consistency_detail = (
            f"unsat_core: {consistency_check.get('unsat_core', [])}"
        )
    else:
        consistency_status = "absent"
        consistency_detail = "no axiom_consistency_check in bundle"

    # Extraction provenance — certified layer.
    if extraction.get("verifiable") is True:
        ext_status = "verifiable"
        ext_detail = (
            f"extractor={extraction.get('extractor')!r} — deterministic "
            f"re-extraction reproducible from canonical_tome"
        )
    elif extraction.get("extractor"):
        ext_status = "advisory"
        ext_detail = (
            f"extractor={extraction.get('extractor')!r} — non-deterministic; "
            f"re-extraction not guaranteed to reproduce the axiom set"
        )
    else:
        ext_status = "absent"
        ext_detail = "no extraction provenance in bundle"

    # Source evidence coverage — v1: not yet populated by attest, but
    # the layer is present for forward-compat. Transform receipts
    # surface this via source_chain_hash; bundles will surface it once
    # the evidence-chain layer is wired into sum attest.
    source_evidence_status = "absent"
    source_evidence_detail = (
        "bundle carries no source_chain_hash; "
        "evidence chain not bound to source byte ranges"
    )

    # Semantic preservation — empirical-benchmark layer. Not measured
    # at verify-time; the slider bench (docs/SLIDER_CONTRACT.md)
    # measures it per-rendering. Present here so future tooling can
    # populate when a bench receipt is available.
    semantic_status = "not_measured"
    semantic_detail = (
        "verify does not measure semantic preservation; "
        "see docs/SLIDER_CONTRACT.md for the per-axis benchmark contract"
    )

    # Truth of content — explicitly not-asserted. The whole point of
    # the proof-boundary discipline.
    truth_status = "not_asserted"
    truth_detail = (
        "verify attests the issuer signed this bundle. It does NOT "
        "attest factual truth of the content. See "
        "docs/PROOF_BOUNDARY.md §5."
    )

    # Recommended action — synthesised from the above.
    if crypto_status == "pass" and ext_status == "verifiable":
        recommended = (
            "safe to reuse as attested deterministic transform; "
            "safe to cite the issuer + canonical reconstruction; "
            "NOT safe as factual authority without independent verification"
        )
    elif crypto_status == "pass":
        recommended = (
            "safe to reuse as attested transform with the recorded "
            "extractor; NOT safe to assume deterministic re-extraction; "
            "NOT safe as factual authority"
        )
    elif crypto_status == "absent":
        recommended = (
            "bundle has no cryptographic attestation; treat as "
            "advisory only; re-attest with a signing key before "
            "downstream consumers rely on it"
        )
    else:
        recommended = "review the per-check details above before reuse"

    # Known gaps — surface the things this verifier specifically does
    # NOT prove, so the consumer doesn't infer more than is true.
    known_gaps = [
        "Truth of content (not asserted; verify is signature + reconstruction)",
        "Source evidence coverage (no source_chain binding in this bundle)",
        "Semantic preservation (not measured at verify-time; see slider bench)",
    ]
    if ext_status != "verifiable":
        known_gaps.append(
            "Extraction non-determinism (re-running the extractor may differ)"
        )
    if crypto_status == "absent":
        known_gaps.append(
            "No cryptographic signature (bundle authorship not attested)"
        )

    return {
        "schema": "sum.verify_explained.v1",
        "verified": crypto_status in ("pass", "absent"),
        "checks": {
            "cryptographic_integrity": {
                "status": crypto_status,
                "detail": crypto_detail,
                "epistemic_status": "provable",
            },
            "canonical_reconstruction": {
                "status": canonical_status,
                "detail": canonical_detail,
                "epistemic_status": "provable",
            },
            "axiom_consistency": {
                "status": consistency_status,
                "detail": consistency_detail,
                "epistemic_status": "provable",
            },
            "extraction_provenance": {
                "status": ext_status,
                "detail": ext_detail,
                "epistemic_status": "certified",
            },
            "source_evidence_coverage": {
                "status": source_evidence_status,
                "detail": source_evidence_detail,
                "epistemic_status": "not-asserted",
            },
            "semantic_preservation": {
                "status": semantic_status,
                "detail": semantic_detail,
                "epistemic_status": "empirical-benchmark",
            },
            "truth_of_content": {
                "status": truth_status,
                "detail": truth_detail,
                "epistemic_status": "not-asserted",
            },
        },
        "known_gaps": known_gaps,
        "recommended_action": recommended,
    }


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

    # ── --explain: layered per-dimension report ─────────────────────────
    # Productizes the proof-boundary discipline as user-visible output.
    # Each check carries: status (pass/advisory/absent/...) + detail +
    # epistemic_status (provable / certified / empirical-benchmark /
    # not-asserted). See docs/ZENITH_FRAMING_2026-05-16.md §5 for the
    # destination design.
    if getattr(args, "explain", False):
        explained = _build_verify_explanation(
            bundle=bundle,
            axioms=axioms,
            ed25519_status=ed25519_status,
            hmac_status=hmac_status,
            extraction=extraction,
        )
        json.dump(explained, sys.stdout, indent=2 if args.pretty else None)
        sys.stdout.write("\n")
        from sum_cli.audit_log import emit_audit_event
        emit_audit_event("verify", {
            "ok": True,
            "axiom_count": axioms,
            "state_integer_digits": len(claimed_state_str),
            "branch": bundle.get("branch", "main"),
            "signatures": {"hmac": hmac_status, "ed25519": ed25519_status},
            "extraction": extraction,
            "explain": True,
        })
        return 0

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

    from sum_cli.audit_log import emit_audit_event
    emit_audit_event("verify", {
        "ok": True,
        "axiom_count": axioms,
        "state_integer_digits": len(claimed_state_str),
        "branch": bundle.get("branch", "main"),
        "signatures": {"hmac": hmac_status, "ed25519": ed25519_status},
        "extraction": extraction,
    })
    return 0


# ─── render ──────────────────────────────────────────────────────────
#
# The inverse of `sum attest`. Reads a CanonicalBundle, parses its
# triples back from the canonical tome, and re-emits a tome under
# explicit slider control. Closes the "tags ↔ tomes" symmetry from
# the shell — the engine and the Worker have always supported it,
# the CLI did not until now.
#
# Two paths:
#
#   local (default)
#     Only the density slider is honoured. Non-density axes default
#     to 0.5 (neutral); any deviation requires the LLM extrapolator
#     and is rejected with an actionable error pointing at
#     --use-worker. Output is the deterministic canonical tome with
#     the slider header recorded above the body, matching what
#     `generate_controlled` writes in-process.
#
#   --use-worker URL
#     POSTs {triples, slider_position} to <URL>/api/render and
#     returns the LLM-conditioned tome plus the signed render_receipt
#     (sum.render_receipt.v1). Uses urllib so no new runtime
#     dependency. Errors from the Worker (network, non-2xx, malformed
#     response) surface on stderr with a non-zero exit.
#
# Default stdout is the tome text only — symmetric with `sum attest`'s
# JSON-on-stdout default for its own native artefact. `--json` returns
# a structured envelope (`{tome, sliders, mode, render_receipt?}`) for
# callers that need the receipt. `--output` writes the tome text to a
# file regardless of `--json`.

_TOME_LINE_RE_RENDER = None  # lazy-compiled below in _parse_tome_triples


def _parse_tome_triples(canonical_tome: str) -> list[tuple[str, str, str]]:
    """Parse `The S P O.` lines from a canonical tome back into triples.

    Same regex `cmd_verify` uses for state reconstruction, kept private
    here so the render path stays independent of verify's import graph.
    Lines that don't match the canonical pattern are ignored — the tome
    may include header lines (`@canonical_version: ...`, `# Title`,
    `## Subject`) that are not axioms.
    """
    global _TOME_LINE_RE_RENDER
    if _TOME_LINE_RE_RENDER is None:
        import re
        _TOME_LINE_RE_RENDER = re.compile(r"^The (\S+) (\S+) (.+)\.$")
    triples: list[tuple[str, str, str]] = []
    for line in canonical_tome.splitlines():
        match = _TOME_LINE_RE_RENDER.match(line.strip())
        if match:
            triples.append((match.group(1), match.group(2), match.group(3)))
    return triples


def _post_render_to_worker(
    worker_url: str,
    triples: list[tuple[str, str, str]],
    sliders_dict: dict,
) -> dict:
    """POST to <worker_url>/api/render and return the parsed JSON.

    Pure-stdlib (urllib) so the CLI gains no new runtime dep. Raises
    RuntimeError with an actionable message on network / HTTP / JSON
    error; the caller surfaces it on stderr and returns non-zero.
    """
    import urllib.error
    import urllib.request

    base = worker_url.rstrip("/")
    endpoint = f"{base}/api/render"
    payload = json.dumps({
        "triples": [list(t) for t in triples],
        "slider_position": sliders_dict,
    }).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        method="POST",
        headers={
            "content-type": "application/json",
            # Cloudflare in front of hosted Workers (incl. the
            # default sum-demo.ototao.workers.dev) rejects the
            # default Python urllib User-Agent with HTTP 403 /
            # error 1010. Identify ourselves as a known scripted
            # client so the request passes the bot-detection layer.
            "user-agent": f"sum-cli/{__version__} (+https://github.com/OtotaO/SUM)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        # The Worker returns {"error": "..."} on documented failures.
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = ""
        raise RuntimeError(
            f"worker {endpoint} returned HTTP {e.code}: {err_body or e.reason}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"worker {endpoint} unreachable: {e.reason}") from e

    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"worker {endpoint} returned non-JSON body: {body[:200]!r} ({e})"
        ) from e


def cmd_render(args: argparse.Namespace) -> int:
    raw = _read_input(args.input)
    try:
        bundle = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"sum: bundle is not valid JSON: {e}", file=sys.stderr)
        return 2

    for field in ("canonical_tome", "canonical_format_version"):
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

    triples = _parse_tome_triples(bundle["canonical_tome"])
    if not triples:
        print(
            "sum: bundle's canonical_tome contains zero parseable axiom lines. "
            "Nothing to render.",
            file=sys.stderr,
        )
        return 3

    # Build TomeSliders early so out-of-range values fail fast with a
    # consistent error message before we touch the algebra or network.
    from sum_engine_internal.ensemble.tome_sliders import TomeSliders

    try:
        sliders = TomeSliders(
            density=args.density,
            length=args.length,
            formality=args.formality,
            audience=args.audience,
            perspective=args.perspective,
        )
    except ValueError as e:
        print(f"sum: invalid slider value: {e}", file=sys.stderr)
        return 2

    sliders_dict = {
        "density": sliders.density,
        "length": sliders.length,
        "formality": sliders.formality,
        "audience": sliders.audience,
        "perspective": sliders.perspective,
    }

    # ── Worker path ──────────────────────────────────────────────────
    if args.use_worker:
        try:
            response = _post_render_to_worker(args.use_worker, triples, sliders_dict)
        except RuntimeError as e:
            print(f"sum: {e}", file=sys.stderr)
            return 1

        tome_text = response.get("tome", "")
        if not isinstance(tome_text, str) or not tome_text:
            print(
                "sum: worker response missing or empty 'tome' field",
                file=sys.stderr,
            )
            return 1

        envelope = {
            "tome": tome_text,
            "sliders": sliders_dict,
            "mode": "worker",
            "worker_url": args.use_worker,
        }
        # Surface the worker's render-time metadata so the JSON envelope
        # is self-describing for downstream consumers (cache_status,
        # drift, render_id, the signed receipt). All optional.
        for key in (
            "render_receipt",
            "render_id",
            "drift",
            "cache_status",
            "llm_calls_made",
            "wall_clock_ms",
            "quantized_sliders",
            "triples_used",
        ):
            if key in response:
                envelope[key] = response[key]

        return _emit_render_output(envelope, args)

    # ── Local deterministic path ─────────────────────────────────────
    if sliders.requires_extrapolator():
        non_neutral = [
            f"{name}={getattr(sliders, name)}"
            for name in ("length", "formality", "audience", "perspective")
            if abs(getattr(sliders, name) - 0.5) > 1e-9
        ]
        print(
            "sum: non-neutral LLM-conditioned axes ("
            + ", ".join(non_neutral)
            + ") require an LLM extrapolator. The local CLI render path "
            "actions only the density slider. Either:\n"
            "  • drop the affected sliders to 0.5 (neutral) for a "
            "deterministic local render, or\n"
            "  • pass --use-worker https://sum.ototao.com to render via "
            "the hosted Worker (returns a signed render_receipt).",
            file=sys.stderr,
        )
        return 2

    from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
    from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator

    algebra = GodelStateAlgebra()  # type: ignore[no-untyped-call]
    import math

    state = 1
    for s, p, o in triples:
        prime = algebra.get_or_mint_prime(s, p, o)
        state = math.lcm(state, prime)

    tome_generator = AutoregressiveTomeGenerator(algebra)
    tome_text = tome_generator.generate_controlled(
        state, sliders=sliders, title=args.title,
    )

    envelope = {
        "tome": tome_text,
        "sliders": sliders_dict,
        "mode": "local-deterministic",
        "axiom_count_input": len(triples),
        "title": args.title,
    }
    return _emit_render_output(envelope, args)


def _emit_render_output(envelope: dict, args: argparse.Namespace) -> int:
    """Write the rendered tome to --output (or stdout) and emit either
    plain text or the JSON envelope on stdout depending on --json.

    Both flags compose: `--output tome.md --json` writes the tome text
    to tome.md AND prints the JSON envelope to stdout, so a caller can
    keep the receipt while persisting the prose. Without --json the
    JSON envelope is discarded after extraction.
    """
    tome_text = envelope["tome"]

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(tome_text)
                if not tome_text.endswith("\n"):
                    f.write("\n")
        except OSError as e:
            print(f"sum: cannot write --output {args.output!r}: {e}", file=sys.stderr)
            return 1

    if args.json:
        json.dump(envelope, sys.stdout, indent=2 if args.pretty else None)
        sys.stdout.write("\n")
    elif not args.output:
        # Default path: tome text on stdout, no metadata. Matches the
        # `sum render < bundle.json > tome.md` pipeline shape.
        sys.stdout.write(tome_text)
        if not tome_text.endswith("\n"):
            sys.stdout.write("\n")

    if args.verbose:
        kind = envelope.get("mode", "unknown")
        msg = f"sum: rendered {len(tome_text)} chars via mode={kind}"
        if "render_receipt" in envelope:
            kid = envelope["render_receipt"].get("kid", "?")
            msg += f", receipt kid={kid}"
        print(msg, file=sys.stderr)

    from sum_cli.audit_log import emit_audit_event
    audit_payload: dict = {
        "mode": envelope.get("mode"),
        "axiom_count_input": envelope.get("axiom_count_input"),
        "tome_chars": len(tome_text),
        "sliders": envelope.get("sliders"),
    }
    if "render_receipt" in envelope:
        receipt = envelope["render_receipt"]
        audit_payload["render_receipt_kid"] = receipt.get("kid")
        audit_payload["render_receipt_schema"] = receipt.get("schema")
        audit_payload["worker_url"] = envelope.get("worker_url")
    emit_audit_event("render", audit_payload)
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


# ─── compliance ──────────────────────────────────────────────────────
#
# Per-regime validators consuming sum.audit_log.v1. The audit log is
# regime-agnostic substrate; ``sum compliance check`` is the actionable
# layer that turns it into a pass/fail verdict for a specific regime.
# Exit code 0 when ok=true, 1 otherwise — pipe-friendly for CI gates.

_COMPLIANCE_REGIMES: dict[str, str] = {
    "eu-ai-act-article-12": (
        "EU AI Act (Regulation (EU) 2024/1689) Article 12 — record-"
        "keeping for high-risk AI systems. Pins per-row traceability "
        "fields (timestamp, operation, cli_version), schema, and "
        "operation-specific anchors (source_uri / axiom_count / "
        "state_integer_digits / mode)."
    ),
    "gdpr-article-30": (
        "GDPR (Regulation (EU) 2016/679) Article 30 — Records of "
        "Processing Activities. Pins the per-row floor enabling Art "
        "30 reporting (schema, timestamp, ISO-8601-UTC parseability, "
        "processing-category indicator, processor identity). The "
        "controller separately maintains record-set-scope metadata "
        "(Art 30(1)(a)–(g) controller name, purposes, categories, "
        "recipients, transfers, retention, security measures) "
        "out-of-band; this validator does not pin those."
    ),
    "hipaa-164-312-b": (
        "HIPAA Security Rule 45 CFR § 164.312(b) — Audit Controls "
        "(Technical Safeguards). Pins the per-row form requirements "
        "for an audit recording that supports examination of "
        "activity (schema, timestamp, ISO-8601-UTC, activity type, "
        "system component identification, per-operation examination "
        "anchors). Deployment-scope obligations — auditor function, "
        "retention, ePHI inventory — live outside this validator."
    ),
    "iso-27001-8-15": (
        "ISO/IEC 27001:2022 Annex A.8.15 — Logging. Pins the per-"
        "row form floor an audit log must satisfy for the recording "
        "to count as a 'produced' log under A.8.15 (schema, "
        "timestamp, ISO-8601-UTC, activity, system component). The "
        "'stored', 'protected', 'analysed' verbs map to deployment-"
        "scope obligations (file-system policy, access control, "
        "SIEM integration) outside this validator."
    ),
    "soc-2-cc-7-2": (
        "SOC 2 Trust Services Criteria CC7.2 — System Operations. "
        "Pins the per-row form floor required to enable the "
        "monitoring criterion (schema, timestamp, ISO-8601-UTC, "
        "activity classification, system component identification). "
        "The detection / monitoring / analysis activities themselves "
        "(SIEM rules, alert routing, oncall rotations) live at "
        "deployment scope outside this validator."
    ),
    "pci-dss-4-req-10": (
        "PCI DSS v4.0 Requirement 10 — Log and Monitor All Access "
        "to System Components and Cardholder Data. Pins per-row "
        "content visible in audit_log.v1 against Req 10.2.2 (event "
        "content) plus 10.6 (consistent time): schema, timestamp, "
        "ISO-8601-UTC, event type, origination, event-content "
        "completeness. NOT pinned: 10.1 organisational policies; "
        "10.2.1.* specific event-type coverage; 10.2.2 user "
        "identification (audit_log.v1 has no user_id field); 10.3 "
        "log file protection; 10.4 log review process; 10.5 12-"
        "month retention; 10.7 failure detection / alerting."
    ),
}


def _compliance_validators():
    """Return the regime → validate-callable dispatch table.

    Built lazily to avoid importing compliance modules at CLI
    startup (each module imports the dataclass infrastructure;
    deferring keeps `sum --help` cold-start fast). Registered
    regimes must match :data:`_COMPLIANCE_REGIMES` keys exactly —
    a mismatch surfaces as a KeyError at dispatch, intentional
    (better than silent fallthrough)."""
    from sum_engine_internal.compliance import (  # local import — see docstring
        eu_ai_act_article_12,
        gdpr_article_30,
        hipaa_164_312_b,
        iso_27001_8_15,
        pci_dss_4_req_10,
        soc_2_cc_7_2,
    )
    return {
        "eu-ai-act-article-12": eu_ai_act_article_12.validate,
        "gdpr-article-30": gdpr_article_30.validate,
        "hipaa-164-312-b": hipaa_164_312_b.validate,
        "iso-27001-8-15": iso_27001_8_15.validate,
        "soc-2-cc-7-2": soc_2_cc_7_2.validate,
        "pci-dss-4-req-10": pci_dss_4_req_10.validate,
    }


def cmd_compliance_check(args: argparse.Namespace) -> int:
    if args.regime not in _COMPLIANCE_REGIMES:
        print(
            f"sum: unknown regime {args.regime!r}. Known: "
            f"{sorted(_COMPLIANCE_REGIMES)}",
            file=sys.stderr,
        )
        return 2

    if args.audit_log == "-":
        text = sys.stdin.read()
    else:
        try:
            with open(args.audit_log, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as e:
            print(f"sum: cannot read --audit-log {args.audit_log!r}: {e}", file=sys.stderr)
            return 2

    rows: list[dict] = []
    parse_errors: list[tuple[int, str]] = []
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            parse_errors.append((i, str(e)))

    validators = _compliance_validators()
    try:
        validate = validators[args.regime]
    except KeyError:
        # Defensive — _COMPLIANCE_REGIMES gate above should have caught this.
        # Reaching here means a regime is registered in _COMPLIANCE_REGIMES
        # but missing from _compliance_validators() — a wiring drift.
        print(
            f"sum: regime {args.regime!r} listed but not wired "
            f"(internal: missing from _compliance_validators dispatch)",
            file=sys.stderr,
        )
        return 2
    report = validate(rows)

    out = report.to_dict()
    if parse_errors:
        # Surface parse errors alongside rule violations — both are
        # compliance-relevant. A malformed JSONL line is itself a
        # traceability defect.
        out["parse_errors"] = [
            {"line_index": idx, "error": msg} for idx, msg in parse_errors
        ]

    json.dump(out, sys.stdout, indent=2 if args.pretty else None)
    sys.stdout.write("\n")
    return 0 if (report.ok and not parse_errors) else 1


# ─── transform — generic transform-registry dispatch ────────────────
#
# Closes the audit's "no CLI integration" gap. Every registered
# transform in sum_engine_internal.transforms becomes shell-invokable
# via `sum transform <verb>`. Output is JSON on stdout; receipts are
# embedded in the JSON when signing keys are configured via env
# vars (SUM_TRANSFORM_SIGNING_JWK, SUM_TRANSFORM_SIGNING_KID).


def cmd_transform_list(args: argparse.Namespace) -> int:
    """List registered transforms. Emits sum.transform_registry.v1."""
    from sum_engine_internal.transforms import get_transform, list_transforms

    out = {
        "schema": "sum.transform_registry.v1",
        "transforms": [
            {
                "id": name,
                "requires_llm": get_transform(name).requires_llm,
                "digital_source_type": get_transform(name).digital_source_type,
            }
            for name in list_transforms()
        ],
    }
    json.dump(out, sys.stdout, indent=2 if getattr(args, "pretty", False) else None)
    sys.stdout.write("\n")
    return 0


def cmd_transform_apply(args: argparse.Namespace) -> int:
    """Run a transform with input from stdin or --input <file>. Emits
    a JSON envelope with the output + optional signed receipt."""
    import asyncio

    from sum_engine_internal.transforms import (
        TransformEnv,
        get_transform,
        list_transforms,
    )
    from sum_engine_internal.transform_receipt import (
        build_payload,
        canonical_hash,
        compute_source_chain_hash,
        sign_transform_receipt,
    )

    try:
        transform = get_transform(args.transform_name)
    except KeyError:
        print(
            f"sum: unknown transform {args.transform_name!r}; "
            f"known: {list_transforms()}",
            file=sys.stderr,
        )
        return 2

    # Read input + parameters from stdin / files.
    if args.input == "-":
        raw_input = json.load(sys.stdin)
    else:
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                raw_input = json.load(f)
        except OSError as e:
            print(f"sum: cannot read --input {args.input!r}: {e}", file=sys.stderr)
            return 2
        except json.JSONDecodeError as e:
            print(f"sum: --input is not valid JSON: {e}", file=sys.stderr)
            return 2

    parameters: dict = {}
    if args.parameters:
        try:
            parameters = json.loads(args.parameters)
        except json.JSONDecodeError as e:
            print(f"sum: --parameters is not valid JSON: {e}", file=sys.stderr)
            return 2
        if not isinstance(parameters, dict):
            print("sum: --parameters must be a JSON object", file=sys.stderr)
            return 2

    # T4 wiring: parse + validate --source-chain upfront so usage
    # errors (missing file, non-list) surface as rc=2 (usage error)
    # rather than rc=1 (transform error). Hash computation happens
    # after transform.apply runs.
    source_chain_data = None
    if getattr(args, "source_chain", None):
        try:
            with open(args.source_chain, "r", encoding="utf-8") as f:
                source_chain_data = json.load(f)
        except OSError as e:
            print(f"sum: cannot read --source-chain {args.source_chain!r}: {e}", file=sys.stderr)
            return 2
        except json.JSONDecodeError as e:
            print(f"sum: --source-chain is not valid JSON: {e}", file=sys.stderr)
            return 2
        if not isinstance(source_chain_data, list):
            print(
                "sum: --source-chain must be a JSON array of EvidenceLink "
                "records (each {claim, provenance: {source_uri, byte_start, "
                "byte_end}})",
                file=sys.stderr,
            )
            return 2

    # Build the TransformEnv from env vars (LLM keys + signing JWK +
    # routed-model selector). SUM_TRANSFORM_MODEL is the BYO-keys / free-
    # provider escape valve — set it to e.g.
    # 'meta-llama/Llama-3.3-70B-Instruct' (+ HF_TOKEN) to route through
    # Hugging Face Inference Providers, or 'ollama:llama3.1' for a free
    # local Ollama install. Full matrix: docs/BYOK_AND_FREE_PROVIDERS.md.
    env = TransformEnv(
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        cf_ai_gateway_base=os.environ.get("CF_AI_GATEWAY_BASE"),
        model=os.environ.get("SUM_TRANSFORM_MODEL"),
    )
    signing_jwk_raw = os.environ.get("SUM_TRANSFORM_SIGNING_JWK")
    signing_kid = os.environ.get("SUM_TRANSFORM_SIGNING_KID")
    if signing_jwk_raw and signing_kid:
        try:
            env.private_jwk = json.loads(signing_jwk_raw)
            env.kid = signing_kid
        except json.JSONDecodeError:
            print(
                "sum: SUM_TRANSFORM_SIGNING_JWK is not valid JSON; "
                "receipt will be omitted.",
                file=sys.stderr,
            )

    try:
        result = asyncio.run(transform.apply(raw_input, parameters, env))
    except Exception as e:  # noqa: BLE001 — surface as a clean CLI error
        print(f"sum: transform {args.transform_name!r} failed: {e}", file=sys.stderr)
        return 1

    # Compute hashes for the receipt payload.
    parameters_hash = canonical_hash(transform.canonicalize_parameters(parameters))
    input_hash = canonical_hash(transform.canonicalize_input(raw_input))
    output_hash = canonical_hash(transform.canonicalize_output(result.output))

    # T4: compute source_chain_hash now that source_chain_data was
    # parsed + validated upfront (above, before transform.apply).
    source_chain_hash = None
    if source_chain_data is not None:
        source_chain_hash = compute_source_chain_hash(source_chain_data)

    envelope = {
        "output": result.output,
        "model": result.model,
        "provider": result.provider,
        "digital_source_type": result.digital_source_type,
        "llm_calls_made": result.llm_calls_made,
        "extra": result.extra,
    }
    if env.private_jwk and env.kid:
        payload = build_payload(
            transform=args.transform_name,
            parameters_hash=parameters_hash,
            input_hash=input_hash,
            output_hash=output_hash,
            model=result.model,
            provider=result.provider,
            digital_source_type=result.digital_source_type,
            source_chain_hash=source_chain_hash,
        )
        envelope["transform_receipt"] = sign_transform_receipt(
            payload, private_jwk=env.private_jwk, kid=env.kid,
        )
    else:
        # No signing config — still surface the hashes so callers can
        # cross-check application-layer integrity without a receipt.
        envelope["parameters_hash"] = parameters_hash
        envelope["input_hash"] = input_hash
        envelope["output_hash"] = output_hash
        if source_chain_hash:
            envelope["source_chain_hash"] = source_chain_hash

    json.dump(envelope, sys.stdout, indent=2 if args.pretty else None)
    sys.stdout.write("\n")
    return 0


def cmd_compliance_regimes(args: argparse.Namespace) -> int:
    out = {
        "schema": "sum.compliance_regimes.v1",
        "regimes": [
            {"id": rid, "description": desc}
            for rid, desc in sorted(_COMPLIANCE_REGIMES.items())
        ],
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _read_text_arg(path: str) -> Optional[str]:
    """Read a text/code file ('-' for stdin) as UTF-8. Prints a usage
    error and returns None on failure (caller returns rc=2)."""
    if path == "-":
        return sys.stdin.read()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        print(f"sum: cannot read {path!r}: {e}", file=sys.stderr)
        return None


def cmd_frontier(args: argparse.Namespace) -> int:
    """Build a render frontier over a source and one or more rendered
    versions, score each, and cycle through them.

    The versions (``--version`` files, text or code) are given
    most-faithful first — that ordering is the compression control. Each
    version's meaning-loss is *measured* against the source by a named
    proxy (lexical, offline). With ``--scrub T`` (T in [0,1]) the command
    prints only the single version at that position on the
    faithful→compressed path — the cycler. Without it, it emits a
    ``sum.render_frontier.v1`` JSON overview with the measured numbers.

    Honest boundary: the per-version ``meaning_loss`` is a per-document
    *measurement*, not a guarantee; the marginal distribution-free
    guarantee is a separate ``sum.meaning_risk_receipt.v1`` over a named
    corpus. The emitted JSON carries that note.
    """
    try:
        from sum_engine_internal.research.frontier import RenderFrontier
        from sum_engine_internal.research.meaning.meaning_loss import (
            LexicalCoverageScorer,
        )
    except ImportError as e:
        print(
            f"sum: `sum frontier` needs the [research] extra "
            f"(pip install 'sum-engine[research]'): {e}",
            file=sys.stderr,
        )
        return 2

    if args.scorer != "lexical":
        print(
            f"sum: scorer {args.scorer!r} is not available offline; only "
            f"'lexical' ships without a judge (entailment scoring needs an "
            f"NLI judge — operator-gated). See docs/PRODUCT_VISION.md.",
            file=sys.stderr,
        )
        return 2
    scorer = LexicalCoverageScorer()

    source = _read_text_arg(args.source)
    if source is None:
        return 2
    if not args.version:
        print("sum: at least one --version is required", file=sys.stderr)
        return 2

    renderings: list[tuple[str, dict, str]] = []
    for path in args.version:
        text = _read_text_arg(path)
        if text is None:
            return 2
        label = os.path.basename(path) or path
        renderings.append((label, {"file": path}, text))

    try:
        frontier = RenderFrontier.from_renderings(source, renderings, scorer)
    except ValueError as e:
        print(f"sum: {e}", file=sys.stderr)
        return 2

    # --scrub T: print just the version at position T (the cycler).
    if args.scrub is not None:
        point = frontier.scrub(args.scrub)
        sys.stdout.write(point.rendering)
        if not point.rendering.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    out = {"schema": "sum.render_frontier.v1", **frontier.as_dict()}
    json.dump(out, sys.stdout, indent=2 if getattr(args, "pretty", False) else None)
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
            "  sum render < bundle.json > tome.md        # inverse: bundle → prose\n"
            "  sum render --density 0.5 < bundle.json    # half the axioms, lex-prefix\n"
            "  sum render --length 0.9 --use-worker https://sum.ototao.com < bundle.json --json\n"
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
    p_attest.add_argument(
        "--format", choices=["auto", "raw"], default="auto",
        help=(
            "Input format routing. ``auto`` (default) uses the file "
            "extension to route through the omni-format pivot: PDF, "
            "HTML, DOCX, EPUB, .ipynb, .json are converted to markdown "
            "via markitdown (requires `pip install "
            "'sum-engine[omni-format]'`); plaintext / .md / .txt / "
            "stdin pass through unchanged. ``raw`` reads bytes as "
            "UTF-8 with no conversion (the original behaviour). "
            "Source URI is anchored to the original input bytes "
            "either way — a receipt for a PDF binds to the PDF, not "
            "to its markdown intermediate."
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
    p_attest_batch.add_argument(
        "--format", choices=["auto", "raw"], default="auto",
        help=(
            "Input format routing. Same semantics as `sum attest --format`. "
            "Default `auto` routes by extension: PDF/HTML/DOCX/EPUB/etc. "
            "go through markitdown; plaintext/markdown pass through. "
            "`raw` disables conversion."
        ),
    )
    p_attest_batch.add_argument(
        "--dedup-threshold", type=float, default=None, metavar="J",
        help=(
            "Skip near-duplicate inputs whose Jaccard similarity to "
            "an earlier-accepted file equals or exceeds J (range "
            "(0.0, 1.0]). Each file is sketched with a 128-permutation "
            "MinHash over word 3-shingles; skipped files are reported "
            "on stderr in `sum: file=<path> dedup_skipped jaccard=<j> "
            "duplicate_of=<earlier_path>` form. Default: disabled. "
            "Recommended: 0.85 (catches near-dups without mistaking "
            "two docs about the same topic for the same doc)."
        ),
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
    p_verify.add_argument(
        "--explain", action="store_true",
        help=(
            "Emit a layered per-dimension verification report "
            "(sum.verify_explained.v1) instead of the default summary. "
            "Productizes the proof-boundary discipline: each verification "
            "dimension (cryptographic integrity / canonical reconstruction / "
            "extraction provenance / source coverage / semantic preservation / "
            "truth of content) carries its own status + detail + epistemic-"
            "status tag. See docs/ZENITH_FRAMING_2026-05-16.md §5 for the "
            "destination design of this surface."
        ),
    )
    p_verify.set_defaults(func=cmd_verify)

    # render — bundle → tome under explicit slider control. Inverse of attest.
    p_render = subparsers.add_parser(
        "render",
        help="Render a bundle's axioms back into prose under explicit slider control.",
        description=(
            "Reads a CanonicalBundle from stdin (or --input), parses its "
            "axioms back from the canonical tome, and re-emits a tome "
            "under the supplied 5-axis slider position (density / length "
            "/ formality / audience / perspective). The local path is "
            "deterministic and actions only the density slider — "
            "non-neutral length / formality / audience / perspective "
            "values are LLM-gated and require --use-worker URL to "
            "produce. With --use-worker, posts to <URL>/api/render and "
            "returns the LLM-conditioned tome plus the signed "
            "render_receipt (sum.render_receipt.v1). "
            "Default stdout: the tome text. With --json: a structured "
            "envelope including the receipt when present. --output PATH "
            "writes the tome text to a file."
        ),
    )
    p_render.add_argument("--input", "-i", help="Read bundle from this path instead of stdin ('-' for stdin).")
    p_render.add_argument(
        "--density", type=float, default=1.0, metavar="F",
        help="Axiom-coverage slider in [0, 1]. 1.0 keeps all axioms (default); 0.0 keeps none.",
    )
    p_render.add_argument(
        "--length", type=float, default=0.5, metavar="F",
        help="Length slider in [0, 1]. 0.5 = neutral (default). Non-0.5 requires --use-worker.",
    )
    p_render.add_argument(
        "--formality", type=float, default=0.5, metavar="F",
        help="Formality slider in [0, 1]. 0.5 = neutral (default). Non-0.5 requires --use-worker.",
    )
    p_render.add_argument(
        "--audience", type=float, default=0.5, metavar="F",
        help="Audience slider in [0, 1]. 0.5 = neutral (default). Non-0.5 requires --use-worker.",
    )
    p_render.add_argument(
        "--perspective", type=float, default=0.5, metavar="F",
        help="Perspective slider in [0, 1]. 0.5 = neutral (default). Non-0.5 requires --use-worker.",
    )
    p_render.add_argument("--title", default="Rendered Tome", help="Tome title. Default: 'Rendered Tome'.")
    p_render.add_argument(
        "--output", "-o", metavar="PATH", default=None,
        help="Write tome text to this path. Without it, tome text goes to stdout.",
    )
    p_render.add_argument(
        "--use-worker", metavar="URL", default=None,
        help=(
            "POST triples + slider position to <URL>/api/render and "
            "return the LLM-conditioned tome + signed render_receipt. "
            "Required when any non-density slider is non-neutral. "
            "Hosted Worker: https://sum.ototao.com."
        ),
    )
    p_render.add_argument(
        "--json", action="store_true",
        help="Emit a JSON envelope on stdout (tome, sliders, mode, render_receipt if from worker).",
    )
    p_render.add_argument("--pretty", action="store_true", help="Pretty-print --json output.")
    p_render.add_argument("--verbose", "-v", action="store_true", help="Emit diagnostics on stderr.")
    p_render.set_defaults(func=cmd_render)

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

    # compliance — regime validators consuming sum.audit_log.v1.
    p_compliance = subparsers.add_parser(
        "compliance",
        help="Validate a sum.audit_log.v1 stream against a compliance regime.",
        description=(
            "Apply a per-regime validator to a sum.audit_log.v1 JSONL stream "
            "and emit a sum.compliance_report.v1 verdict. The audit log is "
            "regime-agnostic substrate; this verb is the actionable layer "
            "that turns it into a compliance-grade pass/fail."
        ),
    )
    p_compliance_sub = p_compliance.add_subparsers(
        dest="compliance_cmd", required=True, metavar="<compliance-cmd>",
    )

    p_comp_check = p_compliance_sub.add_parser(
        "check",
        help="Validate an audit-log stream against a regime; emit a JSON report.",
        description=(
            "Read a sum.audit_log.v1 JSONL stream from --audit-log (or stdin "
            "if '-'), validate against the named regime, emit a "
            "sum.compliance_report.v1 JSON object on stdout. Exit code is "
            "0 when ok=true, 1 otherwise — pipe-friendly for CI gates."
        ),
    )
    p_comp_check.add_argument(
        "--regime",
        required=True,
        choices=sorted(_COMPLIANCE_REGIMES.keys()),
        help=(
            "Compliance regime to validate against. Six regimes are "
            "wired: EU AI Act Article 12, GDPR Article 30, HIPAA "
            "164.312(b), ISO 27001 Annex A.8.15, SOC 2 CC 7.2, PCI "
            "DSS 4.0 Requirement 10. Use `sum compliance regimes` "
            "to list the canonical regime identifiers."
        ),
    )
    p_comp_check.add_argument(
        "--audit-log",
        required=True,
        help="Path to a sum.audit_log.v1 JSONL file ('-' for stdin).",
    )
    p_comp_check.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the report JSON.",
    )
    p_comp_check.set_defaults(func=cmd_compliance_check)

    p_comp_regimes = p_compliance_sub.add_parser(
        "regimes",
        help="List available compliance regimes.",
        description=(
            "Emit the set of regime identifiers this CLI can validate "
            "against. Adding a new regime appends to this list; "
            "existing identifiers are stable."
        ),
    )
    p_comp_regimes.set_defaults(func=cmd_compliance_regimes)

    # transform — generic transform-registry dispatch (T1c).
    p_transform = subparsers.add_parser(
        "transform",
        help="Run a registered transform (slider / extract / compose / …).",
        description=(
            "Generic dispatch surface for the transform registry. Every "
            "transform in sum_engine_internal.transforms is callable via "
            "`sum transform apply <name>`. Output is JSON on stdout; if "
            "SUM_TRANSFORM_SIGNING_JWK + SUM_TRANSFORM_SIGNING_KID are "
            "set, the response includes a signed sum.transform_receipt.v1 "
            "envelope. See docs/TRANSFORM_REGISTRY.md."
        ),
    )
    p_transform_sub = p_transform.add_subparsers(
        dest="transform_cmd", required=True, metavar="<transform-cmd>",
    )

    p_xf_list = p_transform_sub.add_parser(
        "list",
        help="List registered transforms.",
        description=(
            "Emit a sum.transform_registry.v1 JSON document listing every "
            "registered transform with its id, requires_llm flag, and "
            "digital_source_type."
        ),
    )
    p_xf_list.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print the JSON.",
    )
    p_xf_list.set_defaults(func=cmd_transform_list)

    p_xf_apply = p_transform_sub.add_parser(
        "apply",
        help="Apply a transform to input from stdin / a file.",
        description=(
            "Run the named transform with input from --input <file> "
            "(or '-' for stdin) and --parameters <json>. Emits a JSON "
            "envelope with the transform's output, the model + provider "
            "that served it, and either a signed transform_receipt (if "
            "signing keys are configured via SUM_TRANSFORM_SIGNING_JWK + "
            "SUM_TRANSFORM_SIGNING_KID env vars) or the bare hash fields "
            "for application-layer integrity checks. Exit 0 on success, "
            "1 on transform error, 2 on usage error."
        ),
    )
    p_xf_apply.add_argument(
        "transform_name",
        help="Transform id — one of: slider / extract / compose. Use "
             "`sum transform list` for the full set.",
    )
    p_xf_apply.add_argument(
        "--input", required=True,
        help="Path to a JSON file with the transform's input shape, "
             "or '-' to read from stdin.",
    )
    p_xf_apply.add_argument(
        "--parameters", default="{}",
        help="JSON object with the transform's parameters (default '{}').",
    )
    p_xf_apply.add_argument(
        "--source-chain", default=None, metavar="PATH",
        help=(
            "Optional path to a JSON file containing a list of "
            "EvidenceLink-shaped records: "
            "[{\"claim\": \"s||p||o\", "
            "\"provenance\": {\"source_uri\": \"...\", "
            "\"byte_start\": N, \"byte_end\": M}}, ...]. "
            "When supplied, the canonical hash of this chain is "
            "bound to the receipt's `source_chain_hash` field, "
            "binding the receipt to specific byte ranges of source "
            "documents. See docs/TRANSFORM_RECEIPT_FORMAT.md §1.1."
        ),
    )
    p_xf_apply.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print the JSON envelope.",
    )
    p_xf_apply.set_defaults(func=cmd_transform_apply)

    # frontier — cycle through versions of a text/code, each scored.
    p_frontier = subparsers.add_parser(
        "frontier",
        help="Build + cycle a render frontier over a source and its versions.",
        description=(
            "Score one or more rendered versions of a source (text or "
            "code) against it and cycle through them. Versions are given "
            "most-faithful first via repeated --version; that order is "
            "the compression control. With --scrub T (0..1) the command "
            "prints just the version at that position on the "
            "faithful→compressed path; otherwise it emits a "
            "sum.render_frontier.v1 JSON overview with each version's "
            "MEASURED meaning-loss (a per-document measurement under a "
            "named proxy — not a guarantee; the marginal distribution-"
            "free guarantee is a separate sum.meaning_risk_receipt.v1). "
            "Needs the [research] extra. See docs/PRODUCT_VISION.md."
        ),
    )
    p_frontier.add_argument(
        "--source", required=True,
        help="Path to the source text/code file ('-' for stdin).",
    )
    p_frontier.add_argument(
        "--version", action="append", metavar="PATH", default=[],
        help="A rendered version file. Repeat, most-faithful first.",
    )
    p_frontier.add_argument(
        "--scrub", type=float, default=None, metavar="T",
        help="Position 0..1 on the faithful→compressed path; print just "
             "that version's text (the cycler). Omit for the JSON overview.",
    )
    p_frontier.add_argument(
        "--scorer", default="lexical", choices=["lexical"],
        help="Meaning-loss proxy (default 'lexical', offline). Entailment "
             "scoring needs an NLI judge and is operator-gated.",
    )
    p_frontier.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print the JSON overview.",
    )
    p_frontier.set_defaults(func=cmd_frontier)

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
