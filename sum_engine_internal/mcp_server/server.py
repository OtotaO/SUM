"""SUM MCP server — FastMCP tool registration.

Five tools, all stateless, all thin façades over the existing
``sum_engine_internal`` + ``sum_cli.main`` surfaces. Producing a
bundle here yields the same canonical bytes the CLI produces, so
cross-runtime verifiers (Node ``standalone_verifier``, browser
``single_file_demo``) accept MCP-attested bundles unchanged.

Why FastMCP and not the lower-level ``mcp.server.Server``: the
SUM tool surface is a small, stable set of named verbs with
typed inputs — exactly the shape FastMCP's decorator API was
designed for. We do not need streaming, sampling, or
long-running tasks for v1; if those become relevant for ledger
walks or long extraction runs, a v2 can drop down to the lower
API without breaking tool callers.

Error policy: tools return result dicts on success and dicts
with an ``error`` key on user-side failures (empty input, bundle
schema rejection, missing extractor). Programming errors
propagate as exceptions and FastMCP turns them into MCP error
responses — losing the difference between a bug and a
predictable rejection is exactly what the receipts trust
contract forbids.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from mcp.server.fastmcp import FastMCP


_SUPPORTED_CANONICAL_FORMAT = "1.0.0"
_SUPPORTED_PRIME_SCHEME = "sha256_64_v1"
_TOME_LINE_PATTERN = re.compile(r"^The (\S+) (\S+) (.+)\.$")


def build_server() -> FastMCP:
    """Build and register the SUM MCP server.

    Returns the configured ``FastMCP`` instance. Caller is
    responsible for running it (the ``main()`` below uses the
    stdio transport, which is the standard MCP wire for local
    LLM clients).
    """
    mcp = FastMCP(
        name="sum",
        instructions=(
            "SUM verifiable knowledge distillation engine. Exposes "
            "extract / attest / verify / inspect / schema. The "
            "extract path is fast and offline (sieve extractor). The "
            "attest path produces a signed CanonicalBundle that any "
            "Python/Node/browser SUM verifier accepts. The verify "
            "path returns a structured result dict, not an exit code."
        ),
    )

    @mcp.tool()
    def extract(text: str, extractor: str = "auto") -> dict:
        """Extract (subject, predicate, object) triples from prose.

        Args:
            text: Natural-language input. Empty input returns
                an error result, not an exception.
            extractor: ``"auto"`` (default), ``"sieve"`` (offline,
                deterministic, requires spaCy installed), or
                ``"llm"`` (OpenAI structured-output, requires
                ``OPENAI_API_KEY`` env). ``"auto"`` picks sieve
                if spaCy is importable, otherwise llm if the env
                var is set, otherwise returns an error.

        Returns:
            ``{"triples": [[s, p, o], ...], "extractor": <chosen>,
            "count": <int>}`` on success.
            ``{"error": <message>}`` on user-side failure.
        """
        text = (text or "").strip()
        if not text:
            return {"error": "empty input"}

        from sum_cli.main import _extract, _pick_extractor
        try:
            chosen = _pick_extractor(None if extractor == "auto" else extractor)
        except SystemExit as e:
            return {"error": str(e)}

        try:
            triples = _extract(text, chosen, model=None)
        except RuntimeError as e:
            return {"error": str(e)}

        return {
            "triples": [list(t) for t in triples],
            "extractor": chosen,
            "count": len(triples),
        }

    @mcp.tool()
    def attest(
        text: str,
        extractor: str = "auto",
        branch: str = "main",
        title: str | None = None,
        signing_key: str | None = None,
    ) -> dict:
        """Extract triples and produce a signed CanonicalBundle.

        Mirrors ``sum attest`` in the CLI. The bundle this tool
        returns verifies byte-identically via ``sum verify`` and
        the Node / browser verifiers — same canonical codec,
        same Ed25519/HMAC signing path.

        Args:
            text: Natural-language input. Will be content-addressed
                via ``sha256:<hex>`` for the source URI.
            extractor: ``"auto"``, ``"sieve"``, or ``"llm"``.
            branch: Branch name in the bundle metadata. Defaults
                to ``"main"``.
            title: Optional human-readable bundle title.
            signing_key: Optional shared secret for the HMAC
                signature path. Ed25519 always uses an in-memory
                ephemeral key when no PEM is supplied; the HMAC
                path is omitted entirely if this is None.

        Returns:
            ``{"bundle": <CanonicalBundle JSON>, "axioms": <int>,
            "source_uri": <sha256:...>, "extractor": <chosen>}``
            on success. ``{"error": <message>}`` on failure.
        """
        text = (text or "").strip()
        if not text:
            return {"error": "empty input"}

        from sum_cli import __version__ as cli_version
        from sum_cli.main import _extract, _pick_extractor
        from datetime import datetime, timezone

        try:
            chosen = _pick_extractor(None if extractor == "auto" else extractor)
            triples = _extract(text, chosen, model=None)
        except (RuntimeError, SystemExit) as e:
            return {"error": str(e)}

        if not triples:
            return {
                "error": (
                    "extractor returned zero triples. Input may be too "
                    "short, negated, or hedged — see "
                    "docs/FEATURE_CATALOG.md entries 6-9."
                ),
            }

        from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
        from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator
        from sum_engine_internal.infrastructure.canonical_codec import CanonicalCodec

        algebra = GodelStateAlgebra()
        tome_gen = AutoregressiveTomeGenerator(algebra)
        codec = CanonicalCodec(algebra, tome_gen, signing_key=signing_key)
        state = algebra.encode_chunk_state(list(triples))
        bundle = codec.export_bundle(state, branch=branch, title=title)

        source_uri = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        bundle["sum_cli"] = {
            "extractor": chosen,
            "source_uri": source_uri,
            "cli_version": cli_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "produced_by": "mcp_server",
        }

        return {
            "bundle": bundle,
            "axioms": len(triples),
            "source_uri": source_uri,
            "extractor": chosen,
        }

    @mcp.tool()
    def verify(bundle: dict, signing_key: str | None = None, strict: bool = False) -> dict:
        """Verify a CanonicalBundle's signatures and structural integrity.

        Runs the same six-step verification the CLI's ``sum verify``
        runs: schema-version gate, prime-scheme gate, Ed25519
        signature, HMAC signature (if signing_key supplied),
        canonical-tome → state-integer reconstruction, axiom-count
        match.

        Args:
            bundle: The CanonicalBundle dict (parsed from JSON).
            signing_key: Optional HMAC key. Without it, an embedded
                HMAC signature is reported as ``"skipped"``, not
                failed — except in strict mode.
            strict: If True, presence-without-key for HMAC and
                absence-of-both-signatures both reject.

        Returns:
            ``{"ok": <bool>, "axioms": <int>, "signatures": {...},
            "branch": ..., "errors": [<reason>, ...]}``. Always a
            dict — never raises on a malformed bundle, the contract
            is "verifier said no" not "verifier crashed".
        """
        from sum_cli.main import _verify_ed25519_bundle, _verify_hmac_bundle

        errors: list[str] = []

        for field in ("canonical_tome", "state_integer", "canonical_format_version"):
            if field not in bundle:
                errors.append(f"bundle missing required field: {field}")
        if errors:
            return {"ok": False, "errors": errors}

        if bundle["canonical_format_version"] != _SUPPORTED_CANONICAL_FORMAT:
            errors.append(
                f"unsupported canonical_format_version "
                f"{bundle['canonical_format_version']!r} "
                f"(this server speaks {_SUPPORTED_CANONICAL_FORMAT})"
            )
        declared_scheme = bundle.get("prime_scheme", _SUPPORTED_PRIME_SCHEME)
        if declared_scheme != _SUPPORTED_PRIME_SCHEME:
            errors.append(
                f"unsupported prime_scheme {declared_scheme!r} "
                f"(this server speaks {_SUPPORTED_PRIME_SCHEME})"
            )
        if errors:
            return {"ok": False, "errors": errors}

        ed25519_status = _verify_ed25519_bundle(bundle)
        hmac_status = _verify_hmac_bundle(bundle, signing_key)

        if ed25519_status == "invalid":
            errors.append("Ed25519 signature invalid — bundle does not match embedded public key")
        if hmac_status == "invalid":
            errors.append("HMAC signature invalid — bundle tampered or signed with a different key")
        if strict:
            if ed25519_status == "absent" and hmac_status == "absent":
                errors.append("strict: no signatures present to verify")
            if hmac_status == "skipped":
                errors.append("strict: HMAC signature present but no signing_key supplied")

        if errors:
            return {
                "ok": False,
                "signatures": {"ed25519": ed25519_status, "hmac": hmac_status},
                "errors": errors,
            }

        from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra

        algebra = GodelStateAlgebra()
        state = 1
        axioms = 0
        for line in bundle["canonical_tome"].splitlines():
            match = _TOME_LINE_PATTERN.match(line.strip())
            if not match:
                continue
            prime = algebra.get_or_mint_prime(match.group(1), match.group(2), match.group(3))
            state = math.lcm(state, prime)
            axioms += 1

        try:
            claimed_state = int(bundle["state_integer"])
        except (TypeError, ValueError):
            return {
                "ok": False,
                "signatures": {"ed25519": ed25519_status, "hmac": hmac_status},
                "errors": [f"state_integer is not an integer: {bundle['state_integer']!r}"],
            }

        expected_count = bundle.get("axiom_count")
        if expected_count is not None and axioms != expected_count:
            errors.append(
                f"axiom count mismatch: parsed {axioms}, claimed {expected_count}"
            )

        if state != claimed_state:
            errors.append(
                f"state integer mismatch: claimed {str(claimed_state)[:32]}…, "
                f"reconstructed {str(state)[:32]}…"
            )

        return {
            "ok": not errors,
            "axioms": axioms,
            "state_integer_digits": len(str(claimed_state)),
            "branch": bundle.get("branch", "main"),
            "bundle_version": bundle.get("bundle_version", "unknown"),
            "signatures": {"ed25519": ed25519_status, "hmac": hmac_status},
            "errors": errors,
        }

    @mcp.tool()
    def inspect(bundle: dict) -> dict:
        """Read-only summary of a bundle's headline fields.

        Does not verify signatures or reconstruct state — for that,
        call ``verify``. This is the equivalent of ``sum inspect
        --bundle`` in the CLI: a quick "what's in this bundle"
        view that an LLM agent can use before deciding whether to
        verify, dispatch, or reject.

        Args:
            bundle: The CanonicalBundle dict.

        Returns:
            Headline metadata: branch, axiom_count, bundle_version,
            canonical_format_version, prime_scheme, signature
            presence (not validity — that's verify's job),
            sum_cli sidecar if present.
        """
        return {
            "branch": bundle.get("branch", "main"),
            "title": bundle.get("title"),
            "axiom_count": bundle.get("axiom_count"),
            "bundle_version": bundle.get("bundle_version"),
            "canonical_format_version": bundle.get("canonical_format_version"),
            "prime_scheme": bundle.get("prime_scheme"),
            "state_integer_digits": (
                len(str(bundle["state_integer"]))
                if "state_integer" in bundle
                else None
            ),
            "tome_lines": (
                len(bundle["canonical_tome"].splitlines())
                if isinstance(bundle.get("canonical_tome"), str)
                else None
            ),
            "signatures_present": {
                "ed25519": "ed25519_signature" in bundle or "public_key" in bundle,
                "hmac": "hmac_signature" in bundle,
            },
            "sum_cli": bundle.get("sum_cli"),
        }

    @mcp.tool()
    def schema(name: str = "list") -> dict:
        """Return one of the SUM canonical schemas.

        Args:
            name: ``"list"`` returns the catalogue of names this
                server knows. Otherwise: ``"sum.canonical_bundle.v1"``,
                ``"sum.render_receipt.v1"``, ``"sum.merkle_inclusion.v1"``.

        Returns:
            For ``"list"``: ``{"schemas": [<name>, ...]}``.
            For a named schema: a dict with ``schema``, ``version``,
            ``fields`` describing field semantics. This is a
            human-readable catalogue, not a strict JSON Schema —
            ``docs/RENDER_RECEIPT_FORMAT.md`` and
            ``docs/MERKLE_SIDECAR_FORMAT.md`` are the wire-spec
            sources of truth.
        """
        catalogue = {
            "sum.canonical_bundle.v1": {
                "schema": "sum.canonical_bundle.v1",
                "version": _SUPPORTED_CANONICAL_FORMAT,
                "prime_scheme": _SUPPORTED_PRIME_SCHEME,
                "fields": {
                    "canonical_tome": "Newline-delimited 'The {s} {p} {o}.' lines",
                    "state_integer": "LCM of all axiom primes (decimal string)",
                    "axiom_count": "Number of axioms in the tome",
                    "branch": "Branch name (default 'main')",
                    "title": "Optional human-readable title",
                    "canonical_format_version": "Format version (currently 1.0.0)",
                    "prime_scheme": "Prime-derivation scheme (currently sha256_64_v1)",
                    "ed25519_signature": "Optional — base64url Ed25519 over canonical bytes",
                    "public_key": "Optional — base64url Ed25519 public key",
                    "hmac_signature": "Optional — hex HMAC-SHA256",
                },
                "spec": "docs/PROOF_BOUNDARY.md §1.3.1, §1.4",
            },
            "sum.render_receipt.v1": {
                "schema": "sum.render_receipt.v1",
                "fields": {
                    "schema": "Always 'sum.render_receipt.v1'",
                    "render_id": "UUID of the render call",
                    "issued_at": "ISO 8601 UTC timestamp",
                    "engine_version": "Worker version that signed",
                    "input_triples_hash": "sha256 over JCS-canonical triples",
                    "input_slider_position": "5-axis slider snapshot",
                    "output_tome_hash": "sha256 over the rendered tome",
                    "digital_source_type": "C2PA digital_source_type alignment",
                    "kid": "JWKS key ID used to sign",
                    "alg": "Signing algorithm (currently EdDSA)",
                },
                "spec": "docs/RENDER_RECEIPT_FORMAT.md",
                "envelope": "RFC 7515 §A.5 detached JWS over RFC 8785 JCS bytes",
            },
            "sum.merkle_inclusion.v1": {
                "schema": "sum.merkle_inclusion.v1",
                "fields": {
                    "leaf_index": "0-based index in the lex-sorted leaf array",
                    "leaf_hash": "sha256(LEAF_DOMAIN || canonical_fact_key) hex",
                    "siblings": "List of {position: 'left'|'right', hash: hex}",
                },
                "spec": "docs/MERKLE_SIDECAR_FORMAT.md",
                "domain_prefixes": {
                    "LEAF_DOMAIN": "SUM-MERKLE-FACT-LEAF-v1\\0",
                    "NODE_DOMAIN": "SUM-MERKLE-FACT-NODE-v1\\0",
                },
            },
        }

        if name == "list":
            return {"schemas": sorted(catalogue.keys())}
        if name not in catalogue:
            return {
                "error": f"unknown schema {name!r}",
                "known": sorted(catalogue.keys()),
            }
        return catalogue[name]

    return mcp


def main() -> int:
    """stdio entry point. Equivalent to running ``sum-mcp``.

    Uses FastMCP's built-in stdio transport (newline-delimited
    JSON-RPC 2.0 over stdin/stdout per the MCP wire spec). Local
    MCP clients (Claude Desktop, Claude Code, Cursor, Continue)
    spawn this as a subprocess and talk to it on the pipes.
    """
    server = build_server()
    server.run()
    return 0
