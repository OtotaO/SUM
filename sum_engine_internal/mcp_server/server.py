"""SUM MCP server v2 — hardened tool surface.

v2 hardening over v1:

  1. **Input size caps.** ``MAX_TEXT_CHARS`` rejects oversized
     inputs before any extractor runs; ``MAX_TOME_CHARS`` and
     ``MAX_AXIOM_COUNT`` reject oversized bundles before
     reconstruction.
  2. **Tagged error classes.** Every error result carries
     ``error_class`` from a fixed enum; callers branch on the
     tag, never on the string detail.
  3. **Network opt-in.** The LLM extractor is disabled by
     default. Setting ``SUM_MCP_ALLOW_NETWORK=1`` at server
     start enables it; even then, ``extractor="llm"`` must be
     explicit per call. Prevents a prompt-injected LLM client
     from spending the user's tokens via the auto path.
  4. **Concurrency lock.** spaCy's nlp pipeline is not
     thread-safe under concurrent calls; a module-level
     asyncio lock serialises extractor invocations.
  5. **Catch-all per tool.** Every tool body wraps a single
     ``try``/``except Exception`` returning
     ``error_class="internal"`` — no traceback, no internal
     paths leaked. Server stays up.
  6. **Forward-compat policy.** Bundles with unknown top-level
     fields under ``canonical_format_version=1.0.0`` are
     accepted (additive); future *major* version bumps fail
     closed.
  7. **Single result-construction point.** Every result goes
     through ``success_result`` / ``error_result``, both of
     which emit a structured stderr audit line — values
     redacted, only shapes and timing.
  8. **Network-disallowed signaling.** A request for the LLM
     extractor when ``SUM_MCP_ALLOW_NETWORK`` is unset returns
     ``error_class="network_disallowed"`` rather than failing
     opaquely — the operator's intent ("no network calls") is
     preserved as a first-class error.

Wire-format compatibility note: v1 callers pattern-matching on
``ok: bool`` need to update — v2 uses ``error_class`` presence
as the failure signal. ``ok`` is still emitted by ``verify``
because that tool's purpose is to return a boolean verdict;
elsewhere (``extract``, ``attest``, ``inspect``, ``schema``)
the failure signal is ``"error_class" in result``.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import time
from typing import Any

from mcp.server.fastmcp import FastMCP

from sum_engine_internal.mcp_server.errors import (
    ErrorClass,
    error_result,
    success_result,
)


_SUPPORTED_CANONICAL_FORMAT = "1.0.0"
_SUPPORTED_PRIME_SCHEME = "sha256_64_v1"
_TOME_LINE_PATTERN = re.compile(r"^The (\S+) (\S+) (.+)\.$")

# Hardening caps. Values picked above any legitimate use case
# and below any DoS surface.
MAX_TEXT_CHARS: int = 200_000           # ~50 pages of dense prose
MAX_TOME_CHARS: int = 10_000_000        # ~10 MB tome
MAX_AXIOM_COUNT: int = 100_000          # > the substrate's practical ceiling per PROOF_BOUNDARY §2.2
MAX_STATE_INTEGER_DIGITS: int = 1_000_000  # > 100 k axioms × ~6 digits per prime

# Network-side-effect opt-in. Default fail-closed.
NETWORK_ALLOWED: bool = os.environ.get("SUM_MCP_ALLOW_NETWORK") == "1"

# Concurrency guard around the spaCy nlp pipeline. spaCy is
# documented as not thread-safe under concurrent calls; FastMCP
# may run multiple tool invocations concurrently on its asyncio
# loop. A single module-level lock serialises extraction.
_EXTRACTOR_LOCK = asyncio.Lock()


def build_server() -> FastMCP:
    """Build and register the v2-hardened SUM MCP server.

    Returns the configured ``FastMCP`` instance. Caller runs it
    via ``main()`` (stdio transport).
    """
    mcp = FastMCP(
        name="sum",
        instructions=(
            "SUM verifiable knowledge distillation engine (v2 hardened). "
            "Tools: extract / attest / verify / inspect / render / schema. "
            "Default extractor is offline-only (sieve). The LLM extractor "
            "is disabled unless SUM_MCP_ALLOW_NETWORK=1 was set when the "
            "server started. Every tool returns either a tool-specific "
            "success shape or an `error_class` tag from a fixed enum: "
            "schema, signature, structural, input_too_large, "
            "extractor_unavailable, network_disallowed, revoked, internal. "
            "Branch on `error_class`, not on `errors[i]` substrings."
        ),
    )

    # ------------------------------------------------------------------
    # extract
    # ------------------------------------------------------------------

    @mcp.tool()
    async def extract(text: str, extractor: str = "auto") -> dict:
        """Extract (subject, predicate, object) triples from prose.

        Args:
            text: Natural-language input. Hard-capped at
                ``MAX_TEXT_CHARS`` (200 000 chars). Empty input
                returns ``error_class="schema"``.
            extractor: ``"auto"`` (default; sieve only),
                ``"sieve"`` (offline, deterministic), or
                ``"llm"`` (network; disabled unless
                ``SUM_MCP_ALLOW_NETWORK=1``).

        Returns:
            Success: ``{triples, extractor, count}``.
            Failure: ``{error_class, errors}``.
        """
        t0 = time.perf_counter()
        try:
            err = _validate_text(text)
            if err is not None:
                return error_result("extract", t0, *err)

            err = _validate_extractor_choice(extractor)
            if err is not None:
                return error_result("extract", t0, *err)

            chosen = _resolve_extractor(extractor)

            async with _EXTRACTOR_LOCK:
                from sum_cli.main import _extract
                triples = await asyncio.get_running_loop().run_in_executor(
                    None, _extract, text.strip(), chosen, None
                )

            return success_result(
                "extract",
                t0,
                triples=[list(t) for t in triples],
                extractor=chosen,
                count=len(triples),
            )
        except Exception as exc:
            return error_result(
                "extract",
                t0,
                ErrorClass.INTERNAL,
                f"{type(exc).__name__}",
            )

    # ------------------------------------------------------------------
    # attest
    # ------------------------------------------------------------------

    @mcp.tool()
    async def attest(
        text: str,
        extractor: str = "auto",
        branch: str = "main",
        title: str | None = None,
        signing_key: str | None = None,
    ) -> dict:
        """Extract triples and produce a signed CanonicalBundle.

        Mirrors ``sum attest``. Bundle bytes are byte-identical
        to the CLI path, so they verify through every existing
        Python/Node/browser SUM verifier.

        Args:
            text: Natural-language input, ≤``MAX_TEXT_CHARS``.
            extractor: As ``extract``.
            branch: Bundle branch metadata (default ``"main"``).
            title: Optional bundle title.
            signing_key: Optional shared secret for HMAC-SHA256
                attestation. Ed25519 is intentionally omitted
                when no key path is supplied — see
                docs/MCP_INTEGRATION.md trust-model section.

        Returns:
            Success: ``{bundle, axioms, source_uri, extractor}``.
            Failure: ``{error_class, errors}``.
        """
        t0 = time.perf_counter()
        try:
            err = _validate_text(text)
            if err is not None:
                return error_result("attest", t0, *err)

            err = _validate_extractor_choice(extractor)
            if err is not None:
                return error_result("attest", t0, *err)

            chosen = _resolve_extractor(extractor)

            async with _EXTRACTOR_LOCK:
                from sum_cli.main import _extract
                triples = await asyncio.get_running_loop().run_in_executor(
                    None, _extract, text.strip(), chosen, None
                )

            if not triples:
                return error_result(
                    "attest",
                    t0,
                    ErrorClass.STRUCTURAL,
                    "extractor returned zero triples (input may be too "
                    "short, negated, or hedged — see "
                    "docs/FEATURE_CATALOG.md entries 6-9).",
                )

            from datetime import datetime, timezone
            from sum_cli import __version__ as cli_version
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
                "produced_by": "mcp_server.v2",
            }

            return success_result(
                "attest",
                t0,
                bundle=bundle,
                axioms=len(triples),
                source_uri=source_uri,
                extractor=chosen,
            )
        except Exception as exc:
            return error_result(
                "attest", t0, ErrorClass.INTERNAL, type(exc).__name__
            )

    # ------------------------------------------------------------------
    # verify
    # ------------------------------------------------------------------

    @mcp.tool()
    def verify(bundle: dict, signing_key: str | None = None, strict: bool = False) -> dict:
        """Verify a CanonicalBundle's signatures and structural integrity.

        Args:
            bundle: The CanonicalBundle dict.
            signing_key: Optional HMAC key.
            strict: Reject bundles with no signatures or with
                an HMAC signature present without a key.

        Returns:
            Always: ``{ok, axioms, signatures, ...}``. On failure
            also ``{error_class, errors}``. The ``ok`` boolean is
            preserved on this tool because ``verify``'s purpose is
            specifically to return a verdict.
        """
        t0 = time.perf_counter()
        try:
            if not isinstance(bundle, dict):
                return error_result(
                    "verify", t0, ErrorClass.SCHEMA,
                    f"bundle must be a dict, got {type(bundle).__name__}",
                    ok=False,
                )

            for field in ("canonical_tome", "state_integer", "canonical_format_version"):
                if field not in bundle:
                    return error_result(
                        "verify", t0, ErrorClass.SCHEMA,
                        f"bundle missing required field: {field}",
                        ok=False,
                    )

            if bundle["canonical_format_version"] != _SUPPORTED_CANONICAL_FORMAT:
                # Forward-compat: accept additive 1.x versions
                # later, reject any other major version now.
                ver = str(bundle["canonical_format_version"])
                if not ver.startswith("1."):
                    return error_result(
                        "verify", t0, ErrorClass.SCHEMA,
                        f"unsupported canonical_format_version {ver!r} "
                        f"(this server speaks {_SUPPORTED_CANONICAL_FORMAT})",
                        ok=False,
                    )

            declared_scheme = bundle.get("prime_scheme", _SUPPORTED_PRIME_SCHEME)
            if declared_scheme != _SUPPORTED_PRIME_SCHEME:
                return error_result(
                    "verify", t0, ErrorClass.SCHEMA,
                    f"unsupported prime_scheme {declared_scheme!r}",
                    ok=False,
                )

            # Bundle size caps — applied AFTER schema gate so the
            # error class for a missing-field-on-huge-bundle is
            # SCHEMA, not INPUT_TOO_LARGE.
            tome_str = bundle.get("canonical_tome", "")
            if not isinstance(tome_str, str) or len(tome_str) > MAX_TOME_CHARS:
                return error_result(
                    "verify", t0, ErrorClass.INPUT_TOO_LARGE,
                    f"canonical_tome exceeds {MAX_TOME_CHARS} chars",
                    ok=False,
                )

            state_str = bundle["state_integer"]
            if not isinstance(state_str, str) or len(state_str) > MAX_STATE_INTEGER_DIGITS:
                return error_result(
                    "verify", t0, ErrorClass.INPUT_TOO_LARGE,
                    f"state_integer exceeds {MAX_STATE_INTEGER_DIGITS} digits",
                    ok=False,
                )

            from sum_cli.main import _verify_ed25519_bundle, _verify_hmac_bundle

            ed25519_status = _verify_ed25519_bundle(bundle)
            hmac_status = _verify_hmac_bundle(bundle, signing_key)

            if ed25519_status == "invalid":
                return error_result(
                    "verify", t0, ErrorClass.SIGNATURE,
                    "Ed25519 signature invalid — bundle does not match embedded public key",
                    ok=False,
                    signatures={"ed25519": ed25519_status, "hmac": hmac_status},
                )
            if hmac_status == "invalid":
                return error_result(
                    "verify", t0, ErrorClass.SIGNATURE,
                    "HMAC signature invalid — bundle tampered or signed with a different key",
                    ok=False,
                    signatures={"ed25519": ed25519_status, "hmac": hmac_status},
                )
            if strict:
                if ed25519_status == "absent" and hmac_status == "absent":
                    return error_result(
                        "verify", t0, ErrorClass.SIGNATURE,
                        "strict: no signatures present to verify",
                        ok=False,
                        signatures={"ed25519": ed25519_status, "hmac": hmac_status},
                    )
                if hmac_status == "skipped":
                    return error_result(
                        "verify", t0, ErrorClass.SIGNATURE,
                        "strict: HMAC signature present but no signing_key supplied",
                        ok=False,
                        signatures={"ed25519": ed25519_status, "hmac": hmac_status},
                    )

            from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra

            algebra = GodelStateAlgebra()
            state = 1
            axioms = 0
            for line in tome_str.splitlines():
                match = _TOME_LINE_PATTERN.match(line.strip())
                if not match:
                    continue
                if axioms >= MAX_AXIOM_COUNT:
                    return error_result(
                        "verify", t0, ErrorClass.INPUT_TOO_LARGE,
                        f"axiom count exceeds {MAX_AXIOM_COUNT}",
                        ok=False,
                    )
                prime = algebra.get_or_mint_prime(match.group(1), match.group(2), match.group(3))
                state = math.lcm(state, prime)
                axioms += 1

            try:
                claimed_state = int(state_str)
            except (TypeError, ValueError):
                return error_result(
                    "verify", t0, ErrorClass.SCHEMA,
                    f"state_integer is not an integer: {state_str!r}",
                    ok=False,
                    signatures={"ed25519": ed25519_status, "hmac": hmac_status},
                )

            expected_count = bundle.get("axiom_count")
            if expected_count is not None and axioms != expected_count:
                return error_result(
                    "verify", t0, ErrorClass.STRUCTURAL,
                    f"axiom count mismatch: parsed {axioms}, claimed {expected_count}",
                    ok=False,
                    axioms=axioms,
                    signatures={"ed25519": ed25519_status, "hmac": hmac_status},
                )

            if state != claimed_state:
                return error_result(
                    "verify", t0, ErrorClass.STRUCTURAL,
                    f"state integer mismatch: claimed {str(claimed_state)[:32]}…, "
                    f"reconstructed {str(state)[:32]}…",
                    ok=False,
                    axioms=axioms,
                    signatures={"ed25519": ed25519_status, "hmac": hmac_status},
                )

            return success_result(
                "verify",
                t0,
                ok=True,
                axioms=axioms,
                state_integer_digits=len(state_str),
                branch=bundle.get("branch", "main"),
                bundle_version=bundle.get("bundle_version", "unknown"),
                signatures={"ed25519": ed25519_status, "hmac": hmac_status},
            )
        except Exception as exc:
            return error_result(
                "verify", t0, ErrorClass.INTERNAL, type(exc).__name__, ok=False
            )

    # ------------------------------------------------------------------
    # inspect
    # ------------------------------------------------------------------

    @mcp.tool()
    def inspect(bundle: dict) -> dict:
        """Read-only summary of a bundle's headline fields."""
        t0 = time.perf_counter()
        try:
            if not isinstance(bundle, dict):
                return error_result(
                    "inspect", t0, ErrorClass.SCHEMA,
                    f"bundle must be a dict, got {type(bundle).__name__}",
                )

            tome = bundle.get("canonical_tome")
            tome_lines = (
                len(tome.splitlines()) if isinstance(tome, str) else None
            )
            state_int = bundle.get("state_integer")
            state_digits = (
                len(state_int) if isinstance(state_int, str) else None
            )

            return success_result(
                "inspect",
                t0,
                branch=bundle.get("branch", "main"),
                title=bundle.get("title"),
                axiom_count=bundle.get("axiom_count"),
                bundle_version=bundle.get("bundle_version"),
                canonical_format_version=bundle.get("canonical_format_version"),
                prime_scheme=bundle.get("prime_scheme"),
                state_integer_digits=state_digits,
                tome_lines=tome_lines,
                signatures_present={
                    "ed25519": "ed25519_signature" in bundle or "public_key" in bundle,
                    "hmac": "hmac_signature" in bundle,
                },
                sum_cli=bundle.get("sum_cli"),
            )
        except Exception as exc:
            return error_result(
                "inspect", t0, ErrorClass.INTERNAL, type(exc).__name__
            )

    # ------------------------------------------------------------------
    # render
    # ------------------------------------------------------------------

    @mcp.tool()
    async def render(
        bundle: dict,
        density: float = 1.0,
        length: float = 0.5,
        formality: float = 0.5,
        audience: float = 0.5,
        perspective: float = 0.5,
        title: str = "Rendered Tome",
    ) -> dict:
        """Render a CanonicalBundle's axioms back into prose under
        explicit slider control. The MCP analogue of ``sum render``.

        Local-only path: actions the density slider deterministically
        (lex-prefix subsetting); non-neutral length / formality /
        audience / perspective return ``error_class="schema"`` because
        the LLM-conditioned axes require a Worker, which the MCP
        server does not currently broker (see
        ``docs/MCP_INTEGRATION.md`` for the rationale — keeping the
        MCP server fully offline by default preserves the
        ``SUM_MCP_ALLOW_NETWORK`` opt-in property).

        Args:
            bundle: CanonicalBundle dict (same shape ``verify`` accepts).
            density: Axiom-coverage slider in [0, 1]. 1.0 keeps all
                axioms (default); 0.0 keeps none.
            length / formality / audience / perspective: 0.5 = neutral
                (default). Non-0.5 rejected with a SCHEMA error
                pointing at the local-only constraint.
            title: Tome title. Default: "Rendered Tome".

        Returns:
            Success: ``{tome, sliders, mode, axiom_count_input, title}``.
            Failure: ``{error_class, errors}``.
        """
        t0 = time.perf_counter()
        try:
            if not isinstance(bundle, dict):
                return error_result(
                    "render", t0, ErrorClass.SCHEMA,
                    f"bundle must be a dict, got {type(bundle).__name__}",
                )

            for field in ("canonical_tome", "canonical_format_version"):
                if field not in bundle:
                    return error_result(
                        "render", t0, ErrorClass.SCHEMA,
                        f"bundle missing required field: {field}",
                    )

            ver = str(bundle["canonical_format_version"])
            if not ver.startswith("1."):
                return error_result(
                    "render", t0, ErrorClass.SCHEMA,
                    f"unsupported canonical_format_version {ver!r} "
                    f"(this server speaks {_SUPPORTED_CANONICAL_FORMAT})",
                )

            tome_str = bundle["canonical_tome"]
            if not isinstance(tome_str, str) or len(tome_str) > MAX_TOME_CHARS:
                return error_result(
                    "render", t0, ErrorClass.INPUT_TOO_LARGE,
                    f"canonical_tome exceeds {MAX_TOME_CHARS} chars",
                )

            from sum_engine_internal.ensemble.tome_sliders import TomeSliders

            try:
                sliders = TomeSliders(
                    density=density, length=length,
                    formality=formality, audience=audience,
                    perspective=perspective,
                )
            except (ValueError, TypeError) as e:
                return error_result(
                    "render", t0, ErrorClass.SCHEMA,
                    f"invalid slider value: {e}",
                )

            if sliders.requires_extrapolator():
                non_neutral = [
                    f"{name}={getattr(sliders, name)}"
                    for name in ("length", "formality", "audience", "perspective")
                    if abs(getattr(sliders, name) - 0.5) > 1e-9
                ]
                return error_result(
                    "render", t0, ErrorClass.SCHEMA,
                    "non-neutral LLM-conditioned axes ("
                    + ", ".join(non_neutral)
                    + ") require an LLM extrapolator. The MCP server's "
                    "render tool is local-only (deterministic density "
                    "slider) — drop the affected sliders to 0.5, or "
                    "use the Worker's POST /api/render endpoint for "
                    "LLM-conditioned rendering.",
                )

            triples: list[tuple[str, str, str]] = []
            for line in tome_str.splitlines():
                m = _TOME_LINE_PATTERN.match(line.strip())
                if m:
                    if len(triples) >= MAX_AXIOM_COUNT:
                        return error_result(
                            "render", t0, ErrorClass.INPUT_TOO_LARGE,
                            f"axiom count exceeds {MAX_AXIOM_COUNT}",
                        )
                    triples.append((m.group(1), m.group(2), m.group(3)))

            if not triples:
                return error_result(
                    "render", t0, ErrorClass.STRUCTURAL,
                    "bundle's canonical_tome contains zero parseable "
                    "axiom lines — nothing to render.",
                )

            from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
            from sum_engine_internal.ensemble.tome_generator import AutoregressiveTomeGenerator

            algebra = GodelStateAlgebra()
            state = 1
            for s, p, o in triples:
                state = math.lcm(state, algebra.get_or_mint_prime(s, p, o))

            tome_gen = AutoregressiveTomeGenerator(algebra)
            tome_text = tome_gen.generate_controlled(state, sliders=sliders, title=title)

            return success_result(
                "render",
                t0,
                tome=tome_text,
                sliders={
                    "density": sliders.density,
                    "length": sliders.length,
                    "formality": sliders.formality,
                    "audience": sliders.audience,
                    "perspective": sliders.perspective,
                },
                mode="local-deterministic",
                axiom_count_input=len(triples),
                title=title,
            )
        except Exception as exc:
            return error_result(
                "render", t0, ErrorClass.INTERNAL, type(exc).__name__
            )

    # ------------------------------------------------------------------
    # schema
    # ------------------------------------------------------------------

    @mcp.tool()
    def schema(name: str = "list") -> dict:
        """Return one of the SUM canonical schemas."""
        t0 = time.perf_counter()
        try:
            if not isinstance(name, str):
                return error_result(
                    "schema", t0, ErrorClass.SCHEMA,
                    f"name must be a string, got {type(name).__name__}",
                )

            catalogue = _build_schema_catalogue()
            if name == "list":
                return success_result(
                    "schema", t0, schemas=sorted(catalogue.keys())
                )
            if name not in catalogue:
                return error_result(
                    "schema", t0, ErrorClass.SCHEMA,
                    f"unknown schema {name!r}",
                    known=sorted(catalogue.keys()),
                )
            return success_result("schema", t0, **catalogue[name])
        except Exception as exc:
            return error_result(
                "schema", t0, ErrorClass.INTERNAL, type(exc).__name__
            )

    return mcp


# ----------------------------------------------------------------------
# Validation helpers (single source of truth, easy to fuzz)
# ----------------------------------------------------------------------


def _validate_text(text: Any) -> tuple[ErrorClass, str] | None:
    """Validate the ``text`` argument shared by extract + attest.

    Returns None on success; a (class, message) tuple on
    failure, ready for ``error_result``.
    """
    if not isinstance(text, str):
        return (ErrorClass.SCHEMA, f"text must be a string, got {type(text).__name__}")
    if len(text) > MAX_TEXT_CHARS:
        return (
            ErrorClass.INPUT_TOO_LARGE,
            f"text exceeds {MAX_TEXT_CHARS} chars (got {len(text)})",
        )
    if not text.strip():
        return (ErrorClass.SCHEMA, "text is empty after stripping whitespace")
    return None


def _validate_extractor_choice(extractor: Any) -> tuple[ErrorClass, str] | None:
    """Validate the ``extractor`` argument."""
    if not isinstance(extractor, str):
        return (
            ErrorClass.SCHEMA,
            f"extractor must be a string, got {type(extractor).__name__}",
        )
    if extractor not in {"auto", "sieve", "llm"}:
        return (
            ErrorClass.SCHEMA,
            f"extractor must be one of auto/sieve/llm, got {extractor!r}",
        )
    if extractor == "llm" and not NETWORK_ALLOWED:
        return (
            ErrorClass.NETWORK_DISALLOWED,
            "extractor='llm' requires SUM_MCP_ALLOW_NETWORK=1 at server start "
            "(prevents prompt-injected clients from spending API tokens)",
        )
    return None


def _resolve_extractor(extractor: str) -> str:
    """Pick the concrete extractor name. Network-safe by default.

    ``extractor="auto"`` resolves to ``"sieve"`` unconditionally
    here — even if ``OPENAI_API_KEY`` is set. The CLI's
    ``_pick_extractor`` falls through to ``llm`` on auto when
    the env var is set; the MCP path does not, because a
    prompt-injected client should not be able to drive
    network calls via the auto path. Set
    ``SUM_MCP_ALLOW_NETWORK=1`` and pass ``extractor="llm"``
    explicitly to use the LLM path.
    """
    if extractor == "auto":
        return "sieve"
    if extractor == "llm" and not NETWORK_ALLOWED:
        # _validate_extractor_choice should have caught this;
        # belt-and-braces.
        raise RuntimeError("extractor='llm' requested without NETWORK_ALLOWED")
    return extractor


def _build_schema_catalogue() -> dict:
    return {
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


def main() -> int:
    """stdio entry point. Equivalent to running ``sum-mcp``."""
    server = build_server()
    server.run()
    return 0
