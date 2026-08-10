"""Meaning-layer MCP tools — the receipt family for agent swarms.

Registered on the same hardened server as the legacy bundle surface
(``server.py``). These tools expose SUM's flagship layer over stdio:

  verify_receipt        the four sum_verify schemas (meaning-risk, render,
                        transform, chain), dispatched exactly like
                        ``python -m sum_verify`` (Stage A + optional side-band
                        replay); returns the same honest verdict shape the CLI
                        prints, proxy caveat included. The research-tier
                        perspective schema is not in this offline path.
  meaning_diff          per-document kept / dropped / added readout under a
                        named entailment judge (``sum meaning-diff``).
  depth_frontier        the whole faithful-to-compressed ladder with
                        per-rung diffs (``sum depth-diff``).
  mint_meaning_receipt  guided issuance, BYO private key ONLY.
  mint_chain_receipt    compose hop receipts into a certified chain,
                        BYO private key ONLY.

Concurrency contract (the swarm requirement, stated honestly):

  * Verification is pure crypto + integer math over per-call inputs — no
    shared mutable state, safe under arbitrary parallelism. ``verify_receipt``
    runs in the default executor so concurrent calls genuinely overlap.
  * The model judge (NLI / embedding) is NOT assumed re-entrant: torch
    inference shares model state, so every judge call serialises behind one
    in-process lock. For true parallel judging run N server processes.
  * Every result carries a ``concurrency`` hint field saying which of the
    two regimes the tool ran under, so an orchestrating agent can plan.

Key policy (non-negotiable): the mint tools accept a caller-supplied
private JWK and NEVER generate, store, or log key material. A request
without a usable private key is refused with ``error_class="schema"`` —
key custody stays with the caller. The audit line logs shapes and timing
only (``errors.py``), never values, so the key never reaches stderr either.

Optional-extra boundaries (each tool tags the gap instead of crashing):
``verify_receipt`` needs ``[verify]`` (joserfc + cryptography);
``meaning_diff`` / ``depth_frontier`` / the mint tools need ``[research]``;
the NLI judge needs ``[judge]``. Missing extras return
``error_class="extractor_unavailable"`` with the pip install hint —
that class means "the capability's backing dependency is not installed."
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from sum_engine_internal.mcp_server.errors import (
    ErrorClass,
    error_result,
    success_result,
)

# Reuse the server's text cap so prose limits stay in one story.
MAX_TEXT_CHARS: int = 200_000
MAX_LOSSES: int = 100_000       # far above any real corpus, below DoS
MAX_HOPS: int = 64              # chains are short; 64 is generous
MAX_VERSIONS: int = 16          # frontier rungs per call

# One lock for every model-judge invocation (torch inference is not
# assumed re-entrant-safe). Verification never takes this lock.
_JUDGE_LOCK = asyncio.Lock()

_VERIFY_HINT = (
    "verification is pure crypto + integer math over per-call inputs; "
    "safe under arbitrary parallelism"
)
_JUDGE_HINT = (
    "model-judge calls serialise behind one in-process lock; for true "
    "parallel judging run N server processes"
)

_DIFF_SCOPE = (
    "per-document MEASUREMENT under the named judge; not a certified bound. "
    "Proxy is blind to arrangement, sound, connotation, implicature. For a "
    "(1-delta) distribution-free bound use a sum.meaning_risk_receipt.v1 "
    "over a named corpus."
)

_PROXY_CAVEAT = (
    "verified=true is a cryptographic fact (signature + replayed bound), "
    "not evidence meaning was preserved. The bound is over a named proxy; "
    "vs human judgments the proxy correlated only modestly at summary level "
    "(Spearman rho ~0.27-0.33 on SummEval; NLI ~0.29 replicates on FRANK; "
    "the embedding judge is corpus-dependent, near zero on abstractive "
    "FRANK-XSum). Not a substitute for human review."
)

_EMBEDDING_CAVEAT = (
    "the embedding judge is brittle at the claim level and corpus-dependent "
    "(near zero correlation on abstractive FRANK-XSum; F18 paraphrase "
    "misranking); for a load-bearing readout use scorer='nli'"
)


def _need_verify_extra() -> "tuple[ErrorClass, str] | None":
    try:
        import joserfc  # noqa: F401
        return None
    except ImportError:
        return (
            ErrorClass.EXTRACTOR_UNAVAILABLE,
            "receipt verification needs the [verify] extra: "
            "pip install 'sum-engine[verify]'",
        )


def _need_research_extra() -> "tuple[ErrorClass, str] | None":
    try:
        import sum_engine_internal.research.meaning  # noqa: F401
        return None
    except ImportError:
        return (
            ErrorClass.EXTRACTOR_UNAVAILABLE,
            "this tool needs the [research] extra: "
            "pip install 'sum-engine[research,receipt-verify]'",
        )


def _load_scorer(scorer: str) -> "tuple[Any, tuple[ErrorClass, str] | None]":
    """Load an ENTAILMENT judge. lexical is rejected up front: it has no
    per-claim entailment, so a kept/dropped/added readout under it would be
    fabricated structure (the same rule the CLI enforces)."""
    if not isinstance(scorer, str) or scorer not in {"nli", "embedding"}:
        return None, (
            ErrorClass.SCHEMA,
            "scorer must be 'nli' or 'embedding' (lexical has no per-claim "
            "entailment; the CLI rejects it for diffs too). nli needs the "
            "[judge] extra and is the load-bearing choice.",
        )
    try:
        from sum_cli.main import _load_meaning_scorer
        loaded, err = _load_meaning_scorer(scorer)
        if err:
            return None, (ErrorClass.EXTRACTOR_UNAVAILABLE, err)
        return loaded, None
    except ImportError as exc:
        return None, (
            ErrorClass.EXTRACTOR_UNAVAILABLE,
            f"scorer {scorer!r} unavailable: {type(exc).__name__} — nli "
            "needs pip install 'sum-engine[research,judge]'",
        )


def _validate_prose(name: str, value: Any) -> "tuple[ErrorClass, str] | None":
    if not isinstance(value, str):
        return (ErrorClass.SCHEMA, f"{name} must be a string, got {type(value).__name__}")
    if len(value) > MAX_TEXT_CHARS:
        return (
            ErrorClass.INPUT_TOO_LARGE,
            f"{name} exceeds {MAX_TEXT_CHARS} chars (got {len(value)})",
        )
    if not value.strip():
        return (ErrorClass.SCHEMA, f"{name} is empty after stripping whitespace")
    return None


def _validate_losses(losses: Any) -> "tuple[ErrorClass, str] | None":
    if not isinstance(losses, list) or not losses:
        return (ErrorClass.SCHEMA, "losses must be a non-empty list of numbers in [0, 1]")
    if len(losses) > MAX_LOSSES:
        return (ErrorClass.INPUT_TOO_LARGE, f"losses exceeds {MAX_LOSSES} entries")
    for x in losses:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return (ErrorClass.SCHEMA, "losses entries must be numbers")
        if not (0.0 <= float(x) <= 1.0):
            return (ErrorClass.SCHEMA, "losses entries must lie in [0, 1]")
    return None


def _require_private_jwk(private_jwk: Any) -> "tuple[ErrorClass, str] | None":
    """BYO key, or nothing. This server never generates or stores keys."""
    if not isinstance(private_jwk, dict) or not private_jwk.get("d"):
        return (
            ErrorClass.SCHEMA,
            "mint tools require a caller-supplied Ed25519 PRIVATE JWK "
            "(kty=OKP, crv=Ed25519, with 'd'). This server never generates "
            "or stores key material — key custody stays with the caller "
            "(generate one offline, e.g. `sum mint-meaning --gen-key DIR`).",
        )
    if private_jwk.get("kty") != "OKP" or private_jwk.get("crv") != "Ed25519":
        return (
            ErrorClass.SCHEMA,
            "private_jwk must be an Ed25519 OKP JWK (kty='OKP', crv='Ed25519')",
        )
    return None


def _public_jwks_of(private_jwk: dict, kid: str) -> dict:
    pub = {k: v for k, v in private_jwk.items() if k != "d"}
    pub.setdefault("kty", "OKP")
    pub.setdefault("crv", "Ed25519")
    pub["kid"] = kid
    pub.setdefault("alg", "EdDSA")
    pub.setdefault("use", "sig")
    return {"keys": [pub]}


def _classify_verify_error(exc: Exception) -> ErrorClass:
    """Map sum_verify's exception taxonomy onto the server's error classes."""
    name = type(exc).__name__
    if name in {"UnsupportedSchemaError"}:
        return ErrorClass.SCHEMA
    if name in {"JoseEnvelopeError", "ReceiptVerifyError"}:
        return ErrorClass.SIGNATURE
    if name in {
        "MeaningReceiptReplayError",
        "MeaningReceiptDisclosureError",
        "ChainReceiptReplayError",
        "ChainReceiptDisclosureError",
    }:
        return ErrorClass.STRUCTURAL
    return ErrorClass.INTERNAL


def _chain_verdict(payload: dict, *, hops_given: bool, losses_given: bool) -> dict:
    verdict: dict[str, Any] = {
        "verified": True,
        "schema": "sum.chain_receipt.v1",
        "replayed": hops_given or losses_given,
        "hops_replayed": hops_given,
        "end_to_end_replayed": losses_given,
        "n_hops": payload.get("n_hops"),
    }
    if "budget_micro" in payload:
        verdict["budget"] = payload["budget_micro"] / 1_000_000
    if "joint_delta_micro" in payload:
        verdict["joint_confidence"] = max(
            0.0, 1.0 - payload["joint_delta_micro"] / 1_000_000
        )
    verdict["budget_scope"] = payload.get("budget_scope")
    if "not_covered" in payload:
        verdict["not_covered"] = payload.get("not_covered")
    return verdict


def _flat_verdict(schema: str, payload: Any, *, losses_given: bool) -> dict:
    verdict: dict[str, Any] = {"verified": True, "schema": schema}
    verdict["replayed"] = losses_given and schema == "sum.meaning_risk_receipt.v1"
    if isinstance(payload, dict):
        for k in ("scorer", "not_covered"):
            if k in payload:
                verdict[k] = payload[k]
        if schema == "sum.meaning_risk_receipt.v1":
            if "risk_upper_bound_micro" in payload:
                verdict["risk_upper_bound"] = payload["risk_upper_bound_micro"] / 1_000_000
            if "controlled" in payload:
                verdict["controlled"] = payload["controlled"]
    if schema == "sum.meaning_risk_receipt.v1":
        verdict["proxy_caveat"] = _PROXY_CAVEAT
    return verdict


def register_meaning_tools(mcp: Any) -> None:
    """Register the meaning-layer tools on an existing FastMCP server."""

    # ------------------------------------------------------------------
    # verify_receipt
    # ------------------------------------------------------------------

    @mcp.tool()
    async def verify_receipt(
        receipt: dict,
        jwks: dict,
        losses: "list[float] | None" = None,
        hops: "list[dict] | None" = None,
        max_age_seconds: "int | None" = None,
    ) -> dict:
        """Verify a SUM receipt offline (the four schemas ``sum_verify``
        handles: meaning-risk, render, transform, chain).

        The fifth family schema, ``sum.perspective_risk_receipt.v1``, is
        research-tier and is not verifiable through this dependency-light
        offline path; it is rejected with ``error_class="schema"`` naming the
        supported schemas.

        Dispatches exactly like ``python -m sum_verify``: Stage A
        (signature / schema / disclosure) always; Stage B replay when the
        side-band evidence is supplied. Returns the same honest verdict
        shape the CLI prints — including the ``proxy_caveat`` on
        meaning-risk receipts and the ``budget_scope`` honesty line on
        chains. verified=true is a cryptographic fact, never proof that
        meaning was preserved.

        Args:
            receipt: The signed envelope (dict with ``schema``/``kid``/
                ``payload``/``jws``).
            jwks: Issuer public JWKS ``{"keys": [...]}``.
            losses: Optional committed loss vector — replays a
                meaning-risk receipt's bound, or a chain's end_to_end leg.
            hops: Optional ordered per-hop envelopes for a chain receipt
                (hash-checked, verified, mirrors compared).
            max_age_seconds: Optional replay-defense window on
                ``signed_at``.

        Returns:
            Success: the CLI verdict shape + ``concurrency`` hint.
            Failure: ``{verified: false, error_class, errors}``.
        """
        t0 = time.perf_counter()
        try:
            err = _need_verify_extra()
            if err is not None:
                return error_result("verify_receipt", t0, *err, verified=False)
            if not isinstance(receipt, dict) or not isinstance(jwks, dict):
                return error_result(
                    "verify_receipt", t0, ErrorClass.SCHEMA,
                    "receipt and jwks must both be dicts", verified=False,
                )
            import sum_verify

            schema = receipt.get("schema")
            if schema not in sum_verify.SUPPORTED_SCHEMAS:
                return error_result(
                    "verify_receipt", t0, ErrorClass.SCHEMA,
                    f"unsupported schema {schema!r}; this server handles "
                    f"{', '.join(sorted(sum_verify.SUPPORTED_SCHEMAS))}",
                    verified=False,
                )
            if losses is not None:
                lerr = _validate_losses(losses)
                if lerr is not None:
                    return error_result("verify_receipt", t0, *lerr, verified=False)
            if hops is not None:
                if not isinstance(hops, list) or not all(isinstance(h, dict) for h in hops):
                    return error_result(
                        "verify_receipt", t0, ErrorClass.SCHEMA,
                        "hops must be a list of hop receipt envelopes (dicts)",
                        verified=False,
                    )
                if len(hops) > MAX_HOPS:
                    return error_result(
                        "verify_receipt", t0, ErrorClass.INPUT_TOO_LARGE,
                        f"hops exceeds {MAX_HOPS} entries", verified=False,
                    )

            def _run() -> dict:
                if schema == "sum.chain_receipt.v1":
                    payload = sum_verify.verify_chain_receipt(
                        receipt, jwks,
                        hop_envelopes=hops,
                        end_to_end_losses=losses,
                        max_age_seconds=max_age_seconds,
                    )
                    return _chain_verdict(
                        payload,
                        hops_given=hops is not None,
                        losses_given=losses is not None,
                    )
                result = sum_verify.verify(
                    receipt, jwks, losses=losses, max_age_seconds=max_age_seconds
                )
                payload = result if isinstance(result, dict) else getattr(result, "payload", {})
                verdict = _flat_verdict(schema, payload, losses_given=losses is not None)
                if losses is not None and schema == "sum.meaning_risk_receipt.v1":
                    verdict["n"] = len(losses)
                return verdict

            # Pure crypto + math: no judge lock; run in the executor so
            # concurrent verifications genuinely overlap.
            try:
                verdict = await asyncio.get_running_loop().run_in_executor(None, _run)
            except Exception as exc:  # verification failure -> tagged verdict
                return error_result(
                    "verify_receipt", t0, _classify_verify_error(exc),
                    f"{type(exc).__name__}: {exc}", verified=False,
                )

            return success_result(
                "verify_receipt", t0,
                **verdict,
                concurrency=_VERIFY_HINT,
            )
        except Exception as exc:
            return error_result(
                "verify_receipt", t0, ErrorClass.INTERNAL, type(exc).__name__,
                verified=False,
            )

    # ------------------------------------------------------------------
    # meaning_diff
    # ------------------------------------------------------------------

    @mcp.tool()
    async def meaning_diff(source: str, rendering: str, scorer: str = "nli") -> dict:
        """Per-document kept / dropped / added readout under a named judge.

        The MCP analogue of ``sum meaning-diff``. A MEASUREMENT for THIS
        document — never a certified bound; the scope field says so on
        every result. scorer='nli' (needs the [judge] extra) is the
        load-bearing choice; 'embedding' is allowed but its result carries
        the brittleness caveat; 'lexical' is rejected (no per-claim
        entailment). Judge calls serialise behind one in-process lock —
        for parallel judging run N server processes.

        Returns:
            Success: ``{loss, recall, fidelity, source_claims,
            preserved_claims, dropped_claims, added_claims, scorer,
            scorer_version, scope, concurrency}``.
            Failure: ``{error_class, errors}``.
        """
        t0 = time.perf_counter()
        try:
            for name, value in (("source", source), ("rendering", rendering)):
                err = _validate_prose(name, value)
                if err is not None:
                    return error_result("meaning_diff", t0, *err)
            rerr = _need_research_extra()
            if rerr is not None:
                return error_result("meaning_diff", t0, *rerr)
            loaded, serr = _load_scorer(scorer)
            if serr is not None:
                return error_result("meaning_diff", t0, *serr)

            async with _JUDGE_LOCK:
                r = await asyncio.get_running_loop().run_in_executor(
                    None, loaded.explain, source, rendering
                )

            fields: dict[str, Any] = {
                "loss": r.loss,
                "recall": r.recall,
                "fidelity": r.fidelity,
                "source_claims": list(r.source_claims),
                "preserved_claims": list(r.preserved_claims),
                "dropped_claims": list(r.dropped_claims),
                # MeaningReadout names this ``unsupported_claims`` (transform
                # sentences the source does not ground). The wire key stays
                # ``added_claims`` for the documented tool contract.
                "added_claims": list(r.unsupported_claims),
                "scorer": loaded.name,
                "scorer_version": loaded.version,
                "scope": _DIFF_SCOPE,
                "concurrency": _JUDGE_HINT,
            }
            if scorer == "embedding":
                fields["judge_caveat"] = _EMBEDDING_CAVEAT
            return success_result("meaning_diff", t0, **fields)
        except Exception as exc:
            return error_result(
                "meaning_diff", t0, ErrorClass.INTERNAL, type(exc).__name__
            )

    # ------------------------------------------------------------------
    # depth_frontier
    # ------------------------------------------------------------------

    @mcp.tool()
    async def depth_frontier(
        source: str, versions: "list[str]", scorer: str = "nli"
    ) -> dict:
        """Kept / dropped / added at EVERY rung of a faithful-to-compressed
        ladder, plus the loss-per-compression slope.

        The MCP analogue of ``sum depth-diff``: pass the source and the
        pre-made renderings most-faithful first. A per-document
        MEASUREMENT under the named judge, never a certified bound.
        Judge calls serialise behind one in-process lock.

        Returns:
            Success: ``{rungs: [...], scorer, scorer_version, scope,
            concurrency}`` where each rung mirrors ``sum depth-diff
            --json`` (label, compression_ratio, meaning_loss, claims...).
            Failure: ``{error_class, errors}``.
        """
        t0 = time.perf_counter()
        try:
            err = _validate_prose("source", source)
            if err is not None:
                return error_result("depth_frontier", t0, *err)
            if not isinstance(versions, list) or not versions:
                return error_result(
                    "depth_frontier", t0, ErrorClass.SCHEMA,
                    "versions must be a non-empty list of rendering strings "
                    "(most-faithful first)",
                )
            if len(versions) > MAX_VERSIONS:
                return error_result(
                    "depth_frontier", t0, ErrorClass.INPUT_TOO_LARGE,
                    f"versions exceeds {MAX_VERSIONS} entries",
                )
            for i, v in enumerate(versions):
                err = _validate_prose(f"versions[{i}]", v)
                if err is not None:
                    return error_result("depth_frontier", t0, *err)
            rerr = _need_research_extra()
            if rerr is not None:
                return error_result("depth_frontier", t0, *rerr)
            loaded, serr = _load_scorer(scorer)
            if serr is not None:
                return error_result("depth_frontier", t0, *serr)

            from sum_engine_internal.research.frontier import RenderFrontier

            renderings = [(f"v{i}", {}, text) for i, text in enumerate(versions)]

            def _run() -> list:
                frontier = RenderFrontier.from_renderings(source, renderings, loaded)
                return [r.as_dict() for r in frontier.depth_diff(loaded)]

            async with _JUDGE_LOCK:
                rungs = await asyncio.get_running_loop().run_in_executor(None, _run)

            return success_result(
                "depth_frontier", t0,
                rungs=rungs,
                scorer=loaded.name,
                scorer_version=loaded.version,
                scope=_DIFF_SCOPE,
                concurrency=_JUDGE_HINT,
            )
        except Exception as exc:
            return error_result(
                "depth_frontier", t0, ErrorClass.INTERNAL, type(exc).__name__
            )

    # ------------------------------------------------------------------
    # mint_meaning_receipt
    # ------------------------------------------------------------------

    @mcp.tool()
    async def mint_meaning_receipt(
        private_jwk: dict,
        kid: str,
        corpus_id: str,
        transform: str,
        loss_definition: str,
        losses: "list[float] | None" = None,
        pairs: "list[dict] | None" = None,
        scorer: str = "nli",
        scorer_name: "str | None" = None,
        scorer_version: str = "unversioned",
        delta: float = 0.05,
        method: str = "empirical_bernstein",
        alpha_target: "float | None" = None,
    ) -> dict:
        """Mint a signed ``sum.meaning_risk_receipt.v1`` (BYO private key).

        The MCP analogue of ``sum mint-meaning``. Two modes: BYO ``losses``
        (requires ``scorer_name`` naming the judge that produced them), or
        ``pairs`` = [{"source":..., "rendering":...}] scored here under
        ``scorer`` (judge lock applies). The receipt self-verifies through
        the sum_verify path before it is returned; if that fails you get an
        error, never a bad receipt. This server NEVER generates or stores
        keys — supply an Ed25519 private JWK; only the derived public JWKS
        is returned.

        Returns:
            Success: ``{receipt, public_jwks, verdict, losses, warnings,
            concurrency}``.
            Failure: ``{error_class, errors}``.
        """
        t0 = time.perf_counter()
        try:
            kerr = _require_private_jwk(private_jwk)
            if kerr is not None:
                return error_result("mint_meaning_receipt", t0, *kerr)
            for name, value in (
                ("kid", kid), ("corpus_id", corpus_id),
                ("transform", transform), ("loss_definition", loss_definition),
            ):
                if not isinstance(value, str) or not value.strip():
                    return error_result(
                        "mint_meaning_receipt", t0, ErrorClass.SCHEMA,
                        f"{name} must be a non-empty string",
                    )
            if (losses is None) == (pairs is None):
                return error_result(
                    "mint_meaning_receipt", t0, ErrorClass.SCHEMA,
                    "provide exactly one of losses (BYO, with scorer_name) "
                    "or pairs (scored here)",
                )
            if method not in {"auto", "hoeffding", "clopper_pearson", "empirical_bernstein"}:
                return error_result(
                    "mint_meaning_receipt", t0, ErrorClass.SCHEMA,
                    f"unknown method {method!r}",
                )
            rerr = _need_research_extra()
            if rerr is not None:
                return error_result("mint_meaning_receipt", t0, *rerr)
            verr = _need_verify_extra()
            if verr is not None:
                return error_result("mint_meaning_receipt", t0, *verr)

            from sum_engine_internal.research.meaning import (
                build_payload,
                certify_meaning_risk,
                score_pairs,
                sign_meaning_risk_receipt,
            )
            import sum_verify

            if losses is not None:
                lerr = _validate_losses(losses)
                if lerr is not None:
                    return error_result("mint_meaning_receipt", t0, *lerr)
                if not isinstance(scorer_name, str) or not scorer_name.strip():
                    return error_result(
                        "mint_meaning_receipt", t0, ErrorClass.SCHEMA,
                        "BYO losses require scorer_name as a NON-EMPTY STRING "
                        "(name the judge that produced them; it rides the "
                        "signed payload, so it must be a string, not a dict/"
                        "list — the receipt is conditional on it)",
                    )
                if not isinstance(scorer_version, str) or not scorer_version.strip():
                    return error_result(
                        "mint_meaning_receipt", t0, ErrorClass.SCHEMA,
                        "scorer_version must be a non-empty string (it rides "
                        "the signed payload)",
                    )
                loss_vec = [float(x) for x in losses]
                judge_name, judge_version = scorer_name, scorer_version
            else:
                if not isinstance(pairs, list) or not pairs or not all(
                    isinstance(p, dict) and isinstance(p.get("source"), str)
                    and isinstance(p.get("rendering"), str)
                    for p in pairs
                ):
                    return error_result(
                        "mint_meaning_receipt", t0, ErrorClass.SCHEMA,
                        'pairs must be a non-empty list of {"source": str, '
                        '"rendering": str}',
                    )
                for i, p in enumerate(pairs):
                    for field in ("source", "rendering"):
                        err = _validate_prose(f"pairs[{i}].{field}", p[field])
                        if err is not None:
                            return error_result("mint_meaning_receipt", t0, *err)
                loaded, serr = _load_scorer(scorer)
                if serr is not None:
                    return error_result("mint_meaning_receipt", t0, *serr)
                tuples = [(p["source"], p["rendering"]) for p in pairs]
                async with _JUDGE_LOCK:
                    loss_vec = await asyncio.get_running_loop().run_in_executor(
                        None, score_pairs, tuples, loaded
                    )
                judge_name, judge_version = loaded.name, loaded.version

            def _mint() -> "tuple[dict, dict, dict]":
                guarantee = certify_meaning_risk(
                    loss_vec, scorer_name=judge_name,
                    scorer_version=judge_version, delta=delta, method=method,
                )
                payload = build_payload(
                    guarantee=guarantee, losses=loss_vec, corpus_id=corpus_id,
                    transform=transform, alpha_target=alpha_target,
                    loss_definition=loss_definition,
                )
                receipt = sign_meaning_risk_receipt(
                    payload, private_jwk=private_jwk, kid=kid
                )
                public_jwks = _public_jwks_of(private_jwk, kid)
                # Self-verify through the real verifier before handing it out.
                verified_payload = sum_verify.verify_meaning_risk_receipt(
                    receipt, public_jwks, losses=loss_vec
                )
                return receipt, public_jwks, verified_payload

            receipt, public_jwks, verified_payload = (
                await asyncio.get_running_loop().run_in_executor(None, _mint)
            )

            warnings: list[str] = []
            n = len(loss_vec)
            if n < 30:
                warnings.append(
                    f"n={n} is small: distribution-free bounds are wide at "
                    "this size and the receipt may certify little; that is "
                    "honest, not an error."
                )
            if verified_payload.get("risk_upper_bound_micro", 0) >= 999_999:
                warnings.append(
                    "risk_upper_bound is ~1.0 — the certificate is vacuous "
                    "(it bounds the loss by the trivial maximum). Do not "
                    "present it as evidence of preservation."
                )

            return success_result(
                "mint_meaning_receipt", t0,
                receipt=receipt,
                public_jwks=public_jwks,
                verdict={
                    "verified": True,
                    "replayed": True,
                    "risk_upper_bound": verified_payload["risk_upper_bound_micro"] / 1_000_000,
                    "n": n,
                    "method": verified_payload.get("method"),
                    "proxy_caveat": _PROXY_CAVEAT,
                },
                losses=[round(float(x), 6) for x in loss_vec],
                warnings=warnings,
                concurrency=_JUDGE_HINT if pairs is not None else _VERIFY_HINT,
            )
        except Exception as exc:
            return error_result(
                "mint_meaning_receipt", t0, ErrorClass.INTERNAL, type(exc).__name__
            )

    # ------------------------------------------------------------------
    # mint_chain_receipt
    # ------------------------------------------------------------------

    @mcp.tool()
    async def mint_chain_receipt(
        private_jwk: dict,
        kid: str,
        hop_envelopes: "list[dict]",
        end_to_end_losses: "list[float] | None" = None,
        scorer_name: "str | None" = None,
        scorer_version: str = "1",
        loss_definition: "str | None" = None,
        delta: float = 0.05,
        method: str = "hoeffding",
        hops_jwks: "dict | None" = None,
    ) -> dict:
        """Compose signed hop receipts into a ``sum.chain_receipt.v1``
        (BYO private key).

        The MCP analogue of ``sum mint-chain``: ordered hop envelopes (in
        transformation order, >= 2) become one signed chain with an
        integer-exact Bonferroni budget; optionally a directly-measured
        end-to-end leg (requires ``scorer_name`` + ``loss_definition``).
        The mandatory ``budget_scope`` honesty field rides the payload: the
        budget bounds the SUM of per-hop expected losses, NOT the
        end-to-end loss (directed loss, no triangle inequality). The chain
        self-verifies (with ``hops_jwks`` merged in when the hops were
        signed by other keys) before it is returned. BYO key only.

        Returns:
            Success: ``{receipt, public_jwks, verdict, concurrency}``.
            Failure: ``{error_class, errors}``.
        """
        t0 = time.perf_counter()
        try:
            kerr = _require_private_jwk(private_jwk)
            if kerr is not None:
                return error_result("mint_chain_receipt", t0, *kerr)
            if not isinstance(kid, str) or not kid.strip():
                return error_result(
                    "mint_chain_receipt", t0, ErrorClass.SCHEMA,
                    "kid must be a non-empty string",
                )
            if (
                not isinstance(hop_envelopes, list)
                or len(hop_envelopes) < 2
                or not all(isinstance(h, dict) for h in hop_envelopes)
            ):
                return error_result(
                    "mint_chain_receipt", t0, ErrorClass.SCHEMA,
                    "hop_envelopes must be >= 2 signed meaning-risk receipt "
                    "envelopes in transformation order",
                )
            if len(hop_envelopes) > MAX_HOPS:
                return error_result(
                    "mint_chain_receipt", t0, ErrorClass.INPUT_TOO_LARGE,
                    f"hop_envelopes exceeds {MAX_HOPS} entries",
                )
            if end_to_end_losses is not None:
                lerr = _validate_losses(end_to_end_losses)
                if lerr is not None:
                    return error_result("mint_chain_receipt", t0, *lerr)
                if (
                    not isinstance(scorer_name, str) or not scorer_name.strip()
                    or not isinstance(loss_definition, str)
                    or not loss_definition.strip()
                ):
                    return error_result(
                        "mint_chain_receipt", t0, ErrorClass.SCHEMA,
                        "end_to_end_losses require scorer_name and "
                        "loss_definition as NON-EMPTY STRINGS (the leg is a "
                        "separate DIRECT measurement with its own named judge; "
                        "both ride the signed payload)",
                    )
                if not isinstance(scorer_version, str) or not scorer_version.strip():
                    return error_result(
                        "mint_chain_receipt", t0, ErrorClass.SCHEMA,
                        "scorer_version must be a non-empty string (it rides "
                        "the signed end-to-end leg)",
                    )
            rerr = _need_research_extra()
            if rerr is not None:
                return error_result("mint_chain_receipt", t0, *rerr)
            verr = _need_verify_extra()
            if verr is not None:
                return error_result("mint_chain_receipt", t0, *verr)

            from sum_engine_internal.research.meaning import (
                build_chain_payload,
                build_end_to_end_leg,
                sign_chain_receipt,
            )
            import sum_verify

            def _mint() -> "tuple[dict, dict, dict]":
                leg = None
                if end_to_end_losses is not None:
                    leg = build_end_to_end_leg(
                        [float(x) for x in end_to_end_losses],
                        scorer_name=scorer_name,
                        scorer_version=scorer_version,
                        loss_definition=loss_definition,
                        delta=delta,
                        method=method,
                    )
                payload = build_chain_payload(hop_envelopes, end_to_end=leg)
                receipt = sign_chain_receipt(
                    payload, private_jwk=private_jwk, kid=kid
                )
                public_jwks = _public_jwks_of(private_jwk, kid)
                if isinstance(hops_jwks, dict):
                    for key in hops_jwks.get("keys", []):
                        if not isinstance(key, dict):
                            continue
                        # Strip private members before merging: this field is
                        # named public_jwks and callers are invited to
                        # distribute it. A caller easily passes the PRIVATE hop
                        # signing JWKS by mistake (the dict it minted the hops
                        # with); republishing 'd' would leak the hop key. This
                        # is the same filter _public_jwks_of applies to the
                        # chain key (#7).
                        pub_key = {k: v for k, v in key.items() if k != "d"}
                        if pub_key not in public_jwks["keys"]:
                            public_jwks["keys"].append(pub_key)
                verified_payload = sum_verify.verify_chain_receipt(
                    receipt, public_jwks,
                    hop_envelopes=hop_envelopes,
                    end_to_end_losses=(
                        [float(x) for x in end_to_end_losses]
                        if end_to_end_losses is not None else None
                    ),
                )
                return receipt, public_jwks, verified_payload

            try:
                receipt, public_jwks, verified_payload = (
                    await asyncio.get_running_loop().run_in_executor(None, _mint)
                )
            except Exception as exc:
                return error_result(
                    "mint_chain_receipt", t0, _classify_verify_error(exc),
                    f"{type(exc).__name__}: {exc}",
                )

            verdict = _chain_verdict(
                verified_payload,
                hops_given=True,
                losses_given=end_to_end_losses is not None,
            )
            return success_result(
                "mint_chain_receipt", t0,
                receipt=receipt,
                public_jwks=public_jwks,
                verdict=verdict,
                concurrency=_VERIFY_HINT,
            )
        except Exception as exc:
            return error_result(
                "mint_chain_receipt", t0, ErrorClass.INTERNAL, type(exc).__name__
            )
