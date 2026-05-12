"""Transform protocol — the contract every registered transform implements.

Companion spec: ``docs/TRANSFORM_REGISTRY.md`` + ``docs/TRANSFORM_RECEIPT_FORMAT.md``.

The protocol exists because the slider product is one specific instance
of a more general pattern in SUM:

    (bundle | text) × transform × parameters → signed artifact

Every operation that produces a signed transform receipt
(``sum.transform_receipt.v1``) implements ``Transform`` and is registered
via ``register()`` in ``sum_engine_internal.transforms.__init__``.

This module pins the data shapes. Implementation lives in the per-
transform modules (``slider.py``, ``extract.py``, ``compose.py``, …).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Protocol, runtime_checkable


# Mirrors the Worker's Provider union (worker/src/receipt/sign.ts) +
# the render-receipt + transform-receipt spec §1.3.
Provider = Literal[
    "anthropic",
    "openai",
    "cf-ai-gateway-anthropic",
    "cf-ai-gateway-openai",
    "canonical-path",
]

# C2PA digitalSourceType (v2.2). `trainedAlgorithmicMedia` for LLM-served
# transforms; `algorithmicMedia` for deterministic ones.
DigitalSourceType = Literal["trainedAlgorithmicMedia", "algorithmicMedia"]


@dataclass
class TransformEnv:
    """Capability bag passed to every transform.

    A transform takes only what it declares it needs:
      - ``requires_llm = False`` transforms ignore the LLM fields;
      - ``requires_llm = True`` transforms raise if no key is available.

    The dispatch surface (HTTP / CLI / MCP) is responsible for
    populating this from request headers (BYO-keys) and env vars
    (operator-funded). User-supplied keys take precedence; see
    ``worker/src/routes/render.ts`` for the precedence rule.
    """
    # Receipt signing — required to produce a signed receipt. Absence
    # is non-fatal: the transform still runs and returns output, but
    # the caller gets no ``transform_receipt`` field. (Mirrors the
    # render-receipt path's existing behaviour.)
    private_jwk: dict[str, Any] | None = None
    kid: str | None = None

    # LLM capabilities (optional; only used by requires_llm transforms).
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    cf_ai_gateway_base: str | None = None
    default_anthropic_model: str = "claude-haiku-4-5-20251001"
    default_openai_model: str = "gpt-4o-mini"

    # Provider preference (optional). When set, the transform tries
    # this provider first and fails fast if it's not configured —
    # matches the Worker's resolveProvider() behaviour.
    preferred_provider: Provider | None = None


@dataclass
class TransformResult:
    """What a transform returns to the dispatch surface.

    The dispatch surface uses this to build the
    ``sum.transform_receipt.v1`` payload + return value to the caller.
    Honest-provenance: ``model`` and ``provider`` reflect what
    *actually* served, never the configured-default.
    """
    output: Any
    """The transform's primary output (tome string, tag-set list,
    merged bundle, etc.). Caller is responsible for any further
    serialisation; the transform exposes
    ``canonicalize_output(output)`` to produce the bytes that get
    hashed into the receipt's ``output_hash``."""

    model: str
    """The model the API echoed back, OR ``canonical-deterministic-v0``
    for non-LLM transforms. Same honest-provenance rule as
    ``sum.render_receipt.v1``."""

    provider: Provider
    """What actually served the call. ``canonical-path`` for
    deterministic transforms."""

    digital_source_type: DigitalSourceType
    """C2PA v2.2 distinguishes ``trainedAlgorithmicMedia`` (LLM) from
    ``algorithmicMedia`` (pure algorithm). Mirrors render-receipt."""

    llm_calls_made: int = 0
    """For telemetry / cost accounting. 0 for deterministic transforms."""

    cache_status: Literal["hit", "miss", "bypass"] = "miss"
    """Whether the dispatch surface served this from cache. The
    transform itself never sees a HIT (it's not called on HIT); this
    field is informational for caching dispatch surfaces."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Per-transform optional auxiliary data (e.g. slider drift
    measurements, extractor provenance per tag). Not signed into the
    receipt; surface for the HTTP response body."""


@runtime_checkable
class Transform(Protocol):
    """The interface every registered transform satisfies.

    Implementations are typically dataclasses (or plain classes with
    `name`, `requires_llm`, `digital_source_type` as class attrs).
    The registry stores instances, not classes — making it easy to
    parametrise (e.g. multiple slider configurations) without
    subclassing.
    """

    name: str
    """Registry id, e.g. ``"slider"``, ``"extract"``, ``"compose"``.
    MUST match the ``transform`` field of the receipts this transform
    produces."""

    requires_llm: bool
    """True iff the transform may call an LLM. False for
    pure-algorithmic transforms (compose, density-only slider, etc.)."""

    digital_source_type: DigitalSourceType
    """C2PA classification. ``trainedAlgorithmicMedia`` for LLM-
    mediated transforms; ``algorithmicMedia`` for deterministic
    transforms. Receipts inherit this from the transform."""

    def canonicalize_parameters(self, params: dict[str, Any]) -> bytes:
        """JCS-canonical bytes of the parameters object. The receipt's
        ``parameters_hash`` is ``sha256-`` + hex of this output. Per
        the spec, sort keys alphabetically; render numeric values
        with the transform's documented precision rule (the slider
        transform, for example, rounds non-density axes to bin
        centres before hashing)."""
        ...

    def canonicalize_input(self, raw_input: Any) -> bytes:
        """Canonical bytes of the input. For a CanonicalBundle, this
        is ``state_integer ‖ canonical_tome``; for raw text, UTF-8
        bytes. Each transform pins its accepted input shape."""
        ...

    def canonicalize_output(self, output: Any) -> bytes:
        """Canonical bytes of the output. For a tome, UTF-8 bytes;
        for a tag set, JCS of sorted unique triples; for a merged
        bundle, state_integer ‖ canonical_tome. Each transform pins
        its own canonicalisation."""
        ...

    async def apply(
        self,
        input: Any,
        parameters: dict[str, Any],
        env: TransformEnv,
    ) -> TransformResult:
        """Run the transform. Async because LLM-mediated transforms
        call out to the network; pure-algorithmic transforms can
        return immediately (`async def f(): return ...`) and still
        satisfy the protocol."""
        ...


def run_sync(coro: Awaitable[TransformResult]) -> TransformResult:
    """Convenience for callers that want a sync interface to an async
    transform. Uses ``asyncio.run`` for top-level callers; if there's
    already a running event loop (e.g. the caller is inside a
    notebook), returns the awaited result via ``asyncio.run_coroutine
    _threadsafe`` would be required — we keep this helper for simple
    cases and document the limitation."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "run_sync called inside a running event loop; await the "
        "transform's apply() directly from your async context."
    )
