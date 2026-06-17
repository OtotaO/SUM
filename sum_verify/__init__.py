"""``sum_verify`` — the small, stable, dependency-light SUM receipt verifier.

This is the package an integrator pins. It exposes a *stable* public API
for verifying SUM's signed receipts without installing the CLI, the
research extras, or a numeric stack (no numpy / scipy / torch). The only
runtime dependencies are ``cryptography`` (already a SUM core dep) and
``joserfc`` (the ``sum-engine[verify]`` extra).

    pip install "sum-engine[verify]"

Usage — verify any SUM receipt, dispatched by its ``schema`` field:

    import json
    from sum_verify import verify

    receipt = json.load(open("receipt.json"))
    jwks    = json.load(open("jwks.json"))          # the issuer's /.well-known/jwks.json

    # Signature + structural checks only:
    payload = verify(receipt, jwks)

    # Meaning-risk receipts also replay the conformal bound offline when
    # you hand them the committed per-pair losses side-band:
    losses  = json.load(open("losses.json"))         # bare list or {"losses": [...]}
    payload = verify(receipt, jwks, losses=losses)

Supported schemas (``SUPPORTED_SCHEMAS``):
  - ``sum.meaning_risk_receipt.v1``  — signed, replayable bound on a named
    meaning-loss proxy (the flagship; replays offline here);
  - ``sum.render_receipt.v1``        — signed render provenance;
  - ``sum.transform_receipt.v1``     — signed transform provenance.

What a verified receipt proves — and does NOT
---------------------------------------------
PROVES: the payload was signed by the holder of ``kid``'s private key;
the envelope is well-formed and unexpired; and — for a meaning-risk
receipt replayed with its losses — that the committed losses hash to the
anchor and re-certify to the stated bound by exact integer equality.

Does NOT prove that *meaning was preserved*. A meaning-risk receipt
bounds a NAMED PROXY for meaning-loss, marginally (on average over the
calibration corpus), under exchangeability — never per-document, and
never the layers its ``not_covered`` field declares out of scope
(arrangement, sound, connotation, implicature). Where that proxy has been
measured against human faithfulness judgments (SummEval), it correlated
only MODESTLY (Spearman rho ~= 0.27-0.33) — directionally valid, not a
substitute for human review. The verifier ENFORCES that those disclosures
are present; it does not let a bare bound through. (The CLI surfaces this
as a ``proxy_caveat`` on every verified meaning-risk verdict.)

Stability promise: ``__version__`` tracks THIS module's public surface
and the receipt wire formats it accepts — independent of the engine's
release version. A bump to the engine that does not change a supported
wire format does not bump this. Names exported in ``__all__`` are the
pinnable contract.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

from typing import Any, Sequence

from sum_engine_internal.infrastructure.jose_envelope import JoseEnvelopeError
from sum_engine_internal.render_receipt.verifier import (
    SUPPORTED_SCHEMA as RENDER_SCHEMA,
)
from sum_engine_internal.render_receipt.verifier import (
    VerifyError as ReceiptVerifyError,
)
from sum_engine_internal.render_receipt.verifier import verify_receipt as verify_render_receipt
from sum_engine_internal.transform_receipt.format import (
    SUPPORTED_SCHEMA as TRANSFORM_SCHEMA,
)
from sum_engine_internal.transform_receipt.verifier import verify_transform_receipt
from sum_verify._meaning import (
    SUPPORTED_SCHEMA as MEANING_RISK_SCHEMA,
)
from sum_verify._meaning import (
    MeaningReceiptDisclosureError,
    MeaningReceiptReplayError,
    verify_meaning_risk_receipt,
)

# Version of THIS verify surface + the wire formats it accepts. SemVer.
# Bump minor when a new supported schema is added; major on a
# backwards-incompatible change to an accepted format or the public API.
__version__ = "1.0.0"

SUPPORTED_SCHEMAS: tuple[str, ...] = (
    MEANING_RISK_SCHEMA,
    RENDER_SCHEMA,
    TRANSFORM_SCHEMA,
)

__all__ = [
    "__version__",
    "SUPPORTED_SCHEMAS",
    "verify",
    "verify_meaning_risk_receipt",
    "verify_render_receipt",
    "verify_transform_receipt",
    "UnsupportedSchemaError",
    "MeaningReceiptReplayError",
    "MeaningReceiptDisclosureError",
    "JoseEnvelopeError",
    "ReceiptVerifyError",
]


class UnsupportedSchemaError(ValueError):
    """Raised by :func:`verify` when the envelope's ``schema`` is not one
    of :data:`SUPPORTED_SCHEMAS`. Carries the offending schema so a caller
    can branch on it."""

    def __init__(self, schema: Any) -> None:
        self.schema = schema
        super().__init__(
            f"unsupported receipt schema {schema!r}; sum_verify handles "
            f"{', '.join(SUPPORTED_SCHEMAS)}"
        )


def verify(
    envelope: Any,
    jwks: Any,
    *,
    losses: Sequence[float] | None = None,
    max_age_seconds: int | None = None,
) -> Any:
    """Verify any supported SUM receipt, dispatched on its ``schema``.

    Returns the verified payload dict for a meaning-risk receipt, or the
    verifier's ``VerifyResult`` for a render / transform receipt (whose
    ``.payload`` carries the verified body). Raises the schema-specific
    error on failure (:class:`JoseEnvelopeError` /
    :class:`ReceiptVerifyError` for the cryptographic layer,
    :class:`MeaningReceiptReplayError` /
    :class:`MeaningReceiptDisclosureError` for meaning-risk replay), or
    :class:`UnsupportedSchemaError` for an unrecognised schema.

    ``losses`` is honoured only by the meaning-risk path (the other
    receipt types carry no replayable bound); it is ignored elsewhere.
    """
    schema = envelope.get("schema") if isinstance(envelope, dict) else None
    if schema == MEANING_RISK_SCHEMA:
        return verify_meaning_risk_receipt(
            envelope, jwks, losses=losses, max_age_seconds=max_age_seconds
        )
    if schema == RENDER_SCHEMA:
        return verify_render_receipt(envelope, jwks, max_age_seconds=max_age_seconds)
    if schema == TRANSFORM_SCHEMA:
        return verify_transform_receipt(
            envelope, jwks, max_age_seconds=max_age_seconds
        )
    raise UnsupportedSchemaError(schema)
