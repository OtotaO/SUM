"""ShareableRender — self-contained, verifier-friendly JSON snapshot
of a transform run.

A daily-use friction killer: today, dragging the slider on the live
demo produces a render that's ephemeral — refresh kills it. T5
introduces a stable, content-addressed JSON snapshot that captures
everything a downstream verifier needs to reproduce + verify the
render:

    {
      "schema": "sum.shareable_render.v1",
      "transform": "slider" | "extract" | "compose" | ...,
      "input": <transform's input shape>,
      "parameters": <transform's parameters shape>,
      "output": <transform's output shape>,
      "receipt": <sum.transform_receipt.v1 envelope>,
      "created_at": <ISO-8601 UTC>
    }

The receipt's ``transform_id`` is the natural filename / URL slug
for the share. Two shares of the same render produce byte-identical
JSON (modulo ``created_at``) because every component is canonicalised.

v0 scope (this PR, T5):
  - Python-side ShareableRender dataclass + round-trip JSON.
  - Helper to verify a share against a JWKS: signature check on the
    embedded receipt + integrity checks on the embedded input /
    output / parameters against the receipt's hashes.

Deferred (T5b / Worker side):
  - HTTP surface: ``POST /api/share`` to store a render; ``GET
    /share/<transform_id>`` to retrieve. KV-backed; content-addressed.
  - Demo UI: "Share this render" button after a successful render.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


SUPPORTED_SCHEMA = "sum.shareable_render.v1"


@dataclass(frozen=True)
class ShareableRender:
    """A self-contained, verifier-friendly snapshot of a transform run."""
    transform: str
    input: Any
    parameters: dict[str, Any]
    output: Any
    receipt: dict[str, Any]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )[:-4] + "Z"  # millisecond precision
    )
    schema: str = SUPPORTED_SCHEMA

    @property
    def share_id(self) -> str:
        """Stable content-addressed identifier for this share.
        Equals the receipt's ``transform_id`` when present; falls
        back to a hash of (transform, input, parameters, output)
        when no receipt is embedded (e.g. an unsigned local share)."""
        receipt_id = (self.receipt or {}).get("payload", {}).get("transform_id")
        if receipt_id:
            return receipt_id
        # Fallback: hash the four signed-equivalent fields.
        h = hashlib.sha256()
        for chunk in (
            self.transform,
            json.dumps(self.input, sort_keys=True, separators=(",", ":")),
            json.dumps(self.parameters, sort_keys=True, separators=(",", ":")),
            json.dumps(self.output, sort_keys=True, separators=(",", ":")),
        ):
            h.update(chunk.encode("utf-8"))
            h.update(b"|")
        return h.hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSON serialisation. Schema-prefixed so a
        downstream consumer can dispatch on ``schema``."""
        d = asdict(self)
        # Move `schema` to first position for readability when
        # serialised (JCS will re-sort; this only affects raw JSON).
        return {
            "schema": d["schema"],
            "transform": d["transform"],
            "input": d["input"],
            "parameters": d["parameters"],
            "output": d["output"],
            "receipt": d["receipt"],
            "created_at": d["created_at"],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialise to a JSON string. Indent=2 by default for
        human-readable share files (the demo will paste these).
        Pass indent=None for compact form (KV storage)."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShareableRender":
        """Construct from a dict (typically parsed JSON). Schema is
        validated; unknown schemas raise ValueError."""
        if d.get("schema") != SUPPORTED_SCHEMA:
            raise ValueError(
                f"unsupported share schema: {d.get('schema')!r}; "
                f"expected {SUPPORTED_SCHEMA!r}"
            )
        required = ("transform", "input", "parameters", "output", "receipt")
        for k in required:
            if k not in d:
                raise ValueError(f"share missing required field: {k!r}")
        return cls(
            transform=d["transform"],
            input=d["input"],
            parameters=d["parameters"],
            output=d["output"],
            receipt=d["receipt"],
            created_at=d.get("created_at", ""),
        )

    @classmethod
    def from_json(cls, s: str) -> "ShareableRender":
        return cls.from_dict(json.loads(s))


@dataclass(frozen=True)
class ShareVerifyResult:
    """Result of verifying a ShareableRender. ``signature_verified``
    is True when the embedded receipt's signature checks against
    the supplied JWKS. ``integrity_checks`` is a dict mapping each
    optional integrity check name to True/False; the application-
    layer recomputed hashes are compared to the receipt's signed
    hashes.

    A consumer that wants only "is this signed correctly" reads
    ``signature_verified``. A consumer that wants "and does the
    embedded input/output/parameters match the receipt" reads the
    full ``integrity_checks`` dict.
    """
    signature_verified: bool
    integrity_checks: dict[str, bool]
    receipt_kid: str | None = None
    receipt_signed_at: str | None = None


def verify_share(
    share: ShareableRender,
    jwks: dict[str, Any],
) -> ShareVerifyResult:
    """Verify a ShareableRender end-to-end.

    Two layers of check:

      1. The receipt's JWS signature against the supplied JWKS.
         Raises ``transform_receipt.VerifyError`` on failure (signature
         invalid, unknown kid, schema unknown, etc.).

      2. Application-layer integrity: recompute parameters_hash /
         input_hash / output_hash from the share's embedded fields
         and compare to the receipt's signed values. Each check is
         reported in ``integrity_checks``.

    Returns a ``ShareVerifyResult`` describing both layers. Callers
    that want strict acceptance check ``signature_verified is True
    and all(integrity_checks.values())``.
    """
    from sum_engine_internal.transform_receipt import (
        canonical_hash,
        verify_transform_receipt,
    )
    from sum_engine_internal.transforms import get_transform

    # Layer 1: signature.
    verify_result = verify_transform_receipt(share.receipt, jwks)
    sig_ok = verify_result.verified
    payload = share.receipt.get("payload", {})

    # Layer 2: per-field integrity checks. Recompute each hash from
    # the share's embedded fields using the transform's own
    # canonicalisers, then compare to the receipt's signed values.
    integrity: dict[str, bool] = {}
    try:
        transform = get_transform(share.transform)
    except KeyError:
        # Receipt's transform name doesn't match any registered
        # transform — could be a future transform a forward-compat
        # verifier should accept-with-warning. Mark integrity
        # checks as inconclusive (False on all).
        integrity = {
            "transform_known": False,
        }
        return ShareVerifyResult(
            signature_verified=sig_ok,
            integrity_checks=integrity,
            receipt_kid=verify_result.kid,
            receipt_signed_at=payload.get("signed_at"),
        )

    expected_params_hash = canonical_hash(
        transform.canonicalize_parameters(share.parameters)
    )
    integrity["parameters_hash"] = (
        expected_params_hash == payload.get("parameters_hash")
    )

    expected_input_hash = canonical_hash(
        transform.canonicalize_input(share.input)
    )
    integrity["input_hash"] = (
        expected_input_hash == payload.get("input_hash")
    )

    expected_output_hash = canonical_hash(
        transform.canonicalize_output(share.output)
    )
    integrity["output_hash"] = (
        expected_output_hash == payload.get("output_hash")
    )

    return ShareVerifyResult(
        signature_verified=sig_ok,
        integrity_checks=integrity,
        receipt_kid=verify_result.kid,
        receipt_signed_at=payload.get("signed_at"),
    )
