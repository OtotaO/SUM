"""Wire format + sign/verify for ``sum.meaning_risk_receipt.v1``.

Companion to ``docs/MEANING_RISK_RECEIPT_FORMAT.md``. A meaning-risk
receipt is a signed, replayable certificate that the *expected
meaning-loss* of a transform — measured by a named proxy, bounded
distribution-free — does not exceed a stated ceiling.

It reuses the existing trust stack wholesale:

  - ``infrastructure.jcs.canonicalize``     — RFC 8785 canonical bytes
  - ``infrastructure.jose_envelope``         — Ed25519 / detached-JWS sign + verify
  - ``research.meaning.conformal_meaning``   — the certified bound itself

and adds exactly one new thing the other receipts don't have: a
**replay anchor**. The payload commits ``losses_hash`` — the JCS hash of
the rounded per-pair loss vector — so a verifier handed the same loss
vector side-band can (a) confirm it matches the hash, then (b) re-run
``certify_meaning_risk`` and confirm the bound reproduces. Because the
certifier is deterministic in the losses, this is byte-exact on the same
commit. That replay property is what turns "measured" into
"measured *and* independently reproducible" — the gap the slider
contract's T2/T3 names.

What a verified meaning-risk receipt proves — and does NOT
----------------------------------------------------------
PROVES (cryptographically + by replay):
  - the payload was signed by the holder of ``kid``'s private key;
  - the committed losses hash to ``losses_hash``;
  - re-running the named certifier on those losses reproduces
    ``risk_upper_bound`` at the stated ``delta`` and ``method``.

Does NOT prove:
  - that meaning was preserved — only that a *named proxy* for
    meaning-loss is bounded *on average* (marginally) under
    *exchangeability* with the calibration corpus;
  - anything about arrangement (*naẓm*), sound, connotation, or
    implicature — those layers are explicitly outside the proxy and the
    payload's ``not_covered`` field says so.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from sum_engine_internal.infrastructure.jcs import canonicalize
from sum_engine_internal.infrastructure.jose_envelope import (
    sign_jose_envelope,
    verify_jose_envelope,
)
from sum_engine_internal.research.meaning.conformal_meaning import (
    MeaningRiskGuarantee,
    certify_meaning_risk,
)

SUPPORTED_SCHEMA = "sum.meaning_risk_receipt.v1"

# Per-pair losses are rounded to this many decimals before hashing AND
# before re-certification, so the replay anchor is float-stable across
# runtimes (the same gotcha the render-receipt format documents for
# integer-vs-float zero). 6 decimals is far finer than any proxy's
# meaningful resolution.
_LOSS_DECIMALS = 6

# The layers a proxy-based meaning-loss bound structurally cannot cover.
# Shipped as a constant so every receipt declares the same boundary and
# a verifier can assert it is present.
DEFAULT_NOT_COVERED: tuple[str, ...] = (
    "arrangement",   # naẓm — meaning from word order / structure
    "sound",         # prosody, meter, rhyme, phonetic texture
    "connotation",   # tone, register, emotional colour
    "implicature",   # what is conveyed without being asserted
)

_DEFAULT_DISCLOSURE = (
    "This certificate bounds a NAMED PROXY for meaning-loss, not meaning "
    "itself. The bound is marginal (the average over the calibration "
    "corpus), not per-document, and is valid only under exchangeability "
    "between that corpus and deployment. It does not cover arrangement "
    "(naẓm), sound, connotation, or implicature."
)


# Rate / probability quantities cross the wire as INTEGER micro-units
# (1e-6 resolution), never as floats. SUM's Node JCS canonicaliser
# rejects floats outright — cross-runtime float formatting is the
# integer-vs-float-zero hazard the render-receipt format warns about, so
# a float-bearing payload would not be cross-runtime verifiable. Integers
# canonicalise byte-identically in every runtime. _MICRO_SCALE matches
# the old 6-decimal rounding exactly (1e6 = 10 ** _LOSS_DECIMALS).
_MICRO_SCALE = 10 ** _LOSS_DECIMALS


def _to_micro(x: float) -> int:
    """Quantise a value in [0, 1] to integer micro-units: round(x * 1e6).
    The single float→wire boundary; the result is JCS-safe everywhere."""
    return int(round(float(x) * _MICRO_SCALE))


def _from_micro(m: int) -> float:
    """Inverse of :func:`_to_micro` for verify-side recompute. The float
    lives only in the conformal layer, never on the wire."""
    return int(m) / _MICRO_SCALE


def _losses_micro(losses: Sequence[float]) -> list[int]:
    """Per-pair losses as integer micro-units — the vector ``losses_hash``
    commits and the certifier-replay compares. Float-free, so the anchor
    is byte-stable across runtimes."""
    return [_to_micro(x) for x in losses]


def _quantized(losses: Sequence[float]) -> list[float]:
    """The float vector corresponding EXACTLY to the committed micro
    ints — what the certifier replays over, so ``build_payload`` and
    ``verify`` agree on the bound to the last micro-unit."""
    return [_from_micro(m) for m in _losses_micro(losses)]


def losses_hash(losses: Sequence[float]) -> str:
    """``"sha256-<hex>"`` over the JCS-canonical bytes of the INTEGER
    micro-unit loss vector. The replay anchor committed in the payload."""
    import hashlib

    return "sha256-" + hashlib.sha256(
        canonicalize(_losses_micro(losses))
    ).hexdigest()


def build_payload(
    *,
    guarantee: MeaningRiskGuarantee,
    losses: Sequence[float],
    corpus_id: str,
    transform: str,
    alpha_target: float | None = None,
    loss_definition: str,
    not_covered: Sequence[str] = DEFAULT_NOT_COVERED,
    disclosure: str = _DEFAULT_DISCLOSURE,
    signed_at: str | None = None,
) -> dict[str, Any]:
    """Assemble a ``sum.meaning_risk_receipt.v1`` payload from a
    certified guarantee and the losses it was certified over.

    ``corpus_id`` names the calibration envelope (e.g.
    ``"abrahamic-parallel-translations-v0"``) — the exchangeability
    scope the bound is valid within. ``transform`` is a free string
    naming what produced the pairs (e.g. ``"slider:density=0.5"``).
    ``alpha_target`` is the risk level the operator wanted controlled;
    when supplied, ``controlled`` records whether the certified ceiling
    met it. ``loss_definition`` is a one-line human description of what
    the proxy's [0, 1] number means (required — it is the semantics a
    reader needs). ``not_covered`` must be non-empty — declaring the
    proxy's structural blind spots is the honesty contract the receipt
    exists to keep (an empty list would let a receipt silently imply it
    covers arrangement / sound / connotation / implicature).

    The bound written into the payload is **(re)computed over the
    quantised committed loss vector** (``_quantized`` — the integer
    micro-units the hash commits), not the raw one. This is the
    load-bearing replay-determinism fix: the producer's
    bound and a verifier's replay are then computed over byte-identical
    inputs — the exact vector ``losses_hash`` commits — so Stage B
    reproduces the bound on the same commit even for raw losses that
    straddle the rounding boundary. The re-certification also re-rejects
    non-finite / out-of-[0,1] losses (via ``certify_meaning_risk``)
    before anything is signed, so a corrupted side-band vector cannot be
    committed. The passed ``guarantee`` supplies the scorer identity and
    certification parameters (delta / method / n).

    ``signed_at`` defaults to current UTC, stamped in the exact
    millisecond-precision ISO-8601 + ``Z`` shape the transform-receipt
    format uses, so timestamps are byte-identical across runtimes.
    """
    if not not_covered:
        raise ValueError(
            "not_covered must be non-empty — the proxy's structural "
            "blind spots (arrangement / sound / connotation / "
            "implicature) must be declared, not left implicit"
        )
    if guarantee.n != len(losses):
        raise ValueError(
            f"guarantee.n={guarantee.n} disagrees with len(losses)="
            f"{len(losses)}; the certificate must be over exactly the "
            f"committed losses"
        )
    if signed_at is None:
        now = datetime.now(timezone.utc)
        signed_at = (
            now.strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{now.microsecond // 1000:03d}Z"
        )

    # Re-certify over the EXACT quantised inputs the wire commits, so the
    # producer's bound replays byte-for-byte. The loss vector is quantised
    # via _quantized; `delta` and `alpha_target` are SCALARS that also
    # cross the wire as micro-units, so they must be quantised here too —
    # verify recomputes them as _from_micro(*_micro), and build must
    # certify at the identical value or an off-grid delta (e.g. a
    # Bonferroni-corrected threshold, 1/30) shifts the bound by ~1 micro
    # and false-rejects a genuine receipt. Also re-validates the losses
    # (certify rejects NaN/inf/out-of-range) before signing.
    rounded = _quantized(losses)
    delta_q = _from_micro(_to_micro(guarantee.delta))
    canonical = certify_meaning_risk(
        rounded,
        scorer_name=guarantee.scorer_name,
        scorer_version=guarantee.scorer_version,
        delta=delta_q,
        method=guarantee.method,
    )

    # Every value is int | str | bool | list[str] — float-free, so the
    # payload canonicalises byte-identically in Python and Node. The four
    # rate/probability quantities ride as integer micro-units (see
    # _to_micro); n is a count, controlled a bool, the rest strings.
    payload: dict[str, Any] = {
        "scorer": guarantee.scorer_name,
        "scorer_version": guarantee.scorer_version,
        "loss_definition": loss_definition,
        "n": guarantee.n,
        "method": guarantee.method,
        "delta_micro": _to_micro(guarantee.delta),
        "point_estimate_micro": _to_micro(canonical.point_estimate),
        "risk_upper_bound_micro": _to_micro(canonical.risk_upper_bound),
        "losses_hash": losses_hash(losses),
        "corpus_id": corpus_id,
        "transform": transform,
        "not_covered": list(not_covered),
        "disclosure": disclosure,
        "signed_at": signed_at,
    }
    if alpha_target is not None:
        # Evaluate `controlled` against the QUANTISED alpha (the value
        # verify recomputes from alpha_target_micro), not the raw one —
        # an off-grid alpha straddling the bound would otherwise flip the
        # flag between build and verify.
        alpha_q = _from_micro(_to_micro(alpha_target))
        payload["alpha_target_micro"] = _to_micro(alpha_target)
        payload["controlled"] = bool(canonical.controls(alpha_q))
    return payload


def sign_meaning_risk_receipt(
    payload: dict[str, Any],
    *,
    private_jwk: dict[str, Any],
    kid: str,
) -> dict[str, Any]:
    """Produce a signed ``sum.meaning_risk_receipt.v1`` envelope from a
    payload built by ``build_payload``. Returns the four-key envelope
    ``{schema, kid, payload, jws}``."""
    envelope = sign_jose_envelope(payload, private_jwk=private_jwk, kid=kid)
    envelope["schema"] = SUPPORTED_SCHEMA
    return envelope


class MeaningReceiptReplayError(Exception):
    """Raised when a meaning-risk receipt is cryptographically valid but
    the supplied losses do not reproduce its committed hash, bound, or a
    field derived from them (``risk_upper_bound_micro``,
    ``point_estimate_micro``, ``n``, ``controlled``). Distinct from a
    signature failure: the
    receipt is genuine, but the side-band evidence does not back its
    claim."""


class MeaningReceiptDisclosureError(Exception):
    """Raised when a meaning-risk receipt is cryptographically valid but
    omits the disclosure fields the format requires (``not_covered``
    non-empty, ``disclosure`` non-empty). The receipt exists to bound a
    *named proxy* while declaring what it cannot cover; a receipt that
    passes every cryptographic check yet discloses nothing reads as a
    bare bound — the exact 'reads as a meaning guarantee' failure the
    module exists to prevent. Structural, so it is enforced on every
    verify, with or without side-band losses."""


def verify_meaning_risk_receipt(
    envelope: Any,
    jwks: Any,
    *,
    losses: Sequence[float] | None = None,
    max_age_seconds: int | None = None,
) -> dict[str, Any]:
    """Verify a ``sum.meaning_risk_receipt.v1`` envelope.

    Always performs the full JOSE verification (signature, schema, header
    invariants) via ``verify_jose_envelope``. When ``losses`` are
    supplied side-band, ALSO performs the replay check:

      1. confirm ``losses_hash(losses)`` equals ``payload.losses_hash``;
      2. re-run ``certify_meaning_risk`` on the quantised losses at the
         payload's ``delta_micro`` (re-floated) and ``method``;
      3. confirm the reproduced ``risk_upper_bound_micro`` and
         ``point_estimate_micro`` equal the payload's — exact integer
         equality, no epsilon.

    Returns the verified payload dict on success.

    Raises
    ------
    JoseEnvelopeError
        On any cryptographic / structural failure (propagated).
    MeaningReceiptDisclosureError
        When the payload omits a required disclosure field
        (``not_covered`` non-empty, ``disclosure`` non-empty). Enforced
        on every verify, with or without ``losses``.
    MeaningReceiptReplayError
        When ``losses`` are supplied but fail the hash, bound,
        ``point_estimate``, ``n``, or ``controlled`` replay.
    """
    result = verify_jose_envelope(
        envelope,
        jwks,
        supported_schema=SUPPORTED_SCHEMA,
        max_age_seconds=max_age_seconds,
    )
    payload = result.payload

    # ---- structural disclosure invariants (always, losses or not) ----
    # The receipt's whole purpose is to bound a NAMED proxy while
    # declaring what it cannot cover. A signed-but-disclosure-free
    # receipt reads as a bare bound; reject it.
    not_covered = payload.get("not_covered")
    if not isinstance(not_covered, list) or not not_covered:
        raise MeaningReceiptDisclosureError(
            "payload.not_covered must be a non-empty list declaring the "
            f"proxy's structural blind spots; got {not_covered!r}"
        )
    disclosure = payload.get("disclosure")
    if not isinstance(disclosure, str) or not disclosure.strip():
        raise MeaningReceiptDisclosureError(
            "payload.disclosure must be a non-empty string stating the "
            f"proxy / marginal / exchangeability caveat; got {disclosure!r}"
        )

    if losses is None:
        return payload

    # ---- replay step 1: hash anchor ----
    recomputed_hash = losses_hash(losses)
    if recomputed_hash != payload.get("losses_hash"):
        raise MeaningReceiptReplayError(
            f"losses_hash mismatch: supplied losses hash to "
            f"{recomputed_hash} but receipt commits "
            f"{payload.get('losses_hash')!r}"
        )

    # ---- replay step 2: re-certify over the ROUNDED committed vector ----
    # Same vector build_payload certified over (and the hash commits),
    # so the bound reproduces byte-for-byte. ValueError here means the
    # committed losses are not valid [0,1] data — surface it as a replay
    # failure, not a bare exception (the receipt's contract).
    rounded = _quantized(losses)
    try:
        replay = certify_meaning_risk(
            rounded,
            scorer_name=str(payload.get("scorer", "")),
            scorer_version=str(payload.get("scorer_version", "")),
            delta=_from_micro(int(payload["delta_micro"])),
            method=payload["method"],
        )
    except ValueError as e:
        raise MeaningReceiptReplayError(
            f"committed losses are not valid [0,1] data: {e}"
        ) from e

    # ---- replay step 3: bound + point estimate match (integer micro) ----
    # Exact integer equality — no epsilon, no float rounding-mode argument.
    want_ub = int(payload["risk_upper_bound_micro"])
    got_ub = _to_micro(replay.risk_upper_bound)
    if want_ub != got_ub:
        raise MeaningReceiptReplayError(
            f"risk_upper_bound does not replay: receipt claims {want_ub} "
            f"micro but re-certification yields {got_ub} micro"
        )
    want_pe = int(payload["point_estimate_micro"])
    got_pe = _to_micro(replay.point_estimate)
    if want_pe != got_pe:
        raise MeaningReceiptReplayError(
            f"point_estimate does not replay: receipt claims {want_pe} "
            f"micro but re-certification yields {got_pe} micro"
        )

    # ---- replay step 4: sample size must match the committed losses ----
    # n is the field a reader uses to judge finite-sample confidence;
    # an inflated n misrepresents calibration size even when the bound
    # is honest. replay.n == len(rounded) == len(losses) by construction.
    if int(payload["n"]) != replay.n:
        raise MeaningReceiptReplayError(
            f"n does not replay: receipt claims n={payload['n']} but the "
            f"committed losses contain {replay.n} samples"
        )

    # ---- replay step 5: the operational pass/fail decision ----
    # `controlled` is the field a consumer acts on. It is a pure
    # function of (risk_upper_bound, alpha_target) — recompute it from
    # the replayed bound so a flipped flag cannot ride a valid signature.
    if "alpha_target_micro" in payload:
        alpha = _from_micro(int(payload["alpha_target_micro"]))
        expected_controlled = replay.controls(alpha)
        if bool(payload.get("controlled")) != expected_controlled:
            raise MeaningReceiptReplayError(
                f"controlled does not replay: receipt claims "
                f"controlled={payload.get('controlled')} at "
                f"alpha_target_micro={payload['alpha_target_micro']}, but "
                f"the replayed bound {_to_micro(replay.risk_upper_bound)} "
                f"micro gives controlled={expected_controlled}"
            )

    return payload
