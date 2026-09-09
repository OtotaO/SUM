"""Frozen compatibility for replaying pre-2026-09-09 research fixtures only.

Current issuance must use the research builders directly. Historical fixture
generators explicitly call this adapter so stronger new disclosure and evidence
defaults do not silently change the bytes of already signed evidence. The old
sampling wording below is historical, not valid current issuance guidance; see
docs/MEANING_RISK_RECEIPT_FORMAT.md for its erratum.
"""

HISTORICAL_MEANING_DISCLOSURE = (
    "This certificate bounds a NAMED PROXY for meaning-loss, not meaning "
    "itself. The bound is marginal (the average over the calibration "
    "corpus), not per-document, and is valid only under exchangeability "
    "between that corpus and deployment. It does not cover arrangement "
    "(naẓm), sound, connotation, or implicature."
)

HISTORICAL_CHAIN_DISCLOSURE = (
    "This certificate binds an ordered chain of per-hop meaning-risk "
    "certificates and their composed additive budget. Every per-hop "
    "caveat applies unchanged: each bound is over a NAMED PROXY for "
    "meaning-loss, marginal over its calibration corpus, valid under "
    "exchangeability. Composition adds no new knowledge about any "
    "single document's fate across the chain."
)


def historical_fixture_payload(payload: dict, *, disclosure: str | None = None) -> dict:
    """Restore only the historical fixture wire profile, never new issuance."""
    if payload.get("signed_at", "") >= "2026-09-09" or not payload.get("signed_at"):
        raise ValueError("historical fixture adapter requires a pre-correction signed_at")
    result = dict(payload)
    for field in ("statistical_scope", "sampling_status", "evaluation_evidence_status"):
        result.pop(field, None)
    if any(field in result for field in ("sampling_contract", "evaluation_manifest")):
        raise ValueError("historical fixture adapter cannot discard supplied evidence")
    if disclosure is not None:
        result["disclosure"] = disclosure
    return result
