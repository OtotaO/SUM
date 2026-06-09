"""sum_engine_internal.research.meaning — measuring the loss of meaning.

The frontier *below* fact-preservation. SUM's slider contract certifies
whether (subject, predicate, object) triples survive a transform; this
package opens the layer beneath the proposition — the meaning a reader
reconstructs from what was compressed — and does so under the same
discipline the rest of the repo holds: every number is a *named,
versioned proxy*, every guarantee is *distribution-free and marginal*,
and every claim states what it does **not** cover (arrangement / *naẓm*,
sound, connotation, implicature).

The three composable pieces:

  - ``meaning_loss`` — named bounded proxies for per-pair meaning-loss.
    ``LexicalCoverageScorer`` (deterministic, dependency-free, a runnable
    placeholder) and ``EntailmentScorer`` (the real path: bidirectional-
    entailment loss over an injected NLI judge, in the spirit of
    semantic-entropy meaning-clustering, Farquhar et al., *Nature* 2024).

  - ``conformal_meaning`` — the dual of the slider's rate kernel: a
    distribution-free *upper* bound on expected meaning-loss
    (``certify_meaning_risk``), reusing the adversarially-hardened
    Hoeffding / Clopper–Pearson bounds.

  - ``receipt`` — a signed, **same-commit-replayable** certificate over
    that bound (``sum.meaning_risk_receipt.v1``), reusing the existing
    JCS + Ed25519 + detached-JWS trust stack and adding a replay anchor
    so a third party can reproduce the bound byte-for-byte.

Together they form a signed, same-commit-replayable certificate that
bounds a *named proxy* for meaning-loss — computed in **checkable text
space** (entailment-between-texts or lexical coverage), not from model
internals — composing a distribution-free proxy bound with a replayable
receipt. The certificate does not measure meaning; the proxy is named so
a reader always knows what was bounded. We are not aware of a prior
artifact combining a distribution-free meaning-loss-proxy bound with a
replayable signed receipt. See ``docs/MEANING_LOSS_FRONTIER.md`` for the
chart of the whole frontier and ``docs/MEANING_RISK_RECEIPT_FORMAT.md``
for the wire spec.

Stability: research-grade (behind the ``[research]`` extra). APIs may
change between minor releases.
"""
from sum_engine_internal.research.meaning.meaning_loss import (
    EntailmentScorer,
    LexicalCoverageScorer,
    MeaningScorer,
    score_pairs,
)
from sum_engine_internal.research.meaning.conformal_meaning import (
    GroupedMeaningRisk,
    MeaningRiskGuarantee,
    certify_meaning_risk,
    certify_meaning_risk_by_group,
    empirical_risk_coverage,
)
from sum_engine_internal.research.meaning.receipt import (
    DEFAULT_NOT_COVERED,
    SUPPORTED_SCHEMA,
    MeaningReceiptDisclosureError,
    MeaningReceiptReplayError,
    build_payload,
    losses_hash,
    sign_meaning_risk_receipt,
    verify_meaning_risk_receipt,
)
from sum_engine_internal.research.meaning.perspective_receipt import (
    SUPPORTED_SCHEMA as PERSPECTIVE_SCHEMA,
    build_perspective_payload,
    evidence_hash,
    sign_perspective_risk_receipt,
    verify_perspective_risk_receipt,
)
from sum_engine_internal.research.meaning.drift_budget import (
    AdditiveAuditResult,
    CertifiedDriftBudget,
    ChainDriftReadout,
    HopLoss,
    audit_additive_vs_end_to_end,
    compose_drift_budget,
    compose_drift_budget_from_payloads,
    measure_chain_drift,
)

__all__ = [
    # meaning_loss
    "MeaningScorer",
    "LexicalCoverageScorer",
    "EntailmentScorer",
    "score_pairs",
    # conformal_meaning
    "MeaningRiskGuarantee",
    "certify_meaning_risk",
    "empirical_risk_coverage",
    "GroupedMeaningRisk",
    "certify_meaning_risk_by_group",
    # receipt
    "SUPPORTED_SCHEMA",
    "DEFAULT_NOT_COVERED",
    "MeaningReceiptReplayError",
    "MeaningReceiptDisclosureError",
    "build_payload",
    "losses_hash",
    "sign_meaning_risk_receipt",
    "verify_meaning_risk_receipt",
    # perspective_receipt
    "PERSPECTIVE_SCHEMA",
    "build_perspective_payload",
    "evidence_hash",
    "sign_perspective_risk_receipt",
    "verify_perspective_risk_receipt",
    # drift_budget — multi-hop composition
    "HopLoss",
    "ChainDriftReadout",
    "measure_chain_drift",
    "CertifiedDriftBudget",
    "compose_drift_budget",
    "compose_drift_budget_from_payloads",
    "AdditiveAuditResult",
    "audit_additive_vs_end_to_end",
]
