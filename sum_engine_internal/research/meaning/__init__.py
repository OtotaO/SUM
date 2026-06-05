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

Together they form the first verifiable certificate over a meaning-space
(not token-space) loss. See ``docs/MEANING_LOSS_FRONTIER.md`` for the
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
    MeaningRiskGuarantee,
    certify_meaning_risk,
    empirical_risk_coverage,
)
from sum_engine_internal.research.meaning.receipt import (
    DEFAULT_NOT_COVERED,
    SUPPORTED_SCHEMA,
    MeaningReceiptReplayError,
    build_payload,
    losses_hash,
    sign_meaning_risk_receipt,
    verify_meaning_risk_receipt,
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
    # receipt
    "SUPPORTED_SCHEMA",
    "DEFAULT_NOT_COVERED",
    "MeaningReceiptReplayError",
    "build_payload",
    "losses_hash",
    "sign_meaning_risk_receipt",
    "verify_meaning_risk_receipt",
]
