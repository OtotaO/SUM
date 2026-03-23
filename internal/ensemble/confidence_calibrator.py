"""
Confidence Calibrator — Multi-Signal Automatic Confidence Scoring

Phase 24: Replaces the static 0.5 default with a research-informed
multi-signal calibration pipeline that runs at zero cost.

Signals:
    1. Source-type heuristic — URL domain → base confidence
    2. Redundancy boost — same axiom from N sources → +0.05 per source
    3. Contradiction penalty — conflicts with existing axioms → ×0.5
    4. Linguistic certainty — hedging words in source text → multiplier

Optional:
    4. LLM verbalized confidence — ask model to self-rate (opt-in)

Usage:
    calibrator = ConfidenceCalibrator()
    score = await calibrator.calibrate(
        axiom_key="earth||orbits||sun",
        source_url="https://nasa.gov/solar-system",
        ledger=ledger,
    )

Author: ototao
License: Apache License 2.0
"""

import re
import logging
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ─── Source-Type Confidence Map ───────────────────────────────────────
# Based on NAACL 2024 research on source credibility and calibration.
# These represent base confidence scores for different source types.

SOURCE_PATTERNS = [
    # (compiled regex for domain, base_confidence, label)
    # ── Specific domains first (before generic TLDs) ──
    (re.compile(r"arxiv\.org"), 0.88, "arxiv"),
    (re.compile(r"pubmed|ncbi\.nlm\.nih"), 0.88, "pubmed"),
    (re.compile(r"doi\.org"), 0.87, "doi"),
    (re.compile(r"(reuters|apnews|bbc)\."), 0.75, "news_wire"),
    (re.compile(r"(en\.)?wikipedia\.org"), 0.70, "wikipedia"),
    (re.compile(r"(medium\.com|substack|blogspot|wordpress)"), 0.40, "blog"),
    (re.compile(r"(reddit\.com|twitter\.com|x\.com)"), 0.35, "social"),
    # ── Generic TLDs last ──
    (re.compile(r"\.(edu|ac\.\w+)$"), 0.90, "academic"),
    (re.compile(r"\.(gov|mil)$"), 0.85, "government"),
    (re.compile(r"\.(org)$"), 0.65, "organization"),
    (re.compile(r"\.(com|net|io)$"), 0.50, "commercial"),
]

DEFAULT_CONFIDENCE = 0.50
REDUNDANCY_BOOST = 0.05
REDUNDANCY_CAP = 1.0
CONTRADICTION_PENALTY = 0.5


class ConfidenceCalibrator:
    """Multi-signal automatic confidence scoring engine.

    Computes a calibrated confidence score for an axiom based on:
    1. Source URL domain classification
    2. Redundancy across independent sources
    3. Contradiction detection against existing state
    4. Linguistic certainty (hedging markers in source text)
    """

    def source_type_score(self, source_url: str) -> float:
        """Score confidence based on the URL's domain type.

        Returns a base confidence in [0.0, 1.0] based on domain
        pattern matching against known source categories.
        """
        if not source_url:
            return DEFAULT_CONFIDENCE

        try:
            parsed = urlparse(source_url)
            domain = parsed.netloc.lower() or parsed.path.lower()
        except Exception:
            return DEFAULT_CONFIDENCE

        for pattern, score, label in SOURCE_PATTERNS:
            if pattern.search(domain):
                logger.debug(
                    "Source %s classified as %s (confidence=%.2f)",
                    domain, label, score,
                )
                return score

        return DEFAULT_CONFIDENCE

    async def redundancy_boost(
        self, axiom_key: str, ledger
    ) -> float:
        """Boost confidence if the axiom has been ingested from multiple sources.

        Each independent ingestion adds +0.05, capped at 1.0.
        Returns the additive boost (not the total score).
        """
        if not ledger:
            return 0.0

        try:
            chain = await ledger.get_axiom_provenance(axiom_key)
            # Count unique sources
            unique_sources = set()
            for entry in chain:
                src = entry.get("source_url", "")
                if src:
                    unique_sources.add(src)

            n = len(unique_sources)
            if n <= 1:
                return 0.0
            # Boost for each additional source beyond the first
            return min((n - 1) * REDUNDANCY_BOOST, 0.5)
        except Exception as e:
            logger.warning("Redundancy check failed: %s", e)
            return 0.0

    def contradiction_penalty(
        self,
        axiom_key: str,
        current_state: int,
        algebra,
    ) -> float:
        """Apply a penalty if the axiom contradicts existing knowledge.

        Returns a multiplier in (0.0, 1.0]. 1.0 = no contradiction.
        """
        if current_state <= 1 or not algebra:
            return 1.0

        # Check if axiom contains a predicate that might conflict
        parts = axiom_key.split("||")
        if len(parts) != 3:
            return 1.0

        subject, predicate, obj = parts

        # Look for conflicting axioms: same subject+predicate, different object
        try:
            active_axioms = algebra.get_active_axioms(current_state)
            for existing in active_axioms:
                ex_parts = existing.split("||")
                if len(ex_parts) == 3:
                    ex_s, ex_p, ex_o = ex_parts
                    if (ex_s == subject and ex_p == predicate
                            and ex_o != obj):
                        logger.info(
                            "Contradiction detected: '%s' vs existing '%s'",
                            axiom_key, existing,
                        )
                        return CONTRADICTION_PENALTY
        except Exception as e:
            logger.warning("Contradiction check failed: %s", e)

        return 1.0

    async def calibrate(
        self,
        axiom_key: str,
        source_url: str = "",
        current_state: int = 1,
        algebra=None,
        ledger=None,
        manual_confidence: Optional[float] = None,
        linguistic_certainty: float = 1.0,
    ) -> float:
        """Compute calibrated confidence using all available signals.

        Pipeline:
            1. Start with source-type base score
            2. Add redundancy boost (if ledger available)
            3. Apply contradiction penalty (if algebra + state available)
            4. Apply linguistic certainty multiplier (if < 1.0)
            5. Clamp to [0.0, 1.0]

        If manual_confidence is provided, it is used as-is (no calibration).

        Args:
            linguistic_certainty: Output from detect_hedging() in
                syntactic_sieve.py. 1.0 = definite, <1.0 = hedged.
                This is a metadata-only signal.

        Returns:
            float: Calibrated confidence score in [0.0, 1.0]
        """
        if manual_confidence is not None:
            return max(0.0, min(1.0, manual_confidence))

        # Signal 1: Source type
        base = self.source_type_score(source_url)

        # Signal 2: Redundancy boost
        boost = await self.redundancy_boost(axiom_key, ledger)

        # Signal 3: Contradiction penalty
        penalty = self.contradiction_penalty(
            axiom_key, current_state, algebra
        )

        # Signal 4: Linguistic certainty
        ling = max(0.0, min(1.0, linguistic_certainty))

        # Combine: (base + redundancy_boost) × contradiction × linguistic
        score = (base + boost) * penalty * ling

        # Clamp
        return max(0.0, min(REDUNDANCY_CAP, score))
