"""Local, deterministic, zero-$ entailment judges for ``EntailmentScorer``.

F18 (``docs/DOGFOOD_FINDINGS_2026-06-06.md``) showed the lexical scorer
does not just over-report a faithful paraphrase — it *misranks* it. The
fix is the ``EntailmentScorer`` driven by a real judge. The load-bearing
realisation: that judge does **not** need a paid API. A local
sentence-embedding model (``transformers``, run offline in eval mode) is
deterministic and fixes the misranking — empirically, on the F18 corpus,
the faithful paraphrase drops from 0.739 (lexical) to 0.200, below the
tag's 0.400, restoring a sensible ranking.

``EmbeddingJudge`` wraps a local mean-pooled sentence-embedding model as
an ``entails(premise, hypothesis) -> bool`` decision: the hypothesis is
"entailed" iff its embedding's maximum cosine similarity to any premise
*sentence* meets ``threshold``.

Honest boundary (this judge is a *similarity* proxy, named so):
  - It measures **semantic similarity, not strict directional
    entailment** — it can accept a fluent on-topic contradiction. The
    scorer it produces is named ``bidirectional-entailment[minilm-cosine-<t>]``
    so a receipt records exactly what judged it. For strict NLI, plug a
    local NLI cross-encoder through the same ``entails`` interface.
  - **Determinism / replay:** eval-mode + fixed weights is deterministic
    on a given machine, but floating-point ops can differ across hardware
    / library versions, so a boolean near ``threshold`` could flip
    cross-machine. This is fine for the *frontier* (a per-document
    MEASUREMENT) and for ranking. For a SIGNED ``sum.meaning_risk_receipt.v1``
    whose replay must reproduce cross-machine, treat a model judge's
    reproducibility as **machine-pinned** — a documented boundary, not a
    guarantee. (The signed-receipt path is separately operator-gated.)
  - ``transformers`` + ``torch`` are optional — the ``[judge]`` extra —
    and imported lazily, so the base install is unaffected.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sum_engine_internal.research.meaning.meaning_loss import (
    EntailmentScorer,
    _sentences,
)

# all-MiniLM-L6-v2 is a small (~80MB), widely-cached sentence-embedding
# model. A BERT-family model — we mean-pool its token embeddings rather
# than depend on the (version-fragile) sentence-transformers wrapper.
DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class EmbeddingJudge:
    """A local, deterministic ``entails`` judge backed by a mean-pooled
    sentence-embedding model. ``threshold`` is the cosine-similarity bar
    for "entailed". Model + tokenizer load lazily on first use."""

    threshold: float = 0.5
    model_id: str = DEFAULT_MODEL_ID
    _tok: Any = field(default=None, init=False, repr=False, compare=False)
    _mdl: Any = field(default=None, init=False, repr=False, compare=False)

    def _ensure_loaded(self) -> None:
        if self._mdl is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:  # pragma: no cover - environment-dependent
            raise ImportError(
                "EmbeddingJudge needs the [judge] extra (transformers + "
                "torch): pip install 'sum-engine[judge]'"
            ) from e
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._mdl = AutoModel.from_pretrained(self.model_id).eval()

    def _embed(self, texts: list[str]):
        import torch
        import torch.nn.functional as F

        self._ensure_loaded()
        with torch.no_grad():
            enc = self._tok(
                texts, padding=True, truncation=True, return_tensors="pt"
            )
            out = self._mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return F.normalize(pooled, p=2, dim=1)

    def entails(self, premise: str, hypothesis: str) -> bool:
        """True iff ``hypothesis`` is semantically covered by some sentence
        of ``premise`` (max cosine similarity ≥ ``threshold``)."""
        premise_sents = _sentences(premise) or [premise]
        if not hypothesis.strip():
            return False
        pe = self._embed(premise_sents)
        he = self._embed([hypothesis])
        return float((he @ pe.T).max()) >= self.threshold

    @property
    def name(self) -> str:
        return f"minilm-cosine-{self.threshold:g}"


def embedding_entailment_scorer(
    threshold: float = 0.5,
    model_id: str = DEFAULT_MODEL_ID,
) -> EntailmentScorer:
    """An ``EntailmentScorer`` driven by a local ``EmbeddingJudge`` — the
    zero-$, offline, deterministic, paraphrase-aware meaning-loss scorer
    that fixes F18. Lazily loads the model on first ``loss`` call."""
    judge = EmbeddingJudge(threshold=threshold, model_id=model_id)
    return EntailmentScorer(
        entails=judge.entails,
        judge_name=judge.name,
        judge_version="1",
    )
