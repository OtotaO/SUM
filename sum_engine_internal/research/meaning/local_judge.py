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
    so a receipt records exactly what judged it. For strict directional
    entailment that *also catches a fluent contradiction*, use
    ``NLIJudge`` / ``nli_entailment_scorer`` below — the stronger judge.
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


# A local NLI cross-encoder — the *real* directional entailment judge, a
# strict upgrade over the embedding-similarity proxy. Default: an
# ANLI-trained DeBERTa-v3 (MIT, ~184MB), which the literature (MENLI,
# arXiv:2208.07316) and our own probe confirm does what cosine cannot —
# credit a faithful paraphrase as ENTAILMENT *and* flag a fluent on-topic
# contradiction as CONTRADICTION. This is "what our peer does, and better":
# AEX (arXiv:2603.14283) disclaims rewriting-robustness; a real NLI judge
# is exactly that robustness.
DEFAULT_NLI_MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


@dataclass
class NLIJudge:
    """A local, deterministic ``entails`` judge backed by an NLI sequence-
    classification cross-encoder. ``entails(premise, hypothesis)`` is true
    iff the model's softmax probability of the ENTAILMENT class meets
    ``threshold``. Unlike ``EmbeddingJudge`` (symmetric similarity), this
    is directional 3-way NLI: it distinguishes "follows from" from "merely
    on the same topic", so it catches a fluent contradiction an embedding
    judge would wave through. Model loads lazily on first use.

    Complementarity (measured, 2026-06-06): on the F18 corpus the NLI judge
    flags a fluent on-topic contradiction at loss 1.0 where the embedding
    judge credits it at 0.4 (the hallucination-robustness only directional
    entailment buys) — but, because ``EntailmentScorer`` recall asks
    whether the (shorter) transform *entails* each detailed source
    sentence, strict NLI scores any lossy compression high and so does not
    *rank* faithful compressions as finely as the embedding judge. They
    are complementary: NLI is the contradiction / faithfulness gate,
    embedding is the gist-preservation ranker. MENLI (arXiv:2208.07316)
    recommends blending the two; a blended scorer is the natural next
    refinement."""

    threshold: float = 0.5
    model_id: str = DEFAULT_NLI_MODEL_ID
    _tok: Any = field(default=None, init=False, repr=False, compare=False)
    _mdl: Any = field(default=None, init=False, repr=False, compare=False)
    _entail_idx: int = field(default=-1, init=False, repr=False, compare=False)

    def _ensure_loaded(self) -> None:
        if self._mdl is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as e:  # pragma: no cover - environment-dependent
            raise ImportError(
                "NLIJudge needs the [judge] extra (transformers + torch + "
                "sentencepiece): pip install 'sum-engine[judge]'"
            ) from e
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._mdl = AutoModelForSequenceClassification.from_pretrained(
            self.model_id
        ).eval()
        # Find the ENTAILMENT class index from the model's own label map —
        # NLI models disagree on label order (0=entail vs 2=entail), so we
        # never hard-code it.
        id2label = self._mdl.config.id2label
        match = [i for i, lbl in id2label.items() if "entail" in str(lbl).lower()]
        if not match:
            raise ValueError(
                f"model {self.model_id!r} has no ENTAILMENT label in "
                f"id2label={id2label}; not an NLI model"
            )
        self._entail_idx = int(match[0])

    def entails(self, premise: str, hypothesis: str) -> bool:
        """True iff the NLI model judges ``premise`` to ENTAIL
        ``hypothesis`` with probability ≥ ``threshold``."""
        if not hypothesis.strip():
            return False
        import torch

        self._ensure_loaded()
        with torch.no_grad():
            enc = self._tok(
                premise, hypothesis, return_tensors="pt", truncation=True
            )
            probs = torch.softmax(self._mdl(**enc).logits[0], dim=-1)
        return float(probs[self._entail_idx]) >= self.threshold

    @property
    def name(self) -> str:
        return f"nli:{self.model_id}"


def nli_entailment_scorer(
    threshold: float = 0.5,
    model_id: str = DEFAULT_NLI_MODEL_ID,
) -> EntailmentScorer:
    """An ``EntailmentScorer`` driven by a local NLI cross-encoder
    (``NLIJudge``) — the production-grade, paraphrase-aware, contradiction-
    catching, offline, deterministic meaning-loss scorer. The strict
    upgrade over both lexical (F18) and embedding-similarity scoring."""
    judge = NLIJudge(threshold=threshold, model_id=model_id)
    return EntailmentScorer(
        entails=judge.entails,
        judge_name=judge.name,
        judge_version="1",
    )
