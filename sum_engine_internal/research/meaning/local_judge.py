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

    def entails_batch(self, premise: str, hypotheses: "list[str]") -> "list[bool]":
        """Batched :meth:`entails`: one decision per hypothesis against the
        same ``premise``, embedding the premise sentences ONCE and all
        (non-blank) hypotheses in ONE forward pass — instead of re-embedding
        the premise for every hypothesis (the O(m·n) waste in the per-pair
        loss). Bit-exact with the per-hypothesis ``entails`` calls: padding is
        attention-masked, so each sentence's pooled embedding is independent
        of batch composition, and the max-cosine decision is identical."""
        results = [False] * len(hypotheses)
        nonblank = [(i, h) for i, h in enumerate(hypotheses) if h.strip()]
        if not nonblank:
            return results
        premise_sents = _sentences(premise) or [premise]
        pe = self._embed(premise_sents)
        he = self._embed([h for _, h in nonblank])
        sims = (he @ pe.T).max(dim=1).values  # max cosine per hypothesis
        for k, (i, _) in enumerate(nonblank):
            results[i] = float(sims[k]) >= self.threshold
        return results

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
        entails_batch=judge.entails_batch,  # fast path: embed each sentence once
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


def _entailment_index(id2label: dict) -> int:
    """The index of the ENTAILMENT class in a model's ``id2label`` — robust
    to label-order disagreement (0=entail vs 2=entail) AND to negation
    labels. A binary RTE head with ``{0: "not_entailment", 1: "entailment"}``
    must pick 1, not 0: ``"entail" in "not_entailment"`` is True, so a naive
    substring match silently inverts the judge. Prefer an exact
    ``"entailment"`` label; otherwise an ``entail`` label that is not a
    negation."""
    norm = {int(i): str(lbl).strip().lower() for i, lbl in id2label.items()}
    exact = [i for i, lbl in norm.items() if lbl == "entailment"]
    if exact:
        return exact[0]
    cand = [
        i for i, lbl in norm.items()
        if "entail" in lbl and "not" not in lbl and "non" not in lbl
    ]
    if cand:
        return cand[0]
    raise ValueError(
        f"no ENTAILMENT label in id2label={id2label}; not an NLI model"
    )


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
        self._entail_idx = _entailment_index(self._mdl.config.id2label)

    def entails(self, premise: str, hypothesis: str) -> bool:
        """True iff the NLI model judges ``premise`` to ENTAIL
        ``hypothesis`` with probability ≥ ``threshold``."""
        if not hypothesis.strip():
            return False
        import torch

        self._ensure_loaded()
        # Cap input length explicitly: a tokenizer whose model_max_length is
        # the uncapped sentinel (1e30) makes truncation=True a no-op, so a
        # long premise would overrun max_position_embeddings and OOM-kill /
        # raise. Cap at the model's position limit (≤512 for this class).
        mpe = getattr(self._mdl.config, "max_position_embeddings", 512)
        max_len = int(mpe) if isinstance(mpe, int) and 0 < mpe <= 4096 else 512
        with torch.no_grad():
            enc = self._tok(
                premise, hypothesis, return_tensors="pt",
                truncation=True, max_length=max_len,
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
