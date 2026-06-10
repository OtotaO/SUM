"""Bounded, named proxies for *meaning-loss* between a source text and a
transform of it (a compression, rendering, or translation).

The frontier this opens — the layer *below* fact-preservation. SUM's
slider contract measures whether (subject, predicate, object) triples
survive a transform. But meaning lives beneath the proposition: in what
a reader reconstructs from an ellipsis, in arrangement, in connotation,
in register. None of that is a fact, and none of it is captured by
triple-match. This module is the first rung toward measuring it — and,
crucially, toward measuring it *honestly*.

The honest boundary (read this before using anything here)
----------------------------------------------------------
**No function in this module measures meaning.** Each computes a
*named, versioned proxy* for meaning-loss — a number in [0, 1] where 0
is "the proxy detects no loss" and 1 is "the proxy detects total loss".
The proxy is a stand-in chosen for being computable in *checkable text
space* (so a third party can recompute it), not for being meaning. A
guarantee built on top of one of these scorers (see
``conformal_meaning``) is rigorous *conditional on the named scorer* —
swap the scorer and the number means something else. The scorer's
``name`` and ``version`` therefore travel inside every receipt, so a
reader always knows which proxy was certified.

What these proxies do NOT cover — declared, not hidden:

  - **Arrangement** — the meaning that emerges only from word order and
    structure (what the rhetoricians of the Abrahamic scriptural
    tradition call *naẓm*; what al-Jurjānī argued is where eloquence
    actually lives). A bag-of-units scorer is blind to it by
    construction; an entailment scorer is largely blind to it too.
  - **Sound** — prosody, meter, rhyme, phonetic texture.
  - **Connotation / register** — tone, stance, the emotional colour of
    a word choice.
  - **Implicature** — what is conveyed without being said.

A receipt over these proxies must *say* it does not cover those layers.
That refusal-to-claim is the most defensible move in the whole space:
the literature has no validated measure of arrangement-meaning loss
(there is no ground truth to check one against), so the correct
engineering act is to measure the change we *can* compute and to label
the layers we cannot.

The scorer interface
--------------------
``MeaningScorer`` is a structural protocol: anything with a ``name``, a
``version``, and a ``loss(source, transform) -> float`` is a scorer. Two
ship here:

  - ``LexicalCoverageScorer`` — dependency-free, deterministic. A crude
    *proxy-of-a-proxy* whose only job is to make the whole pipeline
    (scorer → conformal certificate → signed receipt) runnable and
    testable with zero model downloads. It measures bidirectional
    content-unit overlap distance. It does **not** understand
    paraphrase, so it over-reports loss on a faithful reword — a
    limitation it is named for and must never be hidden behind.

  - ``EntailmentScorer`` — the real path. Wraps a caller-supplied
    ``entails(premise, hypothesis) -> bool`` decision (an NLI model, an
    LLM judge — the same shape as the slider bench's v0.4 NLI audit) and
    computes a *bidirectional-entailment* meaning-loss in the spirit of
    semantic-entropy meaning-clustering (Farquhar, Kossen, Kuhn & Gal, *Nature*
    2024 — cluster by mutual entailment, not tokens). The model is
    *injected*, never imported here, so this module stays dependency-
    free and the certified loss stays tied to a named, swappable judge.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, runtime_checkable

# A small, frozen, language-agnostic-ish English stop set. Deliberately
# minimal and *frozen* — the scorer is deterministic only if this set
# never silently changes, so any edit is a scorer-version bump.
_STOP_WORDS: frozenset[str] = frozenset(
    """
    a an the this that these those of to in on at by for with from into over
    and or but nor so yet as if then than is are was were be been being am
    it its it's he she they them his her their we you i me my your our us do
    does did has have had will would shall should can could may might must
    not no out up down off above below near here there which who whom whose
    """.split()
)

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _content_units(text: str) -> list[str]:
    """Lower-cased content words (stop-words removed). The atomic
    "meaning unit" of the lexical proxy. Order is preserved so callers
    that want multiset semantics can keep duplicates; the lexical scorer
    collapses to a set."""
    return [
        w for w in (m.group(0).lower() for m in _WORD_RE.finditer(text))
        if w not in _STOP_WORDS
    ]


def _sentences(text: str) -> list[str]:
    """Naïve sentence split for the entailment scorer's per-unit pass.
    Deterministic; splits on ., !, ? followed by whitespace, plus
    newlines. Empty fragments dropped."""
    rough = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [s.strip() for s in rough if s.strip()]


@runtime_checkable
class MeaningScorer(Protocol):
    """Structural protocol for a named, versioned meaning-loss proxy.

    ``loss`` returns a value in [0, 1]: 0 = proxy detects no loss,
    1 = proxy detects total loss. ``name`` and ``version`` identify the
    proxy and travel inside every certificate built over it — so a
    verifier always knows *which* proxy a bound is conditional on.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def loss(self, source: str, transform: str) -> float: ...


@dataclass(frozen=True, slots=True)
class LexicalCoverageScorer:
    """Dependency-free, deterministic meaning-loss proxy: bidirectional
    content-unit overlap distance.

    Let ``S`` and ``T`` be the *sets* of content units of the source and
    the transform. The loss is a convex blend of two directional gaps::

        drop  = |S \\ T| / |S|     (source meaning the transform no longer carries)
        fab   = |T \\ S| / |T|     (content the transform introduced unseen in source)
        loss  = w_drop * drop + w_fab * fab

    With the defaults (``w_drop=0.7``, ``w_fab=0.3``) the proxy weights
    *omission* (the dominant failure mode of compression) above
    *fabrication* (hallucination), but counts both. Properties this
    proxy satisfies — the MeaningBERT-style sanity contract (Beauchemin
    et al., 2023), pinned by the unit tests in
    ``Tests/research/test_meaning_loss.py`` and fuzzed in
    ``test_meaning_property.py`` (not mechanically proven; "guarantee" is
    reserved for the replayable receipt):

      - identity: ``loss(x, x) == 0``
      - disjoint: no shared unit → ``loss == 1``
      - monotone: deleting a source unit from the transform never
        *decreases* the loss.

    Weights must sum to 1 (validated) so the loss is a proper convex
    blend in [0, 1] and ``disjoint`` lands exactly at 1.

    Honest limitation (named, not hidden): the proxy is *lexical*, and a
    2026-06-06 dogfood (docs/DOGFOOD_FINDINGS_2026-06-06.md, F18) showed
    it does not merely over-report a faithful paraphrase — it **misranks**
    it: a faithful reword scored *higher* loss (0.762) than a crude
    extractive trim (0.394), and higher even than a near-empty tag
    (0.742), because a reword shares few words with the source (high drop
    AND high fabrication). So this scorer is trustworthy only for
    *extractive* (sentence-dropping) compression; for paraphrase it
    inverts the ranking and would steer a user wrong. It is a runnable
    placeholder, not a meaning measure; deployments that care about
    paraphrase MUST plug ``EntailmentScorer`` (an NLI/LLM judge).
    """

    w_drop: float = 0.7
    w_fab: float = 0.3

    def __post_init__(self) -> None:
        if not math.isclose(self.w_drop + self.w_fab, 1.0, abs_tol=1e-9):
            raise ValueError(
                f"w_drop + w_fab must sum to 1.0 (a convex blend); got "
                f"{self.w_drop} + {self.w_fab} = {self.w_drop + self.w_fab}"
            )

    @property
    def name(self) -> str:
        return "lexical-coverage-bidirectional"

    @property
    def version(self) -> str:
        # Bump on ANY change to _STOP_WORDS, tokenisation, or the blend
        # — the certified number is only reproducible against a pinned
        # scorer version.
        return "1"

    def loss(self, source: str, transform: str) -> float:
        s = set(_content_units(source))
        t = set(_content_units(transform))
        if not s and not t:
            return 0.0  # two empties carry the same (empty) meaning
        if not s:
            # Source carries no content units but transform does: pure
            # fabrication relative to an empty source.
            return 1.0
        drop = len(s - t) / len(s)
        fab = (len(t - s) / len(t)) if t else 1.0
        val = self.w_drop * drop + self.w_fab * fab
        return max(0.0, min(1.0, val))


@dataclass(frozen=True, slots=True)
class EntailmentScorer:
    """Bidirectional-entailment meaning-loss over a caller-supplied NLI
    decision — the real path, kept dependency-free by *injection*.

    ``entails(premise, hypothesis)`` must return ``True`` iff a reader
    of ``premise`` would accept ``hypothesis`` as following from it. This
    is exactly the shape of the slider bench's v0.4 NLI audit
    (``LiveLLMAdapter.check_entailment``) and of any off-the-shelf NLI
    model's entail/neutral/contradict head reduced to a boolean.

    The loss decomposes meaning preservation into two directions over
    sentence-level units:

        recall   = mean over source sentences s of [transform ⊨ s]
                   — did the transform keep what the source asserted?
        fidelity = mean over transform sentences t of [source ⊨ t]
                   — did the transform avoid asserting what the source did not?
        loss     = 1 - (w_recall * recall + w_fidelity * fidelity)

    Defaults weight recall (omission) above fidelity (fabrication),
    mirroring the lexical scorer. The weights must sum to 1 (validated):
    on identity (recall = fidelity = 1) the loss is ``1 - (w_recall +
    w_fidelity)``, which is 0 — the sanity contract — only when they sum
    to 1. Because the judge is named and versioned by the caller
    (``judge_name`` / ``judge_version``), the certified bound stays
    explicitly conditional on it — swap the judge, bump the version, and
    the certificate's provenance changes with it.

    Determinism note: the certificate replay property (re-run → identical
    bound) holds only if ``entails`` is itself deterministic for fixed
    inputs. A sampling LLM judge breaks replay; a temperature-0 judge or
    a local NLI model preserves it. The scorer does not enforce this —
    it is part of the proof boundary the receipt discloses.
    """

    entails: Callable[[str, str], bool]
    judge_name: str
    judge_version: str = "unspecified"
    w_recall: float = 0.6
    w_fidelity: float = 0.4
    # Optional batch hook: ``entails_batch(premise, hypotheses) -> [bool]``,
    # one decision per hypothesis against the same premise. When a judge
    # provides it (the EmbeddingJudge does), ``loss``/``explain`` take a fast
    # path that embeds each unique sentence ONCE instead of O(m·n) times —
    # bit-exact, ~25× on the embedding path. Falls back to the scalar
    # ``entails`` callback (e.g. the NLI cross-encoder) when absent.
    entails_batch: Callable[[str, "Sequence[str]"], "list[bool]"] | None = None

    def __post_init__(self) -> None:
        if not math.isclose(self.w_recall + self.w_fidelity, 1.0, abs_tol=1e-9):
            raise ValueError(
                f"w_recall + w_fidelity must sum to 1.0 so identity "
                f"loss(x, x) == 0 holds; got {self.w_recall} + "
                f"{self.w_fidelity} = {self.w_recall + self.w_fidelity}"
            )

    @property
    def name(self) -> str:
        return f"bidirectional-entailment[{self.judge_name}]"

    @property
    def version(self) -> str:
        return self.judge_version

    def _decisions(self, premise: str, hypotheses: "Sequence[str]") -> "list[bool]":
        """Per-hypothesis entailment decisions against one ``premise``. Uses
        the batch hook (one embed pass over the unique sentences) when the
        judge provides it; else the scalar ``entails`` callback. Bit-exact:
        identical premise/hypothesis pairs, identical decisions."""
        if self.entails_batch is not None:
            return list(self.entails_batch(premise, list(hypotheses)))
        return [self.entails(premise, h) for h in hypotheses]

    def loss(self, source: str, transform: str) -> float:
        src_units = _sentences(source)
        tr_units = _sentences(transform)
        if not src_units and not tr_units:
            return 0.0
        if not src_units:
            return 1.0
        recall = sum(self._decisions(transform, src_units)) / len(src_units)
        if tr_units:
            fidelity = sum(self._decisions(source, tr_units)) / len(tr_units)
        else:
            # Transform asserts nothing: no fabrication, but recall
            # already captures the total omission.
            fidelity = 1.0
        preservation = self.w_recall * recall + self.w_fidelity * fidelity
        return max(0.0, min(1.0, 1.0 - preservation))

    def explain(self, source: str, transform: str) -> "MeaningReadout":
        """Per-DOCUMENT readout: the same bidirectional-entailment loss this
        scorer certifies, decomposed into the human-legible "what was kept /
        dropped / added". A MEASUREMENT for one document, **not** a certified
        bound (use a meaning_risk receipt over a corpus for a 1-δ guarantee)."""
        return explain_meaning_loss(
            source, transform, entails=self.entails,
            judge_name=self.judge_name, judge_version=self.judge_version,
            w_recall=self.w_recall, w_fidelity=self.w_fidelity,
            entails_batch=self.entails_batch,
        )


@dataclass(frozen=True)
class MeaningReadout:
    """A per-DOCUMENT meaning readout — a MEASUREMENT for ONE (source,
    transform) pair, the #1 thing real users ask for ("what changed in MY
    text?"). It decomposes the bidirectional-entailment ``loss`` into the
    sentences the transform DROPPED (source claims it failed to preserve) and
    the sentences it ADDED without grounding (transform claims the source does
    not support). ``loss`` equals what ``EntailmentScorer.loss`` returns for
    the same inputs — same number, made legible.

    Honest boundary (load-bearing): this is a per-document MEASUREMENT under a
    NAMED judge, NOT a certified bound and NOT "meaning" itself. The
    distribution-free (1-δ) guarantee lives in a meaning_risk receipt over a
    corpus; ``scope`` says so, and any surface that shows a readout must too.
    """
    loss: float
    preservation: float
    recall: float
    fidelity: float
    source_claims: int
    preserved_claims: int
    dropped_claims: tuple[str, ...]       # source sentences NOT preserved → "what was lost"
    transform_claims: int
    unsupported_claims: tuple[str, ...]   # transform sentences NOT grounded → "what was added"
    judge: str
    judge_version: str

    @property
    def scope(self) -> str:
        return ("measured for THIS document under the named judge — a "
                "per-document MEASUREMENT, not a certified bound or a "
                "guarantee; for a (1-δ) bound use a meaning_risk receipt over "
                "a named corpus")


def explain_meaning_loss(
    source: str,
    transform: str,
    *,
    entails: "Callable[[str, str], bool]",
    judge_name: str,
    judge_version: str = "unspecified",
    w_recall: float = 0.6,
    w_fidelity: float = 0.4,
    entails_batch: "Callable[[str, Sequence[str]], list[bool]] | None" = None,
) -> MeaningReadout:
    """Build a :class:`MeaningReadout` for one (source, transform) pair via the
    injected ``entails`` judge. Mirrors ``EntailmentScorer.loss`` exactly
    (including its empty-input edge cases) so the readout's ``loss`` is the
    per-pair loss a certificate is built from — just decomposed.

    When ``entails_batch`` is supplied it is used for the per-sentence
    decisions (one embed pass over the unique sentences instead of O(m·n)) —
    bit-exact with the scalar path; the EmbeddingJudge provides it."""
    if not math.isclose(w_recall + w_fidelity, 1.0, abs_tol=1e-9):
        raise ValueError(
            f"w_recall + w_fidelity must sum to 1.0; got {w_recall} + {w_fidelity}"
        )
    src = _sentences(source)
    tr = _sentences(transform)
    if entails_batch is not None:
        src_keep = entails_batch(transform, src) if src else []
        tr_keep = entails_batch(source, tr) if tr else []
        dropped = tuple(s for s, k in zip(src, src_keep) if not k)
        unsupported = tuple(t for t, k in zip(tr, tr_keep) if not k)
    else:
        dropped = tuple(s for s in src if not entails(transform, s))
        unsupported = tuple(t for t in tr if not entails(source, t))
    if not src and not tr:
        recall = fidelity = 1.0                 # identity-empty → loss 0
    elif not src:
        recall, fidelity = 0.0, 1.0             # source empty but transform asserts → loss 1
    else:
        recall = 1.0 - len(dropped) / len(src)
        fidelity = 1.0 if not tr else 1.0 - len(unsupported) / len(tr)
    loss = max(0.0, min(1.0, 1.0 - (w_recall * recall + w_fidelity * fidelity)))
    return MeaningReadout(
        loss=loss, preservation=1.0 - loss, recall=recall, fidelity=fidelity,
        source_claims=len(src), preserved_claims=len(src) - len(dropped),
        dropped_claims=dropped, transform_claims=len(tr),
        unsupported_claims=unsupported, judge=judge_name, judge_version=judge_version,
    )


def score_pairs(
    pairs: Sequence[tuple[str, str]],
    scorer: MeaningScorer,
) -> list[float]:
    """Apply ``scorer`` to every ``(source, transform)`` pair, returning
    the per-pair meaning-loss in [0, 1]. The list this returns is the
    calibration sample the conformal layer certifies — and the exact
    bytes a verifier recomputes to replay the bound."""
    return [scorer.loss(src, tr) for src, tr in pairs]
