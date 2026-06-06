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

    Honest limitation (named, not hidden): the proxy is *lexical*. A
    faithful paraphrase that preserves meaning with different words
    scores as loss. It is a runnable placeholder, not a meaning measure;
    deployments that care about paraphrase plug ``EntailmentScorer``.
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

    def loss(self, source: str, transform: str) -> float:
        src_units = _sentences(source)
        tr_units = _sentences(transform)
        if not src_units and not tr_units:
            return 0.0
        if not src_units:
            return 1.0
        recall = sum(
            1 for s in src_units if self.entails(transform, s)
        ) / len(src_units)
        if tr_units:
            fidelity = sum(
                1 for t in tr_units if self.entails(source, t)
            ) / len(tr_units)
        else:
            # Transform asserts nothing: no fabrication, but recall
            # already captures the total omission.
            fidelity = 1.0
        preservation = self.w_recall * recall + self.w_fidelity * fidelity
        return max(0.0, min(1.0, 1.0 - preservation))


def score_pairs(
    pairs: Sequence[tuple[str, str]],
    scorer: MeaningScorer,
) -> list[float]:
    """Apply ``scorer`` to every ``(source, transform)`` pair, returning
    the per-pair meaning-loss in [0, 1]. The list this returns is the
    calibration sample the conformal layer certifies — and the exact
    bytes a verifier recomputes to replay the bound."""
    return [scorer.loss(src, tr) for src, tr in pairs]
