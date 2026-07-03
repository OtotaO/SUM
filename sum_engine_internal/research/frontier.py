"""The *render frontier* — the workbench's load-bearing abstraction.

The product vision (``docs/PRODUCT_VISION.md``) describes a slider, a
number box, a frontier scrubber, and an API/MCP/CURL surface. They are
all **views over one object**: for a source text, an *ordered path* of
renderings from most-faithful to most-compressed, each point carrying
its measured numbers. Build the frontier once; every surface is a thin
view.

    source ─▶  RenderFrontier
                 point[0]  most faithful   (params, rendering, measured loss)
                 …
                 point[k]  most compressed (params, rendering, measured loss)

  - the **slider** picks a point;
  - the **number box** shows the params + measured numbers of a point;
  - the **scrubber** drags ``t ∈ [0, 1]`` across the path;
  - the **API** serialises the whole object (``as_dict``).

Process intensification: this module composes parts SUM already owns
(the slider's parameter axes; the meaning-loss scorer) and adds no new
substrate. Strategic abstraction: the frontier is the single object the
whole workbench rests on.

The honest line (proof boundary)
--------------------------------
A frontier point's ``meaning_loss`` is a **per-document measurement** —
a point estimate for *this* text under a *named proxy*. It is **not** a
guarantee and **not** a certified bound. The distribution-free,
marginal, corpus-level guarantee is a separate object — a
``sum.meaning_risk_receipt.v1`` for a given parameter setting across a
named corpus (see ``research.meaning``). A surface built on this module
must label the two differently and never present a per-document
"guarantee" (there is no such thing). The scorer's name + version travel
on the frontier so a reader always knows which proxy produced the
number, and the layers the proxy cannot cover (arrangement, sound,
connotation, implicature) remain its declared blind spots.

Rendering itself (turning a source + params into compressed text) needs
the slider's LLM / canonical path; this module takes the renderings as
input (or an injected ``render_fn``) so it stays dependency-free and
offline-testable — the same injection discipline as ``EntailmentScorer``.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from sum_engine_internal.research.meaning.meaning_loss import MeaningScorer


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """One rendering on the frontier.

    ``position`` is the point's place on the faithful→compressed path,
    ``0.0`` = most faithful, ``1.0`` = most compressed (assigned by
    index, the caller's compression-control order — *not* by the
    measured loss, which is the outcome, not the control).

    ``meaning_loss`` is a per-document MEASUREMENT under the frontier's
    named scorer, in [0, 1]; it is not a certified bound.

    ``fact_preservation`` is optional (the slider's triple-survival
    fraction, computed elsewhere via the ``extract`` transform); ``None``
    when not supplied.
    """
    label: str
    params: Mapping[str, Any]
    rendering: str
    meaning_loss: float
    position: float
    fact_preservation: float | None = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "label": self.label,
            "params": dict(self.params),
            "rendering": self.rendering,
            "meaning_loss": self.meaning_loss,
            "position": self.position,
        }
        if self.fact_preservation is not None:
            d["fact_preservation"] = self.fact_preservation
        return d


@dataclass(frozen=True, slots=True)
class RungDiff:
    """One rung of a *semantic depth diff* — a per-DOCUMENT MEASUREMENT.

    For one rendering on the frontier, the human-legible decomposition of how
    its meaning differs from the source under a NAMED entailment proxy, plus the
    local rate at which the measured loss changes as compression deepens.

    Honest boundary (load-bearing — these labels are the moat):

    - ``meaning_loss`` is a **directed** loss to the source (an asymmetric
      recall/fidelity blend) under the named judge. It is **not** a metric (no
      triangle inequality) and **not** a certified bound. Do not compose it
      across rungs by addition or triangle inequality.
    - ``loss_per_compression`` is a finite difference of that measured loss over
      the change in **compression** between this rung and the previous one on
      the path, where ``compression = 1 - words(rendering)/words(source)``. It
      is a divided difference over caller-chosen rungs, **not** a derivative of
      a certified rate-distortion curve and **not** an information-bottleneck
      rate. Its **sign is meaningful**: negative means the more-compressed rung
      measured *lower* loss (recovery, or judge noise) — not a smooth descent.
      ``None`` for the first rung (no left neighbour) and when compression did
      not change between two rungs.
    - kept/dropped/added are sentence-level decisions by the named proxy, whose
      human correlation is modest and which is blind to arrangement, sound,
      connotation, and implicature.
    """
    label: str
    position: float
    compression_ratio: float          # words(rendering)/words(source); 1.0 = no length reduction
    meaning_loss: float               # directed loss to source under the named judge (asymmetric; NOT a metric)
    recall: float
    fidelity: float
    source_claims: int
    preserved_claims: int
    dropped_claims: tuple[str, ...]   # source sentences the proxy says were NOT preserved
    added_claims: tuple[str, ...]     # rendering sentences the named proxy could not ground in the source (unsupported; non-entailment, not proof of fabrication)
    loss_per_compression: float | None

    def as_dict(self) -> dict[str, Any]:
        lpc = self.loss_per_compression
        return {
            "label": self.label,
            "position": self.position,
            "compression_ratio": round(self.compression_ratio, 6),
            "meaning_loss": round(self.meaning_loss, 6),
            "meaning_loss_note": (
                "directed loss to source under the named judge; asymmetric, "
                "NOT a metric — do not compose by triangle inequality"
            ),
            "recall": round(self.recall, 6),
            "fidelity": round(self.fidelity, 6),
            "preserved_claims": self.preserved_claims,
            "source_claims": self.source_claims,
            "dropped_claims": list(self.dropped_claims),
            "added_claims": list(self.added_claims),
            "added_claims_note": (
                "rendering sentences the named proxy could not ground in the "
                "source; non-entailment under the proxy, not proof of fabrication"
            ),
            "loss_per_compression": None if lpc is None else round(lpc, 6),
            "loss_per_compression_note": (
                "finite difference of measured loss over Δcompression vs the "
                "previous rung; NOT a rate-distortion derivative; sign meaningful"
            ),
        }


@dataclass(frozen=True, slots=True)
class RenderFrontier:
    """An ordered path of renderings of one source, most-faithful first.

    Construct with :meth:`from_renderings` (you already have the
    renderings) or :meth:`from_render_fn` (inject a render function).
    Both score every point with the supplied ``MeaningScorer`` and record
    the scorer's identity so a reader knows which proxy produced the
    numbers.
    """
    source: str
    scorer_name: str
    scorer_version: str
    points: tuple[FrontierPoint, ...] = field(default_factory=tuple)

    # ---- constructors ----

    @classmethod
    def from_renderings(
        cls,
        source: str,
        renderings: Sequence[tuple[str, Mapping[str, Any], str]],
        scorer: MeaningScorer,
        *,
        fact_preservations: Sequence[float | None] | None = None,
    ) -> "RenderFrontier":
        """Build a frontier from an **ordered** list of renderings.

        ``renderings`` is a sequence of ``(label, params, rendering)``,
        ordered most-faithful → most-compressed (the caller's compression
        control). Each point's ``meaning_loss`` is computed as
        ``scorer.loss(source, rendering)``. ``position`` is assigned by
        index. ``fact_preservations``, if given, must align with
        ``renderings`` by index.
        """
        n = len(renderings)
        if n == 0:
            raise ValueError("a frontier needs at least one rendering")
        if fact_preservations is not None and len(fact_preservations) != n:
            raise ValueError(
                f"fact_preservations has {len(fact_preservations)} entries "
                f"but there are {n} renderings"
            )
        pts: list[FrontierPoint] = []
        for i, (label, params, rendering) in enumerate(renderings):
            loss = float(scorer.loss(source, rendering))
            if not (0.0 <= loss <= 1.0):
                raise ValueError(
                    f"scorer returned out-of-range loss {loss} for point "
                    f"{label!r}; a MeaningScorer must return [0, 1]"
                )
            position = 0.0 if n == 1 else i / (n - 1)
            fp = None if fact_preservations is None else fact_preservations[i]
            pts.append(
                FrontierPoint(
                    label=label,
                    params=dict(params),
                    rendering=rendering,
                    meaning_loss=loss,
                    position=position,
                    fact_preservation=(None if fp is None else float(fp)),
                )
            )
        return cls(
            source=source,
            scorer_name=scorer.name,
            scorer_version=scorer.version,
            points=tuple(pts),
        )

    @classmethod
    def from_render_fn(
        cls,
        source: str,
        settings: Sequence[tuple[str, Mapping[str, Any]]],
        render_fn: Callable[[str, Mapping[str, Any]], str],
        scorer: MeaningScorer,
    ) -> "RenderFrontier":
        """Build a frontier by *rendering* each setting via an injected
        function. ``settings`` is an ordered sequence of
        ``(label, params)``; ``render_fn(source, params) -> str`` is the
        renderer (the slider's LLM or canonical path — injected so this
        module stays dependency-free and deterministic when the renderer
        is). Ordering is the caller's compression control.
        """
        renderings = [
            (label, params, render_fn(source, params))
            for label, params in settings
        ]
        return cls.from_renderings(source, renderings, scorer)

    # ---- views ----

    def __len__(self) -> int:
        return len(self.points)

    @property
    def faithful(self) -> FrontierPoint:
        """The most-faithful point (position 0.0)."""
        return self.points[0]

    @property
    def compressed(self) -> FrontierPoint:
        """The most-compressed point (position 1.0)."""
        return self.points[-1]

    def scrub(self, t: float) -> FrontierPoint:
        """Return the frontier point nearest position ``t`` (clamped to
        [0, 1]). ``t=0`` is most faithful, ``t=1`` most compressed — the
        scrubber's read."""
        t = max(0.0, min(1.0, float(t)))
        n = len(self.points)
        if n == 1:
            return self.points[0]
        idx = round(t * (n - 1))
        return self.points[idx]

    def at(self, label: str) -> FrontierPoint:
        """Return the point with the given ``label`` (a named perspective
        is just a labelled point). Raises ``KeyError`` if absent."""
        for p in self.points:
            if p.label == label:
                return p
        raise KeyError(f"no frontier point labelled {label!r}")

    def losses(self) -> list[float]:
        """Per-point measured meaning-loss, in frontier order. These are
        *measurements*, not a certified bound (see module docstring)."""
        return [p.meaning_loss for p in self.points]

    def depth_diff(self, scorer: MeaningScorer) -> list[RungDiff]:
        """The **semantic depth diff**: for every rung on the faithful→
        compressed path, what the named proxy says the rendering KEPT, DROPPED,
        and ADDED versus the source, plus the local rate at which measured loss
        rises as compression deepens.

        Connect-the-pieces (no new substrate): this iterates the existing
        points, calls the scorer's ``explain`` — the same per-claim readout
        ``sum meaning-diff`` produces for one pair — once per rung, and computes
        ``loss_per_compression`` as a finite difference over a real compression
        coordinate ``1 - words(rendering)/words(source)``. It deliberately does
        **not** use the index-assigned ``position`` as the rate axis (position
        is ordinal, not a compression rate), so the slope is intrinsic and not
        an artifact of how many rungs the caller chose.

        ``scorer`` must expose ``explain`` (an entailment judge). A lexical
        word-overlap scorer has no per-claim entailment and is rejected. The
        whole readout is a per-document MEASUREMENT under a named proxy, never a
        certified bound (see the module docstring); the only (1-δ) guarantee is
        a separate ``sum.meaning_risk_receipt.v1`` over a named corpus.
        """
        explain = getattr(scorer, "explain", None)
        if not callable(explain):
            raise TypeError(
                "depth_diff needs an entailment judge exposing .explain() "
                "(e.g. EntailmentScorer / --scorer nli|embedding); a lexical "
                "word-overlap scorer has no per-claim entailment"
            )
        src_words = max(1, len(self.source.split()))
        rungs: list[RungDiff] = []
        prev_loss: float | None = None
        prev_comp: float | None = None
        for p in self.points:
            r = explain(self.source, p.rendering)
            ratio = len(p.rendering.split()) / src_words
            comp = 1.0 - ratio  # 0.0 = no length reduction, →1.0 = maximal compression
            lpc: float | None = None
            if prev_loss is not None and prev_comp is not None:
                d_comp = comp - prev_comp
                if abs(d_comp) >= 1e-9:
                    lpc = (r.loss - prev_loss) / d_comp
            rungs.append(
                RungDiff(
                    label=p.label,
                    position=p.position,
                    compression_ratio=ratio,
                    meaning_loss=r.loss,
                    recall=r.recall,
                    fidelity=r.fidelity,
                    source_claims=r.source_claims,
                    preserved_claims=r.preserved_claims,
                    dropped_claims=r.dropped_claims,
                    added_claims=r.unsupported_claims,
                    loss_per_compression=lpc,
                )
            )
            prev_loss = r.loss
            prev_comp = comp
        return rungs

    def as_dict(self) -> dict[str, Any]:
        """Serialise for the API / MCP / CURL surface. The same object
        every door returns."""
        return {
            "source": self.source,
            "scorer": self.scorer_name,
            "scorer_version": self.scorer_version,
            "n": len(self.points),
            "measurement_note": (
                "meaning_loss is a per-document measurement under the "
                "named scorer, not a certified bound; the marginal "
                "distribution-free guarantee is a sum.meaning_risk_"
                "receipt.v1 over a named corpus"
            ),
            "points": [p.as_dict() for p in self.points],
        }
