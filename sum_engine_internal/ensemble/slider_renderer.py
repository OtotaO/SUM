"""Slider Renderer — Tags → Tomes with axis-conditioned LLM rendering.

The Phase E genesis-vision module. Takes a set of canonicalised triples
plus a TomeSliders position; returns a rendered tome with per-axis
round-trip drift measured against the source triples. Caches by
quantized slider position so adjacent drag positions hit the same cell.

Architecture (filled by STATE 4):

    render(triples, sliders, llm) ─┬─→ check cache (quantize sliders)
                                    │     hit  → return cached RenderResult
                                    │     miss → continue
                                    │
                                    ├─→ apply density (deterministic prefix)
                                    │
                                    ├─→ build_system_prompt(quantized_sliders)
                                    │
                                    ├─→ llm.chat.completions.create(...)
                                    │     → tome (string)
                                    │
                                    ├─→ re-extract triples from tome
                                    │   (sieve or LLM extractor)
                                    │
                                    ├─→ compute drift = |source △ reextracted|
                                    │   per axis (drift_density, drift_length,
                                    │   drift_formality, drift_audience,
                                    │   drift_perspective)
                                    │
                                    ├─→ store in cache under quantize(sliders)
                                    │
                                    └─→ return RenderResult

Why drift is computed PER AXIS rather than as a single scalar:
each axis has different acceptable tolerance. Density at 0.3 should
preserve 30% of source triples (drift "expected"). Audience at 0.9
should preserve 100% (any drift is a hallucination). The UI uses
per-axis drift to color individual sliders red when they violate
their axis-specific contract.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Optional, Protocol, Sequence

from sum_engine_internal.ensemble.tome_sliders import (
    SLIDER_BINS_PER_AXIS,
    TomeSliders,
    apply_density,
    build_system_prompt,
    quantize,
    snap_to_bin,
)


# ─── Type contracts ───────────────────────────────────────────────────


# A canonicalised triple in (subject, predicate, object) form. Strings
# are already lowercased + underscore-joined per the sieve's
# canonicalisation rules. The renderer never re-canonicalises; the
# caller owns input shape.
Triple = tuple[str, str, str]


class CacheStatus(str, Enum):
    """Where the rendered output came from."""

    HIT = "hit"
    MISS = "miss"
    BYPASS = "bypass"   # caller passed cache=None; never even checked


class DriftAxis(str, Enum):
    """Names match TomeSliders fields."""

    DENSITY = "density"
    LENGTH = "length"
    FORMALITY = "formality"
    AUDIENCE = "audience"
    PERSPECTIVE = "perspective"


@dataclass(frozen=True)
class AxisDrift:
    """Per-axis drift measurement for one render.

    `value` is the drift metric IN THE UNITS THE AXIS DEFINES — they are
    not all the same scale. Density drift is (1 - retained_fraction);
    LLM-axis drifts are |source_set △ reextracted_set| / |union|.
    The threshold is what the SLIDER_CONTRACT.md says counts as
    'unacceptable' for that axis at the slider's current position.
    """

    axis: DriftAxis
    value: float                      # measured drift, axis-defined units
    threshold: float                  # contract limit at this slider position
    classification: str               # "ok" | "warn" | "fail"
    explanation: Optional[str] = None # human-readable why-it-failed, if classification != "ok"


@dataclass(frozen=True)
class RenderResult:
    """Output of slider_renderer.render().

    `tome`              — the generated narrative text. May be the canonical
                          deterministic rendering (when all LLM axes are at
                          neutral) or an LLM-conditioned rendering.
    `triples_used`      — the post-density triples actually fed to the LLM.
    `reextracted_triples`— triples re-extracted from the rendered tome,
                          in tome-extraction order. For the canonical path
                          this equals triples_used (no LLM transformation,
                          no semantic drift). For the LLM path this is what
                          actually survived the round-trip — load-bearing
                          for any honest fact-preservation claim.
    `claimed_triples`   — v0.3: LLM self-attestation of triples it
                          considered preserved in the tome. Empty tuple
                          on the canonical path (no LLM call) and on
                          legacy `chat_completion`-only renders. Cross-
                          checked against `reextracted_triples` for an
                          adversarial-divergence signal: claimed but not
                          re-extracted = LLM hallucinated preservation;
                          re-extracted but not claimed = LLM encoded
                          facts it didn't itemise.
    `drift`             — per-axis measurement against source triples.
    `cache_status`      — provenance of the result.
    `llm_calls_made`    — 0 on cache hit; 1 on miss (one LLM call per render
                          — the axes condition the system prompt rather than
                          firing N parallel calls).
    `wall_clock_ms`     — total time including cache lookup, LLM call, and
                          drift measurement.
    `quantized_sliders` — what the cache key actually used (post-snap).
    `render_id`         — content-addressed identifier of this render
                          (sha256 of triples_used + quantized_sliders + tome).
    """

    tome: str
    triples_used: tuple[Triple, ...]
    reextracted_triples: tuple[Triple, ...]
    claimed_triples: tuple[Triple, ...]
    drift: tuple[AxisDrift, ...]
    cache_status: CacheStatus
    llm_calls_made: int
    wall_clock_ms: int
    quantized_sliders: TomeSliders
    render_id: str


# ─── Cache protocol ───────────────────────────────────────────────────


class SliderCache(Protocol):
    """Protocol any cache backend must satisfy. STATE 4 ships an
    in-memory implementation (single-process); the Worker provides
    a KV-backed implementation in worker/src/cache/bin_cache.ts that
    matches the same key shape.
    """

    async def get(self, key: str) -> Optional[RenderResult]: ...
    async def put(self, key: str, value: RenderResult, ttl_seconds: int) -> None: ...
    async def stats(self) -> dict[str, int]: ...


# ─── LLM client protocol ──────────────────────────────────────────────


class LLMChatClient(Protocol):
    """Subset of the OpenAI-compatible API the renderer requires.

    Renderer never imports `LiveLLMAdapter` directly — it depends on
    this protocol so tests can inject a deterministic fake without
    pulling in the real LLM call path.

    v0.3: gained `chat_completion_structured` for constrained-decoding
    rendering. The renderer prefers structured when available; the
    plain `chat_completion` is retained for backwards compatibility
    and as a fallback path for caller-supplied LLM clients that don't
    yet support structured outputs.
    """

    async def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
    ) -> str: ...

    async def chat_completion_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
    ) -> tuple[str, list[Triple]]: ...


# ─── Extractor protocol ───────────────────────────────────────────────


# Re-extraction is needed to compute drift. The renderer doesn't know
# whether to use the sieve (deterministic) or the LLM (high recall);
# caller decides by injecting an extractor function.
TripleExtractor = Callable[[str], Awaitable[list[Triple]]]


# ─── Public API ───────────────────────────────────────────────────────


def cache_key(triples: Sequence[Triple], quantized: TomeSliders) -> str:
    """Derive the cache key from canonicalised triples + quantized
    sliders. Pure function — same input always returns same key.

    Format: sha256_hex(json.dumps({sorted_triples, quantized_sliders}))
    truncated to 32 chars (16 bytes) for compact key strings.

    The sort is critical for cache-key stability: callers may pass
    the same triple set in any order; we want them to hit the same
    cache cell. The post-density triples are what gets keyed (so a
    slider position with density=0.3 caches a different cell than
    density=1.0 even with identical input).
    """
    sorted_triples = sorted(tuple(t) for t in triples)
    payload = {
        "triples": sorted_triples,
        "sliders": {
            "density": quantized.density,
            "length": quantized.length,
            "formality": quantized.formality,
            "audience": quantized.audience,
            "perspective": quantized.perspective,
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def _axiom_key(t: Triple) -> str:
    return f"{t[0]}||{t[1]}||{t[2]}"


def _format_triples_for_llm(triples: Sequence[Triple]) -> str:
    """Render triples as a numbered list for the LLM user prompt. The
    LLM is instructed (via the system prompt from build_system_prompt)
    to preserve every fact, so the format only needs to be unambiguous."""
    return "FACTS:\n" + "\n".join(
        f"{i+1}. ({s}, {p}, {o})" for i, (s, p, o) in enumerate(triples)
    )


def _deterministic_tome(triples: Sequence[Triple]) -> str:
    """Canonical-path tome generator: one sentence per triple, no LLM.
    Used when all LLM-axes are at neutral (sliders.requires_extrapolator()
    is False) — i.e. only density is non-default."""
    if not triples:
        return ""
    return " ".join(f"The {s} {p} {o}." for (s, p, o) in triples)


# ─── Lookup tables for register / jargon / pronoun classifiers ─────────
#
# These are deliberately small and hand-curated. The contract doc says
# "lookup table" not "comprehensive dictionary"; STATE 5 bench data will
# tell us if they're sufficient. Replacing with a Wikipedia-derived
# frequency table is a v0.2 swap that doesn't change the API.

_FORMAL_MARKERS: frozenset[str] = frozenset({
    "moreover", "furthermore", "thus", "hence", "therefore", "subsequently",
    "demonstrate", "demonstrates", "established", "indicate", "indicates",
    "constitute", "constitutes", "exhibit", "exhibits", "facilitate",
    "facilitates", "utilize", "utilizes", "wherein", "whereby", "whereas",
    "notwithstanding", "consequently", "accordingly", "respectively",
    "approximately", "considerable", "considerably", "substantial",
    "observed", "noted", "presented",
})

_CASUAL_MARKERS: frozenset[str] = frozenset({
    "stuff", "things", "really", "pretty", "kinda", "gonna", "wanna",
    "lots", "tons", "loads", "ok", "okay", "yeah", "yep", "nope",
    "bunch", "bit", "guy", "guys", "folks", "huge", "totally",
    "basically", "honestly", "literally", "cool", "awesome",
})

# First-person pronouns. Lowercased; matched against tokenised words.
_FIRST_PERSON: frozenset[str] = frozenset({
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
    "i'm", "i've", "i'd", "i'll", "we're", "we've", "we'd", "we'll",
})

# All English pronouns we count toward the denominator.
_ALL_PRONOUNS: frozenset[str] = _FIRST_PERSON | frozenset({
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    "one", "ones", "oneself",
})

def _load_common_words() -> frozenset[str]:
    """Load the bundled top-5000 frequency-table from the package data
    directory. One-shot at module import; the result is immutable.

    Tries the 5000-word table first, falls back to the 2000-word table
    for partial-checkout robustness, then a minimal stoplist as last
    resort. The bench will report measurement degradation on degraded
    fallback but the renderer keeps working.
    """
    from importlib.resources import files
    data_pkg = files("sum_engine_internal.ensemble.data")
    for filename in ("common_english_5000.txt", "common_english_2000.txt"):
        try:
            text = (data_pkg / filename).read_text(encoding="utf-8")
            return frozenset(w.strip().lower() for w in text.splitlines() if w.strip())
        except (FileNotFoundError, ModuleNotFoundError):
            continue
    # Last-resort fallback: most common 50 English words.
    return frozenset((
        "the of and to in that is was he for it with as his on be at by "
        "i this had not are but from or have an they which one you were "
        "her all she there would their we him been has when who will more"
    ).split())


# Top-5000 most common English words by frequency in the Brown corpus.
# Words NOT in this set AND longer than 4 chars count as "jargon" for
# the audience-axis density measurement. See data/common_english_5000.txt.
_COMMON_WORDS: frozenset[str] = _load_common_words()


def _tokens(text: str) -> list[str]:
    """Lowercase token stream. Apostrophes preserved so contractions like
    'i'm' match _FIRST_PERSON entries."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def _formal_score(text: str) -> float:
    """Map text to [0,1] register score. 1.0 = pure formal markers, 0.0 =
    pure casual markers, 0.5 = equal or no markers found.

    Deterministic; pure function of token contents."""
    toks = _tokens(text)
    if not toks:
        return 0.5
    formal_hits = sum(1 for t in toks if t in _FORMAL_MARKERS)
    casual_hits = sum(1 for t in toks if t in _CASUAL_MARKERS)
    total_marker_hits = formal_hits + casual_hits
    if total_marker_hits == 0:
        return 0.5
    return formal_hits / total_marker_hits


def _jargon_density(text: str) -> float:
    """Ratio of jargon tokens to total tokens. Jargon = not in
    _COMMON_WORDS AND longer than 4 chars. Returns 0.0 for empty text."""
    toks = _tokens(text)
    if not toks:
        return 0.0
    jargon = sum(1 for t in toks if len(t) > 4 and t not in _COMMON_WORDS)
    return jargon / len(toks)


def _first_person_ratio(text: str) -> float:
    """Ratio of first-person pronouns to all pronouns. Returns 0.5 (the
    neutral midpoint) when no pronouns appear, so absence of pronouns
    doesn't read as 'pure third-person' (which would be a false positive
    for the perspective=1.0 contract)."""
    toks = _tokens(text)
    pronoun_count = sum(1 for t in toks if t in _ALL_PRONOUNS)
    if pronoun_count == 0:
        return 0.5
    first_count = sum(1 for t in toks if t in _FIRST_PERSON)
    return first_count / pronoun_count


# Length bands per docs/SLIDER_CONTRACT.md §Length. Words PER source
# triple, calibrated against the Phase E.1 STATE 5b bench run
# (gpt-4o-mini, 8 multi-fact paragraphs, median 6 source triples).
# The original per-triple-linear bands assumed the LLM scaled response
# length linearly with input fact count; empirically it doesn't —
# below position 0.5 there's a floor (~6 words per triple), above
# 0.5 the LLM scales aggressively.
_LENGTH_BANDS: dict[float, tuple[int, int]] = {
    0.1: (4,  10),    # ~6 wpt: telegraphic / one-line-per-fact floor
    0.3: (5,  12),    # ~7 wpt: brief — LLM rarely compresses below this
    0.5: (4,  10),    # ~6 wpt: LLM natural baseline (no directive)
    0.7: (30, 60),    # ~40 wpt: expanded prose
    0.9: (80, 140),   # ~100 wpt: essay-length
}

# Per-axis thresholds — copied from SLIDER_CONTRACT.md. Above threshold ⇒ "fail".
_THRESHOLDS: dict[str, float] = {
    "density": 0.001,
    "length": 0.5,
    "formality": 0.25,
    "audience": 0.10,
    "perspective": 0.20,
}


def _classify(value: float, threshold: float) -> str:
    """Three-band classifier: ok ≤ threshold, warn ≤ 1.5×threshold, fail beyond."""
    if value <= threshold:
        return "ok"
    if value <= threshold * 1.5:
        return "warn"
    return "fail"


def fact_preservation(
    source_triples: Sequence[Triple],
    reextracted_triples: Sequence[Triple],
) -> float:
    """STRICT set-based round-trip preservation: fraction of source
    triples whose exact `(subject, predicate, object)` key appears
    in re-extracted set. **Brittle to surface-form drift** — e.g.
    source `(alice, graduated, 2012)` vs re-extracted
    `(alice, graduated_in, 2012)` counts as not preserved despite
    identical meaning. Use `fact_preservation_normalized` or
    `semantic_fact_preservation` for an honest measurement; this
    function is retained as a regression check on extractor
    stability.

    Returns 1.0 when source is empty.
    """
    src = {_axiom_key(t) for t in source_triples}
    if not src:
        return 1.0
    reext = {_axiom_key(t) for t in reextracted_triples}
    return len(src & reext) / len(src)


# ── Triple normalization (A3 layer) ──────────────────────────────────
#
# Catches the common surface-form drifts the LLM extractor produces
# under directive pressure (length=0.7, formality=0.3, etc.):
#
#   "graduated"     ↔ "graduated_in"  ↔ "graduated_from"
#   "born_in"       ↔ "was_born_in"   ↔ "was_born"
#   "the_alice"     ↔ "alice"
#   "wrote"         ↔ "authored"     ← NOT caught (true synonym)
#                                       — A1 semantic layer covers this
#
# Free, deterministic, pure rule-based. Empirically catches ~30–50%
# of the "drift" we measured at non-neutral axis positions. The A1
# embedding layer below catches what remains.

_PREDICATE_PREFIX_TRIM: tuple[str, ...] = (
    "was_", "were_", "is_", "are_", "has_", "have_", "had_", "be_",
)
_PREDICATE_SUFFIX_TRIM: tuple[str, ...] = (
    "_in", "_on", "_at", "_for", "_by", "_with", "_to", "_from", "_of",
    "_into", "_onto", "_upon", "_about", "_during", "_after", "_before",
    "_through", "_between", "_against",
)
_ARTICLES: frozenset[str] = frozenset({"a", "an", "the"})


def _normalize_predicate(p: str) -> str:
    """Strip auxiliary prefixes (was_, has_, ...) and preposition
    suffixes (_in, _from, ...) so morphologically-related predicates
    collapse to one form. No stemming — past tense is preserved."""
    p = p.strip().lower()
    for prefix in _PREDICATE_PREFIX_TRIM:
        if p.startswith(prefix) and len(p) > len(prefix):
            p = p[len(prefix):]
            break
    for suffix in _PREDICATE_SUFFIX_TRIM:
        if p.endswith(suffix) and len(p) > len(suffix):
            p = p[: -len(suffix)]
            break
    return p


def _normalize_entity(e: str) -> str:
    """Lowercase + strip leading/trailing articles. Internal underscores
    preserved (so 'the_United_States' becomes 'united_states' but
    'state_of_emergency' is unchanged)."""
    e = e.strip().lower()
    tokens = e.split("_")
    while tokens and tokens[0] in _ARTICLES:
        tokens = tokens[1:]
    while tokens and tokens[-1] in _ARTICLES:
        tokens = tokens[:-1]
    return "_".join(tokens) if tokens else e


def _normalize_triple(t: Triple) -> Triple:
    """Pure function. Same input → same output. Catches preposition
    and auxiliary-verb drift in the predicate plus article noise in
    entities. Does NOT catch true synonyms — that's the semantic
    layer's job."""
    return (_normalize_entity(t[0]), _normalize_predicate(t[1]), _normalize_entity(t[2]))


def fact_preservation_normalized(
    source_triples: Sequence[Triple],
    reextracted_triples: Sequence[Triple],
) -> float:
    """A3 layer: set-based preservation after predicate/entity
    normalization. Catches the common LLM-extractor surface-form
    drifts (graduated / graduated_in / was_graduated) for free.
    Composes with `semantic_fact_preservation` (A1) which handles
    the harder synonym/paraphrase cases.
    """
    src = {_axiom_key(_normalize_triple(t)) for t in source_triples}
    if not src:
        return 1.0
    reext = {_axiom_key(_normalize_triple(t)) for t in reextracted_triples}
    return len(src & reext) / len(src)


# ── Semantic preservation (A1 layer) ─────────────────────────────────


def _cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. No numpy dep — bench scale is small."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# Default cosine threshold. Per the paraphrase-detection literature
# survey (MDPI 2025, sentence-transformers docs), 0.80–0.90 covers
# F1-optimal for text-embedding-3-small-class models; 0.85 is the
# midpoint defensible at v0.2. Tunable per bench.
SEMANTIC_FACT_THRESHOLD: float = 0.85


async def semantic_fact_preservation(
    source_triples: Sequence[Triple],
    reextracted_triples: Sequence[Triple],
    embed_fn: Callable[[str], Awaitable[list[float]]],
    threshold: float = SEMANTIC_FACT_THRESHOLD,
) -> float:
    """A1 layer: semantic preservation via cosine similarity on
    triple-as-text embeddings. For each source triple, greedily
    finds the best-matching unused re-extracted triple by cosine
    similarity. Counts as preserved if best-match similarity ≥
    `threshold`.

    Greedy one-to-one assignment (a re-extracted triple can match at
    most one source triple) so a single overly-generic re-extracted
    fact can't claim credit for multiple source facts.

    Returns 1.0 when source is empty; 0.0 when re-extracted is empty
    but source isn't.
    """
    src_count = len(source_triples)
    if src_count == 0:
        return 1.0
    if not reextracted_triples:
        return 0.0

    src_strs = [f"{s} {p} {o}" for (s, p, o) in source_triples]
    reext_strs = [f"{s} {p} {o}" for (s, p, o) in reextracted_triples]

    # Batch embeddings to halve the round-trip count.
    import asyncio  # local import keeps module-import-time clean
    all_embs = await asyncio.gather(*[embed_fn(s) for s in src_strs + reext_strs])
    src_embs = all_embs[: len(src_strs)]
    reext_embs = all_embs[len(src_strs):]

    used: set[int] = set()
    matched = 0
    for se in src_embs:
        best_j = -1
        best_sim = -1.0
        for j, re_emb in enumerate(reext_embs):
            if j in used:
                continue
            sim = _cosine_sim(se, re_emb)
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_j >= 0 and best_sim >= threshold:
            used.add(best_j)
            matched += 1
    return matched / src_count


def order_preservation(
    source_triples: Sequence[Triple],
    reextracted_triples: Sequence[Triple],
) -> float:
    """Pairwise order-preservation: of the triples that survive the
    round-trip, what fraction retained their relative order from
    source to tome?

    Defends against the MontageLie attack (Zheng et al., 2025) where
    an adversarial render preserves every source triple but reorders
    them into a deceptive narrative. Set-based fact_preservation
    scores 1.000 on such a render; this metric drops toward 0.5
    (random shuffling) or below.

    Returns 1.0 when all preserved pairs retained order, 0.0 when
    all were reversed, NaN when fewer than 2 source triples survive
    (no pairs to compare).

    Source order is the position of each triple in the input
    sequence; tome order is the position in the re-extracted
    sequence (LLM-emitted order during re-extraction).
    """
    source_pos: dict[str, int] = {}
    for i, t in enumerate(source_triples):
        source_pos.setdefault(_axiom_key(t), i)
    tome_pos: dict[str, int] = {}
    for i, t in enumerate(reextracted_triples):
        tome_pos.setdefault(_axiom_key(t), i)

    preserved = sorted(set(source_pos) & set(tome_pos))
    if len(preserved) < 2:
        return float("nan")

    correct = total = 0
    for i, a in enumerate(preserved):
        for b in preserved[i + 1:]:
            sa, sb = source_pos[a], source_pos[b]
            ta, tb = tome_pos[a], tome_pos[b]
            if sa == sb or ta == tb:
                continue
            if (sa < sb) == (ta < tb):
                correct += 1
            total += 1
    return correct / total if total > 0 else float("nan")


def measure_drift(
    source_triples: Sequence[Triple],
    reextracted_triples: Sequence[Triple],
    tome: str,
    sliders: TomeSliders,
) -> tuple[AxisDrift, ...]:
    """Per-axis drift between source and re-extracted triples + the tome
    text. Pure function. Deterministic per input. Formulas in
    docs/SLIDER_CONTRACT.md.

    Note: callers must pass `quantized` sliders for thresholds to map
    onto the 5-bin grid. Passing un-quantized sliders works but length
    band lookup will fail with KeyError if value isn't a bin centre."""
    source_keys = {_axiom_key(t) for t in source_triples}
    reextracted_keys = {_axiom_key(t) for t in reextracted_triples}

    # ── Density ──────────────────────────────────────────────────────
    n_source = len(source_keys)
    expected_retained = math.floor(n_source * sliders.density) if n_source else 0
    actual_retained = len(source_keys & reextracted_keys)
    if expected_retained == 0:
        # density rounds to 0 ⇒ no triples expected to survive; drift is
        # zero unless the LLM hallucinated some in.
        density_drift = 0.0 if actual_retained == 0 else float(actual_retained)
    else:
        density_drift = abs(1.0 - (actual_retained / expected_retained))

    # ── Length ───────────────────────────────────────────────────────
    n_words = len(tome.split())
    band_lo, band_hi = _LENGTH_BANDS[snap_to_bin(sliders.length)]
    target_words = ((band_lo + band_hi) / 2) * max(n_source, 1)
    if target_words == 0:
        length_drift = 0.0
    else:
        length_drift = abs(n_words - target_words) / target_words

    # ── Formality ────────────────────────────────────────────────────
    target_formal = sliders.formality
    actual_formal = _formal_score(tome)
    formality_drift = abs(target_formal - actual_formal)

    # ── Audience ─────────────────────────────────────────────────────
    target_jargon = sliders.audience * 0.30  # contract: max ~30% at audience=1.0
    actual_jargon = _jargon_density(tome)
    audience_drift = abs(target_jargon - actual_jargon)

    # ── Perspective ──────────────────────────────────────────────────
    target_first_person = 1.0 - sliders.perspective
    actual_first_person = _first_person_ratio(tome)
    perspective_drift = abs(target_first_person - actual_first_person)

    return (
        AxisDrift(DriftAxis.DENSITY, density_drift, _THRESHOLDS["density"],
                  _classify(density_drift, _THRESHOLDS["density"])),
        AxisDrift(DriftAxis.LENGTH, length_drift, _THRESHOLDS["length"],
                  _classify(length_drift, _THRESHOLDS["length"])),
        AxisDrift(DriftAxis.FORMALITY, formality_drift, _THRESHOLDS["formality"],
                  _classify(formality_drift, _THRESHOLDS["formality"])),
        AxisDrift(DriftAxis.AUDIENCE, audience_drift, _THRESHOLDS["audience"],
                  _classify(audience_drift, _THRESHOLDS["audience"])),
        AxisDrift(DriftAxis.PERSPECTIVE, perspective_drift, _THRESHOLDS["perspective"],
                  _classify(perspective_drift, _THRESHOLDS["perspective"])),
    )


async def render(
    triples: Sequence[Triple],
    sliders: TomeSliders,
    llm: LLMChatClient,
    extractor: TripleExtractor,
    cache: Optional[SliderCache] = None,
    cache_ttl_seconds: int = 24 * 60 * 60,
) -> RenderResult:
    """Render a tome from triples at the requested slider position.

    Cache-first; canonical path (no LLM) when only density is non-default;
    LLM path otherwise. See module docstring for the pipeline diagram and
    docs/SLIDER_CONTRACT.md for the per-axis drift contract."""
    t_start = time.monotonic()
    quantized = quantize(sliders)
    triples_tuple = tuple(tuple(t) for t in triples)
    key = cache_key(triples_tuple, quantized)

    # ── Cache lookup ─────────────────────────────────────────────────
    if cache is not None:
        hit = await cache.get(key)
        if hit is not None:
            # Re-stamp cache_status without rebuilding the rest. Frozen
            # dataclass ⇒ construct a new copy.
            return RenderResult(
                tome=hit.tome,
                triples_used=hit.triples_used,
                reextracted_triples=hit.reextracted_triples,
                claimed_triples=hit.claimed_triples,
                drift=hit.drift,
                cache_status=CacheStatus.HIT,
                llm_calls_made=0,
                wall_clock_ms=int((time.monotonic() - t_start) * 1000),
                quantized_sliders=hit.quantized_sliders,
                render_id=hit.render_id,
            )

    # ── Apply density (deterministic axiom subset) ───────────────────
    keys_in_order = [_axiom_key(t) for t in triples_tuple]
    kept_key_set = set(apply_density(keys_in_order, sliders.density))
    kept_triples: tuple[Triple, ...] = tuple(
        t for t in triples_tuple if _axiom_key(t) in kept_key_set
    )

    # ── Choose path: canonical (deterministic) vs LLM ────────────────
    claimed: list[Triple] = []
    if not sliders.requires_extrapolator():
        # All LLM axes are neutral ⇒ skip LLM entirely. The deterministic
        # tome is its own re-extraction (no drift introduced). Claimed
        # equals re-extracted equals the post-density set on this path.
        tome = _deterministic_tome(kept_triples)
        reextracted: list[Triple] = list(kept_triples)
        claimed = list(kept_triples)
        llm_calls = 0
    else:
        sys_prompt = build_system_prompt(quantized)
        user_prompt = _format_triples_for_llm(kept_triples)
        # v0.3: prefer structured decoding when the LLM client supports
        # it. Pydantic-schema-enforced render returns (tome, claimed)
        # in one call; we still cross-check via independent re-extraction
        # so the LLM's self-attestation is a signal, not the source of
        # truth. Falls back to plain chat_completion + empty claimed
        # for legacy clients (e.g. tests without the structured method).
        if hasattr(llm, "chat_completion_structured"):
            tome, claimed_raw = await llm.chat_completion_structured(
                sys_prompt, user_prompt,
            )
            claimed = [tuple(t) for t in claimed_raw]
        else:
            tome = await llm.chat_completion(sys_prompt, user_prompt)
            claimed = []
        reextracted = await extractor(tome)
        llm_calls = 1

    # measure_drift compares re-extracted facts against the ORIGINAL
    # source set (not the post-density kept set), because density drift
    # is "did the renderer keep the right COUNT of source facts?" — only
    # answerable against the original. Passing kept_triples here would
    # double-apply density and break the formula at non-1.0 settings.
    drift = measure_drift(triples_tuple, reextracted, tome, quantized)

    render_id = hashlib.sha256(
        (key + tome).encode("utf-8")
    ).hexdigest()[:16]

    cache_status = CacheStatus.MISS if cache is not None else CacheStatus.BYPASS
    result = RenderResult(
        tome=tome,
        triples_used=kept_triples,
        reextracted_triples=tuple(tuple(t) for t in reextracted),
        claimed_triples=tuple(tuple(t) for t in claimed),
        drift=drift,
        cache_status=cache_status,
        llm_calls_made=llm_calls,
        wall_clock_ms=int((time.monotonic() - t_start) * 1000),
        quantized_sliders=quantized,
        render_id=render_id,
    )

    if cache is not None:
        await cache.put(key, result, cache_ttl_seconds)

    return result


# ─── In-memory cache (default for tests + single-process use) ─────────


@dataclass
class InMemorySliderCache:
    """Simple dict-backed cache. Not for production multi-process use.
    The Worker uses worker/src/cache/bin_cache.ts (KV-backed) for that.
    """

    _data: dict[str, RenderResult] = field(default_factory=dict)
    _hits: int = 0
    _misses: int = 0

    async def get(self, key: str) -> Optional[RenderResult]:
        result = self._data.get(key)
        if result is None:
            self._misses += 1
        else:
            self._hits += 1
        return result

    async def put(self, key: str, value: RenderResult, ttl_seconds: int) -> None:
        # In-memory: no TTL enforcement (process is the lifetime).
        # The Worker KV cache enforces TTL on its side.
        self._data[key] = value

    async def stats(self) -> dict[str, int]:
        return {
            "size": len(self._data),
            "hits": self._hits,
            "misses": self._misses,
        }
