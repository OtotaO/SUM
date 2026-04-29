"""Unit tests for the syntactic-sieve noise filter.

Pinned behaviours:

  1. ``_is_noise_component`` returns True for empty/whitespace,
     single-char strings, components with markdown/code/table
     punctuation (``|``, ``\\``, `````, ``*``, ``#``, ``<``, ``>``,
     ``[``, ``]``, ``{``, ``}``, ``(``, ``)``, ``=``, ``/``), and
     path-like substrings (``://``, ``.md``, ``.py``, ``](``).

  2. ``_is_clean_triple`` rejects a triple if ANY component is noise.

  3. ``DeterministicSieve.extract_triplets`` filters noisy triples
     before returning. The README's known noise cases
     (``('|', 'close', 'this')``, ``('#', 'license', 'apache')``,
     etc.) are confirmed absent from the output.

  4. Legitimate prose triples (``('alice', 'like', 'cats')``) are
     NOT filtered — the heuristic is conservative.

Author: ototao
License: Apache License 2.0
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Component-level filter
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "component,is_noise",
    [
        # ---- Round-1 (PR #91) ----
        # Empty / whitespace
        ("", True),
        ("   ", True),
        ("\t\n", True),
        # Single-character noise
        ("|", True),
        ("\\", True),
        ("#", True),
        ("*", True),
        ("=", True),
        ("/", True),
        ("a", True),  # lone letter — too thin
        ("1", True),  # bare digit
        # Multi-char alpha — kept
        ("alice", False),
        ("marie_curie", False),
        ("relativity", False),
        ("proposed", False),
        # Forbidden punctuation embedded
        ("alice|bob", True),
        ("foo*bar", True),
        ("foo[bar]", True),
        ("foo(bar)", True),
        ("foo{bar}", True),
        ("foo<bar>", True),
        ("foo=bar", True),
        ("foo/bar", True),
        # Path / URL / link residue
        ("https://example.com", True),
        ("docs/PROOF.md", True),  # contains '/' AND '.md'
        ("foo](bar", True),
        # Pure punctuation (no alphanumeric)
        ("...", True),
        ("§§§", True),
        # Borderline: alphanumeric with one allowed punctuation
        ("bundle.is_delta", False),  # dot is fine; legit code identifier
        # ---- Round-2 (this PR) — quantity / range / hash filters ----
        # Currency / measurement / range markers
        ("~$0.12", True),       # currency tilde
        ("21.1×_figure", True), # multiplication sign
        ("0.94–0.96", True),    # en-dash range
        ("1.0→0.9", True),      # arrow (transformation)
        ("80%+", True),         # percent + plus
        ("%_drift", True),      # bare percent
        ("$5.00", True),        # currency
        ("10°C", True),         # degree
        ("±0.05", True),        # plus-minus
        ("≈3.14", True),        # approx
        # Pure decimal numbers (NOT bare integers — those are years)
        ("0.5750", True),
        ("10.2", True),
        ("21.1", True),
        ("1.0", True),
        # Letter-prefixed versions like ``v0.4.0`` are intentionally kept:
        # the bare-decimal regex requires a leading digit, and these
        # often appear as legitimate semantic content in release notes.
        ("v0.4.0", False),
        ("shipped_v1.2.0", False),
        # Bare integer (year-like) — KEPT, legit semantic content
        ("2026", False),
        ("1989", False),
        # Long hash / hex strings
        ("a" * 81, True),       # over length cap
        ("deadbeefdeadbeefdeadbeef", True),  # hex-only ≥16 chars
        ("0123456789abcdef0123456789", True),  # hex-only sha-ish
        # Hex < 16 chars: kept (could be a real word like 'cafe')
        ("cafe", False),
        ("dead", False),
        # Reasonable-length tokens that mix digits + letters but no noise marker
        ("v1", False),         # short version → kept
        ("ipv6", False),
        ("3d_render", False),
    ],
)
def test_is_noise_component(component, is_noise):
    from sum_engine_internal.algorithms.syntactic_sieve import _is_noise_component
    assert _is_noise_component(component) is is_noise, (
        f"_is_noise_component({component!r}) returned the wrong answer "
        f"(expected {is_noise})"
    )


# --------------------------------------------------------------------------
# Triple-level filter
# --------------------------------------------------------------------------


def test_is_clean_triple_rejects_any_noisy_component():
    from sum_engine_internal.algorithms.syntactic_sieve import _is_clean_triple
    # All three clean → kept
    assert _is_clean_triple(("alice", "likes", "cats"))
    # One noisy → rejected
    assert not _is_clean_triple(("|", "close", "this"))
    assert not _is_clean_triple(("alice", "*", "bob"))
    assert not _is_clean_triple(("alice", "wrote", "[Hamlet]"))


# --------------------------------------------------------------------------
# Sieve integration: README round-trip
# --------------------------------------------------------------------------


spacy = pytest.importorskip("spacy")


def _has_en_core_web_sm() -> bool:
    try:
        spacy.load("en_core_web_sm")
        return True
    except (OSError, ImportError):
        return False


@pytest.mark.skipif(
    not _has_en_core_web_sm(),
    reason="en_core_web_sm not installed",
)
def test_sieve_filters_known_readme_noise():
    """The README's known noise patterns (table-cell pipe, lone
    punctuation, link residue) MUST NOT appear in extract_triplets'
    output. Documents the canonical sieve-quality contract."""
    from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve

    sieve = DeterministicSieve()
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    triples = sieve.extract_triplets(text)

    forbidden_subjects = {"|", "#", "*", "\\", "✓", "§", "1"}
    seen_subjects = {s for s, _, _ in triples}
    overlap = seen_subjects & forbidden_subjects
    assert not overlap, f"sieve emitted noise subject(s): {overlap}"

    # Forbid components containing pipe (catches ('|', ...) AND
    # multi-char pipe-bearing strings).
    bad = [t for t in triples if any("|" in c for c in t)]
    assert not bad, f"sieve emitted pipe-bearing triples: {bad[:5]}"

    # Forbid markdown-link residue subjects ending in `.md`](`
    md_residue = [t for t in triples if any(".md" in c for c in t)]
    assert not md_residue, f"sieve emitted .md residue triples: {md_residue[:5]}"


@pytest.mark.skipif(
    not _has_en_core_web_sm(),
    reason="en_core_web_sm not installed",
)
def test_sieve_keeps_legitimate_prose_triples():
    """The filter MUST NOT strip clean prose-derived triples. If
    over-eager, it would silently drop content."""
    from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve

    sieve = DeterministicSieve()
    text = (
        "Marie Curie won two Nobel Prizes. "
        "Einstein proposed relativity. "
        "Shakespeare wrote Hamlet."
    )
    triples = sieve.extract_triplets(text)
    # Don't pin exact triples (spaCy version-dependent); just assert
    # that we got SOME multi-component clean triples back.
    assert len(triples) >= 2, f"sieve over-filtered prose: only {triples}"
    for s, p, o in triples:
        assert s and p and o, f"empty component in {(s, p, o)}"
        assert all(c.isalnum() or c == "_" for c in s.replace(" ", "")), s
