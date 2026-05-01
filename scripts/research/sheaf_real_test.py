"""v1 sheaf-Laplacian detector against real prose, real sieve extraction.

No LLM API calls. No Worker round-trip. The test:

  1. Write 4 paraphrases of the same paragraph, all preserving the
     same load-bearing facts. (The paraphrases are HUMAN-AUTHORED
     here so the experiment is reproducible without paying an LLM
     vendor.)
  2. Attest paraphrase 1 via `sum attest` → source bundle.
  3. Render the bundle at density {1.0, 0.7, 0.5} → 3 deterministic
     SUM renders.
  4. Run the sieve on paraphrases 2, 3, 4 (treating each as an
     independent "render" of the same fact-set).
  5. Build cochains over the source bundle's induced graph, compute
     the v1 Laplacian quadratic form for each render.
  6. Report the consistency profile honestly.

Honest expectations BEFORE running:
  - render 1 (density=1.0): V should be very small (re-extraction of
    a canonical tome should mention every source entity).
  - render 2 (density=0.7): V > render 1 (some axioms dropped).
  - render 3 (density=0.5): V > render 2 (more axioms dropped).
  - paraphrases 2-4: V depends on whether the sieve extracts the
    SAME entity tokens from paraphrased prose. If sieve canonicalises
    "Alice" and "the woman named Alice" to the same token, V is small.
    If not, V will be larger — and that's a HONEST signal of
    sieve-canonicalisation drift, not detector failure.

The point of this test: does the v1 signal mean anything on real
prose, or does sieve extraction noise drown it out? If V on
paraphrases is comparable to V on heavily-density-filtered renders,
v1 is NOT useful as a hallucination detector on naturalistic input
— it can't distinguish lawful paraphrase variation from genuine
axiom dropout. That would be a falsification.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from sum_engine_internal.research.sheaf_laplacian import (
    KnowledgeSheaf,
    consistency_profile,
    laplacian_quadratic_form,
    cochain_from_extracted,
    per_edge_discrepancy,
)


# ── Source corpus ─────────────────────────────────────────────────────
#
# Four paraphrases of the same factual content. All preserve:
#   alice graduated MIT 2012
#   bob owns dog
#   carol writes python
#   einstein proposed relativity

PARAPHRASES = [
    # Paraphrase 1 — the source bundle's prose
    "Alice graduated from MIT in 2012. Bob owns a dog. "
    "Carol writes Python. Einstein proposed relativity.",

    # Paraphrase 2 — short reorderings, same facts
    "Einstein proposed relativity. Bob owns a dog. "
    "Alice graduated from MIT in 2012. Carol writes Python.",

    # Paraphrase 3 — slightly more verbose, same facts
    "In the year 2012, Alice graduated from MIT. Bob is the owner of a dog. "
    "Carol writes Python code. Einstein proposed the theory of relativity.",

    # Paraphrase 4 — minor lexical variation, same facts
    "Alice graduated MIT in 2012. Bob owns the dog. "
    "Carol writes Python. Einstein proposed relativity.",
]


def shell(cmd: list[str], stdin_text: str | None = None) -> str:
    """Run a shell command, return stdout."""
    proc = subprocess.run(
        cmd,
        input=stdin_text,
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\nstderr: {proc.stderr}"
        )
    return proc.stdout


def attest_to_bundle(prose: str) -> dict:
    """Run `sum attest --extractor=sieve` on prose, return parsed bundle."""
    out = shell(
        ["python", "-m", "sum_cli.main", "attest", "--extractor=sieve"],
        stdin_text=prose,
    )
    return json.loads(out)


def render_bundle(bundle: dict, density: float) -> str:
    """Run `sum render --density D` on the bundle, return tome text."""
    return shell(
        ["python", "-m", "sum_cli.main", "render", f"--density={density}"],
        stdin_text=json.dumps(bundle),
    )


def parse_tome_triples(tome: str) -> list[tuple[str, str, str]]:
    """Re-parse `The S P O.` lines from a canonical tome."""
    import re
    pattern = re.compile(r"^The (\S+) (\S+) (.+)\.$")
    triples = []
    for line in tome.splitlines():
        m = pattern.match(line.strip())
        if m:
            triples.append((m.group(1), m.group(2), m.group(3)))
    return triples


def sieve_extract(prose: str) -> list[tuple[str, str, str]]:
    """Run the sieve extractor on prose, return triples."""
    # Use sum_engine_internal.algorithms.syntactic_sieve directly to get
    # triples without the bundle envelope.
    from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve
    sieve = DeterministicSieve()
    return list(sieve.extract_triplets(prose))


def main():
    print("=" * 72)
    print("v1 sheaf-Laplacian detector — real-prose test")
    print("=" * 72)

    # ── Step 1: attest paraphrase 1 to get the source bundle ──────────
    print("\n[1] Attesting source paraphrase via `sum attest --extractor=sieve`…")
    source_bundle = attest_to_bundle(PARAPHRASES[0])
    source_triples = parse_tome_triples(source_bundle["canonical_tome"])
    print(f"    source bundle: {len(source_triples)} triples")
    for t in source_triples:
        print(f"      {t}")
    state_int = source_bundle["state_integer"]
    print(f"    state_integer: ...{state_int[-32:]}")

    # ── Step 2: 3 deterministic renders at varying density ────────────
    print("\n[2] Rendering source bundle at density ∈ {1.0, 0.7, 0.5}…")
    deterministic_renders: dict[float, list[tuple[str, str, str]]] = {}
    for d in (1.0, 0.7, 0.5):
        tome = render_bundle(source_bundle, d)
        re_extracted = parse_tome_triples(tome)
        deterministic_renders[d] = re_extracted
        print(f"    density={d}: {len(re_extracted)} triples re-extracted from tome")

    # ── Step 3: sieve-extract from paraphrases 2-4 ────────────────────
    print("\n[3] Sieve-extracting from paraphrases 2-4…")
    paraphrase_extractions: list[list[tuple[str, str, str]]] = []
    for i, prose in enumerate(PARAPHRASES[1:], start=2):
        triples = sieve_extract(prose)
        paraphrase_extractions.append(triples)
        print(f"    paraphrase {i}: {len(triples)} triples")
        for t in triples:
            print(f"      {t}")

    # ── Step 4: build the manifold and apply v1 detector ──────────────
    print("\n[4] Building 6-render manifold + computing consistency profile…")
    print("      render 1: density=1.0 deterministic")
    print("      render 2: density=0.7 deterministic (some axioms dropped)")
    print("      render 3: density=0.5 deterministic (more axioms dropped)")
    print("      render 4: paraphrase 2 (reordered)")
    print("      render 5: paraphrase 3 (more verbose)")
    print("      render 6: paraphrase 4 (minor lexical variation)")

    manifold = [
        deterministic_renders[1.0],
        deterministic_renders[0.7],
        deterministic_renders[0.5],
        *paraphrase_extractions,
    ]

    profile = consistency_profile(source_triples, manifold)
    print(f"\n    profile.render_count = {profile['render_count']}")
    print(f"    profile.mean_laplacian = {profile['mean_laplacian']:.4f}")
    print(f"    profile.std_laplacian = {profile['std_laplacian']:.4f}")
    print(f"    profile.max_per_render = {profile['max_per_render']:.4f}")
    print(f"    profile.argmax_render_idx = {profile['argmax_render_idx']}")

    # Per-render breakdown
    print("\n[5] Per-render Laplacian quadratic form:")
    sheaf = KnowledgeSheaf.from_triples(source_triples)
    LABELS = [
        "render 1  density=1.0",
        "render 2  density=0.7",
        "render 3  density=0.5",
        "render 4  paraphrase 2 (reordered)",
        "render 5  paraphrase 3 (more verbose)",
        "render 6  paraphrase 4 (lexical variation)",
    ]
    for label, triples in zip(LABELS, manifold):
        x = cochain_from_extracted(sheaf, triples)
        v = laplacian_quadratic_form(sheaf, x)
        # Top edge if non-zero
        if v > 1e-9:
            top = per_edge_discrepancy(sheaf, x)[0]
            print(f"    {label:<40} V = {v:>6.3f}  top edge: {top[0]} (score {top[1]:.3f})")
        else:
            print(f"    {label:<40} V = {v:>6.3f}  (consistent)")

    # ── Step 5: honest interpretation ─────────────────────────────────
    print("\n[6] Interpretation:")
    p_d10 = profile["per_render_v"][0]
    p_d07 = profile["per_render_v"][1]
    p_d05 = profile["per_render_v"][2]
    p_para = profile["per_render_v"][3:]

    if p_d10 < 1e-9:
        print("    ✓ density=1.0 render is consistent (V = 0). Round-trip clean.")
    else:
        print(f"    ⚠ density=1.0 render has V={p_d10:.3f} — re-extraction "
              f"missed an entity. Surprising; investigate sieve.")

    if p_d07 > p_d10 and p_d05 > p_d07:
        print("    ✓ density-decreasing renders show V increasing as predicted.")
    else:
        print("    ⚠ density ordering does NOT match prediction.")

    avg_paraphrase = sum(p_para) / len(p_para) if p_para else 0.0
    print(f"    paraphrase-extraction mean V = {avg_paraphrase:.3f}")
    print(f"    density=0.5 V = {p_d05:.3f}")
    if avg_paraphrase < p_d05:
        ratio = p_d05 / avg_paraphrase if avg_paraphrase > 1e-9 else float("inf")
        print(f"    ✓ density=0.5 V > paraphrase mean (ratio = {ratio:.2f}x).")
        print(f"      v1 distinguishes lawful paraphrase variation from")
        print(f"      genuine axiom dropout on this small sample.")
    elif abs(avg_paraphrase - p_d05) < 0.1:
        print(f"    ⚠ paraphrase V ≈ density=0.5 V — v1 cannot distinguish")
        print(f"      paraphrase noise from genuine axiom drop. Falsification")
        print(f"      on this corpus; v2 (semantic stalks) needed.")
    else:
        print(f"    ⚠ paraphrase V > density=0.5 V — sieve extraction noise")
        print(f"      dominates the signal. v1 is NOT useful here.")


if __name__ == "__main__":
    main()
