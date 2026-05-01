"""Synthetic micro-benchmark: does v1 separate clean from adversarial?

This is the small empirical test the spec's P1 / P2 predictions
demand. Six hand-built fact-sets, each with:

  - 3 "clean" renderings (paraphrased; all source entities present)
  - 5 "adversarial" renderings, one per perturbation class:
      A1: entity-swap (one source entity replaced by another)
      A2: predicate-flip (predicate of one triple changed)
      A3: fact-fabrication (extra triple over off-graph entities)
      A4: triple-drop (one triple removed; isolated endpoints disappear)
      A5: consistent-swap (the SAME swap applied to all 'adversarial'
                           renders — the consistent-hallucination case
                           the spec §6 explicitly disclaims as out of scope)

Reports:
  - per-class mean & std of the Laplacian quadratic form V
  - clean-vs-adversarial separation ratio (mean ratio of V_adv / V_clean)
  - per-edge top-1 localization accuracy on classes the detector catches
  - explicit honest negative findings on classes v1 misses

This is NOT calibrated against ground-truth labels yet — it's the
"does the math give us a signal at all" check before investing in
the proper Week-2 benchmark.
"""
from __future__ import annotations

import statistics
from sum_engine_internal.research.sheaf_laplacian import (
    KnowledgeSheaf,
    consistency_profile,
    laplacian_quadratic_form,
    cochain_from_extracted,
    per_edge_discrepancy,
)


# Six hand-built fact-sets. Each is a connected directed graph with
# 4 entities and 3 predicates so each entity participates in ≥1 edge.

FACT_SETS = [
    # 1. Linear chain
    [("alice", "knows", "bob"),
     ("bob", "manages", "carol"),
     ("carol", "owns", "dog")],
    # 2. Star with center
    [("hub", "links", "spoke_a"),
     ("hub", "links", "spoke_b"),
     ("hub", "links", "spoke_c")],
    # 3. Two-step provenance
    [("paper", "cites", "theorem"),
     ("theorem", "proves", "claim"),
     ("claim", "implies", "result")],
    # 4. Branching
    [("ceo", "leads", "company"),
     ("company", "owns", "factory"),
     ("company", "owns", "office")],
    # 5. Causality chain
    [("storm", "caused", "outage"),
     ("outage", "delayed", "shipment"),
     ("shipment", "missed", "deadline")],
    # 6. Knowledge claim
    [("einstein", "proposed", "relativity"),
     ("relativity", "explains", "gravitation"),
     ("gravitation", "bends", "spacetime")],
]


def clean_renders(triples):
    """3 paraphrases that mention all source entities. Trivially clean
    in v1 (presence stalks): each cochain = (1, 1, ..., 1) ⇒ V = 0."""
    return [list(triples), list(triples), list(triples)]


def adversarial_swap(triples):
    """A1: swap the FIRST source entity with a different source entity.

    Concrete: in fact-set 1, replace 'alice' (subject of edge 0) with
    'carol' everywhere in the render. The render now mentions
    {carol, bob, carol, dog} = {carol, bob, dog} as entities. The
    source vertex 'alice' drops to 0 in the cochain.
    """
    swap_from = triples[0][0]               # first subject
    # pick a swap target that's also in the source vertex set
    other_entities = [t[0] for t in triples[1:]] + [t[2] for t in triples[1:]]
    swap_to = next((e for e in other_entities if e != swap_from), swap_from + "_x")
    return [(s if s != swap_from else swap_to,
             p,
             o if o != swap_from else swap_to)
            for (s, p, o) in triples]


def adversarial_predicate_flip(triples):
    """A2: change the predicate of the first triple. v1 ignores
    predicates so this is a known-MISSED class."""
    s, p, o = triples[0]
    return [(s, p + "_FLIPPED", o), *triples[1:]]


def adversarial_fact_fabrication(triples):
    """A3: add a triple with off-graph entities. v1 ignores anything
    outside the source vertex set so this is also known-MISSED."""
    return [*triples, ("ghost_subject", "fabricated", "ghost_object")]


def adversarial_triple_drop(triples):
    """A4: drop the first triple. If its endpoints don't appear in
    another edge, they vanish from the render."""
    return list(triples[1:])


def adversarial_consistent_swap(triples):
    """A5: same as A1 but applied identically to ALL renders in the
    manifold. v1 cannot distinguish this from clean (no per-render
    variance) — the consistent-hallucination failure mode the spec
    §6 explicitly disclaims."""
    return adversarial_swap(triples)


PERTURBATION_FNS = {
    "A1_entity_swap":             adversarial_swap,
    "A2_predicate_flip":          adversarial_predicate_flip,
    "A3_fact_fabrication":        adversarial_fact_fabrication,
    "A4_triple_drop":             adversarial_triple_drop,
    "A5_consistent_swap_x3":      adversarial_consistent_swap,
}


def bench_one(triples, perturbation_name, perturbation_fn):
    """For one fact-set + one perturbation class, measure V on
    (3 clean) and on (1 perturbed) and report the gap."""
    sheaf = KnowledgeSheaf.from_triples(triples)
    clean = clean_renders(triples)
    if perturbation_name == "A5_consistent_swap_x3":
        # For A5, the manifold is 3 perturbed renders ("consistent
        # hallucination" — every render makes the same lie). Compare
        # to the 3-clean baseline.
        adversarial_manifold = [perturbation_fn(triples) for _ in range(3)]
    else:
        adversarial_manifold = [perturbation_fn(triples)]

    profile_clean = consistency_profile(triples, clean)
    profile_adv = consistency_profile(triples, adversarial_manifold)
    return {
        "perturbation": perturbation_name,
        "n_entities": len(sheaf.vertices),
        "n_edges": len(sheaf.edges),
        "V_clean_mean": profile_clean["mean_laplacian"],
        "V_adv_mean": profile_adv["mean_laplacian"],
        "V_adv_max": profile_adv["max_per_render"],
        "top1_localization": profile_adv["per_edge_top3_argmax_render"][:1] if profile_adv["per_edge_top3_argmax_render"] else [],
    }


def main():
    print(f"{'fact-set':<10} {'perturbation':<25} {'V_clean':>8} {'V_adv':>8} {'V_adv_max':>10} {'caught?':<8}")
    print("-" * 80)

    rows: list[dict] = []
    for fs_i, triples in enumerate(FACT_SETS):
        for pert_name, pert_fn in PERTURBATION_FNS.items():
            r = bench_one(triples, pert_name, pert_fn)
            r["fact_set"] = fs_i + 1
            rows.append(r)
            caught = "✓" if r["V_adv_mean"] > r["V_clean_mean"] + 1e-9 else "✗"
            print(
                f"{fs_i+1:<10} {pert_name:<25} "
                f"{r['V_clean_mean']:>8.3f} {r['V_adv_mean']:>8.3f} "
                f"{r['V_adv_max']:>10.3f} {caught:<8}"
            )

    print()
    print("Per-class summary:")
    print(f"{'perturbation':<25} {'V_adv_mean':>12} {'detect_rate':>14}")
    print("-" * 60)
    for pert_name in PERTURBATION_FNS:
        rs = [r for r in rows if r["perturbation"] == pert_name]
        v_adv = [r["V_adv_mean"] for r in rs]
        caught = sum(1 for r in rs if r["V_adv_mean"] > r["V_clean_mean"] + 1e-9)
        print(
            f"{pert_name:<25} "
            f"{statistics.mean(v_adv):>12.3f} "
            f"{caught}/{len(rs):>10}"
        )

    print()
    print("Localization spot-check (top-1 edge from the worst-V adversarial render):")
    for r in rows[:6]:                  # show first 6 across mixed perturbations
        if r["top1_localization"]:
            edge, score = r["top1_localization"][0]
            print(f"  fact-set {r['fact_set']:<2} {r['perturbation']:<25} "
                  f"top edge: {edge}  score={score:.3f}")

    return rows


if __name__ == "__main__":
    main()
