# Categorical Foundations — the §2.5.1 fixed-point in compact-closed vocabulary

**Status: epistemic-status `categorical-reading`.** This document recasts SUM's empirically-grounded §2.5 closure result and its T1 / T4 receipts in the categorical-algebra vocabulary of Coecke / Sadrzadeh / Clark (DisCoCat, 2010) and the follow-up Frobenius-algebra extension (Kartsaklis / Sadrzadeh, 2015). It is *additive* over the proof-boundary discipline — every load-bearing claim already lives in PROOF_BOUNDARY.md with its existing epistemic status. This document offers a vocabulary, not a new claim.

Per the proof-boundary discipline, "categorical reading" is a fourth status alongside `provable`, `empirical-benchmark`, and `certified`. It means: *this statement is a recasting of an existing claim in mathematical vocabulary that lets external categorical-algebra reviewers locate it inside their tradition.* Recastings do not change what is proved or measured; they let third-party reviewers in adjacent fields ground their priors.

## 1. The thesis

The §2.5 closure result, ratified by the T1 + T4 receipts on `main`, is:

> Across all three measured corpora, `extract ∘ generate ∘ extract = extract` is **composition-stable** under K-step iteration, with `drift_K = drift_1` to within DKW 95% bound by a margin ≥17×. The best-fitting composition law is **fixed-point** (`drift_K = drift_1`).

In categorical vocabulary: the `extract` morphism is an **idempotent endomorphism** on the bundle-object, witnessed empirically by T1 (iteration-stability) and structurally by T4 (fixed-point composition law). This is a real categorical structure, and naming it as such lets reviewers familiar with compact-closed-category arguments place SUM inside the DisCoCat (Coecke / Sadrzadeh / Clark, 2010) tradition rather than inventing a new framework to read it.

## 2. The relevant fragment of DisCoCat

Coecke / Sadrzadeh / Clark (arXiv:1003.4394, 2010) construct a category-theoretic framework that pairs:
- A **distributional semantics**: word meanings as vectors in a high-dimensional space (Firth: "you shall know a word by the company it keeps").
- A **Pregroup grammar** (Lambek, 1999): grammatical types with left and right adjoints, juxtaposition as monoidal tensor, type reductions verifying well-formedness.

Both are **compact closed categories** (Kelly / Laplaza, 1980). DisCoCat takes the product category `FVect × P` whose objects are pairs `(V, p)` of a vector space and a grammatical type. Sentence meaning is a morphism that "lifts" grammatical type reductions to linear maps on the tensor product of word-meaning vector spaces. Sentences become vectors in a fixed sentence space `S`, comparable by inner product regardless of grammatical structure.

The 2010 paper's §6 also names a **Boolean-restricted variant**: replace `FVect` with `FRel` (sets and relations, with cartesian product as tensor) and the framework collapses to a Montague-style truth-valued semantics. *This is the variant SUM's substrate fits.*

## 3. SUM's primitives in the FRel × P fragment

SUM's compose substrate (`sum.transform.compose`, `compose._bundle_triples`, the LCM-of-primes encoding in `GodelStateAlgebra.encode_chunk_state`) does not work with weighted vectors in `R^n`. It works with **subsets of a universe of canonical axiom keys** — each axiom is either present (encoded as its prime factor) or absent (omitted from the LCM). This is precisely the Boolean-restricted case: `B = {0, 1}` valuations, no real-valued weights, set-flavoured composition.

The categorical mapping is:

| SUM primitive | FRel × P object/morphism |
|---|---|
| Universe of canonical axiom keys `X` | An object `X` in `FRel` whose underlying set is the universe of `(subject, predicate, object)` triples after canonicalisation |
| A bundle's axiom set | A subset `A ⊆ X` = a relation `{*} → X` = a morphism in `FRel` |
| `compose --op union` (LCM-merge of state integers for distinct primes) | The set-union operation `A ∪ B`; commutative, associative, idempotent, with identity `∅`. A **commutative idempotent monoid** on subsets of `X` |
| Canonical tome reconstruction | Identity morphism on the bundle-object: `(parse ∘ canonical)(S) = S` |
| `extract` from prose to axiom set | A morphism `Text → Subsets(X)` in `FRel × P` |
| `generate` from axiom set to prose | A morphism `Subsets(X) → Text` (lossy on prose form; lossless on axiom set under §2.5 closure) |

The grammatical-type half of `FRel × P` is not actively used by SUM — sieve extracts SVO triples without performing a Pregroup analysis. This is a deliberate non-adoption (see §6).

## 4. The §2.5 closure as an idempotent endomorphism

Define the round-trip morphism on the axiom-set object:

```
roundtrip := extract ∘ generate : Subsets(X) → Subsets(X)
```

The §2.5 closure claim, in this vocabulary, is:

```
roundtrip ∘ extract  =  extract              (1)
```

which, since `extract` is a morphism `Text → Subsets(X)`, is the statement that the composite `roundtrip ∘ extract` equals the original `extract`. Applying `roundtrip` to the output of `extract` is a no-op on the axiom-set side.

Equivalently, restricted to the image of `extract`, the morphism `roundtrip` satisfies:

```
roundtrip ∘ roundtrip  =  roundtrip          (2)
```

which is the textbook **idempotent endomorphism** condition. T1 verifies (1) and (2) up to K=10 iterations on three corpora; T4 ratifies (2) as the best-fitting composition law over the four candidates (additive / multiplicative-survival / saturating / fixed-point) by minimum sum-of-squared-residuals against per-K median drift.

In a compact closed category, idempotent endomorphisms correspond to **projections** onto sub-objects. Reading (1) categorically: `extract` projects `Text` onto the axiom-set sub-object, and `generate` is a section of that projection — composing them is the projection itself. This is the structural content of "canonical-tome reconstruction is a fixed point under the round-trip."

## 5. The Frobenius-algebra extension and what it does *not* give SUM

Kartsaklis / Sadrzadeh (2015, [arXiv:1505.00138](https://arxiv.org/abs/1505.00138)) extend DisCoCat with **Frobenius algebras** on the meaning spaces. A Frobenius algebra on an object `A` in a symmetric monoidal category is a multiplication `μ : A ⊗ A → A` and a comultiplication `Δ : A → A ⊗ A` satisfying the Frobenius law `(Δ ⊗ id) ∘ (id ⊗ μ) = μ ∘ Δ = (id ⊗ Δ) ∘ (μ ⊗ id)`. It is "special" when `μ ∘ Δ = id` and "commutative" when both operations commute with the symmetry.

In `FVect` with a chosen basis `{eᵢ}`, the canonical special commutative Frobenius algebra is `Δ(eᵢ) = eᵢ ⊗ eᵢ` (copy on basis) and `μ(eᵢ ⊗ eⱼ) = δᵢⱼ eᵢ` (match on basis). This is the categorical-quantum-mechanics "spider theorem" / "classical structures = orthonormal bases" reading (Coecke / Pavlovic / Vicary).

**The temptation:** read SUM's compose-union as a Frobenius multiplication. This is **over-claim** and the document deliberately does not make it. The reasons:
- SUM's compose-union is set union, which is the *additive / disjunctive* side of the Boolean lattice. The canonical Frobenius multiplication on `FVect` (and by analogy `FRel`) is the *matching / copying* operation, which is structurally different. The two sides are related (in ZX-calculus and similar diagrammatic settings they form complementary Frobenius algebras), but they are not the same operation.
- A Frobenius algebra requires a *natural* comultiplication. SUM's substrate has no canonical "split a state integer into two states whose LCM is the original" operation — many factorisations exist, none preferred.
- The honest claim is the weaker, more defensible one: SUM exhibits the algebraic structure of a **commutative idempotent monoid** (semilattice) on the universe of canonical axiom keys. That structure is sufficient to ground the §2.5 closure reading without invoking the full Frobenius machinery.

What Kartsaklis / Sadrzadeh's Frobenius extension *does* give DisCoCat is a treatment of relative pronouns, intransitive verbs, and information structure (theme / rheme) — linguistic phenomena involving copying and matching across sentence positions. These are downstream from SUM's substrate; they would matter if SUM were to add sentence-similarity comparison or grammatical-type analysis (it has not; see §6).

## 6. Why we cite this framework but do not adopt it as substrate

A production implementation of DisCoCat-style pipelines exists: **lambeq** (Kartsaklis et al., [arXiv:2110.04236](https://arxiv.org/abs/2110.04236), [github.com/CQCL/lambeq](https://github.com/CQCL/lambeq), v0.5.0 May 2025, Quantinuum-maintained). It ships a CCG parser (Bobcat), `ccg2discocat` conversion, string-diagram rewriting, and quantum-circuit ansätze for QNLP experiments.

SUM does not depend on lambeq and does not plan to. Reasons:

1. **Crypto trust loop is orthogonal.** lambeq solves semantic comparison; SUM's load-bearing surface is third-party-verifiable attestation over the canonical bytes. Neither framework addresses the other's problem.
2. **Tensor scaling.** Full DisCoCat verb tensors are `dim(V) × dim(S) × dim(W)` — combinatorial in the basis. lambeq mitigates this with low-rank ansätze and quantum-circuit encodings, but at SUM's target corpus scales (journalists distilling ≤10 sources at a time) the Boolean-restricted FRel × P framework already covers the use case without the tensor cost.
3. **Pregroup grammar is not in the dogfood path.** Sieve extracts SVO triples without performing grammatical type analysis. The §2.5 closure result is achieved without it. Adding Pregroup typing would multiply complexity with no proven outcome gain for the named buyers.
4. **Empirical NLP has moved.** Sentence-transformers (BERT, MPNet, BGE) dominate practical sentence-similarity work. If SUM ever ships a `sum compare` surface (deferred per `docs/DOGFOOD_FINDINGS_2026-05-17.md` and downstream), sentence-transformers are the off-the-shelf path, not DisCoCat. The DisCoCat / lambeq line is a viable research alternative — empirically validated on small-corpus tasks (e.g., 4-class sentiment, [arXiv:2209.03152](https://arxiv.org/abs/2209.03152)) — but has not displaced transformer-based methods in industrial NLP.

Citing the framework therefore serves a different goal: it lets reviewers familiar with compact-closed categories, Pregroups, and the Coecke / Sadrzadeh tradition place SUM's §2.5 closure inside their vocabulary without requiring them to first understand the Gödel-prime substrate from scratch. The framework is a vocabulary, not a dependency.

## 7. Open conjectures

The categorical reading suggests two follow-on questions that are *open* in the sense that no SUM artifact today proves either. They are worth recording because each is potentially closable by future work and either result would matter:

1. **Conjecture (categorical proof of §2.5 closure).** Is the §2.5 fixed-point empirically observed in T1 + T4 a *theorem* in `FRel × P`, derivable from properties of the canonical-tome reconstruction morphism alone (i.e., from the structure of LCM-of-distinct-primes encoding plus the canonical reconstruction algorithm), independent of empirical witnessing? If yes, the empirical receipts become a confirmation of a categorical theorem rather than the only ground for the claim. If no — i.e., there exist axiom-set / generator pairs where the fixed-point fails — then the empirical scope of T1 + T4 is the true boundary, which is the current load-bearing state anyway.

2. **Conjecture (Frobenius-induced projection).** Does the Boolean lattice on canonical axiom keys carry a special commutative Frobenius algebra structure (in the FRel sense) such that the `extract` morphism is the projection induced by that structure? Resolving this would clarify whether SUM's substrate has the "classical structure" of Coecke / Pavlovic / Vicary, with the corresponding diagrammatic-calculus consequences. The author's prior expectation: probably yes for the union-side, but the answer is not in any receipt today.

Both are research questions for a categorical-algebra collaborator if one materialises. Neither blocks any current SUM outcome. They are recorded here so a future reader can pick them up.

## 8. What this document does NOT claim

- It does not claim SUM is "based on" DisCoCat. SUM is based on the Gödel-prime encoding of canonical axioms; DisCoCat is a reading of that substrate's algebraic structure.
- It does not claim Frobenius-algebra structure for SUM's compose-union. The honest claim is the weaker commutative-idempotent-monoid structure.
- It does not propose adopting Pregroup grammar, lambeq, or any DisCoCat-derived library as a SUM dependency.
- It does not strengthen any §2 claim in `docs/PROOF_BOUNDARY.md`. Every load-bearing claim retains its existing epistemic status. This document is additive vocabulary, not new evidence.

## References (primary sources)

- Coecke, B., Sadrzadeh, M., Clark, S. (2010). *Mathematical Foundations for a Compositional Distributional Model of Meaning.* [arXiv:1003.4394](https://arxiv.org/abs/1003.4394). The DisCoCat foundational paper. §6 names the FRel × P Boolean-restricted variant that SUM's substrate fits.
- Kartsaklis, D., Sadrzadeh, M. (2015). *Compositional Distributional Semantics with Compact Closed Categories and Frobenius Algebras.* [arXiv:1505.00138](https://arxiv.org/abs/1505.00138). The Frobenius-algebra extension. SUM does not adopt the Frobenius structure (see §5); the reference is for vocabulary continuity.
- Lambek, J. (1999). *Type Grammar Revisited.* In Logical Aspects of Computational Linguistics, LNCS 1582. The Pregroup formalism that DisCoCat's grammatical half is built on.
- Coecke, B., Paquette, E. O. (2010). *Categories for the Practicing Physicist.* [arXiv:0905.3010](https://arxiv.org/abs/0905.3010). Background on compact closed categories and the diagrammatic calculus used in §4.
- Kelly, G. M., Laplaza, M. L. (1980). *Coherence for Compact Closed Categories.* J. Pure Appl. Algebra 19, 193–213. The textbook reference for compact closed categories.
- Selinger, P. (2010). *A Survey of Graphical Languages for Monoidal Categories.* In New Structures for Physics, Springer LNP 813. The graphical-calculus survey cited by the 2010 DisCoCat paper.
- Kartsaklis, D. et al. (2021). *lambeq: An Efficient High-Level Python Library for Quantum NLP.* [arXiv:2110.04236](https://arxiv.org/abs/2110.04236). The production DisCoCat / QNLP Python library SUM cites but does not depend on. Current version 0.5.0, May 2025, at [github.com/CQCL/lambeq](https://github.com/CQCL/lambeq).
- Cervantes-Vega, V. M. et al. (2022). *A multiclass Q-NLP sentiment analysis experiment using DisCoCat.* [arXiv:2209.03152](https://arxiv.org/abs/2209.03152). Empirical extension of DisCoCat to four-class sentiment; representative of the empirical scale at which DisCoCat-line work currently operates.

## SUM cross-references

- `docs/PROOF_BOUNDARY.md` §2.5 + §2.5.1.a/b/c/d — the load-bearing empirical claims this document gives a vocabulary for.
- `docs/BENCH_HARDENING_FROM_QCVV.md` T1 + T4 — the bench-hardening tasks whose receipts witness the §2.5 closure under composition.
- `docs/DRIFT_METRIC_COMPOSITION.md` — T4's fixed-point composition-law analysis; the empirical evidence that the §2.5 closure is composition-stable.
- `fixtures/bench_receipts/s25_iterated_K10_seed_v1_2026-05-21.json` + `_seed_v2_` + `_seed_long_paragraphs_` — T1 receipts.
- `fixtures/bench_receipts/drift_composition_2026-05-22.json` — T4 receipt.
