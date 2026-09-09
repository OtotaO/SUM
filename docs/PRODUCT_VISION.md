# SUM product vision: retained sources, useful views, portable review

Updated 2026-09-09 after the repository audit and the operator's authorization to complete and revise the product. This replaces the June first-increment design; its history remains in git. NORTH_STAR governs the destination. PROOF_BOUNDARY governs claims.

## Destination

Help a person reshape a source for a purpose, inspect consequential changes, restore needed detail, and share a checkable result. Adjustable depth and perspective are views over retained source material. A tag cannot uniquely reconstruct discarded information: expansion must return to the retained source or identify newly retrieved/generated material.

Receipts make the review portable. Source fidelity and the usefulness of the review make it worth doing. A signature proves that a particular key signed particular bytes; a caller must establish trust in that key separately. It does not establish factual truth, human identity, semantic preservation, or statistical applicability.

## The completed loop to maintain

1. Keep the original source unchanged and visible. Preserve exact spans, qualifications, dates, quantities, attribution, and negation through review.
2. Accept an existing rewrite first. Generation is another way to obtain a candidate, with separate progress, error, provider and quota states.
3. Derive each candidate from the same source baseline. Apply density once. Treat candidates as discrete alternatives; order does not imply a Pareto frontier or monotone quality.
4. Show source and candidate together. Link possible changes to actual source/output spans, then record the person's decisions. Browser literal comparison is advisory string evidence, not an entailment judgment or a semantic score.
5. Export the exact reviewed source, output, current review decisions and available receipt/key material. Bind verification state to that version. Edits and failed/superseded operations invalidate prior results.
6. Let a recipient inspect the packet independently. Report signature verification, signed-content bindings, arithmetic replay, model measurement, statistical assumptions and human review separately. Absent checks say not checked.

The browser, CLI and MCP may expose different capabilities. Their shared contracts should be source identity, candidate identity, instrument configuration, evidence references, and review status. Do not imply the browser runs a model judge because the research CLI can.

## What a number means

| Evidence | Scope |
|---|---|
| Literal source/output comparison | Exact or normalized string relationships and possible changes; human interpretation required |
| Named per-document proxy | A measurement under a recorded model, threshold, unitizer, truncation policy and weights; no guarantee for that document |
| Corpus confidence bound | A bound on expected named proxy loss only under the procedure's independent sampling and distribution-match assumptions; arbitrary supplied corpora remain descriptive |
| Signature and byte binding | Integrity relative to a supplied key and the fields actually signed; not a statement of factual truth |
| Human decision | The reviewer's recorded decision about this exact version; unsigned unless a separate signing mechanism attests it |

Historical reference-summary benchmarks demonstrate the measurement and receipt machinery. They do not measure the current SUM generation pipeline. Arithmetic replay of stored losses does not re-run extraction or a judge. A scorer manifest records the instrument configuration; its existence does not establish instrument validity.

## Priorities and acceptance

| Order | Outcome | Acceptance |
|---|---|---|
| 1 | Correct and bounded existing operations | Immutable handles, offline MCP setup errors, bounded bodies, atomic paid admission, explicit verifier policy; meaningful regression tests |
| 2 | Source review and export | A stranger compares their own source and rewrite, inspects spans, makes decisions and exports the exact result; stale results never pass as current |
| 3 | Useful depth alternatives | A small set of real candidates from the retained source; generation and judging costs/time are visible; no interpolation presented as a measurement |
| 4 | Evaluate the actual pipeline | Prespecified sampling and source/generation/scorer manifests; independent human annotations on held-out examples; compare against a practical baseline |
| 5 | Validate recurring use | Observe real users completing and sharing reviews, returning with new sources, and describing the consequence of a missed change |
| 6 | Expand where evidence supports it | Better formats, multi-document conflicts, team workflows or integrations chosen from observed tasks |

A new hosted judge, billing system, knowledge graph platform or standardization effort is not a prerequisite for these outcomes. Preserve the independent cryptographic implementations and shared fixtures. Extract orchestration from the large CLI incrementally when another real consumer needs it.

## Product and commercial hypotheses

Writers/editors may value faster inspection of rewritten material. Small AI teams may value portable release evidence. Neither hypothesis is established by repository stars, simulated customers, the number of implemented features, or a signed benchmark. Measure completed reviews, useful corrections, recipient checks, repeat use and willingness to pay against an existing workflow.

Treat provenance, evaluation and review tools as potential integration partners. Avoid claims that competing standards cannot carry text or custom assertions, or that SUM is already an adopted standard. Standards compatibility and regulatory validators are specific technical capabilities, not compliance certification.

## Secondary interfaces

The extension is an explicit source-capture client over the browser workbench. It should require no background collection, paid server or parallel judging implementation. The terminal UI remains a clearly labelled historical receipt-demo prototype. Internal quantum APIs, Docker recipes and research algorithms stay separate from the supported public workflow until a tested task justifies promotion.
