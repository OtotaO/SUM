"""F3 diagnostic harness — settle which v3.1 hypothesis is load-bearing.

The PR #124 corpus-scale bench surfaced F3 FAIL: v3.1 boundary
deviation gives mean AUC ≈ 0.50 on trusted-target perturbations
across the seed_long_paragraphs corpus, despite the synthetic H12
utility test passing. ``docs/SHEAF_HALLUCINATION_DETECTOR.md``
§3.4.2 named three competing hypotheses for the failure:

  A. Per-doc graphs are too small (5–10 triples); 50/50 partition
     leaves too few trusted edges for boundary_from_weights to
     return a meaningful boundary.
  B. ``cochain_one_hot_v2`` produces zero-vectors at boundary
     vertices not in the trained vocabulary, making harmonic
     extension uninformative.
  C. Random 50/50 partition is harsh; real-world receipt
     distributions are concentrated by source.

This is a diagnostic, not a benchmark. 2×2×2 = 8 cells over the
three hypotheses' axes, each cell run with the SAME trained sheaf
(no training-noise contamination) and the SAME perturbation
classes (A1, A2, A4) × targets (trusted, untrusted) as PR #124.

Per-cell diagnostics let us mechanistically explain *why* a cell
PASSes or FAILs, not just observe the AUC.

Schema: ``sum.sheaf_v3_1_f3_diagnostic.v1``.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from sum_engine_internal.research.sheaf_laplacian_v2 import (
    KnowledgeSheafV2,
    Triple,
    cochain_one_hot_v2,
    train_restriction_maps,
)
from sum_engine_internal.research.sheaf_laplacian_v3 import (
    boundary_deviation,
    boundary_from_weights,
    weights_from_receipts,
)
from sum_engine_internal.algorithms.syntactic_sieve import DeterministicSieve
from sum_engine_internal.infrastructure.jcs import canonicalize as jcs_canonicalize

CORPUS_PATH = REPO / "scripts" / "bench" / "corpora" / "seed_long_paragraphs.json"


# ── Cell-config type system (frozen; pure-function inputs) ──────────


GraphSizeStrategy = Literal["per_doc", "aggregated_4_docs"]
CochainStrategy = Literal["one_hot_default", "trained_embedding"]
PartitionStrategy = Literal["random_50_50", "concentrated_by_doc"]


@dataclass(frozen=True)
class CellConfig:
    """One point in the 2×2×2 diagnostic space.

    Each axis tests one of the three F3 hypotheses:
      - graph_size      → hypothesis A (graph too small)
      - cochain         → hypothesis B (cochain produces zero-vectors)
      - partition       → hypothesis C (random partition too harsh)
    """
    graph_size: GraphSizeStrategy
    cochain: CochainStrategy
    partition: PartitionStrategy

    @property
    def cell_id(self) -> str:
        return f"{self.graph_size}|{self.cochain}|{self.partition}"


@dataclass(frozen=True)
class CellDiagnostics:
    """Mechanistic diagnostics — explain WHY a cell passes or fails.

    Without these, F3 PASS is as opaque as F3 FAIL.

    Mean fields are ``float | None`` so a cell with zero non-
    degenerate graphs emits ``None`` (JSON-safe under JCS) rather
    than ``nan`` (RFC 8785 §3.2.2.3 doesn't define NaN serialization,
    so nan would either crash or silently mis-canonicalize).
    """
    n_graphs: int                              # how many graphs evaluated in this cell
    mean_boundary_size: float | None           # |B| averaged across graphs
    mean_interior_size: float | None           # |I| averaged across graphs
    mean_cochain_norm_boundary: float | None   # ‖x_B‖₂ averaged
    mean_cochain_norm_interior: float | None   # ‖x_I‖₂ averaged
    mean_L_II_rank_ratio: float | None         # rank(L_II) / dim(L_II), averaged
    n_degenerate_partitions: int               # cells where boundary is empty/full


@dataclass(frozen=True)
class CellResult:
    """Per-cell receipt — config + AUCs + diagnostics + verdict."""
    config_cell_id: str
    per_class_target_auc: dict[str, float]     # e.g. "A1|trusted" → 0.62
    trusted_mean_auc: float
    untrusted_mean_auc: float
    diagnostics: CellDiagnostics
    f3_verdict: Literal["PASS", "FAIL"]


@dataclass(frozen=True)
class DiagnosticReport:
    """Full 8-cell report.

    The ``bench_digest`` field is a SHA-256 over the JCS-canonical
    encoding of a *quantized* view of this report (AUCs to 3
    decimals, diagnostic floats to 4 decimals, integers exact).
    Quantization is required because LAPACK threading inside
    np.linalg.lstsq (in v3.1's harmonic_extension) introduces
    ~±0.02 AUC jitter across runs; without quantization the digest
    would be a noise canary, not a reproducibility witness.

    Three uses for the digest:

      1. **Reproducibility canary.** Same machine + same code →
         same digest. Drift indicates upstream change.
      2. **Cross-runtime witness.** When a future Node/browser
         port of v3.1 reproduces these AUCs, the matching digest
         is the K-style portability proof.
      3. **Signable bench artifact.** Ed25519-sign the digest with
         the project's existing JWKS keys → a verifiable receipt
         that the science reproduced. arXiv preprint can cite
         this digest; readers can re-run and verify.

    JCS (RFC 8785) is the same canonicalization the trust loop
    uses for CanonicalBundle and render_receipt.v1 — the digest is
    in the same alphabet as the rest of the substrate.
    """
    schema: str                                # pinned at sum.sheaf_v3_1_f3_diagnostic.v1
    corpus: str
    n_corpus_docs: int
    stalk_dim: int
    lambda_auto_calibrated: float
    cells: tuple[CellResult, ...]
    load_bearing_hypothesis: Literal["A", "B", "C", "none", "multiple"]
    summary: str
    bench_digest: str                          # 64-char lowercase hex (SHA-256)


# ── Stub signatures (filled in STATE 4) ─────────────────────────────


def extract_corpus() -> list[tuple[str, list[Triple]]]:
    """Sieve-extract triples per doc from seed_long_paragraphs."""
    with open(CORPUS_PATH) as f:
        data = json.load(f)
    sieve = DeterministicSieve()
    out: list[tuple[str, list[Triple]]] = []
    for d in data["documents"]:
        triples = list(sieve.extract_triplets(d["text"]))
        if triples:
            out.append((d["id"], triples))
    return out


def train_shared_sheaf(
    all_triples: list[Triple],
) -> tuple[KnowledgeSheafV2, np.ndarray]:
    """Train ONE sheaf on the union vocabulary.

    Reused across all 8 cells to avoid training-noise contamination
    of the variable we're isolating. Same hyperparameters as the
    v2.2 / v3 ROC benches (stalk_dim=8, 200 epochs, seed=0) so cell
    AUCs are directly comparable to PR #114 / PR #124.
    """
    trained, embeddings, _history = train_restriction_maps(
        all_triples,
        stalk_dim=8,
        epochs=200,
        learning_rate=0.005,
        margin=0.5,
        n_negatives_per_positive=3,
        seed=0,
    )
    return trained, embeddings


def auto_calibrate_lambda(
    corpus: list[tuple[str, list[Triple]]],
    trained: KnowledgeSheafV2,
    embeddings: np.ndarray,
) -> float:
    """λ_auto = mean(per-edge Laplacian) — same as v2.2 / v3 benches.

    For each doc, build a per-doc sheaf, project the trained
    embeddings onto its vertex set, and compute the clean Laplacian
    quadratic form divided by the doc's edge count. Mean across
    docs = λ_auto. Calibrates the deficit term so 1 missing entity
    ≈ 1 average per-edge Laplacian contribution.
    """
    from sum_engine_internal.research.sheaf_laplacian_v2 import combined_detector_score
    per_edge_means: list[float] = []
    for _doc_id, source in corpus:
        if not source:
            continue
        try:
            doc_sheaf = KnowledgeSheafV2.from_triples(source, stalk_dim=8)
            doc_emb = np.zeros((len(doc_sheaf.vertices), 8), dtype=np.float64)
            for i, v in enumerate(doc_sheaf.vertices):
                if v in trained.vertex_index:
                    doc_emb[i] = embeddings[trained.vertex_index[v]]
            clean = combined_detector_score(doc_sheaf, doc_emb, source)
            per_edge_means.append(clean["v_laplacian"] / max(len(source), 1))
        except (ValueError, KeyError):
            pass
    return float(np.mean(per_edge_means)) if per_edge_means else 0.05


def build_graphs_for_cell(
    corpus: list[tuple[str, list[Triple]]],
    config: CellConfig,
) -> list[tuple[str, list[Triple]]]:
    """Returns [(graph_id, triples), ...] per the cell's graph_size axis.

    - per_doc          → one graph per doc (16 graphs from corpus)
    - aggregated_4_docs → docs grouped 4 at a time → 4 graphs.
                          Each aggregated graph carries a composite
                          ``graph_id`` of the form
                          ``"agg|<doc_id_1>|<doc_id_2>|...|<doc_id_4>"``
                          so partition_for_cell can derive a stable
                          seed and `concentrated_by_doc` can recover
                          the contributing doc ids.
    """
    if config.graph_size == "per_doc":
        return [(doc_id, list(triples)) for doc_id, triples in corpus]

    if config.graph_size == "aggregated_4_docs":
        out: list[tuple[str, list[Triple]]] = []
        for i in range(0, len(corpus), 4):
            group = corpus[i:i + 4]
            if not group:
                continue
            graph_id = "agg|" + "|".join(doc_id for doc_id, _ in group)
            triples: list[Triple] = []
            seen: set[Triple] = set()
            for _doc_id, ts in group:
                for t in ts:
                    if t not in seen:
                        triples.append(t)
                        seen.add(t)
            out.append((graph_id, triples))
        return out

    raise ValueError(f"unknown graph_size: {config.graph_size!r}")


def partition_for_cell(
    triples: list[Triple],
    graph_id: str,
    config: CellConfig,
    doc_groups: dict[str, list[Triple]] | None = None,
) -> tuple[list[Triple], list[Triple]]:
    """Returns (trusted_edges, untrusted_edges) per the cell's partition axis.

    - random_50_50         → SHA-256-seeded 50/50 (matches PR #124).
    - concentrated_by_doc  → all triples from a doc are uniformly trusted
                             or untrusted; alternates by doc index.
                             For per_doc graphs (one doc per graph) the
                             partition is whole-graph (degenerate by
                             design — surfaces in diagnostics as
                             ``n_degenerate_partitions``). For
                             aggregated graphs, half the contributing
                             docs trusted, half untrusted.

    ``doc_groups`` is only consulted under ``concentrated_by_doc`` for
    aggregated graphs: a mapping from doc_id → its triples within
    this graph, so the partition can be applied at doc granularity.
    """
    stable_seed = int.from_bytes(
        hashlib.sha256(graph_id.encode()).digest()[:4], "big",
    )

    if config.partition == "random_50_50":
        rng = random.Random(stable_seed)
        indices = list(range(len(triples)))
        rng.shuffle(indices)
        n_trusted = len(triples) // 2
        trusted_idx = set(indices[:n_trusted])
        trusted = [t for i, t in enumerate(triples) if i in trusted_idx]
        untrusted = [t for i, t in enumerate(triples) if i not in trusted_idx]
        return trusted, untrusted

    if config.partition == "concentrated_by_doc":
        # Per_doc graphs: graph_id is a single doc_id; alternate trust
        # by hash(doc_id). Half the corpus's docs end up trusted,
        # half untrusted — but each individual graph is uniform.
        if not graph_id.startswith("agg|"):
            if stable_seed % 2 == 0:
                return list(triples), []
            return [], list(triples)
        # Aggregated graphs: split contributing docs in half (lex-sorted
        # for reproducibility), trust the first half, untrust the rest.
        if doc_groups is None:
            # Fallback: degrade to whole-graph partition by seed parity.
            if stable_seed % 2 == 0:
                return list(triples), []
            return [], list(triples)
        sorted_doc_ids = sorted(doc_groups)
        cutoff = len(sorted_doc_ids) // 2
        trusted_doc_ids = set(sorted_doc_ids[:cutoff])
        trusted: list[Triple] = []
        untrusted: list[Triple] = []
        for doc_id, doc_triples in doc_groups.items():
            bucket = trusted if doc_id in trusted_doc_ids else untrusted
            bucket.extend(doc_triples)
        return trusted, untrusted

    raise ValueError(f"unknown partition: {config.partition!r}")


def cochain_for_cell(
    sheaf: KnowledgeSheafV2,
    render: list[Triple],
    config: CellConfig,
    embeddings: np.ndarray,
    trained: KnowledgeSheafV2,
) -> np.ndarray:
    """Returns the cochain x ∈ R^(|V|·d) per the cell's cochain axis.

    - one_hot_default → ``cochain_one_hot_v2(sheaf, render, embedding=local_emb)``
        where ``local_emb`` is the per-doc projection of trained
        embeddings (zero for any vertex not in trained vocab).
        Matches PR #124's bench exactly.

    - trained_embedding → every vertex's row is the trained embedding
        directly — the rendered triples set the cochain support but
        the vertex *values* always come from the trained model.
        Distinction from one_hot_default: this strategy *never*
        zero-pads for "vertex in sheaf but not in render"; only
        for "vertex truly out-of-vocabulary". Tests hypothesis B
        (cochain producing zero-vectors at boundary).
    """
    n_v = len(sheaf.vertices)
    d = sheaf.stalk_dim
    local_emb = np.zeros((n_v, d), dtype=np.float64)
    for i, v in enumerate(sheaf.vertices):
        if v in trained.vertex_index:
            local_emb[i] = embeddings[trained.vertex_index[v]]

    if config.cochain == "one_hot_default":
        return cochain_one_hot_v2(sheaf, render, embedding=local_emb)

    if config.cochain == "trained_embedding":
        # Every in-vocab vertex carries its trained embedding regardless
        # of whether it appears in the render. Out-of-vocab vertices
        # remain zero (only true OOV — not "merely unmentioned").
        # Render mention is implicit: vertices the render didn't touch
        # still contribute to the Laplacian quadratic form via their
        # restriction-map-aware connections to mentioned neighbors.
        return local_emb.copy()

    raise ValueError(f"unknown cochain: {config.cochain!r}")


def run_cell(
    config: CellConfig,
    corpus: list[tuple[str, list[Triple]]],
    trained: KnowledgeSheafV2,
    embeddings: np.ndarray,
    lambda_auto: float,
) -> CellResult:
    """Pure function: cell config + corpus → cell result.

    No global state; same trained_sheaf reused across all cells.
    Each (graph, perturbation_class, target) yields one (clean,
    perturbed) score pair; AUC computed per (class, target);
    cell verdict from trusted_mean_auc.

    The detector under test is v3.1 boundary deviation. v2.2 / v3
    AUCs are not re-measured here — PR #124 already verdicted them
    (F1 MARGINAL, F2 PASS).
    """
    from scripts.research.sheaf_v3_roc_bench import (
        perturb_a1_on_target,
        perturb_a2_on_target,
        perturb_a4_drop_target,
        roc_auc,
    )

    # Cell-local rng for perturbation choices; deterministic per cell.
    rng = random.Random(int.from_bytes(
        hashlib.sha256(config.cell_id.encode()).digest()[:4], "big",
    ))

    all_triples = [t for _, ts in corpus for t in ts]
    all_entities = sorted({e for h, _, t in all_triples for e in (h, t)})
    all_relations = sorted({r for _, r, _ in all_triples})

    graphs = build_graphs_for_cell(corpus, config)

    # Per-cell score collectors keyed "<class>|<target>"
    scores: dict[str, list[tuple[float, int]]] = {
        f"{cls}|{target}": []
        for cls in ("A1", "A2", "A4")
        for target in ("trusted", "untrusted")
    }

    # Diagnostic accumulators
    diag_boundary_sizes: list[int] = []
    diag_interior_sizes: list[int] = []
    diag_xB_norms: list[float] = []
    diag_xI_norms: list[float] = []
    diag_LII_rank_ratios: list[float] = []
    n_degenerate = 0
    n_graphs_evaluated = 0

    for graph_id, graph_triples in graphs:
        if len(graph_triples) < 4:
            continue

        # Build doc_groups dict for the aggregated case so partition
        # can route concentrated_by_doc per-doc within an aggregate.
        doc_groups: dict[str, list[Triple]] | None = None
        if graph_id.startswith("agg|"):
            doc_ids_in_graph = graph_id.split("|")[1:]
            corpus_dict = dict(corpus)
            doc_groups = {
                doc_id: list(corpus_dict.get(doc_id, []))
                for doc_id in doc_ids_in_graph
            }

        trusted, untrusted = partition_for_cell(
            graph_triples, graph_id, config, doc_groups=doc_groups,
        )

        try:
            doc_sheaf = KnowledgeSheafV2.from_triples(graph_triples, stalk_dim=8)
        except (ValueError, KeyError):
            continue

        weights = weights_from_receipts(doc_sheaf, trusted_edges=trusted)
        boundary = boundary_from_weights(doc_sheaf, weights, threshold=0.5)
        is_degenerate = (not boundary) or (len(boundary) == len(doc_sheaf.vertices))
        if is_degenerate:
            n_degenerate += 1

        # ── Diagnostics on the CLEAN render (what does the boundary look like?) ──
        x_clean = cochain_for_cell(doc_sheaf, graph_triples, config, embeddings, trained)
        if not is_degenerate:
            interior = [
                v for v in range(len(doc_sheaf.vertices))
                if v not in set(boundary)
            ]
            x_B = x_clean[boundary]
            x_I = x_clean[interior]
            diag_boundary_sizes.append(len(boundary))
            diag_interior_sizes.append(len(interior))
            diag_xB_norms.append(float(np.linalg.norm(x_B)))
            diag_xI_norms.append(float(np.linalg.norm(x_I)))

            # rank(L_II) / dim(L_II) — needs L_II construction. Cheap on
            # small interiors; reuse v3.1's harmonic-extension internals
            # by hand here (not exposed as a public function, so inline).
            from sum_engine_internal.research.sheaf_laplacian_v2 import (
                sheaf_laplacian_v2,
            )
            from sum_engine_internal.research.sheaf_laplacian_v3 import (
                _block_indices,
            )
            L = sheaf_laplacian_v2(doc_sheaf)
            I_flat = _block_indices(interior, doc_sheaf.stalk_dim)
            L_II = L[np.ix_(I_flat, I_flat)]
            if L_II.size > 0:
                rank = int(np.linalg.matrix_rank(L_II))
                dim = L_II.shape[0]
                diag_LII_rank_ratios.append(rank / dim)

        # ── Score detector on (clean, perturbed_at_trusted) and
        #    (clean, perturbed_at_untrusted) for A1, A2, A4 ──
        def _v31_score(render: list[Triple]) -> float:
            """v3.1 boundary deviation; degenerate cells fall back
            to v3 weighted-Laplacian quadratic form (matches PR #124's
            convention so cells remain comparable)."""
            x = cochain_for_cell(doc_sheaf, render, config, embeddings, trained)
            if is_degenerate:
                from sum_engine_internal.research.sheaf_laplacian_v3 import (
                    weighted_laplacian_quadratic_form_v3,
                )
                return float(weighted_laplacian_quadratic_form_v3(
                    doc_sheaf, x, weights,
                ))
            result = boundary_deviation(doc_sheaf, x, boundary, weights=weights)
            return float(result["deviation"])

        clean_score = _v31_score(graph_triples)

        for target_label, target_pool in (("trusted", trusted), ("untrusted", untrusted)):
            if not target_pool:
                continue
            target_triple = rng.choice(target_pool)
            for cls, perturb_fn in (
                ("A1", lambda r=rng, t=target_triple:
                    perturb_a1_on_target(graph_triples, t, all_entities, r)),
                ("A2", lambda r=rng, t=target_triple:
                    perturb_a2_on_target(graph_triples, t, all_relations, r)),
                ("A4", lambda t=target_triple:
                    perturb_a4_drop_target(graph_triples, t)),
            ):
                perturbed = perturb_fn()
                if perturbed == graph_triples:
                    continue
                try:
                    p_score = _v31_score(perturbed)
                except Exception:
                    continue
                scores[f"{cls}|{target_label}"].append((clean_score, 0))
                scores[f"{cls}|{target_label}"].append((p_score, 1))

        n_graphs_evaluated += 1

    # ── AUC per (class, target) ──
    per_class_target_auc: dict[str, float] = {}
    for key, pairs in scores.items():
        s = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]
        per_class_target_auc[key] = roc_auc(s, labels)

    trusted_aucs = [
        per_class_target_auc.get(f"{c}|trusted", 0.5)
        for c in ("A1", "A2", "A4")
    ]
    untrusted_aucs = [
        per_class_target_auc.get(f"{c}|untrusted", 0.5)
        for c in ("A1", "A2", "A4")
    ]
    trusted_mean = float(np.mean(trusted_aucs))
    untrusted_mean = float(np.mean(untrusted_aucs))

    # ── Verdict (matches PR #124's F3 threshold) ──
    f3_verdict: Literal["PASS", "FAIL"] = "PASS" if trusted_mean >= 0.55 else "FAIL"

    diagnostics = CellDiagnostics(
        n_graphs=n_graphs_evaluated,
        mean_boundary_size=(
            float(np.mean(diag_boundary_sizes)) if diag_boundary_sizes else None
        ),
        mean_interior_size=(
            float(np.mean(diag_interior_sizes)) if diag_interior_sizes else None
        ),
        mean_cochain_norm_boundary=(
            float(np.mean(diag_xB_norms)) if diag_xB_norms else None
        ),
        mean_cochain_norm_interior=(
            float(np.mean(diag_xI_norms)) if diag_xI_norms else None
        ),
        mean_L_II_rank_ratio=(
            float(np.mean(diag_LII_rank_ratios)) if diag_LII_rank_ratios else None
        ),
        n_degenerate_partitions=n_degenerate,
    )

    return CellResult(
        config_cell_id=config.cell_id,
        per_class_target_auc=per_class_target_auc,
        trusted_mean_auc=trusted_mean,
        untrusted_mean_auc=untrusted_mean,
        diagnostics=diagnostics,
        f3_verdict=f3_verdict,
    )


def aggregate_verdict(cells: list[CellResult]) -> tuple[Literal["A", "B", "C", "none", "multiple"], str]:
    """Identify the load-bearing hypothesis from cell results.

    The FAIL baseline is ``per_doc | one_hot_default | random_50_50``
    (the cell PR #124's bench effectively measured). For each
    hypothesis, flip just that axis and check the resulting cell's
    F3 verdict:

      - A flips graph_size  → ``aggregated_4_docs | one_hot_default | random_50_50``
      - B flips cochain     → ``per_doc | trained_embedding | random_50_50``
      - C flips partition   → ``per_doc | one_hot_default | concentrated_by_doc``

    A hypothesis is "load-bearing" iff its single-axis flip
    PASSes (trusted_mean_auc ≥ 0.58, the noise-floor-aware
    threshold above the 0.55 PASS line).

    Returns:
      "A" / "B" / "C"  → exactly one single-axis flip PASSes
      "multiple"       → more than one single-axis flip PASSes
      "none"           → no single-axis flip PASSes
    """
    by_id = {c.config_cell_id: c for c in cells}
    baseline_id = "per_doc|one_hot_default|random_50_50"
    flip_a_id = "aggregated_4_docs|one_hot_default|random_50_50"
    flip_b_id = "per_doc|trained_embedding|random_50_50"
    flip_c_id = "per_doc|one_hot_default|concentrated_by_doc"
    full_flip_id = "aggregated_4_docs|trained_embedding|concentrated_by_doc"

    def _passes(cell_id: str) -> bool:
        c = by_id.get(cell_id)
        return c is not None and c.trusted_mean_auc >= 0.58

    a_pass = _passes(flip_a_id)
    b_pass = _passes(flip_b_id)
    c_pass = _passes(flip_c_id)
    full_pass = _passes(full_flip_id)

    flips = [(name, p) for name, p in [("A", a_pass), ("B", b_pass), ("C", c_pass)] if p]

    baseline = by_id.get(baseline_id)
    full_flip = by_id.get(full_flip_id)
    baseline_auc = baseline.trusted_mean_auc if baseline else float("nan")
    full_auc = full_flip.trusted_mean_auc if full_flip else float("nan")

    summary_lines = [
        f"baseline (PR #124-equivalent) trusted_mean_AUC = {baseline_auc:.3f}",
        f"single-axis flips PASS (≥0.58 trusted_mean_AUC): "
        f"A={a_pass}, B={b_pass}, C={c_pass}",
        f"all-axes-flipped trusted_mean_AUC = {full_auc:.3f} "
        f"({'PASS' if full_pass else 'FAIL'})",
    ]

    if len(flips) == 0:
        verdict: Literal["A", "B", "C", "none", "multiple"] = "none"
        if full_pass:
            summary_lines.append(
                "No single axis flip suffices, but all-three-flipped PASSes — "
                "the F3 fix requires interaction effects across multiple axes."
            )
        else:
            summary_lines.append(
                "No flip combination PASSes — F3 is not driven by any of the "
                "three named hypotheses; the v3.1 math itself or the bench "
                "harness needs reconsideration."
            )
    elif len(flips) == 1:
        verdict = flips[0][0]  # type: ignore[assignment]
        summary_lines.append(
            f"Hypothesis {verdict} is load-bearing — flipping only that axis "
            f"flips F3 from FAIL to PASS."
        )
    else:
        verdict = "multiple"
        names = ", ".join(name for name, _ in flips)
        summary_lines.append(
            f"Multiple single-axis flips PASS ({names}) — F3 has more than "
            f"one independent fix; pick the one with cleanest diagnostics."
        )

    return verdict, "\n".join(summary_lines)


def compute_bench_digest(
    cells: tuple[CellResult, ...],
    corpus: str,
    n_corpus_docs: int,
    stalk_dim: int,
    lambda_auto_calibrated: float,
) -> str:
    """SHA-256 over JCS-canonical bytes of the quantized report payload.

    Quantization (deterministic across runs despite LAPACK jitter):
      - AUC values            → round to 3 decimals
      - diagnostic float      → round to 4 decimals
      - integers / strings    → exact

    Field order is JCS-canonical (lexicographic key sort per RFC
    8785 §3.2.3), so two callers building the digest from
    differently-ordered Python dicts produce identical bytes.

    The digest covers BOTH AUCs AND diagnostics AND config —
    algorithmic change (e.g., new partition strategy) flips it;
    run-to-run noise (same algorithm) does not.
    """
    def _q_auc(x: float) -> float:
        return round(float(x), 3)

    def _q_diag(x: float | None) -> float | None:
        return None if x is None else round(float(x), 4)

    payload: dict[str, Any] = {
        "schema": "sum.sheaf_v3_1_f3_diagnostic.v1",
        "corpus": corpus,
        "n_corpus_docs": int(n_corpus_docs),
        "stalk_dim": int(stalk_dim),
        "lambda_auto_calibrated": _q_diag(lambda_auto_calibrated),
        "cells": [
            {
                "config_cell_id": cell.config_cell_id,
                "per_class_target_auc": {
                    k: _q_auc(v) for k, v in cell.per_class_target_auc.items()
                },
                "trusted_mean_auc": _q_auc(cell.trusted_mean_auc),
                "untrusted_mean_auc": _q_auc(cell.untrusted_mean_auc),
                "diagnostics": {
                    "n_graphs": cell.diagnostics.n_graphs,
                    "mean_boundary_size": _q_diag(cell.diagnostics.mean_boundary_size),
                    "mean_interior_size": _q_diag(cell.diagnostics.mean_interior_size),
                    "mean_cochain_norm_boundary": _q_diag(cell.diagnostics.mean_cochain_norm_boundary),
                    "mean_cochain_norm_interior": _q_diag(cell.diagnostics.mean_cochain_norm_interior),
                    "mean_L_II_rank_ratio": _q_diag(cell.diagnostics.mean_L_II_rank_ratio),
                    "n_degenerate_partitions": cell.diagnostics.n_degenerate_partitions,
                },
                "f3_verdict": cell.f3_verdict,
            }
            for cell in cells
        ],
    }
    canonical_bytes = jcs_canonicalize(payload)
    return hashlib.sha256(canonical_bytes).hexdigest()


def main() -> DiagnosticReport:
    print("=" * 72)
    print("F3 diagnostic — 2×2×2 over (graph_size, cochain, partition)")
    print("=" * 72)

    print("\n[1] Loading corpus + training shared sheaf…")
    corpus = extract_corpus()
    all_triples = [t for _, ts in corpus for t in ts]
    trained, embeddings = train_shared_sheaf(all_triples)
    print(f"    corpus: {len(corpus)} docs, {len(all_triples)} triples")
    print(f"    trained sheaf: |V|={len(trained.vertices)}, "
          f"|R|={len(trained.relations)}, stalk_dim={trained.stalk_dim}")

    print("\n[2] Auto-calibrating λ…")
    lambda_auto = auto_calibrate_lambda(corpus, trained, embeddings)
    print(f"    λ_auto = {lambda_auto:.4f}")

    print("\n[3] Running 8 cells…")
    configs = [
        CellConfig(graph_size=g, cochain=c, partition=p)
        for g in ("per_doc", "aggregated_4_docs")
        for c in ("one_hot_default", "trained_embedding")
        for p in ("random_50_50", "concentrated_by_doc")
    ]
    cells: list[CellResult] = []
    print(f"    {'cell':<60} {'trusted':>8} {'verdict':>8}")
    print("    " + "-" * 80)
    for cfg in configs:
        result = run_cell(cfg, corpus, trained, embeddings, lambda_auto)
        cells.append(result)
        print(f"    {cfg.cell_id:<60} {result.trusted_mean_auc:>8.3f} {result.f3_verdict:>8}")

    print("\n[4] Aggregate verdict — load-bearing hypothesis:")
    load_bearing, summary = aggregate_verdict(cells)
    print(f"    answer: {load_bearing}")
    for line in summary.splitlines():
        print(f"      {line}")

    print("\n[5] Bench digest (JCS + SHA-256 over quantized payload):")
    digest = compute_bench_digest(
        cells=tuple(cells),
        corpus="seed_long_paragraphs",
        n_corpus_docs=len(corpus),
        stalk_dim=trained.stalk_dim,
        lambda_auto_calibrated=lambda_auto,
    )
    print(f"    sha256 = {digest}")

    return DiagnosticReport(
        schema="sum.sheaf_v3_1_f3_diagnostic.v1",
        corpus="seed_long_paragraphs",
        n_corpus_docs=len(corpus),
        stalk_dim=trained.stalk_dim,
        lambda_auto_calibrated=lambda_auto,
        cells=tuple(cells),
        load_bearing_hypothesis=load_bearing,
        summary=summary,
        bench_digest=digest,
    )


if __name__ == "__main__":
    report = main()
    print("\n[6] Receipt JSON:")
    print(json.dumps(asdict(report), indent=2, default=str))
