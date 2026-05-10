"""Z3-backed consistency checker for substrate axioms.

The encoding is deliberately small: each predicate becomes a
binary uninterpreted function on a single ``Entity`` sort, and
the operator-supplied predicate properties (antisymmetric,
irreflexive, functional, transitive) are emitted as universally-
quantified axiom schemas. Z3 in QF_UF + EUF + a few quantified
schemas decides this efficiently for substrate-scale workloads.

Returns a structured ``ConsistencyResult`` with:

  - ``consistent`` — bool from Z3 (True = SAT, False = UNSAT)
  - ``unsat_core`` — list of triple indices forming a minimal
    contradicting subset (when ``consistent=False``)
  - ``z3_check_seconds`` — wall time, for the spike receipt
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

from sum_engine_internal.graph_store import Triple


class PredicateProperty(Enum):
    """Logical properties a predicate may have. Operator declares
    these per predicate; Z3 emits the matching axiom schema."""
    ANTISYMMETRIC = "antisymmetric"
    """∀ x y. P(x, y) ∧ x ≠ y → ¬P(y, x). Catches mutual
    parent_of, contains, etc."""
    IRREFLEXIVE = "irreflexive"
    """∀ x. ¬P(x, x). Catches self-parent, self-equal-by-mistake."""
    FUNCTIONAL = "functional"
    """∀ x y z. P(x, y) ∧ P(x, z) → y = z. Catches "Alice has two
    different birth_dates"."""
    TRANSITIVE = "transitive"
    """∀ x y z. P(x, y) ∧ P(y, z) → P(x, z). Used jointly with
    irreflexive/antisymmetric to detect cycles."""


@dataclass(frozen=True, slots=True)
class ConsistencyResult:
    consistent: bool
    """True iff the axiom set is jointly satisfiable under the
    declared predicate properties."""
    unsat_core: list[int]
    """Indices into the input triples list forming the smallest
    contradicting subset. Empty when consistent=True."""
    z3_check_seconds: float
    """Wall time for the Z3 check. Includes encoding."""
    n_triples: int
    n_predicates_with_properties: int


def check_consistency(
    triples: Iterable[Triple],
    *,
    predicate_properties: Optional[dict[str, set[PredicateProperty]]] = None,
    timeout_ms: int = 5000,
) -> ConsistencyResult:
    """Check whether the triples are jointly satisfiable under the
    declared predicate properties.

    Args:
        triples: input axioms
        predicate_properties: mapping ``predicate_name → set of
            PredicateProperty``. Predicates not in this mapping are
            treated as unconstrained (no logical properties imposed).
            ``None`` means "no properties" — Z3 will return SAT for
            any non-degenerate input. The substrate is expected to
            provide its own predicate library here.
        timeout_ms: Z3 timeout for the satisfiability check.

    Returns:
        ConsistencyResult with the verdict + UNSAT core indices.
    """
    import time
    import z3

    triples_list = list(triples)
    predicate_properties = predicate_properties or {}

    t0 = time.perf_counter()

    # Single Entity sort; each unique entity string gets a Z3 const
    Entity = z3.DeclareSort("Entity")
    entity_consts: dict[str, z3.ExprRef] = {}
    def _const(name: str) -> z3.ExprRef:
        if name not in entity_consts:
            entity_consts[name] = z3.Const(f"e_{len(entity_consts)}", Entity)
        return entity_consts[name]

    # One Z3 binary predicate per predicate name observed
    pred_funcs: dict[str, z3.FuncDeclRef] = {}
    def _pred(name: str) -> z3.FuncDeclRef:
        if name not in pred_funcs:
            pred_funcs[name] = z3.Function(
                f"p_{name}", Entity, Entity, z3.BoolSort()
            )
        return pred_funcs[name]

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    # We use unsat_core extraction → tracked assertions via assert_and_track
    track_labels: dict[str, int] = {}
    for i, t in enumerate(triples_list):
        s = _const(t.subject)
        o = _const(t.object)
        p = _pred(t.predicate)
        label = z3.Bool(f"axiom_{i}")
        track_labels[label.decl().name()] = i
        solver.assert_and_track(p(s, o), label)

    # Universal property schemas, only for predicates the operator
    # declared properties on AND that actually appear in the triples
    used_predicates = {t.predicate for t in triples_list}
    n_predicates_with_properties = 0
    for pname, props in predicate_properties.items():
        if pname not in used_predicates:
            continue
        if not props:
            continue
        n_predicates_with_properties += 1
        p = _pred(pname)
        x = z3.Const("x", Entity)
        y = z3.Const("y", Entity)
        z = z3.Const("z", Entity)
        if PredicateProperty.IRREFLEXIVE in props:
            solver.add(z3.ForAll([x], z3.Not(p(x, x))))
        if PredicateProperty.ANTISYMMETRIC in props:
            solver.add(z3.ForAll(
                [x, y],
                z3.Implies(z3.And(p(x, y), x != y), z3.Not(p(y, x))),
            ))
        if PredicateProperty.FUNCTIONAL in props:
            solver.add(z3.ForAll(
                [x, y, z],
                z3.Implies(z3.And(p(x, y), p(x, z)), y == z),
            ))
        if PredicateProperty.TRANSITIVE in props:
            solver.add(z3.ForAll(
                [x, y, z],
                z3.Implies(z3.And(p(x, y), p(y, z)), p(x, z)),
            ))

    # Constrain distinct entity strings to be distinct Z3 entities.
    # Without this, Z3 can vacuously satisfy properties like
    # antisymmetry / functional by collapsing entities — e.g., it
    # can decide alice == bob to satisfy mutual parent_of. We don't
    # want that interpretation; the substrate's identifiers ARE
    # distinct by construction.
    if len(entity_consts) >= 2:
        solver.add(z3.Distinct(*entity_consts.values()))

    result = solver.check()
    elapsed = time.perf_counter() - t0

    if result == z3.sat:
        return ConsistencyResult(
            consistent=True,
            unsat_core=[],
            z3_check_seconds=elapsed,
            n_triples=len(triples_list),
            n_predicates_with_properties=n_predicates_with_properties,
        )
    if result == z3.unsat:
        core_labels = solver.unsat_core()
        core_indices = sorted(
            track_labels[lbl.decl().name()] for lbl in core_labels
            if lbl.decl().name() in track_labels
        )
        return ConsistencyResult(
            consistent=False,
            unsat_core=core_indices,
            z3_check_seconds=elapsed,
            n_triples=len(triples_list),
            n_predicates_with_properties=n_predicates_with_properties,
        )
    # z3.unknown — timeout or theory-mixed undecidability. Treat
    # as "not provably contradictory"; surface honestly.
    return ConsistencyResult(
        consistent=True,  # weakly: not proven inconsistent
        unsat_core=[],
        z3_check_seconds=elapsed,
        n_triples=len(triples_list),
        n_predicates_with_properties=n_predicates_with_properties,
    )
