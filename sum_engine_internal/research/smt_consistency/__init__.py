"""SMT-backed axiom-set consistency checking.

Detects contradictory axioms before signing. The substrate today
happily signs axiom sets where, e.g., ``(alice, parent_of, bob)``
and ``(bob, parent_of, alice)`` coexist — Z3 with a small
predicate library catches this.

Provable kernel: Nelson-Oppen combination (1979) + modern
CDCL(T) — De Moura & Bjørner, *TACAS* 2008. Decidable for the
QF_UF (quantifier-free uninterpreted functions) fragment that
covers the substrate's axiom-shape.

For each predicate the substrate cares about, the operator
declares its logical properties (antisymmetric, irreflexive,
functional, transitive, etc.); Z3 then checks whether the
asserted axioms are jointly satisfiable. UNSAT ⇒ contradiction;
the UNSAT core is the smallest subset of axioms that produces
the contradiction, surfacing actionable signal for the operator.
"""
from sum_engine_internal.research.smt_consistency.consistency import (
    ConsistencyResult,
    PredicateProperty,
    check_consistency,
)

__all__ = [
    "ConsistencyResult",
    "PredicateProperty",
    "check_consistency",
]
