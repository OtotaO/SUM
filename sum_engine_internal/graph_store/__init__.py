"""Graph-store backend interface for Phase 26.

This module is the spike-stage scaffolding called for in
`docs/PHASE_26_DESIGN.md` §5: a small backend trait that lets us
swap candidate stores (egglog now; Neo4j / PostgreSQL+AGE in a
later spike PR) and a measurement harness that produces
`sum.phase_26_backing_store_spike.v1` receipts.

The interface intentionally captures only what the SUM substrate
actually does today — insert (subject, predicate, object) facts,
count them, query by partial pattern, hash the whole state for
determinism checks. It is NOT a general-purpose graph database
interface; it is a *substrate-shaped* interface, which is what
the Phase 26.1+ engineering will build against.

A backing-store decision belongs to the spike receipt, not to this
file. Adding `EgglogStore` does not commit Phase 26 to egglog; it
makes egglog *measurable* on the same workload as future candidates.
"""
from sum_engine_internal.graph_store.base import (
    GraphStore,
    GraphStoreInfo,
    Triple,
)

__all__ = ["GraphStore", "GraphStoreInfo", "Triple"]
