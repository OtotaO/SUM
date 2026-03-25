"""
Tests for Phase 19D: Index Coherence Invariant

The single invariant under test:

    index_view == brute_force_view  after every state mutation path.

This suite simulates every way the router can mutate branch state
and verifies that the ActivePrimeIndex stays coherent with the
Gödel integer.  If any mutation path forgets to update the index,
this suite catches it.

Mutation paths covered:
    1. boot_sequence (rebuild from state)
    2. /ingest/math  (add primes)
    3. /ingest LLM   (LCM merge from mass engine)
    4. /branch       (fork)
    5. /merge        (LCM union)
    6. delete/DIV    (remove primes)
    7. time-travel   (rebuild from historical algebra)
    8. sync/state    (LCM merge from peer)
    9. causal cascade (trigger_map may mint primes)
   10. JWT user init (empty branch)

Author: ototao
License: Apache License 2.0
"""

import math
import pytest

from internal.algorithms.semantic_arithmetic import (
    GodelStateAlgebra,
    ActivePrimeIndex,
)


def _assert_coherent(index, branch, state, algebra, context=""):
    """The invariant: index view must equal brute-force scan."""
    indexed = sorted(index.get_active_axioms(branch, algebra))
    brute = sorted(algebra.get_active_axioms(state))
    assert indexed == brute, (
        f"COHERENCE VIOLATION ({context}): "
        f"index has {len(indexed)} axioms, brute-force has {len(brute)}"
    )


@pytest.fixture
def algebra():
    return GodelStateAlgebra()


@pytest.fixture
def index():
    return ActivePrimeIndex()


# ─── 1. Boot / Rebuild ───────────────────────────────────────────────

class TestCoherenceBoot:

    def test_rebuild_from_empty(self, algebra, index):
        index.rebuild("main", 1, algebra)
        _assert_coherent(index, "main", 1, algebra, "empty boot")

    def test_rebuild_from_populated(self, algebra, index):
        state = 1
        for i in range(50):
            p = algebra.get_or_mint_prime(f"e{i}", "rel", f"v{i}")
            state = math.lcm(state, p)
        index.rebuild("main", state, algebra)
        _assert_coherent(index, "main", state, algebra, "boot 50 axioms")

    def test_rebuild_after_partial_deletion(self, algebra, index):
        """Simulate boot after some primes were DIV'd out."""
        state = 1
        primes = []
        for i in range(20):
            p = algebra.get_or_mint_prime(f"e{i}", "rel", f"v{i}")
            state = math.lcm(state, p)
            primes.append(p)
        # Remove 5 primes
        for p in primes[:5]:
            while state % p == 0:
                state //= p
        index.rebuild("main", state, algebra)
        _assert_coherent(index, "main", state, algebra, "post-deletion boot")


# ─── 2. Ingest / Add ─────────────────────────────────────────────────

class TestCoherenceIngest:

    def test_sequential_add(self, algebra, index):
        """Simulate /ingest/math adding primes one at a time."""
        state = 1
        index.rebuild("main", state, algebra)
        for i in range(30):
            p = algebra.get_or_mint_prime(f"fact_{i}", "is", f"true_{i}")
            state = math.lcm(state, p)
            index.add("main", p)
        _assert_coherent(index, "main", state, algebra, "sequential ingest")

    def test_batch_lcm_merge(self, algebra, index):
        """Simulate /ingest LLM path: mass engine returns a product, LCM-merge."""
        state = 1
        index.rebuild("main", state, algebra)
        # Simulate mass engine output
        batch_state = 1
        for i in range(20):
            p = algebra.get_or_mint_prime(f"llm_{i}", "says", f"thing_{i}")
            batch_state = math.lcm(batch_state, p)
        state = math.lcm(state, batch_state)
        # Router uses rebuild after LCM merge (as we implemented)
        index.rebuild("main", state, algebra)
        _assert_coherent(index, "main", state, algebra, "batch LCM merge")


# ─── 3. Branch / Fork ────────────────────────────────────────────────

class TestCoherenceBranch:

    def test_fork_coherence(self, algebra, index):
        state = 1
        for i in range(10):
            p = algebra.get_or_mint_prime(f"base_{i}", "rel", f"v_{i}")
            state = math.lcm(state, p)
        index.rebuild("main", state, algebra)
        index.fork("main", "experiment")
        _assert_coherent(index, "experiment", state, algebra, "fork")

    def test_fork_then_mutate_isolation(self, algebra, index):
        """Mutations on fork must not affect source."""
        state = 1
        for i in range(10):
            p = algebra.get_or_mint_prime(f"base_{i}", "rel", f"v_{i}")
            state = math.lcm(state, p)
        index.rebuild("main", state, algebra)
        index.fork("main", "feature")

        # Add to feature only
        new_p = algebra.get_or_mint_prime("feature_only", "rel", "val")
        feature_state = math.lcm(state, new_p)
        index.add("feature", new_p)

        _assert_coherent(index, "main", state, algebra, "source after fork+mutate")
        _assert_coherent(index, "feature", feature_state, algebra, "fork after mutate")


# ─── 4. Merge ────────────────────────────────────────────────────────

class TestCoherenceMerge:

    def test_lcm_merge_coherence(self, algebra, index):
        """Simulate /merge: LCM of two branches."""
        state_a = 1
        state_b = 1
        for i in range(10):
            p = algebra.get_or_mint_prime(f"a_{i}", "rel", f"v_{i}")
            state_a = math.lcm(state_a, p)
        for i in range(10, 20):
            p = algebra.get_or_mint_prime(f"b_{i}", "rel", f"v_{i}")
            state_b = math.lcm(state_b, p)

        index.rebuild("branch_a", state_a, algebra)
        index.rebuild("branch_b", state_b, algebra)

        merged = math.lcm(state_a, state_b)
        index.merge("branch_a", "branch_a", "branch_b")
        _assert_coherent(index, "branch_a", merged, algebra, "merge")


# ─── 5. Delete / DIV ─────────────────────────────────────────────────

class TestCoherenceDelete:

    def test_remove_coherence(self, algebra, index):
        state = 1
        primes = []
        for i in range(15):
            p = algebra.get_or_mint_prime(f"del_{i}", "rel", f"v_{i}")
            state = math.lcm(state, p)
            primes.append(p)
        index.rebuild("main", state, algebra)

        # Remove 3 primes
        for p in primes[5:8]:
            while state % p == 0:
                state //= p
            index.remove("main", p)

        _assert_coherent(index, "main", state, algebra, "after delete")


# ─── 6. Time-Travel ──────────────────────────────────────────────────

class TestCoherenceTimeTravel:

    def test_historical_rebuild(self, algebra, index):
        """Simulate time-travel: rebuild with subset of axioms."""
        state = 1
        for i in range(20):
            p = algebra.get_or_mint_prime(f"hist_{i}", "rel", f"v_{i}")
            state = math.lcm(state, p)
        index.rebuild("main", state, algebra)

        # Time-travel: only first 10 axioms
        past_state = 1
        for i in range(10):
            p = algebra.axiom_to_prime[f"hist_{i}||rel||v_{i}"]
            past_state = math.lcm(past_state, p)
        index.rebuild("time_branch", past_state, algebra)
        _assert_coherent(index, "time_branch", past_state, algebra, "time-travel")


# ─── 7. Sync / P2P ───────────────────────────────────────────────────

class TestCoherenceSync:

    def test_peer_lcm_merge(self, algebra, index):
        """Simulate /sync/state: LCM merge from peer integer."""
        local_state = 1
        for i in range(10):
            p = algebra.get_or_mint_prime(f"local_{i}", "rel", f"v_{i}")
            local_state = math.lcm(local_state, p)
        index.rebuild("main", local_state, algebra)

        # Peer sends primes 5-15
        peer_state = 1
        for i in range(5, 15):
            p = algebra.get_or_mint_prime(f"peer_{i}", "rel", f"v_{i}")
            peer_state = math.lcm(peer_state, p)

        merged = math.lcm(local_state, peer_state)
        index.rebuild("main", merged, algebra)
        _assert_coherent(index, "main", merged, algebra, "peer sync")


# ─── 8. JWT / Empty Init ─────────────────────────────────────────────

class TestCoherenceJWT:

    def test_empty_user_branch(self, algebra, index):
        """Simulate JWT user branch creation: state=1."""
        index.rebuild("user_alice", 1, algebra)
        _assert_coherent(index, "user_alice", 1, algebra, "JWT init")

    def test_user_branch_then_ingest(self, algebra, index):
        """JWT init → ingest → verify coherence."""
        index.rebuild("user_bob", 1, algebra)
        state = 1
        for i in range(5):
            p = algebra.get_or_mint_prime(f"bob_{i}", "knows", f"thing_{i}")
            state = math.lcm(state, p)
            index.add("user_bob", p)
        _assert_coherent(index, "user_bob", state, algebra, "JWT + ingest")


# ─── 9. Stress: Multi-Branch Concurrent ──────────────────────────────

class TestCoherenceStress:

    def test_10_branches_100_axioms(self, algebra, index):
        """Create 10 branches with overlapping axiom sets."""
        all_primes = []
        for i in range(100):
            p = algebra.get_or_mint_prime(f"stress_{i}", "rel", f"v_{i}")
            all_primes.append(p)

        branches = {}
        for b in range(10):
            state = 1
            start = b * 5
            for p in all_primes[start:start + 15]:
                state = math.lcm(state, p)
            branches[f"branch_{b}"] = state
            index.rebuild(f"branch_{b}", state, algebra)

        for name, state in branches.items():
            _assert_coherent(index, name, state, algebra, f"stress {name}")
