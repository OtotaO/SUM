"""
Phase 17 — Horizon III Tests: Universal Vector Alignment & Bare-Metal Core

Validates:
    1. Affine alignment (W*, b*) for cross-model vector space translation
    2. Zig FFI bridge graceful fallback when library is unavailable
    3. Strangler Fig: _deterministic_prime produces identical primes
       regardless of whether the Zig engine is present
    4. Full pipeline round-trip through the Strangler Fig path

Author: ototao
License: Apache License 2.0
"""

import sys
import os
import hashlib
import math
import asyncio
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.ensemble.vector_bridge import ContinuousDiscreteBridge
from internal.infrastructure.zig_bridge import ZigMathEngine


# ─── Phase 16 Reference Vectors (Cross-Language Ground Truth) ────────

REFERENCE_VECTORS = [
    {
        "axiom_key": "alice||likes||cats",
        "prime": 14326936561644797201,
    },
    {
        "axiom_key": "bob||knows||python",
        "prime": 12933559861697884259,
    },
    {
        "axiom_key": "earth||orbits||sun",
        "prime": 10246101339925224733,
    },
]


# ─── Mock Embedding Model ────────────────────────────────────────────

def make_deterministic_embedder(dim: int = 8):
    """Deterministic mock embedder seeded from text hash."""
    async def embed(text: str):
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()
    return embed


# ═══════════════════════════════════════════════════════════════════════
# 1. AFFINE VECTOR ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════

class TestAffineAlignment:

    @pytest.mark.asyncio
    async def test_identity_affine_matches_no_affine(self):
        """Identity W* produces identical results to no affine map."""
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([
            ("Alice", "age", "30"),
            ("Bob", "role", "admin"),
        ])

        embed_fn = make_deterministic_embedder(dim=8)

        # No affine
        bridge_none = ContinuousDiscreteBridge(alg, embed_fn)
        await bridge_none.index_new_primes()
        results_none = await bridge_none.semantic_search_godel_state(
            state, "how old is alice", top_k=5
        )

        # Identity affine (W* = I, no bias)
        identity = np.eye(8, dtype=np.float32)
        bridge_id = ContinuousDiscreteBridge(alg, embed_fn, affine_map=identity)
        await bridge_id.index_new_primes()
        results_id = await bridge_id.semantic_search_godel_state(
            state, "how old is alice", top_k=5
        )

        # Same axioms in same order
        keys_none = [r[0] for r in results_none]
        keys_id = [r[0] for r in results_id]
        assert keys_none == keys_id

        # Similarity scores should be very close (float rounding)
        for (_, s1), (_, s2) in zip(results_none, results_id):
            assert abs(s1 - s2) < 1e-5

    @pytest.mark.asyncio
    async def test_affine_rotation_changes_ranking(self):
        """A non-trivial W* rotation changes the similarity ranking."""
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([
            ("Alice", "age", "30"),
            ("Bob", "role", "admin"),
            ("Carol", "skill", "python"),
        ])

        embed_fn = make_deterministic_embedder(dim=8)

        # No affine
        bridge_none = ContinuousDiscreteBridge(alg, embed_fn)
        await bridge_none.index_new_primes()
        results_none = await bridge_none.semantic_search_godel_state(
            state, "programming", top_k=3
        )

        # Random rotation matrix (orthogonal)
        rng = np.random.RandomState(42)
        Q, _ = np.linalg.qr(rng.randn(8, 8).astype(np.float32))
        # Make it non-trivial by adding a permutation-like distortion
        W = Q @ np.diag(np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float32))

        bridge_rot = ContinuousDiscreteBridge(alg, embed_fn, affine_map=W)
        await bridge_rot.index_new_primes()
        results_rot = await bridge_rot.semantic_search_godel_state(
            state, "programming", top_k=3
        )

        # Scores should differ (W* changes the geometry)
        scores_none = [r[1] for r in results_none]
        scores_rot = [r[1] for r in results_rot]
        assert scores_none != scores_rot

    @pytest.mark.asyncio
    async def test_bias_vector_applied(self):
        """Bias vector b* shifts the embedding space."""
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([("Alice", "age", "30")])

        embed_fn = make_deterministic_embedder(dim=8)

        # Identity W* with a large bias to force a shift
        identity = np.eye(8, dtype=np.float32)
        bias = np.ones(8, dtype=np.float32) * 10.0

        bridge = ContinuousDiscreteBridge(
            alg, embed_fn, affine_map=identity, bias_map=bias
        )
        await bridge.index_new_primes()

        # The indexed embedding should be different from the raw one
        prime = list(bridge.prime_embeddings.keys())[0]
        aligned_vec = bridge.prime_embeddings[prime]

        # Raw embedding (without alignment)
        bridge_raw = ContinuousDiscreteBridge(alg, embed_fn)
        await bridge_raw.index_new_primes()
        raw_vec = bridge_raw.prime_embeddings[prime]

        # They should differ because the bias shifts the vector
        assert not np.allclose(aligned_vec, raw_vec, atol=1e-3)

    @pytest.mark.asyncio
    async def test_aligned_vectors_are_unit_normalized(self):
        """Aligned vectors are re-normalised to unit length."""
        alg = GodelStateAlgebra()
        alg.encode_chunk_state([
            ("Alice", "age", "30"),
            ("Bob", "role", "admin"),
        ])

        embed_fn = make_deterministic_embedder(dim=8)
        rng = np.random.RandomState(123)
        W = rng.randn(8, 8).astype(np.float32) * 5.0  # Large scaling
        bias = rng.randn(8).astype(np.float32) * 3.0

        bridge = ContinuousDiscreteBridge(
            alg, embed_fn, affine_map=W, bias_map=bias
        )
        await bridge.index_new_primes()

        for vec in bridge.prime_embeddings.values():
            norm = np.linalg.norm(vec)
            assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    @pytest.mark.asyncio
    async def test_backward_compatible_no_affine(self):
        """Without affine params, ContinuousDiscreteBridge works exactly as before."""
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([
            ("X", "is", "1"),
            ("Y", "is", "2"),
        ])

        bridge = ContinuousDiscreteBridge(alg, make_deterministic_embedder())
        count = await bridge.index_new_primes()
        assert count == 2

        results = await bridge.semantic_search_godel_state(state, "query", top_k=5)
        assert len(results) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


# ═══════════════════════════════════════════════════════════════════════
# 2. ZIG FFI BRIDGE
# ═══════════════════════════════════════════════════════════════════════

class TestZigBridge:

    def test_zig_engine_graceful_when_no_library(self):
        """ZigMathEngine initialises without crashing when .dylib/.so missing."""
        engine = ZigMathEngine()
        # On this machine (no zig build), lib should be None
        # This test just verifies no exception is raised
        assert engine.lib is None or engine.lib is not None  # Always passes

    def test_zig_engine_returns_none_without_library(self):
        """get_deterministic_prime returns None when library is unavailable."""
        engine = ZigMathEngine()
        if engine.lib is None:
            result = engine.get_deterministic_prime("alice||likes||cats")
            assert result is None

    def test_zig_engine_available_property(self):
        """The .available property correctly reflects library state."""
        engine = ZigMathEngine()
        if engine.lib is None:
            assert not engine.available
        else:
            assert engine.available


# ═══════════════════════════════════════════════════════════════════════
# 3. STRANGLER FIG: DETERMINISTIC PRIME CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════

class TestStranglerFig:

    def test_strangler_fig_matches_reference_vectors(self):
        """
        _deterministic_prime produces identical primes to the Phase 16
        cross-language reference vectors, regardless of Zig availability.
        """
        alg = GodelStateAlgebra()
        for vec in REFERENCE_VECTORS:
            parts = vec["axiom_key"].split("||")
            prime = alg.get_or_mint_prime(parts[0], parts[1], parts[2])
            assert prime == vec["prime"], (
                f"Prime mismatch for {vec['axiom_key']}: "
                f"got {prime}, expected {vec['prime']}"
            )

    def test_strangler_fig_full_pipeline_round_trip(self):
        """Full pipeline: mint → encode → LCM → entailment still works."""
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([
            ("alice", "likes", "cats"),
            ("bob", "knows", "python"),
            ("earth", "orbits", "sun"),
        ])

        # Verify the expected LCM state from Phase 16
        expected_lcm = 1898585074409907150524167558344558620554613878579045806247
        assert state == expected_lcm

        # Entailment still works
        for vec in REFERENCE_VECTORS:
            assert state % vec["prime"] == 0

    def test_strangler_fig_collision_handling(self):
        """Collision resolution still works through the Strangler Fig path."""
        alg = GodelStateAlgebra()
        # Mint two different axioms — they should get different primes
        p1 = alg.get_or_mint_prime("alpha", "is", "1")
        p2 = alg.get_or_mint_prime("beta", "is", "2")
        assert p1 != p2

    def test_strangler_fig_deterministic_across_instances(self):
        """Two independent GodelStateAlgebra instances produce identical primes."""
        alg1 = GodelStateAlgebra()
        alg2 = GodelStateAlgebra()

        for vec in REFERENCE_VECTORS:
            parts = vec["axiom_key"].split("||")
            p1 = alg1.get_or_mint_prime(parts[0], parts[1], parts[2])
            p2 = alg2.get_or_mint_prime(parts[0], parts[1], parts[2])
            assert p1 == p2 == vec["prime"]

    def test_strangler_fig_delete_and_update_still_work(self):
        """Temporal operations work correctly through the Strangler Fig."""
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([
            ("Alice", "age", "30"),
            ("Bob", "age", "40"),
        ])

        # Delete
        p_alice = alg.axiom_to_prime["alice||age||30"]
        new_state = alg.delete_axiom(state, "alice||age||30")
        assert new_state % p_alice != 0

        # Update
        updated_state = alg.update_axiom(state, "alice||age||30", "alice||age||31")
        p_new = alg.axiom_to_prime["alice||age||31"]
        assert updated_state % p_new == 0
        assert updated_state % p_alice != 0


# ═══════════════════════════════════════════════════════════════════════
# 4. INTEGRATION: AFFINE + STRANGLER FIG TOGETHER
# ═══════════════════════════════════════════════════════════════════════

class TestHorizonIIIIntegration:

    @pytest.mark.asyncio
    async def test_affine_bridge_with_strangler_fig_primes(self):
        """
        End-to-end: Strangler Fig primes + affine-aligned vector search
        produces correct results.
        """
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([
            ("alice", "likes", "cats"),
            ("bob", "knows", "python"),
            ("earth", "orbits", "sun"),
        ])

        # Verify primes came through Strangler Fig correctly
        for vec in REFERENCE_VECTORS:
            assert state % vec["prime"] == 0

        # Build an affine-aligned bridge
        identity = np.eye(8, dtype=np.float32)
        bridge = ContinuousDiscreteBridge(
            alg, make_deterministic_embedder(dim=8), affine_map=identity
        )
        await bridge.index_new_primes()
        assert len(bridge.prime_embeddings) == 3

        # Semantic search returns all alive axioms
        results = await bridge.semantic_search_godel_state(
            state, "animals pets", top_k=3
        )
        returned_keys = [r[0] for r in results]
        assert len(returned_keys) == 3

    @pytest.mark.asyncio
    async def test_delete_removes_from_affine_search(self):
        """Deleting an axiom removes it from affine-aligned search."""
        alg = GodelStateAlgebra()
        state = alg.encode_chunk_state([
            ("alice", "likes", "cats"),
            ("bob", "knows", "python"),
        ])

        bridge = ContinuousDiscreteBridge(
            alg, make_deterministic_embedder(dim=8),
            affine_map=np.eye(8, dtype=np.float32),
        )
        await bridge.index_new_primes()

        # Delete alice
        new_state = alg.delete_axiom(state, "alice||likes||cats")
        results = await bridge.semantic_search_godel_state(
            new_state, "cats", top_k=5
        )
        returned_keys = [r[0] for r in results]
        assert "alice||likes||cats" not in returned_keys
        assert "bob||knows||python" in returned_keys
