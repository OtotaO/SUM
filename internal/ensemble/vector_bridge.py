"""
Continuous-Discrete Bridge — Vector Space ↔ Gödel Space

Maps the discrete world of Gödel prime integers to the continuous world
of LLM embedding vectors, enabling semantic (fuzzy) search over a
mathematically exact state.

Architecture:
    - Every minted prime gets a vector embedding of its natural-language
      axiom text.
    - ``semantic_search_godel_state`` filters to primes *alive* in the
      current state (``global_state % prime == 0``) before ranking by
      cosine similarity.
    - Deleted axioms are automatically excluded because their primes no
      longer divide the state.

Horizon III — Universal Vector Alignment:
    - Supports optional affine transformation matrices (W*, b*) from
      Gorbett & Jana (2026) for cross-model linear alignment.
    - Heterogeneous P2P nodes (Llama, Qwen, Mistral, etc.) can perfectly
      align their embeddings into a single Canonical Geometry before
      discrete prime extraction via O(1) linear affine maps.

Author: ototao
License: Apache License 2.0
"""

import logging
from typing import Callable, Awaitable, List, Dict, Optional, Tuple

import numpy as np

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra

logger = logging.getLogger(__name__)


class ContinuousDiscreteBridge:
    """
    Connects the absolute mathematical certainty of Gödel integers
    with the fuzzy, semantic search capabilities of LLM Embeddings.

    The bridge maintains a mapping from each Semantic Prime to its
    vector embedding.  Queries are projected into the same space and
    ranked by cosine similarity — but *only* against primes that are
    currently alive in the global state.

    Horizon III:
        When ``affine_map`` (W*) and optional ``bias_map`` (b*) are
        provided, all embeddings are translated into the Canonical
        Geometry via an O(1) linear affine transformation before
        indexing and search.  This allows heterogeneous LLM nodes
        in a P2P swarm to perfectly align their latent spaces.
    """

    def __init__(
        self,
        algebra: GodelStateAlgebra,
        embedding_model: Callable[[str], Awaitable[List[float]]],
        affine_map: Optional[np.ndarray] = None,
        bias_map: Optional[np.ndarray] = None,
    ):
        """
        Args:
            algebra:         A GodelStateAlgebra instance with minted primes.
            embedding_model: Async callable (text) → List[float] vector.
            affine_map:      Optional W* matrix (d×d) for cross-model alignment.
            bias_map:        Optional b* bias vector (d,) for cross-model alignment.
        """
        self.algebra = algebra
        self.get_embedding = embedding_model
        self.prime_embeddings: Dict[int, np.ndarray] = {}

        # W* and b* matrices from Gorbett & Jana (2026) for cross-model
        # linear alignment into the Canonical Geometry.
        self.affine_map = affine_map
        self.bias_map = bias_map

    # ------------------------------------------------------------------
    # Affine alignment
    # ------------------------------------------------------------------

    def _align_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Applies O(1) Linear Alignment to translate heterogeneous latent
        spaces into the Canonical Geometry.

        If no affine map is configured, returns the input unchanged.

        Args:
            vector: Raw embedding vector (np.ndarray, float32).

        Returns:
            Aligned (and re-normalised) vector.
        """
        if self.affine_map is None:
            return vector

        v = np.dot(vector, self.affine_map)
        if self.bias_map is not None:
            v = v + self.bias_map

        # Re-normalise to unit length for cosine similarity
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm

        return v.astype(np.float32)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index_new_primes(self) -> int:
        """
        Generate embeddings for any un-indexed Semantic Primes.

        Returns:
            Number of newly indexed primes.
        """
        newly_indexed = 0

        for prime, axiom in self.algebra.prime_to_axiom.items():
            if prime not in self.prime_embeddings:
                # "alice||age||30" → "alice age 30"
                natural_text = " ".join(axiom.split("||"))
                raw_vector = await self.get_embedding(natural_text)
                aligned = self._align_vector(
                    np.array(raw_vector, dtype=np.float32)
                )
                self.prime_embeddings[prime] = aligned
                newly_indexed += 1

        if newly_indexed:
            logger.info("Indexed %d new primes into vector space.", newly_indexed)

        return newly_indexed

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    async def semantic_search_godel_state(
        self,
        global_state: int,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Query the exact Gödel state using fuzzy natural language.

        Only primes that are currently *alive* in the global state are
        considered.  Deleted axioms are automatically excluded because
        their primes no longer divide the state.

        Args:
            global_state: The current global Gödel integer.
            query:        Natural language query string.
            top_k:        Maximum number of results.

        Returns:
            List of (axiom_key, similarity_score) tuples, descending.
        """
        raw_query = np.array(
            await self.get_embedding(query), dtype=np.float32
        )
        query_vector = self._align_vector(raw_query)
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []

        active_primes: List[int] = []
        similarities: List[float] = []

        for prime, vector in self.prime_embeddings.items():
            # Only search facts alive in the current state
            if global_state % prime == 0:
                vec_norm = np.linalg.norm(vector)
                if vec_norm == 0:
                    continue
                sim = float(
                    np.dot(query_vector, vector) / (query_norm * vec_norm)
                )
                active_primes.append(prime)
                similarities.append(sim)

        if not active_primes:
            return []

        # Sort by similarity descending
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [
            (self.algebra.prime_to_axiom[active_primes[i]], similarities[i])
            for i in top_indices
        ]
