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

Author: ototao
License: Apache License 2.0
"""

import logging
from typing import Callable, Awaitable, List, Dict, Tuple

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
    """

    def __init__(
        self,
        algebra: GodelStateAlgebra,
        embedding_model: Callable[[str], Awaitable[List[float]]],
    ):
        """
        Args:
            algebra:         A GodelStateAlgebra instance with minted primes.
            embedding_model: Async callable (text) → List[float] vector.
        """
        self.algebra = algebra
        self.get_embedding = embedding_model
        self.prime_embeddings: Dict[int, np.ndarray] = {}

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
                vector = await self.get_embedding(natural_text)
                self.prime_embeddings[prime] = np.array(vector, dtype=np.float32)
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
        query_vector = np.array(
            await self.get_embedding(query), dtype=np.float32
        )
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
