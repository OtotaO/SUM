import sys
sys.set_int_max_str_digits(0)

"""
Semantic Prime Number Theorem (SPNT) & Gödel-State Algebra

Maps irreducible semantic axioms to Prime Numbers. Document states become
massive integers. Merging parallel extractions is an LCM operation, and
entailment checking is Modulo arithmetic.

Mathematical foundations:
    - SPNT Bound: π_sem(N) ~ N / ln(N)
    - Gödel Encoding: state = ∏ prime_i for each axiom_i
    - Lock-Free Merge: LCM(state_A, state_B) deduplicates automatically
    - Entailment: global_state % hypothesis_state == 0

Author: ototao
License: Apache License 2.0
"""

import math
import hashlib
import logging
from typing import Dict, List, Tuple, Set

import sympy

logger = logging.getLogger(__name__)


class SemanticPrimeNumberTheorem:
    """
    The classical PNT states that the density of primes up to N is
    asymptotically N / ln(N).  Here we apply the same bound to
    *semantic primes* — the irreducible, highly-surprising axiomatic
    facts inside a document corpus.

    If an LLM extraction yields more axioms than the SPNT bound,
    it has failed to compress (hallucinated verbosity) and the
    extraction should be rejected.
    """

    @staticmethod
    def asymptotic_bound(total_raw_claims: int) -> int:
        """
        π_sem(N) ~ N / ln(N)

        Calculates the theoretical maximum number of irreducible semantic
        primes for a corpus of N raw claims.

        Args:
            total_raw_claims: Total number of raw claims extracted.

        Returns:
            The SPNT upper bound on truly novel axioms.
        """
        if total_raw_claims <= 3:
            return total_raw_claims
        return int(total_raw_claims / math.log(total_raw_claims))


class GodelStateAlgebra:
    """
    Reduces Mass-Parallel Graph Merging and SMT Theorem Proving
    to Arbitrary-Precision Integer Arithmetic.

    Every unique (subject, predicate, object) triple is assigned a unique
    prime number.  A document chunk's semantic state is the *product* of
    its primes.  Merging states is an LCM.  Entailment is a modulo check.
    Paradox detection looks for mutually exclusive primes that both
    divide the global state.
    """

    def __init__(self):
        # Deterministic primes — no sequential watermark needed
        self.axiom_to_prime: Dict[str, int] = {}
        self.prime_to_axiom: Dict[int, str] = {}

        # Tracks mutually exclusive primes (Level 3 Curvature / Symmetry Breaking).
        # Maps a context key (subject||predicate) to its set of competing object primes.
        self.exclusion_zones: Dict[str, Set[int]] = {}

        # Fractal Crystallization provenance.
        # Maps macro-prime → product of micro-primes for lossless decompression.
        self.macro_provenance: Dict[int, int] = {}

        # Quantum GraphRAG node registry.
        # Maps each node (subject/object) to the product of all primes it participates in.
        self.node_registry: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Prime minting
    # ------------------------------------------------------------------

    def _deterministic_prime(self, axiom_key: str) -> int:
        """
        Generates a globally unique, deterministic prime for a given axiom.

        Uses SHA-256 to hash the axiom key into a 64-bit integer space,
        then finds the next prime.  This allows lock-free LCM merging
        of Gödel States across branches and distributed networks.

        HORIZON III — Strangler Fig:
            Routes to the bare-metal Zig core (nanosecond-speed C-ABI)
            when ``libsum_core`` is compiled and available.  Falls back
            to Python ``sympy.nextprime`` seamlessly otherwise.

        Args:
            axiom_key: The normalised axiom string.

        Returns:
            A deterministic prime number unique to this axiom.
        """
        # ── Bare-Metal Fast Path (Zig C-ABI) ──
        try:
            from internal.infrastructure.zig_bridge import zig_engine
            if zig_engine is not None:
                result = zig_engine.get_deterministic_prime(axiom_key)
                if result is not None:
                    return result
        except ImportError:
            pass

        # ── Legacy Python Fallback ──
        h = hashlib.sha256(axiom_key.encode('utf-8')).digest()
        seed = int.from_bytes(h[:8], byteorder='big')
        return sympy.nextprime(seed)

    def get_or_mint_prime(self, subject: str, predicate: str, object_: str) -> int:
        """
        Mint a new Universal Semantic Prime for an irreducible axiom,
        or return the existing one if we have seen this axiom before.

        Uses cryptographic hashing for deterministic, globally consistent
        prime assignment.  Two isolated instances will always generate
        the exact same prime for the same axiom.

        Args:
            subject:   The entity.
            predicate: The relation / property.
            object_:   The value.

        Returns:
            The prime number assigned to this axiom.
        """
        axiom_key = (
            f"{subject.strip().lower()}||"
            f"{predicate.strip().lower()}||"
            f"{object_.strip().lower()}"
        )

        if axiom_key not in self.axiom_to_prime:
            p = self._deterministic_prime(axiom_key)

            # Collision detection (astronomically rare, but required)
            while p in self.prime_to_axiom and self.prime_to_axiom[p] != axiom_key:
                p = sympy.nextprime(p)

            self.axiom_to_prime[axiom_key] = p
            self.prime_to_axiom[p] = axiom_key

            # Register exclusion zone (subject + predicate).
            # Any two primes in the same zone are mutually exclusive
            # (e.g. "alice||age||30" vs "alice||age||31").
            zone_key = (
                f"{subject.strip().lower()}||"
                f"{predicate.strip().lower()}"
            )
            if zone_key not in self.exclusion_zones:
                self.exclusion_zones[zone_key] = set()
            self.exclusion_zones[zone_key].add(p)

            # Update node registry for GraphRAG traversal
            self._update_node_registry(
                subject.strip().lower(), object_.strip().lower(), p
            )

        return self.axiom_to_prime[axiom_key]

    # ------------------------------------------------------------------
    # Quantum GraphRAG (Node Registry)
    # ------------------------------------------------------------------

    def _update_node_registry(
        self, subject: str, object_: str, prime: int
    ):
        """
        Maintains the Context Integer for each node.

        Every time a prime is minted, both its subject and object nodes
        accumulate the prime into their integer via LCM.
        """
        if subject not in self.node_registry:
            self.node_registry[subject] = 1
        if object_ not in self.node_registry:
            self.node_registry[object_] = 1
        self.node_registry[subject] = math.lcm(
            self.node_registry[subject], prime
        )
        self.node_registry[object_] = math.lcm(
            self.node_registry[object_], prime
        )

    def get_quantum_neighborhood(
        self, global_state: int, nodes: List[str], hops: int = 1
    ) -> int:
        """
        GraphRAG Traversal with N-Hop Support.

        Returns the exact Gödel Integer representing the combined
        topological neighborhood of the requested nodes, filtered to
        only include axioms alive in the current global state.

        The traversal iteratively expands the frontier: at each hop,
        new neighbor nodes are discovered and their edges are collected.

        Args:
            global_state: Current Gödel integer.
            nodes:        Entity names to query.
            hops:         Number of traversal hops (1-hop, 2-hop, etc.).

        Returns:
            Gödel integer representing the neighbourhood context.
        """
        visited: set = set()
        frontier: set = set(n.strip().lower() for n in nodes)
        context_state = 1

        for _hop in range(hops):
            next_frontier: set = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                if node in self.node_registry:
                    node_integer = self.node_registry[node]
                    # Filter for only edges alive in the global state
                    alive_node_integer = math.gcd(
                        global_state, node_integer
                    )
                    context_state = math.lcm(
                        context_state, alive_node_integer
                    )
                    # Discover neighbor nodes for next hop
                    for prime, axiom in self.prime_to_axiom.items():
                        if alive_node_integer % prime == 0:
                            parts = axiom.split("||")
                            if len(parts) == 3:
                                next_frontier.add(parts[0])
                                next_frontier.add(parts[2])
            frontier = next_frontier - visited

        return context_state

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_chunk_state(self, axioms: List[Tuple[str, str, str]]) -> int:
        """
        Convert a local parallel LLM extraction into a single Gödel State
        Integer — the product of each axiom's prime.

        Args:
            axioms: List of (subject, predicate, object) triples.

        Returns:
            A single integer encoding the full semantic state.
        """
        state = 1
        for subj, pred, obj in axioms:
            state *= self.get_or_mint_prime(subj, pred, obj)
        return state

    # ------------------------------------------------------------------
    # Lock-free merge
    # ------------------------------------------------------------------

    def merge_parallel_states(self, state_integers: List[int]) -> int:
        """
        Lock-Free Mass Parallel Merge.

        The global semantic state is the Least Common Multiple (LCM) of
        all chunk states.  This automatically deduplicates overlapping
        knowledge and is both commutative and associative.

        Args:
            state_integers: Gödel integers from parallel chunk extractions.

        Returns:
            The merged global state.
        """
        if not state_integers:
            return 1
        return math.lcm(*state_integers)

    # ------------------------------------------------------------------
    # Entailment
    # ------------------------------------------------------------------

    def verify_entailment(self, global_state: int, hypothesis_state: int) -> bool:
        """
        Epistemic Fidelity Gate (Tags-to-Tomes).

        Does the Global State entail the Hypothesis?  If the modulo is 0,
        the hypothesis is *mathematically proven* to be a subset of the
        global knowledge.

        Args:
            global_state:     The merged global Gödel integer.
            hypothesis_state: The Gödel integer of the claim to verify.

        Returns:
            True if the hypothesis is entailed, False otherwise.
        """
        if hypothesis_state == 0:
            return False
        return global_state % hypothesis_state == 0

    # ------------------------------------------------------------------
    # Curvature / Paradox detection
    # ------------------------------------------------------------------

    def detect_curvature_paradoxes(self, global_state: int) -> List[str]:
        """
        Scan the global integer for Level 3 Curvature (Semantic Contradictions).

        If mutually exclusive claims (e.g. "Alice is 30" AND "Alice is 31")
        both divide the global state evenly, a curvature paradox is
        detected and the SMT solver should be triggered to arbitrate.

        Args:
            global_state: The merged global Gödel integer.

        Returns:
            List of human-readable paradox descriptions (empty = clean).
        """
        paradoxes: List[str] = []

        for zone_key, primes in self.exclusion_zones.items():
            if len(primes) < 2:
                continue

            active_primes = [p for p in primes if global_state % p == 0]

            if len(active_primes) > 1:
                conflicting = [self.prime_to_axiom[p] for p in active_primes]
                paradoxes.append(
                    f"Curvature Contradiction in [{zone_key}]: {conflicting}"
                )

        return paradoxes

    # ------------------------------------------------------------------
    # Hallucination isolation (Sieve of Hallucinations)
    # ------------------------------------------------------------------

    def isolate_hallucinations(
        self, global_state: int, generated_state: int
    ) -> List[str]:
        """
        The Sieve of Hallucinations.

        If ``generated_state`` does not divide ``global_state`` evenly, the
        LLM hallucinated.  This method uses ``math.gcd`` to find the exact
        prime numbers (axioms) the LLM fabricated — in microseconds.

        Algorithm:
            1. ``shared_truth = gcd(global_state, generated_state)``
               — the primes that *are* in the global state.
            2. ``hallucinated_product = generated_state // shared_truth``
               — the product of primes *not* in the global state.
            3. Factor ``hallucinated_product`` against known primes to
               recover the specific axiom strings.

        Args:
            global_state:    The verified global Gödel integer.
            generated_state: The Gödel integer extracted from LLM output.

        Returns:
            List of axiom key strings that were hallucinated (empty = clean).
        """
        if global_state % generated_state == 0:
            return []  # Zero hallucinations

        # Isolate the exact product of all hallucinated primes.
        shared_truth = math.gcd(global_state, generated_state)
        hallucinated_product = generated_state // shared_truth

        hallucinated_axioms: List[str] = []

        for prime, axiom in self.prime_to_axiom.items():
            if hallucinated_product % prime == 0:
                hallucinated_axioms.append(axiom)
                # Divide out the prime to break early if product reaches 1
                while hallucinated_product % prime == 0:
                    hallucinated_product //= prime
            if hallucinated_product == 1:
                break

        return hallucinated_axioms

    # ------------------------------------------------------------------
    # Temporal evolution (Delete / Update)
    # ------------------------------------------------------------------

    def delete_axiom(self, global_state: int, axiom_key: str) -> int:
        """
        Semantic Deletion.

        Removes a fact from the global state by dividing out its prime
        factor.  Handles multiple occurrences if merged carelessly.

        Args:
            global_state: The current global Gödel integer.
            axiom_key:    Normalised axiom key (``subject||predicate||object``).

        Returns:
            The updated global state with the axiom removed.
        """
        axiom_key = axiom_key.strip().lower()

        if axiom_key not in self.axiom_to_prime:
            return global_state  # Axiom was never minted

        prime = self.axiom_to_prime[axiom_key]

        # Completely factor out the prime
        while global_state % prime == 0:
            global_state //= prime

        return global_state

    def update_axiom(
        self, global_state: int, old_axiom_key: str, new_axiom_key: str
    ) -> int:
        """
        Semantic Update (Curvature Resolution).

        Divides out the obsolete fact and multiplies in the new fact —
        a single atomic gauge transformation on the state integer.

        Args:
            global_state:  The current global Gödel integer.
            old_axiom_key: Key of the axiom to replace (``subj||pred||obj``).
            new_axiom_key: Key of the replacement axiom (``subj||pred||obj``).

        Returns:
            The updated global state.

        Raises:
            ValueError: If the new axiom key is malformed.
        """
        parts = new_axiom_key.split("||")
        if len(parts) != 3:
            raise ValueError(
                "Invalid axiom format. Must be subject||predicate||object"
            )

        new_prime = self.get_or_mint_prime(parts[0], parts[1], parts[2])
        state_without_old = self.delete_axiom(global_state, old_axiom_key)

        return math.lcm(state_without_old, new_prime)

    # ------------------------------------------------------------------
    # Gödel Sync Protocol (Distributed Delta)
    # ------------------------------------------------------------------

    def calculate_network_delta(
        self, global_state: int, client_state: int
    ) -> dict:
        """
        O(1) Distributed State Synchronization.

        Compares the server's global integer with the client's integer to
        compute the exact semantic delta:
            shared_truth     = gcd(global, client)
            missing_product  = global // shared_truth   → axioms to ADD
            obsolete_product = client // shared_truth    → axioms to DELETE

        Args:
            global_state: The server's current Gödel integer.
            client_state: The client's cached Gödel integer.

        Returns:
            Dict with ``"add"`` and ``"delete"`` lists of axiom key strings.
        """
        if global_state == client_state:
            return {"add": [], "delete": []}

        shared_truth = math.gcd(global_state, client_state)

        # Primes the client is missing (server has, client doesn't)
        missing_product = global_state // shared_truth

        # Primes the client has that are obsolete
        obsolete_product = client_state // shared_truth

        def _extract_axioms(product: int) -> List[str]:
            axioms: List[str] = []
            for prime, axiom in self.prime_to_axiom.items():
                if product % prime == 0:
                    axioms.append(axiom)
                    while product % prime == 0:
                        product //= prime
                if product == 1:
                    break
            return axioms

        return {
            "add": _extract_axioms(missing_product),
            "delete": _extract_axioms(obsolete_product),
        }

    # ------------------------------------------------------------------
    # Fractal Crystallization (Semantic Zooming)
    # ------------------------------------------------------------------

    def crystallize_axioms(
        self,
        global_state: int,
        micro_axiom_keys: List[str],
        macro_axiom_key: str,
    ) -> int:
        """
        O(1) Fractal Compression (Zoom Out).

        Replaces a cluster of micro-primes with a single Macro-Prime.
        Stores the cluster product as provenance for lossless
        decompression via ``decrystallize_axiom``.

        Args:
            global_state:     Current global Gödel integer.
            micro_axiom_keys: Keys of micro-axioms to compress.
            macro_axiom_key:  Key for the new macro-axiom.

        Returns:
            Updated global state with micro-primes replaced by macro-prime.
        """
        cluster_product = 1

        for key in micro_axiom_keys:
            key = key.strip().lower()
            if (
                key in self.axiom_to_prime
                and global_state % self.axiom_to_prime[key] == 0
            ):
                prime = self.axiom_to_prime[key]
                cluster_product *= prime
                # Completely factor out the micro-prime
                while global_state % prime == 0:
                    global_state //= prime

        if cluster_product == 1:
            return global_state  # No active micro-primes found

        parts = macro_axiom_key.split("||")
        if len(parts) != 3:
            raise ValueError(
                "Invalid macro axiom format. Must be subject||predicate||object"
            )

        macro_prime = self.get_or_mint_prime(parts[0], parts[1], parts[2])

        # Store provenance for lossless decompression
        self.macro_provenance[macro_prime] = cluster_product

        return math.lcm(global_state, macro_prime)

    def decrystallize_axiom(
        self, global_state: int, macro_axiom_key: str
    ) -> int:
        """
        O(1) Semantic Decompression (Zoom In).

        Divides out the Macro-Prime and restores the full cluster of
        micro-primes from stored provenance.

        Args:
            global_state:   Current global Gödel integer.
            macro_axiom_key: Key of the macro-axiom to decompress.

        Returns:
            Updated global state with micro-primes restored.
        """
        macro_axiom_key = macro_axiom_key.strip().lower()

        if macro_axiom_key not in self.axiom_to_prime:
            return global_state

        macro_prime = self.axiom_to_prime[macro_axiom_key]

        if (
            global_state % macro_prime != 0
            or macro_prime not in self.macro_provenance
        ):
            return global_state

        # Factor out the macro-prime
        while global_state % macro_prime == 0:
            global_state //= macro_prime

        # Restore the micro-prime cluster
        return math.lcm(global_state, self.macro_provenance[macro_prime])

