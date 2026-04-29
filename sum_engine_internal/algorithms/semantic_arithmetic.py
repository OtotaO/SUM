import sys
sys.set_int_max_str_digits(0)

"""
Semantic Prime Number Theorem (SPNT) & Gödel-State Algebra

Maps irreducible semantic axioms to Prime Numbers. Document states become
massive integers. Merging parallel extractions is an LCM operation, and
entailment checking is Modulo arithmetic.

Mathematical foundations:
    - SPNT Bound: π_sem(N) ~ N / ln(N)
    - Gödel Encoding: state = lcm(prime_1, prime_2, ...) for each distinct
                       axiom. Primes are pairwise coprime so this equals
                       the product for a set, but is idempotent under
                       duplicates (a *set* of axioms, not a multiset).
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

from sum_engine_internal.infrastructure.scheme_registry import CURRENT_SCHEME

logger = logging.getLogger(__name__)


class PrimeCollisionError(Exception):
    """Hard failure when two distinct axiom keys hash to the same prime.

    In sha256_64_v1, collisions are resolved by advancing to the next
    prime (order-dependent, consensus-unsafe for distributed systems).

    In sha256_128_v2, collisions are fatal.  The 128-bit seed space
    makes birthday-bound collisions require ~2^64 distinct axioms.
    If one ever occurs, something is deeply wrong.
    """


def _get_zig_engine():
    """Lazy import of the Zig FFI bridge singleton."""
    try:
        from sum_engine_internal.infrastructure.zig_bridge import zig_engine
        return zig_engine
    except ImportError:
        return None


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
    # Predicate canonicalization
    # ------------------------------------------------------------------

    @staticmethod
    def _canonicalize_predicate(predicate: str) -> str:
        """Normalize a free-form predicate to the canonical vocabulary."""
        from sum_engine_internal.algorithms.predicate_canon import canonicalize
        return canonicalize(predicate)

    # ------------------------------------------------------------------
    # Prime minting
    # ------------------------------------------------------------------

    def _deterministic_prime(self, axiom_key: str) -> int:
        """
        Generates a globally unique, deterministic prime for a given axiom.

        Dispatches based on the active scheme:
            sha256_64_v1:  SHA-256 → first 8 bytes → nextprime(seed)
            sha256_128_v2: SHA-256 → first 16 bytes → nextprime(seed)

        HORIZON III — Strangler Fig:
            Routes to the bare-metal Zig core (nanosecond-speed C-ABI)
            when ``libsum_core`` is compiled and available.  Falls back
            to Python ``sympy.nextprime`` seamlessly otherwise.

        Args:
            axiom_key: The normalised axiom string.

        Returns:
            A deterministic prime number unique to this axiom.
        """
        if CURRENT_SCHEME == "sha256_128_v2":
            return self._deterministic_prime_v2(axiom_key)
        return self._deterministic_prime_v1(axiom_key)

    def _deterministic_prime_v1(self, axiom_key: str) -> int:
        """v1: SHA-256 → first 8 bytes big-endian → nextprime(seed)."""
        # ── Bare-Metal Fast Path (Zig C-ABI) ──
        zig = _get_zig_engine()
        if zig is not None:
            result = zig.get_deterministic_prime(axiom_key)
            if result is not None:
                return result

        # ── Python Fallback ──
        h = hashlib.sha256(axiom_key.encode('utf-8')).digest()
        seed = int.from_bytes(h[:8], byteorder='big')
        return sympy.nextprime(seed)

    def _deterministic_prime_v2(self, axiom_key: str) -> int:
        """v2: SHA-256 → first 16 bytes big-endian → nextprime(seed).

        Uses sympy.nextprime which internally uses BPSW for large inputs.
        """
        # ── Bare-Metal Fast Path (Zig C-ABI v2) ──
        zig = _get_zig_engine()
        if zig is not None and hasattr(zig, 'get_deterministic_prime_v2'):
            result = zig.get_deterministic_prime_v2(axiom_key)
            if result is not None:
                return result

        # ── Python path ──
        h = hashlib.sha256(axiom_key.encode('utf-8')).digest()
        seed = int.from_bytes(h[:16], byteorder='big')
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

        Raises:
            ValueError: if any component is empty/whitespace-only OR
                contains the ``||`` axiom-key separator. Both cases
                would corrupt the round-trip via the canonical tome:
                empty components produce ``The  pred obj.`` lines that
                fail the verifier's ``^The (\\S+) (\\S+) (.+)\\.$`` regex,
                and ``||`` in a component splits ambiguously when the
                tome generator parses ``axiom_key.split("||")``. Both
                arise from real-world noise (markdown table cells whose
                cell contents look like sentences) and must be rejected
                at the algebra boundary so verification stays
                round-trippable.
        """
        # Defensive validation. Cheap; runs on every prime mint.
        # Three conditions to reject:
        #   1. Empty / whitespace-only component (would emit
        #      ``The  pred obj.`` which fails the verifier regex).
        #   2. Pipe character in any component. Even a lone ``|`` is
        #      unsafe: subject=``'|'`` produces axiom_key
        #      ``'|||close||this'`` which splits to
        #      ``['', '|close', 'this']`` — round-trip drift. Stripping
        #      pipes wholesale is conservative but safe; legitimate
        #      semantic content rarely contains pipes (they're markdown
        #      table syntax or programming punctuation).
        s_stripped = subject.strip()
        p_canon = self._canonicalize_predicate(predicate)
        o_stripped = object_.strip()
        if not s_stripped or not p_canon.strip() or not o_stripped:
            raise ValueError(
                f"axiom has empty/whitespace component: "
                f"subject={subject!r} predicate={predicate!r} object={object_!r}"
            )
        if "|" in s_stripped or "|" in p_canon or "|" in o_stripped:
            raise ValueError(
                f"axiom component contains '|' (would corrupt axiom_key "
                f"round-trip via the '||' separator): "
                f"subject={subject!r} predicate={predicate!r} object={object_!r}"
            )

        axiom_key = (
            f"{s_stripped.lower()}||"
            f"{p_canon}||"
            f"{o_stripped.lower()}"
        )

        # Round-trip self-check: the freshly-built axiom_key MUST
        # split into exactly the components we put in. If not, our
        # validation above missed something — fail loud rather than
        # silently corrupt the prime table.
        round_trip = axiom_key.split("||")
        if (
            len(round_trip) != 3
            or round_trip[0] != s_stripped.lower()
            or round_trip[1] != p_canon
            or round_trip[2] != o_stripped.lower()
        ):
            raise ValueError(
                f"axiom_key round-trip failed: built {axiom_key!r}, "
                f"split returned {round_trip!r}"
            )

        if axiom_key not in self.axiom_to_prime:
            p = self._deterministic_prime(axiom_key)

            # Collision handling is scheme-dependent
            if p in self.prime_to_axiom and self.prime_to_axiom[p] != axiom_key:
                if CURRENT_SCHEME == "sha256_128_v2":
                    # v2: hard fail — collision loop is hidden nondeterminism
                    raise PrimeCollisionError(
                        f"128-bit prime collision detected: prime {p} already "
                        f"assigned to {self.prime_to_axiom[p]!r}, cannot assign "
                        f"to {axiom_key!r}. This should be astronomically rare. "
                        f"Investigate immediately."
                    )
                else:
                    # v1: advance to next prime (order-dependent, legacy behavior)
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
    # State Factorisation
    # ------------------------------------------------------------------

    def get_active_axioms(self, state: int) -> list:
        """
        Factorise a Gödel state to extract all active axiom keys.

        Iterates over all known primes and checks divisibility.
        Returns the list of axiom key strings (e.g. "alice||likes||cats")
        whose primes divide the state.

        Args:
            state: The Gödel integer to factorise.

        Returns:
            List of axiom key strings.
        """
        if state <= 1:
            return []
        active = []
        for prime, axiom in self.prime_to_axiom.items():
            if state % prime == 0:
                active.append(axiom)
        return active

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
        Convert a list of axioms into a single Gödel State Integer via
        iterative LCM over each axiom's derived prime.

        Idempotent under duplicates: encoding the same triple twice
        produces the same state as encoding it once, because
        ``math.lcm(prime, prime) == prime``. This matches the semantic
        model of a *set* of axioms and aligns with every other merge
        operation in the codebase (``math.lcm`` on lines 287, 331, 391,
        399, 592, 617, 706, 742).

        Historical note (pre-M1c cross-runtime harness): this function
        used raw multiplicative accumulation (``state *= prime``), which
        produced ``prime^n`` for duplicated triples. All production
        callers pre-deduplicated their input (the sieve returns
        ``list(set(triples))``), so the difference was invisible in
        running code. The JS port at ``single_file_demo/godel.js`` used
        LCM from day one, and the cross-runtime byte-identity harness
        at ``scripts/verify_godel_cross_runtime.py`` caught the drift
        on the "repeated triple" fixture. The fix aligns the two
        implementations and removes a latent source of non-determinism
        (re-encoding a non-deduplicated list no longer perturbs the
        state integer).

        For input with only distinct triples (the sieve's output shape),
        ``math.lcm(*primes) == product(primes)`` because the primes are
        pairwise coprime, so every existing test observes identical
        state integers before and after this change.

        Args:
            axioms: List of (subject, predicate, object) triples.

        Returns:
            A single integer encoding the full semantic state.
        """
        state = 1
        for subj, pred, obj in axioms:
            try:
                prime = self.get_or_mint_prime(subj, pred, obj)
            except ValueError:
                # Malformed component (empty or contains '||'). Drop
                # silently — we want the bag-of-axioms encoding to
                # survive single-triple noise without a hard crash.
                # The triple is removed from the encoded state but
                # remains in any caller-visible triples list, which is
                # why ``state_for_corpus`` re-filters its returned bag
                # below to match what was actually encoded.
                logger.debug(
                    "encode_chunk_state: skipping malformed axiom: "
                    "subject=%r predicate=%r object=%r",
                    subj, pred, obj,
                )
                continue
            state = math.lcm(state, prime)
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

        # ── Bare-Metal Fast Path (Zig BigInt LCM) ──
        zig = _get_zig_engine()
        if zig is not None:
            result = state_integers[0]
            for s in state_integers[1:]:
                r = zig.bigint_lcm(result, s)
                if r is None:
                    break
                result = r
            else:
                return result

        # ── Legacy Python Fallback ──
        return math.lcm(*state_integers)

    # ------------------------------------------------------------------
    # Chunked-composition (arbitrary-size handling)
    # ------------------------------------------------------------------

    def compose_chunk_states(self, chunk_states: List[int]) -> int:
        """
        Compose per-chunk states into a single corpus-level state.

        This is the chunking primitive: given the Gödel state integers
        produced from N independently-extracted text chunks, return the
        single state integer equivalent to encoding the concatenated
        corpus end-to-end.

        **Algebra invariant** (Item 1 of the arbitrary-size roadmap):
        for any *context-local* extractor f (one that emits the same
        triples for a sentence regardless of surrounding sentences —
        e.g. ``DeterministicSieve``, which walks each ``doc.sents``
        independently),

            state(f(c1) ++ f(c2) ++ ... ++ f(cn))
                == compose_chunk_states([state(f(c1)), ..., state(f(cn))])

        because (a) ``encode_chunk_state`` is LCM-based and idempotent
        under duplicates, and (b) LCM is commutative and associative.
        Asserted as a property test in
        ``Tests/test_chunked_state_composition.py``.

        **Boundary**: this guarantee does NOT hold for cross-chunk
        coreference-aware extractors (e.g. an LLM that resolves "she"
        in chunk 2 against "Marie Curie" introduced in chunk 1). For
        those extractors, chunked encoding is an approximation, not
        an equivalence — see ``docs/PROOF_BOUNDARY.md`` §1.4.

        Implementation note: this is a thin alias of
        ``merge_parallel_states`` chosen to give the chunking pipeline
        a name that matches its role at the call site. Both paths
        prefer the Zig BigInt fast path and fall back to ``math.lcm``.

        Args:
            chunk_states: One state integer per chunk, in any order.

        Returns:
            The single corpus-level state integer (== LCM of inputs).
        """
        return self.merge_parallel_states(chunk_states)

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

        # ── Bare-Metal Fast Path (Zig) ──
        zig = _get_zig_engine()
        if zig is not None:
            if hypothesis_state.bit_length() <= 64:
                # Single prime — optimized streaming mod (no BigInt needed)
                result = zig.is_divisible_by(global_state, hypothesis_state)
                if result is not None:
                    return result
            else:
                # Composite hypothesis — full BigInt modulo
                result = zig.bigint_mod(global_state, hypothesis_state)
                if result is not None:
                    return result == 0

        # ── Legacy Python Fallback ──
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

        # ── Bare-Metal Fast Path (Zig BigInt GCD) ──
        zig = _get_zig_engine()
        if zig is not None:
            zig_gcd = zig.bigint_gcd(global_state, generated_state)
            if zig_gcd is not None:
                shared_truth = zig_gcd
            else:
                shared_truth = math.gcd(global_state, generated_state)
        else:
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

        # ── Bare-Metal Fast Path (Zig BigInt GCD) ──
        zig = _get_zig_engine()
        if zig is not None:
            zig_gcd = zig.bigint_gcd(global_state, client_state)
            if zig_gcd is not None:
                shared_truth = zig_gcd
            else:
                shared_truth = math.gcd(global_state, client_state)
        else:
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


# ======================================================================
# Phase 19D: Active Prime Set Index
# ======================================================================

class ActivePrimeIndex:
    """O(1) active-prime lookup per branch.

    Maintains a ``Set[int]`` of primes known to divide each branch's
    state integer.  This eliminates the O(n) scan over
    ``prime_to_axiom`` that dominated profiling at scale:

        get_active_axioms     : 700 ms → O(k)  where k = active primes
        calculate_network_delta: 3.7 s → O(k)

    Lifecycle:
        rebuild()  — cold-start from a state integer (boot / merge)
        add()      — O(1) on ingest
        remove()   — O(1) on delete
        fork()     — O(k) on branch creation
    """

    def __init__(self):
        self._branches: Dict[str, Set[int]] = {}

    def rebuild(
        self, branch: str, state: int, algebra: GodelStateAlgebra
    ) -> None:
        """Cold-start: scan once to populate the index from a state integer."""
        if state <= 1:
            self._branches[branch] = set()
            return
        self._branches[branch] = {
            p for p in algebra.prime_to_axiom if state % p == 0
        }

    def add(self, branch: str, prime: int) -> None:
        """Register a newly-minted prime as active in a branch."""
        self._branches.setdefault(branch, set()).add(prime)

    def remove(self, branch: str, prime: int) -> None:
        """Remove a prime from a branch (deletion / DIV)."""
        if branch in self._branches:
            self._branches[branch].discard(prime)

    def fork(self, source: str, target: str) -> None:
        """Copy the active set from one branch to another."""
        self._branches[target] = set(self._branches.get(source, set()))

    def merge(self, target: str, *sources: str) -> None:
        """Union active sets from multiple branches into target."""
        merged: Set[int] = set()
        for src in sources:
            merged |= self._branches.get(src, set())
        self._branches[target] = merged

    def get_active_primes(self, branch: str) -> Set[int]:
        """Return the set of active primes for a branch."""
        return self._branches.get(branch, set())

    def get_active_axioms(
        self, branch: str, algebra: GodelStateAlgebra
    ) -> list:
        """Return axiom key strings for all active primes in a branch."""
        return [
            algebra.prime_to_axiom[p]
            for p in self._branches.get(branch, set())
            if p in algebra.prime_to_axiom
        ]

    def assert_coherent(
        self,
        branch: str,
        state: int,
        algebra: "GodelStateAlgebra",
        context: str = "",
    ) -> None:
        """Debug assertion: verify index matches brute-force scan."""
        import os
        if not os.getenv("SUM_DEBUG_INDEX"):
            return

        indexed = sorted(self.get_active_axioms(branch, algebra))
        brute = sorted(algebra.get_active_axioms(state))
        if indexed != brute:
            indexed_set = set(indexed)
            brute_set = set(brute)
            missing = brute_set - indexed_set
            extra = indexed_set - brute_set
            raise AssertionError(
                f"INDEX COHERENCE VIOLATION [{context}] "
                f"branch={branch}: "
                f"index has {len(indexed)} axioms, "
                f"brute-force has {len(brute)} axioms. "
                f"Missing from index: {missing}. "
                f"Extra in index: {extra}."
            )

    def extract_axioms_from_product(
        self, product: int, algebra: GodelStateAlgebra,
        candidate_primes: Set[int] = None,
    ) -> List[str]:
        """Extract axiom keys from a quotient product, optionally narrowed.

        When ``candidate_primes`` is provided, only those primes are
        checked — O(k) instead of O(n).  Falls back to full scan if
        no candidates are given.
        """
        axioms: List[str] = []
        primes_to_check = candidate_primes or set(algebra.prime_to_axiom.keys())
        for prime in primes_to_check:
            if product % prime == 0:
                if prime in algebra.prime_to_axiom:
                    axioms.append(algebra.prime_to_axiom[prime])
                while product % prime == 0:
                    product //= prime
            if product == 1:
                break
        return axioms
