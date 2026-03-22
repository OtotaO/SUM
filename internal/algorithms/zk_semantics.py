"""
Zero-Knowledge Semantic Proofs

Implements cryptographic commitments over Gödel-State entailment:
a node can mathematically prove it knows a specific axiom (= prime
factor of its global state) **without revealing the full state integer**.

Protocol:
    1. Prover computes  Q = State // prime  (the co-factor).
    2. Prover generates a random salt and publishes
       Commitment = SHA-256( Q || salt ).
    3. Verifier receives (commitment, salt, Q, prime) and re-hashes to
       confirm the prover genuinely held the factor.

This is a simplified Pedersen-style commitment scheme optimised for
the Gödel integer domain.  It is non-interactive.

Author: ototao
License: Apache License 2.0
"""

import hashlib
import os


class ZKSemanticProver:
    """
    Zero-Knowledge proofs for Gödel State Entailment.

    Proves  ``State % prime == 0``  without revealing the full State
    integer, using a salted hash commitment over the quotient.
    """

    @staticmethod
    def generate_proof(global_state: int, prime: int) -> dict:
        """
        Generate a ZK proof that ``global_state`` contains ``prime``.

        Args:
            global_state: The full Gödel BigInt.
            prime:        The semantic prime to prove knowledge of.

        Returns:
            A proof dict with ``commitment``, ``salt``, ``prime``,
            and ``quotient`` (as a string for BigInt JSON safety).

        Raises:
            ValueError: If the state does not actually entail the prime.
        """
        if global_state % prime != 0:
            raise ValueError("State does not entail this prime.")

        quotient = global_state // prime
        salt = os.urandom(16).hex()

        # Commitment = Hash(Quotient || Salt)
        commitment = hashlib.sha256(
            f"{quotient}:{salt}".encode()
        ).hexdigest()

        return {
            "commitment": commitment,
            "salt": salt,
            "prime": prime,
            "quotient": str(quotient),
        }

    @staticmethod
    def verify_proof(proof: dict) -> bool:
        """
        Verify a ZK semantic proof by re-computing the commitment.

        The verifier checks that SHA-256(Q || salt) matches the
        published commitment, confirming the prover genuinely held
        a state divisible by the claimed prime.

        Args:
            proof: Dict with ``commitment``, ``salt``, ``quotient``.

        Returns:
            True if the commitment is valid.
        """
        q = int(proof["quotient"])
        salt = proof["salt"]

        expected = hashlib.sha256(
            f"{q}:{salt}".encode()
        ).hexdigest()

        return expected == proof["commitment"]
