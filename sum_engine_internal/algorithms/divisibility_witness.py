"""
Divisibility witness over a Gödel state integer.

Renamed from ``zk_semantics`` / ``ZKSemanticProver`` (2026-09): the old name
claimed a zero-knowledge proof. It is not one, and this module says plainly
what it is.

What ``generate_proof`` publishes: ``prime``, ``quotient = state // prime``,
a random ``salt`` and ``commitment = SHA-256(f"{quotient}:{salt}")``. Anyone
holding the output recovers the full state as ``int(quotient) * prime``, so
the witness REVEALS the state. It is not zero-knowledge, not hiding, and not a
Pedersen commitment.

What ``verify_proof`` checks: only that the commitment is the SHA-256 of the
published quotient and salt. It does not bind ``prime`` (a different prime
with the same quotient and salt still verifies), and it does not bind any
particular state, so it shows internal consistency of the published fields,
not that some known state entails the prime. A caller who holds the state can
check divisibility directly with ``state % prime == 0``.

Behaviour is unchanged by the rename; only names and documentation moved.

Author: ototao
License: Apache License 2.0
"""

import hashlib
import os


class DivisibilityWitness:
    """
    Publishes ``state // prime`` plus a salted SHA-256 commitment to it.

    The witness reveals the state (``quotient * prime == state``) and is not
    a zero-knowledge proof. Verification checks only the hash commitment.
    """

    @staticmethod
    def generate_proof(global_state: int, prime: int) -> dict:
        """
        Build a divisibility witness that ``prime`` divides ``global_state``.

        Args:
            global_state: The full Gödel BigInt.
            prime:        The semantic prime.

        Returns:
            A dict with ``commitment``, ``salt``, ``prime``, and ``quotient``
            (as a string for BigInt JSON safety). The state is recoverable as
            ``int(quotient) * prime``.

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
        Re-compute the hash commitment over the published quotient and salt.

        Returns True when SHA-256(quotient || salt) matches ``commitment``.
        This does NOT bind ``prime`` or any state: it checks that the
        published fields are internally consistent, nothing more.

        Args:
            proof: Dict with ``commitment``, ``salt``, ``quotient``.

        Returns:
            True if the commitment matches.
        """
        q = int(proof["quotient"])
        salt = proof["salt"]

        expected = hashlib.sha256(
            f"{q}:{salt}".encode()
        ).hexdigest()

        return expected == proof["commitment"]
