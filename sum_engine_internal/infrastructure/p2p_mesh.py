"""
P2P Holographic Mesh — Decentralized Epistemic Network

Implements a gossip-based protocol that connects multiple
GlobalKnowledgeOS instances into a distributed hive-mind.

Each node periodically:
    1. Contacts its known peers via the Gödel Sync Protocol.
    2. Receives a delta of missing axioms.
    3. Mints and merges the novel axioms into its local state.
    4. Detects and reports contradictions imported from the network.

Because primes are now Universal and Deterministic (SHA-256 seeded),
two nodes that independently process the same raw text will produce
identical primes, guaranteeing lock-free mathematical consensus.

Author: ototao
License: Apache License 2.0
"""

import asyncio
import math
import logging
from typing import Set, Callable

import httpx

from sum_engine_internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from sum_engine_internal.ensemble.epistemic_arbiter import kos_telemetry
from sum_engine_internal.infrastructure.scheme_registry import CURRENT_SCHEME, is_compatible

logger = logging.getLogger(__name__)


def _zig():
    try:
        from sum_engine_internal.infrastructure.zig_bridge import zig_engine
        return zig_engine
    except ImportError:
        return None


class EpistemicMeshNetwork:
    """
    Decentralized P2P Hive Mind.

    Gossips Gödel Integers across distributed nodes to achieve
    Global Consensus via the Gödel Sync Protocol.
    """

    def __init__(
        self,
        algebra: GodelStateAlgebra,
        get_local_state_fn: Callable,
        update_local_state_fn: Callable,
    ):
        self.peers: Set[str] = set()
        self.algebra = algebra
        self.get_local_state = get_local_state_fn
        self.update_local_state = update_local_state_fn
        self.is_running = False

    # ── Peer Management ──────────────────────────────────────────────

    def add_peer(self, peer_url: str):
        """Register a remote KOS node for gossip syncing."""
        self.peers.add(peer_url.rstrip("/"))
        logger.info("Peer added to Holographic Mesh: %s", peer_url)

    def remove_peer(self, peer_url: str):
        """Remove a peer from the mesh."""
        self.peers.discard(peer_url.rstrip("/"))

    # ── Sync Protocol ────────────────────────────────────────────────

    async def _sync_with_peer(self, peer_url: str, branch: str = "main"):
        """
        Execute one Gödel Sync cycle with a remote peer.

        Exchanges two integers, computes the mathematical delta,
        and absorbs any novel axioms via LCM merge.
        """
        try:
            local_state = self.get_local_state(branch)
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{peer_url}/api/v1/quantum/sync",
                    json={
                        "client_state_integer": hex(local_state),
                        "branch": branch,
                        "prime_scheme": CURRENT_SCHEME,
                    },
                )
                if resp.status_code != 200:
                    return

                data = resp.json()

                # Scheme negotiation: reject incompatible peers
                peer_scheme = data.get("prime_scheme", CURRENT_SCHEME)
                if not is_compatible(peer_scheme):
                    logger.warning(
                        "Rejecting sync with %s: incompatible scheme %s",
                        peer_url, peer_scheme,
                    )
                    return

                # Prefer hex if available, fall back to decimal
                raw_state = data.get("new_global_state_hex") or data.get("new_global_state")
                remote_state = int(raw_state, 16) if raw_state.startswith("0x") else int(raw_state)

                if local_state == remote_state:
                    return  # Perfect consensus — nothing to do

                missing_axioms = data["delta"]["add"]
                if missing_axioms:
                    await kos_telemetry.broadcast(
                        f"🌐 P2P Sync: Absorbing {len(missing_axioms)} "
                        f"novel axioms from {peer_url}"
                    )

                    new_state = local_state
                    for axiom in missing_axioms:
                        parts = axiom.split("||")
                        if len(parts) == 3:
                            p = self.algebra.get_or_mint_prime(
                                parts[0], parts[1], parts[2]
                            )
                            z = _zig()
                            r = z.bigint_lcm(new_state, p) if z else None
                            new_state = r if r is not None else math.lcm(new_state, p)

                    # Detect contradictions imported from the network
                    paradoxes = self.algebra.detect_curvature_paradoxes(
                        new_state
                    )
                    if paradoxes:
                        await kos_telemetry.broadcast(
                            f"🚨 P2P Paradox from {peer_url}: "
                            f"{len(paradoxes)} contradictions detected"
                        )

                    self.update_local_state(branch, new_state)

        except Exception as e:
            logger.debug("P2P sync with %s failed: %s", peer_url, e)

    # ── Gossip Daemon ────────────────────────────────────────────────

    async def start_gossip_daemon(self, interval: int = 15):
        """
        Background task that periodically syncs with all known peers.

        Args:
            interval: Seconds between gossip rounds.
        """
        self.is_running = True
        await kos_telemetry.broadcast(
            "🌍 Holographic P2P Mesh Initialized."
        )
        while self.is_running:
            await asyncio.sleep(interval)
            for peer in list(self.peers):
                await self._sync_with_peer(peer, branch="main")

    def stop_daemon(self):
        """Signal the gossip loop to exit."""
        self.is_running = False
