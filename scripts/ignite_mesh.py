"""
Mesh Igniter — Cross-links P2P Swarm Nodes

Sends POST /peers requests to each node, creating a fully-connected
mesh topology so the Holographic Gossip Protocol begins its 15-second
background sync of Gödel Integers.

Usage:
    python scripts/ignite_mesh.py
"""

import asyncio

import httpx

NODES = [
    "http://localhost:8000",
    "http://localhost:8001",
    "http://localhost:8002",
]


async def link_nodes():
    print("🔗 CROSS-LINKING HOLOGRAPHIC MESH 🔗")
    print()

    async with httpx.AsyncClient(timeout=5.0) as client:
        for host in NODES:
            peers = [n for n in NODES if n != host]
            for peer in peers:
                try:
                    res = await client.post(
                        f"{host}/api/v1/quantum/peers",
                        json={"peer_url": peer},
                    )
                    if res.status_code == 200:
                        print(f"  ✅ {host} → {peer}")
                    else:
                        print(f"  ❌ {host} → {peer}: HTTP {res.status_code}")
                except httpx.ConnectError:
                    print(f"  ❌ {host} offline — is the swarm running?")
                except Exception as e:
                    print(f"  ❌ {host} → {peer}: {e}")

    print()
    print("🌐 Mesh ignited. Gossip daemon will sync every 15 seconds.")
    print("   Run `python scripts/babel_harvester.py` to begin autonomous ingestion.")


if __name__ == "__main__":
    asyncio.run(link_nodes())
