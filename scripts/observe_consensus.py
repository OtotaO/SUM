"""
Planetary Panopticon — Live Swarm Consensus Observer

Polls all three P2P Swarm nodes every second and displays their Gödel
Integers in a live terminal matrix. Shows real-time convergence as the
Holographic Gossip Protocol merges independent mathematical brains into
a single Planetary Integer via LCM.

Usage:
    python scripts/observe_consensus.py

Watch for:
    - Integers growing as the Babel Harvester feeds each node
    - Lengths diverging as different RSS feeds produce different axioms
    - All three snapping to the EXACT same integer every ~15 seconds
      when the P2P Gossip Daemon triggers its LCM sync
"""

import asyncio
import sys
import time

import httpx

NODES = {
    "ALPHA": "http://localhost:8000",
    "BETA":  "http://localhost:8001",
    "GAMMA": "http://localhost:8002",
}

COLORS = {
    "ALPHA": "\033[92m",         # Green
    "BETA":  "\033[96m",         # Cyan
    "GAMMA": "\033[95m",         # Magenta
    "RESET": "\033[0m",
    "WARN":  "\033[93m",         # Yellow
    "SYNC":  "\033[97m\033[1m",  # Bold White
    "DIM":   "\033[90m",         # Dim grey
}


async def fetch_state(client: httpx.AsyncClient, name: str, url: str):
    """Fetch the current Gödel integer from a node."""
    try:
        res = await client.get(f"{url}/api/v1/quantum/state")
        if res.status_code == 200:
            data = res.json()
            return name, str(data.get("global_state_integer", "1"))
    except Exception:
        pass
    return name, "OFFLINE"


def format_integer(state_str: str) -> str:
    """Truncate massive integers for display: first 30 ... last 30 digits."""
    if len(state_str) > 64:
        return state_str[:30] + " ······ " + state_str[-30:]
    return state_str


async def main():
    c = COLORS
    print(f"\n{c['SYNC']}{'=' * 60}{c['RESET']}")
    print(f"{c['SYNC']}   🌐 PLANETARY PANOPTICON — LIVE CONSENSUS MATRIX 🌐{c['RESET']}")
    print(f"{c['SYNC']}{'=' * 60}{c['RESET']}\n")

    consensus_count = 0

    async with httpx.AsyncClient(timeout=2.0) as client:
        while True:
            tasks = [
                fetch_state(client, name, url)
                for name, url in NODES.items()
            ]
            results = await asyncio.gather(*tasks)

            states = {}
            timestamp = time.strftime("%H:%M:%S")

            # Clear screen for live matrix
            sys.stdout.write("\033[H\033[J")

            print(f"{c['SYNC']}{'═' * 60}{c['RESET']}")
            print(f"{c['SYNC']}  🌐 PLANETARY PANOPTICON       {c['DIM']}[{timestamp}]{c['RESET']}")
            print(f"{c['SYNC']}{'═' * 60}{c['RESET']}\n")

            for name, state_str in results:
                states[name] = state_str
                color = c[name]

                if state_str == "OFFLINE":
                    print(f"  {color}⬤ Node {name:<5}{c['RESET']}  {c['WARN']}OFFLINE{c['RESET']}")
                elif state_str == "1":
                    print(f"  {color}⬤ Node {name:<5}{c['RESET']}  {c['DIM']}Empty (State = 1){c['RESET']}")
                else:
                    digits = len(state_str)
                    display = format_integer(state_str)
                    print(f"  {color}⬤ Node {name:<5}{c['RESET']}  [{digits:>5} digits]  {c['DIM']}{display}{c['RESET']}")

            print()

            # ── Convergence Check ──
            active = [s for s in states.values() if s not in ("OFFLINE", "1")]

            if len(active) == 3 and len(set(active)) == 1:
                consensus_count += 1
                print(f"  {c['SYNC']}✅ GLOBAL MATHEMATICAL CONSENSUS ✅{c['RESET']}")
                print(f"  {c['DIM']}All 3 nodes share the exact same Gödel Integer.{c['RESET']}")
                print(f"  {c['DIM']}Consensus streak: {consensus_count}s{c['RESET']}")
            elif len(active) >= 2 and len(set(active)) > 1:
                consensus_count = 0
                print(f"  {c['WARN']}🌀 DIVERGED — Harvesting in progress, awaiting gossip sync...{c['RESET']}")
            elif len(active) == 0:
                consensus_count = 0
                all_offline = all(s == "OFFLINE" for s in states.values())
                if all_offline:
                    print(f"  {c['WARN']}⏳ All nodes offline. Run: bash scripts/launch_swarm.sh{c['RESET']}")
                else:
                    print(f"  {c['DIM']}Nodes booted. Waiting for ingestion...{c['RESET']}")

            print(f"\n  {c['DIM']}Press Ctrl+C to detach observer.{c['RESET']}")

            await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{COLORS['DIM']}Observer detached.{COLORS['RESET']}")
