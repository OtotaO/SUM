"""
Babel Harvester — NLP-Powered Autonomous RSS Ingestion

Pulls live RSS feeds (ArXiv AI, ArXiv Physics, HackerNews) and extracts
rich semantic triplets using the Deterministic Syntactic Sieve (spaCy).
Feeds them into the Quantum Knowledge OS via the /ingest/math endpoint.

Zero LLM calls. Zero API costs. 10,000+ words per second.

Phase 13: Zenith of Process Intensification.

Usage:
    python scripts/babel_harvester.py

Author: ototao
License: Apache License 2.0
"""

import asyncio
import re
import sys
import os

import httpx
import feedparser
import jwt
from datetime import datetime, timedelta

# Allow importing from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from internal.algorithms.syntactic_sieve import DeterministicSieve

# ─── JWT Service Token ────────────────────────────────────────────────
# Generate a long-lived harvester service token offline using the same
# secret as the API.  This gives the harvester its own isolated branch.

SECRET_KEY = os.getenv("SUM_JWT_SECRET", "quantum_supremacy_secret_key_minimum_32b")
harvester_token = jwt.encode(
    {"sub": "babel_harvester", "exp": datetime.utcnow() + timedelta(days=365)},
    SECRET_KEY,
    algorithm="HS256",
)
HEADERS = {"Authorization": f"Bearer {harvester_token}"}

# ─── Node → Feed Mapping ──────────────────────────────────────────────

NODES = {
    "alpha": ("http://localhost:8000", "http://export.arxiv.org/rss/cs.AI"),
    "beta":  ("http://localhost:8001", "http://export.arxiv.org/rss/physics"),
    "gamma": ("http://localhost:8002", "https://news.ycombinator.com/rss"),
}

# ─── Deterministic Sieve (NLP) ────────────────────────────────────────

sieve = DeterministicSieve()


def clean_text(text: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    return re.sub(r"<[^>]+>", " ", text).strip()


async def harvest_feed(node_name: str, node_url: str, feed_url: str):
    """Pull RSS, extract NLP triplets, POST to the node's /ingest/math endpoint."""
    print(f"[{node_name.upper()}] Harvester reading {feed_url}...")

    try:
        parsed = feedparser.parse(feed_url)
    except Exception as e:
        print(f"[{node_name.upper()}] Feed parse error: {e}")
        return

    entries = parsed.entries[:5]  # Top 5 to keep it lightweight
    if not entries:
        print(f"[{node_name.upper()}] No entries found in feed.")
        return

    # NLP extraction at 10,000 words/second
    triplets = []
    for entry in entries:
        raw_text = f"{entry.title}. {entry.get('summary', '')}"
        extracted = sieve.extract_triplets(clean_text(raw_text))
        triplets.extend([list(t) for t in extracted])

    if not triplets:
        print(f"[{node_name.upper()}] No triplets extracted via NLP.")
        return

    # POST to the zero-cost /ingest/math endpoint with JWT auth
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                f"{node_url}/api/v1/quantum/ingest/math",
                json={"triplets": triplets},
                headers=HEADERS,
            )
            if resp.status_code == 200:
                data = resp.json()
                state_preview = str(data.get("new_global_state", ""))[:15]
                print(
                    f"[{node_name.upper()}] ✅ NLP Ingested "
                    f"{data.get('axioms_added', 0)} axioms. "
                    f"State: {state_preview}..."
                )
            else:
                print(
                    f"[{node_name.upper()}] ❌ HTTP {resp.status_code}: "
                    f"{resp.text[:80]}"
                )
        except httpx.ConnectError:
            print(f"[{node_name.upper()}] ❌ Node offline at {node_url}")
        except Exception as e:
            print(f"[{node_name.upper()}] ❌ Error: {e}")


async def main():
    print("🌍 SYNTACTIC SIEVE HARVESTER INITIATED 🌍")
    print(f"   Nodes: {', '.join(NODES.keys())}")
    print(f"   NLP Engine: spaCy en_core_web_sm")
    print(f"   Cost: $0 (zero LLM calls)")
    print()

    cycle = 0
    while True:
        cycle += 1
        print(f"─── Harvest Cycle {cycle} ───")
        tasks = [
            harvest_feed(name, url, feed)
            for name, (url, feed) in NODES.items()
        ]
        await asyncio.gather(*tasks)
        print("💤 Sleeping 60 seconds...\n")
        await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
