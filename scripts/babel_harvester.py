"""
Babel Harvester — Zero-Cost Autonomous RSS Ingestion

Pulls live RSS feeds (ArXiv AI, ArXiv Physics, HackerNews) and converts
entries into (subject, predicate, object) triplets. Feeds them into the
Quantum Knowledge OS via the /ingest/math endpoint — zero LLM calls,
zero API costs, pure mathematical truth.

Usage:
    python scripts/babel_harvester.py
"""

import asyncio
import re

import httpx
import feedparser

# ─── Node → Feed Mapping ──────────────────────────────────────────────

NODES = {
    "alpha": ("http://localhost:8000", "http://export.arxiv.org/rss/cs.AI"),
    "beta":  ("http://localhost:8001", "http://export.arxiv.org/rss/physics"),
    "gamma": ("http://localhost:8002", "https://news.ycombinator.com/rss"),
}


def clean_text(text: str) -> str:
    """Strip HTML tags and non-alphanumeric chars, normalise to lowercase."""
    text = re.sub(r"<[^>]+>", "", text)  # Strip HTML
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip().lower()


async def harvest_feed(node_name: str, node_url: str, feed_url: str):
    """Pull RSS and POST triplets to the node's /ingest/math endpoint."""
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

    # Extract feed source name
    feed_title = clean_text(parsed.feed.get("title", node_name))[:30]

    # Map entries to (subject, predicate, object) triplets
    triplets = []
    for entry in entries:
        title = clean_text(entry.get("title", ""))[:50]
        if title:
            triplets.append([feed_title, "published", title])

        # Extract author if available
        author = clean_text(entry.get("author", ""))[:30]
        if author and title:
            triplets.append([author, "authored", title])

    if not triplets:
        print(f"[{node_name.upper()}] No triplets extracted.")
        return

    # POST to the zero-cost /ingest/math endpoint
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(
                f"{node_url}/api/v1/quantum/ingest/math",
                json={"triplets": triplets, "branch": "main"},
            )
            if resp.status_code == 200:
                data = resp.json()
                state_preview = str(data["new_global_state"])[:20]
                print(
                    f"[{node_name.upper()}] ✅ Ingested {data['axioms_added']} axioms. "
                    f"Gödel State: {state_preview}..."
                )
            else:
                print(f"[{node_name.upper()}] ❌ HTTP {resp.status_code}: {resp.text[:80]}")
        except httpx.ConnectError:
            print(f"[{node_name.upper()}] ❌ Node offline at {node_url}")
        except Exception as e:
            print(f"[{node_name.upper()}] ❌ Error: {e}")


async def main():
    print("🌍 MATH-ONLY BABEL HARVESTER INITIATED 🌍")
    print(f"   Nodes: {', '.join(NODES.keys())}")
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
