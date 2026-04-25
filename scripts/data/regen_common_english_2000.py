"""Regenerate the bundled English-words frequency tables.

One-shot helper for the slider renderer's audience-axis classifier.
Runs against NLTK's Brown corpus (~1M words, 1961, balanced genres).
Output is a deterministic ordering by descending frequency.

Writes both `common_english_2000.txt` and `common_english_5000.txt`
so the loader can fall back to the smaller table on partial checkouts.

Usage:
    python scripts/data/regen_common_english_2000.py

Re-run when the classifier needs a wider vocabulary (E.1 STATE 5b
shipped 2000; v0.2 expanded to 5000).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import nltk

DATA_DIR = Path(__file__).resolve().parents[2] / "sum_engine_internal/ensemble/data"
TARGETS = [2000, 5000]


def main() -> int:
    nltk.download("brown", quiet=True)
    from nltk.corpus import brown

    words = [w.lower() for w in brown.words() if w.isalpha() and len(w) > 1]
    counter = Counter(words)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for n in TARGETS:
        top = [w for w, _ in counter.most_common(n)]
        out = DATA_DIR / f"common_english_{n}.txt"
        out.write_text("\n".join(top) + "\n", encoding="utf-8")
        print(f"wrote {len(top)} words → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
