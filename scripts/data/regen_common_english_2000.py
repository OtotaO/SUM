"""Regenerate the bundled top-2000 English-words frequency table.

One-shot helper for the slider renderer's audience-axis classifier.
Runs against NLTK's Brown corpus (~1M words, 1961, balanced genres).
The output is a deterministic ordering by descending frequency.

Usage:
    python scripts/data/regen_common_english_2000.py

Re-run only when the classifier needs a wider vocabulary (Phase E.1
STATE 5b shipped 2000 words; v0.2 may move to 5000+).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import nltk

OUT = Path(__file__).resolve().parents[2] / "sum_engine_internal/ensemble/data/common_english_2000.txt"
TOP_N = 2000


def main() -> int:
    nltk.download("brown", quiet=True)
    from nltk.corpus import brown

    words = (w.lower() for w in brown.words() if w.isalpha() and len(w) > 1)
    top = [w for w, _ in Counter(words).most_common(TOP_N)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(top) + "\n", encoding="utf-8")
    print(f"wrote {len(top)} words → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
