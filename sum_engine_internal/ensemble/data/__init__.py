"""Bundled data assets for the ensemble layer.

Currently ships:
    common_english_2000.txt — top 2000 words by frequency in the
        Brown corpus (NLTK), used by slider_renderer's audience-axis
        classifier. One word per line, lowercase, sorted by descending
        frequency.

Re-generate via:
    python scripts/data/regen_common_english_2000.py
"""
