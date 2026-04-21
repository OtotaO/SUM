"""SUM CLI — agentic-friendly command line for the SUM knowledge distillation engine.

Install:
    pip install sum-engine[sieve]     # deterministic offline extractor
    pip install sum-engine[llm]       # OpenAI structured-output extractor
    pip install sum-engine[all]       # both + dev tooling

Typical use:
    echo "The printing press was invented by Johannes Gutenberg." \\
        | sum attest > bundle.json
    sum verify < bundle.json          # exit 0 on match, 1 on mismatch

Subcommands live in ``sum_cli.main``; this file exists to make the package
importable and to pin the version string in one place.
"""
from __future__ import annotations

__version__ = "0.1.0"
