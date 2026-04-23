"""SUM CLI — agentic-friendly command line for the SUM knowledge distillation engine.

Install:
    pip install sum-engine[sieve]     # deterministic offline extractor
    pip install sum-engine[llm]       # OpenAI structured-output extractor
    pip install sum-engine[all]       # both + dev tooling

Typical use:
    echo "The printing press was invented by Johannes Gutenberg." \\
        | sum attest > bundle.json
    sum verify < bundle.json          # exit 0 on match, 1 on mismatch

Subcommands live in ``sum_cli.main``. The version string is resolved
dynamically from the installed distribution metadata so it can never
drift from what `pip` actually placed on disk — a prior hardcoded
"0.1.0" survived across the 0.1.0 → 0.2.0 rename release and shipped
a wrong `sum --version` for ~3 minutes before 0.2.1 fixed it.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _dist_version


def _resolve_version() -> str:
    """Read the running package version from installed metadata.

    Returns ``"0+unknown"`` only in the edge case where ``sum_cli`` is
    imported from a source checkout without an editable install (no
    dist-info, no egg-info). That can happen in CI when the test
    harness puts the repo root on PYTHONPATH instead of `pip install
    -e`; the CLI keeps working, the version string is just honest
    about not knowing.
    """
    try:
        return _dist_version("sum-engine")
    except PackageNotFoundError:
        return "0+unknown"


__version__ = _resolve_version()
