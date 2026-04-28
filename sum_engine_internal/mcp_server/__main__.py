"""Entry point for ``python -m sum_engine_internal.mcp_server``.

Equivalent to running the ``sum-mcp`` console script. Stays
useful even without ``pip install`` (e.g. running from a source
checkout) because module-mode invocation does not require the
[project.scripts] entry to be installed on PATH.
"""
from sum_engine_internal.mcp_server.server import main

if __name__ == "__main__":
    raise SystemExit(main())
