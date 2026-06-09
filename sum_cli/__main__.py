"""Enable ``python -m sum_cli`` as an alias for the ``sum`` console script.

Several prospective adopters in the 30-guest adoption simulation (2026-06-09)
reached for ``python3 -m sum_cli`` and hit "package cannot be directly
executed" — the only working forms were the ``sum`` entry point or
``python3 -m sum_cli.main``. This makes the obvious on-ramp work.
"""
from sum_cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
