"""Entry point for the SUM Workbench.

    python -m sum_tui            # run the TUI
    python -m sum_tui --smoke    # headless self-test (compose + demo action), CI-safe

Serve the same app to a browser (the on-ramp to the web front-end):

    textual serve "python -m sum_tui"
"""
from __future__ import annotations

import sys


def _smoke() -> int:
    """Headless: mount the app, fire the demo action, assert no exceptions."""
    import asyncio

    from .app import SumApp

    async def run() -> None:
        app = SumApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#source")
            assert app.query_one("#frontier")
            assert app.query_one("#nutrition")
            await pilot.press("d")        # load the signed golden
            await pilot.pause(2.0)        # let the worker subprocess finish
            await pilot.press("c")        # clear
            await pilot.pause()

    asyncio.run(run())
    print("SMOKE OK: composed, demo action ran, no exceptions.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--smoke" in argv:
        return _smoke()
    from .app import SumApp

    SumApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
