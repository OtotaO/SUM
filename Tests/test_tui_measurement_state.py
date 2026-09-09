"""The prototype must not attach a historical bound to edited or cleared text."""
import asyncio

import pytest

pytest.importorskip("textual")
from textual.widgets import Static, TextArea

from sum_tui.app import SumApp


def test_demo_measurement_is_invalidated_by_source_edits_and_late_responses():
    async def run():
        app = SumApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert str(app.query_one("#lossnum", Static).render()) == "Not measured"
            generation = app._demo_generation
            data = {"verified": True, "replayed": True, "risk_upper_bound": 0.645438, "n": 64}
            app._apply_demo(data, generation)
            await pilot.pause()
            assert "0.6454" in str(app.query_one("#lossnum", Static).render())
            app.query_one("#source", TextArea).text = "My own text"
            await pilot.pause()
            assert str(app.query_one("#lossnum", Static).render()) == "Not measured"
            app._apply_demo(data, generation)
            assert app.query_one("#source", TextArea).text == "My own text"
            generation = app._demo_generation
            app.action_clear()
            app._apply_demo(data, generation)
            assert app.query_one("#source", TextArea).text == ""
            assert str(app.query_one("#lossnum", Static).render()) == "Not measured"
    asyncio.run(run())
