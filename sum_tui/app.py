"""The SUM Workbench Textual app.

Layout (the locked product-vision workbench):

    ① SOURCE        ② TRANSFORM        ④ MEANING-DIFF
    drop / paste    compress slider    kept / dropped / unsupported
                    formality slider   ⑤ FRONTIER
                    simplify slider     rung · compress · loss · Δloss/Δcompress
                    ③ MEANING LOSS
                    big number-box
    ─────────────────────────────────────────────────────────────────
    EPISTEMIC NUTRITION LABEL  (the honesty discipline, surfaced in-UI)
"""
from __future__ import annotations

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    DataTable,
    Digits,
    Footer,
    Header,
    Static,
    TextArea,
)

from . import engine

ACCENT = "#3fd0c9"
GREEN = "#46c46a"
RED = "#e5484d"
AMBER = "#f0a93b"
DIM = "#6b7785"

# Illustrative content for the diff/frontier panels so the workbench reads as a
# complete product offline. Clearly labelled; live numbers come from `sum
# meaning-diff` / `sum frontier` once a judge + a transform are wired.
ILLUSTRATIVE_DIFF = {
    "title": "illustrative — live numbers need sum[research,judge] + a transform (press r)",
    "kept": [
        "The lease begins on March 1 and runs for twelve months.",
        "Rent is 1800 dollars, due on the first of each month.",
    ],
    "dropped": [
        "A late fee of 75 dollars applies after the fifth day.",
        "Either party may terminate with 60 days notice.",
    ],
    "added": [
        "Pets are not allowed under any circumstances.",
    ],
}
ILLUSTRATIVE_FRONTIER = [
    ("full", "0%", "0.000", "  —"),
    ("detailed", "38%", "0.094", "+0.247"),
    ("brief", "61%", "0.231", "+0.598"),
    ("headline", "86%", "0.604", "+1.490"),
]


class Slider(Widget):
    """A focusable one-line slider: ◂/▸ adjust, rendered as a labelled bar."""

    can_focus = True
    value = reactive(0.5)
    BINDINGS = [
        Binding("left", "decr", "−", show=False),
        Binding("right", "incr", "+", show=False),
    ]

    def __init__(self, label: str = "", value: float = 0.5, step: float = 0.05, **kw) -> None:
        super().__init__(**kw)
        self._label = label
        self.step = step
        self.set_reactive(Slider.value, value)

    def watch_value(self) -> None:
        self.refresh()

    def action_incr(self) -> None:
        self.value = min(1.0, round(self.value + self.step, 3))

    def action_decr(self) -> None:
        self.value = max(0.0, round(self.value - self.step, 3))

    def render(self) -> Text:
        width = 16
        filled = max(0, min(width, int(round(self.value * width))))
        bar = "━" * filled + "●" + "─" * (width - filled)
        knob = "▸" if self.has_focus else " "
        return Text.assemble(
            (f"{knob} {self._label:<10}", "bold"),
            (bar, ACCENT),
            (f" {int(self.value * 100):>3d}%", "dim"),
        )


class SumApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "SUM · Workbench"
    # Framing-neutral on purpose: states what the receipt IS (a signed,
    # replayable bound on a named proxy), not which market story sells it.
    SUB_TITLE = "signed, replayable bounds on a named meaning-loss proxy"
    BINDINGS = [
        Binding("d", "load_demo", "Load demo"),
        Binding("r", "run", "Run transform"),
        Binding("c", "clear", "Clear"),
        Binding("question_mark", "help", "Help"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="top"):
            with Vertical(id="col-input", classes="col"):
                yield Static("① SOURCE", classes="panel-title")
                yield TextArea(id="source")
                with Horizontal(classes="btnrow"):
                    yield Button("Load demo  [d]", id="demo", variant="primary")
                    yield Button("Clear  [c]", id="clear")
            with Vertical(id="col-transform", classes="col"):
                yield Static("② TRANSFORM", classes="panel-title")
                yield Slider("compress", value=0.60, id="s-compress")
                yield Slider("formality", value=0.40, id="s-formality")
                yield Slider("simplify", value=0.50, id="s-simplify")
                yield Button("Run ▶  [r]", id="run", variant="success")
                yield Static("③ MEANING LOSS", classes="panel-title")
                with Vertical(id="lossnum-wrap"):
                    yield Digits("0.000", id="lossnum")
                    yield Static("risk upper bound", id="loss-cap", classes="dim")
            with Vertical(id="col-results", classes="col"):
                yield Static("④ MEANING-DIFF", classes="panel-title")
                with VerticalScroll(id="diff"):
                    yield Static(id="diff-body")
                yield Static("⑤ FRONTIER", classes="panel-title")
                yield DataTable(id="frontier")
        yield Static(self._nutrition_idle(), id="nutrition")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#frontier", DataTable)
        table.add_columns("rung", "compress", "loss", "Δloss/Δcompress")
        table.cursor_type = "row"
        self._render_diff(ILLUSTRATIVE_DIFF)
        self._render_frontier(ILLUSTRATIVE_FRONTIER)
        self.query_one("#source", TextArea).text = (
            "Paste a document here — or press  d  to replay the signed "
            "BillSum binding-gate golden, verified offline."
        )
        # Focus an actionable widget (not the TextArea) so single-letter shortcuts
        # fire from launch; they yield to text entry only while editing the source.
        self.query_one("#demo", Button).focus()

    # ── actions ────────────────────────────────────────────────────────────
    def on_button_pressed(self, event: Button.Pressed) -> None:
        action = {"demo": self.action_load_demo, "clear": self.action_clear, "run": self.action_run}
        handler = action.get(event.button.id or "")
        if handler:
            handler()

    def action_load_demo(self) -> None:
        self.query_one("#nutrition", Static).update(
            Text("⟳ replaying the signed golden offline…", style=ACCENT)
        )
        self._load_demo_worker()

    @work(thread=True)
    def _load_demo_worker(self) -> None:
        data = engine.run_demo()
        self.call_from_thread(self._apply_demo, data)

    def _apply_demo(self, data: dict) -> None:
        if not data.get("verified"):
            self.query_one("#nutrition", Static).update(
                Text(f"✗ demo failed: {data.get('error', 'unknown')}", style=RED)
            )
            return
        bound = float(data.get("risk_upper_bound", 0.0) or 0.0)
        self.query_one("#lossnum", Digits).update(f"{bound:.4f}")
        self.query_one("#loss-cap", Static).update(
            f"risk upper bound @95% · n={data.get('n', '?')}"
        )
        self.query_one("#source", TextArea).text = (
            "DEMO · BillSum binding-gate golden (CC0).\n\n"
            "The receipt below was replayed and its Ed25519 signature verified "
            "OFFLINE. 'verified' and 'replayed' are cryptographic facts — not, by "
            "themselves, evidence that meaning was preserved (see the label below)."
        )
        self.query_one("#nutrition", Static).update(self._nutrition_for(data))

    def action_run(self) -> None:
        self.query_one("#nutrition", Static).update(self._nutrition_run_hint())

    def action_clear(self) -> None:
        self.query_one("#source", TextArea).text = ""
        self.query_one("#lossnum", Digits).update("0.000")
        self.query_one("#loss-cap", Static).update("risk upper bound")
        self.query_one("#nutrition", Static).update(self._nutrition_idle())

    def action_help(self) -> None:
        self.query_one("#nutrition", Static).update(self._help_text())

    # ── renderers ──────────────────────────────────────────────────────────
    def _render_diff(self, diff: dict) -> None:
        text = Text()
        text.append(diff.get("title", "") + "\n\n", style=f"italic {DIM}")
        for s in diff.get("kept", []):
            text.append("  ✓ kept         ", style=f"bold {GREEN}")
            text.append(s + "\n", style=GREEN)
        for s in diff.get("dropped", []):
            text.append("  ✗ dropped      ", style=f"bold {RED}")
            text.append(s + "\n", style=RED)
        for s in diff.get("added", []):
            text.append("  ! unsupported  ", style=f"bold {AMBER}")
            text.append(s + "\n", style=AMBER)
        self.query_one("#diff-body", Static).update(text)

    def _render_frontier(self, rows: list[tuple[str, str, str, str]]) -> None:
        table = self.query_one("#frontier", DataTable)
        table.clear()
        for row in rows:
            table.add_row(*row)

    # ── the Epistemic Nutrition Label (honesty discipline, in-UI) ───────────
    def _nutrition_idle(self) -> Text:
        return Text.assemble(
            ("EPISTEMIC NUTRITION LABEL   ", f"bold {ACCENT}"),
            ("press ", DIM), ("d", "bold #d7dde4"),
            (" to replay a real signed receipt offline · ", DIM),
            ("?", "bold #d7dde4"), (" for help", DIM),
        )

    def _nutrition_for(self, data: dict) -> Text:
        not_covered = ", ".join(data.get("not_covered", [])) or "—"
        text = Text()
        text.append("EPISTEMIC NUTRITION LABEL\n", style=f"bold {ACCENT}")
        text.append("  signed ", style=f"bold {GREEN}")
        text.append("✓   ", style=GREEN)
        text.append("replayed ", style=f"bold {GREEN}")
        text.append("✓   ", style=GREEN)
        text.append(f"schema {data.get('schema', '?')}\n", style="#9aa6b2")
        text.append(f"  proxy: {data.get('scorer', '?')}\n", style="#9aa6b2")
        text.append(f"  blind to: {not_covered}\n", style="#9aa6b2")
        text.append("  ⚠ " + str(data.get("proxy_caveat", ""))[:260], style=AMBER)
        return text

    def _nutrition_run_hint(self) -> Text:
        text = Text()
        text.append("LIVE TRANSFORM\n", style=f"bold {AMBER}")
        text.append(
            "  v0.1 wires the offline signed demo (press d). A live transform mints a\n"
            "  sum.meaning_risk_receipt over YOUR text — needs  pip install 'sum-engine[research]'\n"
            "  + a judge ([judge]) + an LLM key (or local Ollama). The sliders above map to the\n"
            "  perspective/compression axes; the receipt is the door this front-end opens.",
            style="#9aa6b2",
        )
        return text

    def _help_text(self) -> Text:
        text = Text()
        text.append("SUM WORKBENCH — keys\n", style=f"bold {ACCENT}")
        rows = [
            ("d", "replay the signed BillSum golden offline (real verify + replay)"),
            ("r", "run a transform (live: needs sum[research,judge] + an LLM key)"),
            ("◂ ▸", "adjust a focused slider   ·   tab cycles panels"),
            ("c", "clear     ·     q  quit"),
            ("web", "textual serve \"python -m sum_tui\"  →  the same UI in a browser"),
        ]
        for key, desc in rows:
            text.append(f"  {key:<5}", style="bold #d7dde4")
            text.append(desc + "\n", style="#9aa6b2")
        return text
