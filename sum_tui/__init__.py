"""SUM Workbench — the terminal front-end for SUM's chain-of-custody-for-meaning.

A Textual TUI over the shipped `sum` / `sum_verify` CLI surface. Realizes the
locked product-vision workbench: source -> transform -> meaning-loss number ->
meaning-diff -> frontier -> signed receipt. Runs offline today (it replays the
real signed BillSum binding-gate golden); the same app serves to the browser via
`textual serve "python -m sum_tui"`, which is the on-ramp to the web front-end.

Run:  python -m sum_tui
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
