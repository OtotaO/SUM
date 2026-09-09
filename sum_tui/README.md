# SUM terminal receipt-demo prototype

This checkout-only Textual app replays and verifies the bundled historical BillSum receipt offline. Its sliders, meaning-diff examples, and candidate table are illustrative. The app does not run a transform or mint a receipt for pasted text. Use the browser workbench for source review and export, or the CLI for research judging.

Install the checkout with verification support and Textual:

```bash
pip install -e '.[verify]'
pip install textual
python -m sum_tui
python -m sum_tui --smoke
```

Keys: `d` replays the signed demo, `r` shows CLI instructions, `c` clears, `?` opens help, and `q` quits. Arrow keys adjust an illustrative slider; Tab changes focus. Unmeasured input says **Not measured**. Editing or clearing the source invalidates a prior demo result and pending demo responses.

The historical bound describes its named BillSum proxy and corpus. It is not a score for the user's input, a factual-truth verdict, or a new model measurement. Verification and arithmetic replay do not re-run the historical judge.

`sum_tui` is excluded from the published wheel. The adapter invokes the lightweight verifier with the current Python interpreter; Textual is a separate developer dependency.
