# SUM Workbench (`sum_tui`)

The terminal front-end for SUM — a [Textual](https://textual.textualize.io) TUI
that realizes the locked product-vision workbench:

```
① SOURCE  →  ② TRANSFORM  →  ③ MEANING LOSS  →  ④ MEANING-DIFF  →  ⑤ FRONTIER  →  signed receipt
```

It is the *door* the substrate has been missing: a one-keystroke way to mint and
verify a meaning receipt, instead of running an orchestration script. Today it
runs **fully offline** by replaying the real signed BillSum binding-gate golden
(`verified` + `replayed` are cryptographic facts, no network, no model). The
honesty discipline is surfaced in-UI as an **Epistemic Nutrition Label** (the
proxy caveat + the proxy's blind spots travel with every number).

## Run

```bash
pip install textual          # the only extra the TUI itself needs
python -m sum_tui            # launch
python -m sum_tui --smoke    # headless self-test (CI-safe)
```

Keys: `d` replay the signed demo · `r` run a transform · `←/→` adjust a focused
slider · `tab` cycle panels · `c` clear · `?` help · `q` quit.

## The web front-end, for free (first cut)

Textual serves the same app to a browser, which is the on-ramp to a dedicated web
UI:

```bash
textual serve "python -m sum_tui"
```

## Status (v0.1)

- **Wired + real:** the offline signed-golden loop (replay + Ed25519 verify),
  the number-box, the Epistemic Nutrition Label.
- **Illustrative (clearly labelled):** the meaning-diff and frontier panels show
  sample structure; live numbers come from `sum meaning-diff` / `sum frontier`
  once a judge (`sum-engine[research,judge]`) and a transform (LLM key or local
  Ollama) are wired through the `Run` action.
- **Architecture:** the TUI never imports the heavy engine — it shells out to the
  shipped `sum` / `sum_verify` CLI, so it stays snappy, dependency-light, and
  degrades honestly when the live judge is absent.

This is **building ahead of a named puller** (operator-directed dream work). It is
deliberately shaped as the low-friction mint/verify surface an adopter would
actually say yes to — so the dream and the adoption strategy point the same way.
