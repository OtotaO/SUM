# Recording the SUM demo GIF

A 15-second screen-cap of the paste → attest → verify loop. Goal: pin it to the top of `README.md` and to the Cloudflare Pages landing so a stranger sees the whole product working before reading a single word. This document is the deterministic runbook for recording it in one take.

## Prerequisites

- **OS:** macOS or Linux with a screen recorder that exports to GIF.
  - macOS: [Kap](https://getkap.co/) (free, exports GIF cleanly) or QuickTime → convert via ffmpeg.
  - Linux: [Peek](https://github.com/phw/peek) or [OBS Studio](https://obsproject.com/) + ffmpeg.
  - Windows: [ScreenToGif](https://www.screentogif.com/).
- **Chrome or Firefox** — any modern browser works; Chrome matches the artifact runtime we target.
- **Python 3** for the local static server.

## One-time setup — 2 minutes

```bash
# From the repo root:
python -m http.server 8765 --directory single_file_demo
# Leave this running. Visit http://localhost:8765/ in the browser.
```

## The deterministic stub — makes extraction beautiful

The live demo falls back to the naive tokeniser when `window.claude.complete` is absent, which produces uneven output unsuitable for a polished recording. For the recording, paste this into the browser DevTools Console **before clicking Attest**:

```js
window.claude = {
  complete: async (prompt) => JSON.stringify([
    ["johannes_gutenberg", "invent", "printing_press"],
    ["marie_curie", "win", "nobel_prizes"],
    ["marie_curie", "be", "physicist"],
    ["ada_lovelace", "write", "first_computer_algorithm"]
  ])
};
```

This stub returns exactly the extraction SUM's sieve + passive-voice fix + apposition handling would produce on the placeholder prose — four triples that showcase three distinct capabilities (passive inversion, apposition secondary fact, clean active baseline). Using the stub instead of a real Claude call guarantees the recording is deterministic across re-takes and keeps the demo offline.

## The recording sequence — 15 seconds total

Set your recorder's output dimensions to **1080 × 720** (or 1280 × 800 for high-DPI source); target 12–15 fps; cap file size at 2 MB after optimisation.

**Second 0–2 — Hero read.** Window is idle, hero visible, textarea empty with the placeholder text showing through.

**Second 2–5 — Paste and press Attest.** Paste the following paragraph (triple-click the placeholder to select, paste to replace):

```
The printing press was invented by Johannes Gutenberg around 1440. Marie Curie, a pioneering physicist, won two Nobel Prizes. Ada Lovelace wrote the first computer algorithm.
```

Press **Attest**. Button text changes to "Extracting…" briefly, then "Minting primes…", then returns.

**Second 5–9 — Results appear.** Four facts crystallise, each with its prime prefix visible. The Gödel state integer fills its card. Let it breathe for 1.5 seconds so the viewer registers the numbers.

**Second 9–11 — Populate verify input.** Open DevTools Console again, run:

```js
document.getElementById('verify-input').value = JSON.stringify(lastBundle, null, 2);
```

Scroll down so the populated verify textarea is visible. Alternatively click **Download bundle**, open the saved file, and paste its contents manually — but the JS approach keeps the recording single-screen.

**Second 11–14 — Verify.** Click **Verify**. Under half a second later the green check appears: *"✓ verified 4 axioms, state integer matches"*.

**Second 14–15 — Hold on the green check.** One beat so the viewer absorbs the proof. Stop recording.

## Post-production

```bash
# Optimise for README embedding (target under 2 MB so GitHub's mobile
# reader handles it without transcoding):
gifsicle -O3 --lossy=80 input.gif -o docs/images/demo.gif
```

Commit to `docs/images/demo.gif`, then update the `README.md` hero with:

```markdown
<p align="center">
  <img src="docs/images/demo.gif" alt="SUM demo: paste a paragraph, attest every fact, verify anywhere." width="720">
</p>
```

(The `<p align="center">` wrapper renders on both GitHub's web UI and on npm/PyPI-style mirrors; GIFs rendered via raw `![](...)` syntax sometimes break on third-party README viewers.)

## Why this script matters

The recording is one asset, re-used in at least three surfaces: the GitHub README hero, the Cloudflare Pages landing poster-frame, and any social / launch surface. Deterministic output (via the stub) means every re-take produces the same primes and the same state integer — so the README can cite a specific prime in the body text and know it matches the GIF. A human curator can also add per-frame annotations ("← passive voice inverted", "← apposition's secondary fact") without worrying the underlying extraction will drift between takes.

## If you don't record today

`docs/images/demo-poster.svg` (shipping alongside this runbook) is a static first-frame placeholder. Use it as the README hero until a real GIF replaces it. The poster frame is already accurate — same hero copy, same placeholder paragraph, same result-card layout — so a visitor who arrives before the GIF is pinned still sees the intended product surface.

## What's new since this runbook was authored

The single-file demo evolved during the Phase E.1 doc-pass + render-
receipt arc. A re-recording should exercise the new surfaces:

- **Render-receipt trust label** (Pre-A3 hygiene cycle, merged in
  PR #48). Three lines next to the rendered tome: "Provenance
  verified" / "Preservation benchmarked: median 1.000; p10 0.769
  long / 0.818 short. Not recomputed for this render." / "Signed
  does not mean true." A re-recording should pause on this label
  for ~2 seconds so a viewer can read it.
- **In-page receipt verifier button** (Phase E.1 v0.9.B, merged in
  PR #52). After a successful `/api/render`, the demo exposes
  "Verify receipt in-page" — runs the same six-step verifier
  against the live JWKS using the vendored `jose` + `canonicalize`
  bundle at `single_file_demo/vendor/sum-verify-deps.js`. A
  re-recording should click the button and capture the green-check
  result.
- **JWKS open-CORS** (Phase E.1 v0.9.A.3, merged in PR #51). The
  in-page verifier fetches `/.well-known/jwks.json` from the live
  Worker; this works from `file://` after the Worker is redeployed
  with the CORS fix.

## Cross-references

- [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) — the
  spec the in-page verifier implements.
- [`single_file_demo/receipt_verifier.js`](../single_file_demo/receipt_verifier.js)
  — the verifier wrapper the demo's button calls.
- [`fixtures/render_receipts/README.md`](../fixtures/render_receipts/README.md)
  — the receipt-fixture set the verifier is tested against.
- [`docs/NEXT_SESSION_PLAYBOOK.md`](NEXT_SESSION_PLAYBOOK.md)
  Pre-A3 hygiene + v0.9.B — the cycles that added the trust label
  + verifier UI.
