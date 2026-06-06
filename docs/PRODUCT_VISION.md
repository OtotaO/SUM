# SUM — product vision (the workbench)

*Status: vision + first-increment design. Captured 2026-06-06 from the
operator's stated vision, with four scoping decisions locked (§2). This
is the standing picture the workbench is built toward; the substrate
docs (CHARTER, SLIDER_CONTRACT, MEANING_LOSS_FRONTIER) remain the
sources of truth for what is proven.*

---

## 1. The vision, in one breath

A box you drop a text into — an essay, an article, a book — and a slider
that renders it anywhere from **a single tag to a full tome and every
point in between**, while a number box beneath shows *precisely what was
preserved and what was lost*, backed by a signed, verifiable receipt.
Other controls reshape the rendering; a scrubber lets you travel the
whole **realm of possibilities** from most-faithful to most-compressed.
Reachable as a **web page, an API, an MCP tool, and a CURL command** —
the same engine behind every door. A companion **browser extension**
auto-tags any page (the *tags↔tomes* origin dream, pointed at the live
web).

The thing that makes it SUM and not a summariser: **every transformation
hands back proof of what it preserved.** The number box is where that
proof becomes legible to a person.

## 2. Scoping decisions (locked 2026-06-06)

| Decision | Choice | Consequence |
|---|---|---|
| **Media scope** | **Text first** (essays / articles / books / pasted text); design the ingestion + receipt schema so media can slot in later without a rewrite | Fastest to a real deliverable; matches where the substrate is proven. Audio/image/video is a named later horizon, not the first build. |
| **Number box** | **Both** — precise numeric *input* of the axis parameters AND the measured *output* (fact-preservation + certified meaning-loss + receipt link). Built as a **strategic abstraction** that compounds, not a one-off widget | The box is a *view over a single frontier point* (§3), so the same object drives input control and the proof readout. |
| **Cycle the realm of possibilities** | **Frontier scrubber** — a continuous path from most-faithful to most-compressed; scrub to your point on the fidelity/brevity trade-off | The frontier is the load-bearing object (§3); named perspectives and A/B variants become later views over the same path. |
| **Build first** | **The summariser workbench** (drop-box + slider + number box + receipts) | Closest to shipping; reuses the most existing substrate. The tag extension rides the same `extract` transform afterwards. |

## 3. The strategic abstraction — the *render frontier*

Process intensification means one object powers every surface. That
object is the **render frontier**: for a source text, an *ordered path*
of renderings from most-faithful to most-compressed, each point carrying
its measured numbers.

```
source ──▶  RenderFrontier
              point[0]  most faithful   (params, rendering, measured loss)
              point[1]  ...
              ...
              point[k]  most compressed (params, rendering, measured loss)
```

Every surface the vision names is a *thin view* over this one object:

| Surface | Is just… |
|---|---|
| The **slider** | picking a point on the frontier |
| The **number box** | the input params + measured numbers *of the current point* |
| The **frontier scrubber** | dragging `t ∈ [0,1]` across the path |
| **Named Perspectives** (later) | labelled points on the path (novice / expert / regulator) |
| The **API / MCP / CURL** | the frontier serialised (`as_dict`) and returned |

Build the frontier once; the workbench is views and styling.

## 4. The honest line through the number box (proof boundary)

The number box shows two *different kinds* of number, and the UI must
never blur them — this is the proof-boundary discipline made visible:

- **Per-document, measured.** For *this* text at *this* setting: a
  fact-preservation fraction and a meaning-loss value under a *named
  proxy*. This is a **measurement**, not a guarantee — a point estimate
  for one document. Honest label: *"measured for this text."*
- **Corpus-level, certified.** A distribution-free *upper bound* on
  expected meaning-loss for *this parameter setting*, marginally over a
  named corpus, sealed in a `sum.meaning_risk_receipt.v1`. Honest label:
  *"certified ≤ X across <corpus>, marginal, under exchangeability."*

The frontier yields the first; the meaning-risk receipt yields the
second. The box surfaces both and *says which is which*. It never shows
a per-document "guarantee" — there is no such thing.

What the box must also disclose (always present, never implied away):
the proxy's name, and the layers it does **not** cover — arrangement,
sound, connotation, implicature.

## 5. What already exists vs. what is new

| Piece | Substrate today | New work |
|---|---|---|
| Tri-modal access (UI / API / MCP / CURL) | Worker `/api/transform`, live demo, MCP server, CURL routes | Unify into one product surface |
| Drop-box for text / essays / books | Text input + slider; long-doc verified (n=16) | Robust long-input ingestion UX |
| Media of all sorts | — (text only) | Later horizon (decision: text first) |
| Slider (5 axes) | density / length / formality / audience / perspective, each with a transform receipt | — |
| Number box | precise axis params exist; **meaning-risk receipt now exists** | Wire the measured + certified numbers into one readout |
| Frontier scrubber | the IB faithful↔compressed framing | **The `RenderFrontier` object (this increment)** + scrubber UI |
| Auto-tag browser extension | `extract` transform (triples); prior extension substrate | Package as a live-page auto-tagger |

## 6. Roadmap (workbench-first, each rung an outcome)

1. **`RenderFrontier` abstraction** *(this increment)* — the backend
   object: source → ordered, scored frontier of renderings; offline-
   runnable (injected render function + scorer), API-serialisable,
   tested. No new substrate — pure composition of the slider params and
   the meaning-loss scorer.
2. **One real, committed receipt** — a public-domain parallel-
   translation corpus + a deterministic judge → a real
   `sum.meaning_risk_receipt.v1` + the cross-runtime golden fixture the
   audit flagged as missing. Turns "mechanism shipped" into
   "demonstrated outcome" with **no paid-judge dependency**.
3. **Workbench UI** — drop-box + slider + number box (both modes) +
   frontier scrubber, as views over the frontier object; the proof
   readout front-and-centre.
4. **API / MCP / CURL parity** — the frontier object returned identically
   through every door.
5. **Auto-tag browser extension** — the `extract` transform on live
   pages, each tagging carrying a receipt.

Operator/$-gated steps (do **not** build until the pull arrives, per the
charter): a real LLM render path for live slider output, and a paid NLI
judge for production meaning-loss. The deterministic offline paths above
make rungs 1–3 demonstrable without either.

## 7. Use-case ledger (who gets what)

| Use case | Beneficiary | Status |
|---|---|---|
| Rewrite for an audience, keep proof of what survived | writer / knowledge worker (**dream**) | substrate present; dogfood-gated |
| "How much meaning did I lose compressing this?" | writer; accessibility / plain-language | mechanism shipped; real-data rung next |
| AI-text transformation disclosure (EU AI Act Art 50) | regulated text-transformers (**funder/compliance**) | receipt stack present; profiles designed |
| Chain-of-custody for AI-transformed text | the moat — what grants fund | our differentiator |
| Agent-to-agent handoff with a fidelity token | developers / agent builders (**substrate**) | deliverable now |
| Auto-tag the web (tags↔tomes) | the origin dream | `extract` present; extension to package |

---

*This document is the compass for product surface; it does not change
any proven claim. Where it describes a number a user sees, the proof-
boundary discipline in `docs/PROOF_BOUNDARY.md` and the honest line in §4
govern what may be said about that number.*
