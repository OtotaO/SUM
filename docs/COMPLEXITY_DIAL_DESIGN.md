# The complexity dial — progressive disclosure as UX-level intensification

*Status: design note, captured 2026-06-07 from an operator brain-dump. A
**direction to hold**, not a build commitment. The binding constraint per
[`CHARTER_2026-05-17.md`](CHARTER_2026-05-17.md) is still a first real
adopter; this is explicitly downstream of (a) the workbench UI existing
and (b) that adopter. It changes **no** proven claim. Source of truth for
the product surface remains [`PRODUCT_VISION.md`](PRODUCT_VISION.md).*

---

## 1. The idea, in one breath

A control on the workbench that sets **how much workbench there is** — a
dial from a near-empty surface (drop box + one slider) up to the full
instrument (every axis, the raw params, the receipt internals, the API
recipe). A *basic → advanced* scale for the interface itself. Novice and
power user drive the **same engine** through a surface dialled to their
need.

## 2. Why it fits SUM specifically (not just good UX hygiene)

This is **progressive disclosure** — the established pattern behind
Blender's modes, Lightroom's panels, a DAW's simple/advanced views. But
for SUM it is more than hygiene; it is *the slider, all the way down.*
SUM's entire thesis is a control that moves through a space of
possibilities while preserving what matters. A dial that moves through
**interface density** is that same gesture turned reflexively on the
tool. The product is a slider; the product's chrome is a slider; it is
coherent rather than cute.

## 3. The load-bearing insight: complexity-mode *is* the perspective axis

The named modes are not a new vocabulary to invent. SUM already has one:
**Perspective Receipts** name points on the frontier as *novice / expert
/ regulator* ([`ZENITH_FRAMING_2026-05-16.md`](ZENITH_FRAMING_2026-05-16.md),
[`PRODUCT_VISION.md`](PRODUCT_VISION.md) §3). Collapse the two:

> The interface's **complexity mode** and the artifact's **target
> perspective** are the same axis.

Put the workbench in **regulator** mode and you get *both* the compliance
fields surfaced *and* the rendering + receipt rendered for a regulator
audience. Put it in **novice** mode and the surface is three controls and
the output is the plain-language render. One dial drives the tool's depth
**and** the output's audience, because they were never two things. That
reuse is worth more than the dial itself — it is the difference between a
settings widget and a unifying primitive.

## 4. Process intensification — but name which layer

The operator's framing ("maximise process intensification at the UX
settings level") is the honest one, and it sits *alongside* the
intensification already in the vision — they are different layers, and
conflating them would muddy both:

| Layer | What gets intensified | Where it lives |
|---|---|---|
| **Object-level** (already designed) | one `RenderFrontier` object powers slider, number box, scrubber, API, MCP | `PRODUCT_VISION.md` §3 |
| **UX-level** (this note) | one **surface** serves novice→expert; no forked "lite" and "pro" apps | this dial |

Object-level says *build the frontier once, the surfaces are views.*
UX-level says *build the surface once, the audiences are dial positions.*
Together: one engine, one object, one surface — the whole product
collapses to its minimal sufficient form per user, which is exactly what
intensification means borrowed from its chemical-engineering origin (more
done per unit of apparatus). Keep them distinctly named; they compound,
they are not the same claim.

## 5. The honest design pushback: 2–3 named modes, not a continuous slider

A *continuous* "complexity" axis is secretly a hard problem. To make a
slider position meaningful you need a **monotonic ordering of every
control by difficulty**, and that ordering does not exist — "show the
formality axis" and "show the raw `delta_micro` field" are not on one
naturally-ordered line. A continuous dial invites endless bikeshedding
over where each control enters.

The 80/20 is **2–3 discrete, named modes** — and §3 already named them:

| Mode | Surface shown | Output framing |
|---|---|---|
| **Novice** | drop box · one density slider · the number box (measured-only) | plain-language render; "measured for this text" |
| **Practitioner** | + all 5 axes · frontier scrubber · the certified bound when a corpus receipt exists | + "certified ≤ X across <corpus>" when present |
| **Programmer / Regulator** | + raw params (`*_micro` fields) · the signed receipt internals · the API / MCP / CURL recipe · the `not_covered` disclosure in full | full proof-boundary surface; verifiable recipe |

Three presets get ~90 % of the value at ~10 % of the design cost, and
each maps onto an audience SUM already serves. A continuous dial can come
later as polish *over* the discrete modes — never as the v1.

## 6. The proof-boundary stays visible at every setting

One rule the dial must not violate: **novice mode hides controls, never
honesty.** The `not_covered` disclosure and the measured-vs-certified
distinction ([`PRODUCT_VISION.md`](PRODUCT_VISION.md) §4) are not
"advanced" — they are the contract. A simpler surface may show them more
briefly (an icon that expands), but it may never *imply away* a per-
document "guarantee" that does not exist. Complexity hides *apparatus*,
not *caveats*.

## 7. Sequencing — capture, don't build

This is a **v2 polish on a v1 that is not yet a web UI.** You cannot add a
complexity dial to a surface that today is a CLI plus a vision. The order
is forced:

1. The workbench UI ships (`PRODUCT_VISION.md` roadmap rung 3).
2. A first real adopter uses it (the charter's binding constraint).
3. *Then* the surface is rich enough to need hiding, and the dial earns
   its place.

Building it before step 1 would be the charter's named **substrate-
velocity / fidget-build trap**
([[feedback_substrate_velocity_trap_2026-06-06]]). The cheapest honest
first increment, when the time comes, is a **two-mode toggle**
(novice / full), not a slider — ship the toggle, learn whether anyone
wants the middle, add the third mode if they do.

## 8. Open questions (for when this is live)

- Does the mode persist per-user, per-document, or per-session?
- Does "regulator" mode change *only* what's shown, or also defaults
  (e.g. auto-attach a Trust Profile)? If the latter, it is no longer pure
  disclosure — flag it.
- Is the dial discoverable without cluttering novice mode? (A single
  unobtrusive corner control is the usual answer.)

## 9. Honest boundary

Nothing here is proven or built. It is a design direction with one real
insight worth keeping — *complexity-mode = perspective axis* — and one
real discipline — *discrete named modes over a continuous slider, hide
apparatus not caveats.* It waits behind the workbench UI and the first
adopter. Captured so it is not lost; deliberately not started.
