# Transform Registry — design

**Status:** designed; first implementation lands with the slider migration (T1).
**Companion:** [`TRANSFORM_RECEIPT_FORMAT.md`](TRANSFORM_RECEIPT_FORMAT.md) — the wire spec for receipts every registered transform produces.
**Justification under the buyer-or-dream filter:** every six grant applications pitch SUM as a *substrate* for verifiable AI output. The slider is one product on that substrate. This document specifies the substrate so the rest of the dream (tags-from-tome, multi-document compose, register/poetry axes, "covers more than any other tag maker" bench) can ship as registered transforms rather than parallel architectures.

---

## 1. The abstraction

Every existing verb in SUM is one shape:

> `(bundle | text) × transform × parameters → signed artifact`

| Existing verb | Input | Transform | Parameters | Output | Today's surface |
|---|---|---|---|---|---|
| `sum attest` | text | extract + encode + sign | extractor choice, source_uri | signed bundle | `POST /api/...` plus CLI |
| Slider render | bundle | recompose tome | density + 4 LLM axes | tome + receipt | `POST /api/render` |
| `sum verify` | bundle | check | — | boolean + error class | CLI |
| Compliance check | audit-log | regime-validate | regime id | compliance report | `sum compliance check` |
| LCM merge (latent) | N bundles | union | — | bundle | not exposed |
| Tag extract (latent) | bundle | project at density=0 | — | tag set | not exposed |
| Translate (reserved) | tome | re-language | target lang | tome | not exposed |
| Restyle (reserved) | tome | re-style | register, genre | tome | not exposed |

**The transform registry** is the single dispatch surface that unifies these. The slider becomes one entry; the rest of the table becomes the next eight months of work, each as a few-hundred-line PR, each riding on the same:

- Cross-runtime byte-stable canonical form (RFC 8785 JCS).
- Ed25519 receipt format (`sum.transform_receipt.v1`).
- JWKS distribution + revocation.
- Audit-log emission (`sum.audit_log.v1` with `operation: "transform"`).
- Compliance hookup (the six per-regime validators consume the audit log regime-agnostically).
- Cache-coherent KV storage (content-addressed by `(transform, parameters, input)`).

No new architecture per transform. Just a transform function + a registration line.

---

## 2. The contract — every transform implements this

A transform is a function (Python + TypeScript ports stay in sync):

```python
# sum_engine_internal/transforms/_base.py
class Transform(Protocol):
    name: str                              # registry id, e.g. "slider"
    requires_llm: bool                     # True iff the transform may call an LLM
    digital_source_type: Literal["trainedAlgorithmicMedia", "algorithmicMedia"]

    def canonicalize_parameters(self, params: dict) -> bytes:
        """Return JCS-canonical bytes of the parameters. The receipt's
        parameters_hash is sha256 of this output."""

    def canonicalize_input(self, raw_input: Any) -> bytes:
        """Return canonical bytes of the input. For a CanonicalBundle,
        this is state_integer ‖ canonical_tome. For raw text, utf8 bytes.
        Pinned per transform so input_hash is reproducible by any caller."""

    def canonicalize_output(self, output: Any) -> bytes:
        """Return canonical bytes of the output. For a tome, utf8 bytes.
        For a tag set, JCS of sorted unique triples. Pinned per transform."""

    async def apply(
        self,
        input: Any,
        parameters: dict,
        env: TransformEnv,     # bag of capabilities: LLM keys, model registry, etc.
    ) -> TransformResult:
        """Run the transform. Returns the output artifact + the
        model/provider/digital_source_type actually used (honest
        provenance per the receipt's `model`+`provider` fields)."""
```

```typescript
// worker/src/transforms/_base.ts — symmetrical
export interface Transform {
  readonly name: string;
  readonly requiresLLM: boolean;
  readonly digitalSourceType: "trainedAlgorithmicMedia" | "algorithmicMedia";

  canonicalizeParameters(params: unknown): Uint8Array;
  canonicalizeInput(rawInput: unknown): Uint8Array;
  canonicalizeOutput(output: unknown): Uint8Array;

  apply(
    input: unknown,
    parameters: Record<string, unknown>,
    env: TransformEnv,
  ): Promise<TransformResult>;
}
```

The Python and TypeScript implementations of the same transform name MUST produce byte-identical `canonicalize_*` outputs for the same inputs. The cross-runtime K-matrix gates this — adding a new transform extends the gate fixture set, not the gate's structure.

---

## 3. The dispatch surface

### 3.1 HTTP API

```
POST /api/transform
{
  "transform": "slider",
  "input": { "triples": [["alice","likes","cats"]], "source_uri": null },
  "parameters": { "density": 1.0, "length": 0.5, "formality": 0.5, "audience": 0.5, "perspective": 0.5 },
  "provider": "anthropic",    // optional
  "force_render": false       // optional
}
→ HTTP 200
{
  "output": { ... transform-specific shape ... },
  "transform_id": "<sha256-trunc-16>",
  "cache_status": "miss" | "hit" | "bypass",
  "wall_clock_ms": 53,
  "transform_receipt": { ... sum.transform_receipt.v1 ... }
}
```

The Worker dispatches by `transform` field through a registry lookup; unknown transform values return HTTP 400 `unknown transform`.

### 3.2 Python CLI

```bash
# Slider — same surface as today, transformed under the hood.
sum transform slider --density=1.0 --length=0.5 --formality=0.5 \
                     --audience=0.5 --perspective=0.5 < bundle.json

# Extract tags from a tome
sum transform extract --multi-school --max-tags=32 < text.txt > tags.json

# Compose multiple bundles
sum transform compose --merge=lcm bundle1.json bundle2.json bundle3.json > sum_of_sums.json
```

`sum render` stays as a CLI alias for `sum transform slider` for backwards compat.

### 3.3 MCP server

Adds one new tool: `transform(name: str, input: ..., parameters: ...) → output + receipt`. The existing `attest` / `verify` / `extract` / `render` tools stay; `transform` is the generic dispatch any MCP-aware agent can call to invoke the registry without per-transform tool definitions.

---

## 4. Cache semantics

Cache key derivation:

```
key = sha256( JCS({
  "transform":      "<name>",
  "input_hash":     "<sha256-hex of canonicalize_input(raw_input)>",
  "parameters_hash": "<sha256-hex of canonicalize_parameters(parameters)>",
  "provider":       "<resolved provider, e.g. 'anthropic'>"
}) )[:32]
```

The provider is included so BYO-keys / operator-funded renders of the same `(transform, input, parameters)` don't serve cross-provider stale entries. The receipt's `provider` field already reflects what served, so HIT/MISS semantics match the trust-loop guarantees.

Cache HIT serves the original transform_receipt verbatim, preserving `signed_at` per §1.3 of the receipt format spec.

---

## 5. The v1 transform set

Five transforms in the v1 registry. Order = priority of implementation:

### 5.1 `slider` (T1 — migration, no new behavior)

Today's `POST /api/render` migrates here verbatim. Receipt shape changes from `sum.render_receipt.v1` to `sum.transform_receipt.v1`. The legacy endpoint stays live as a thin adapter that calls the registry internally.

**Why migration first**: it's the only transform that exists today. Refactoring it into the registry is the foundation; once the registry exists, every subsequent transform is additive.

### 5.2 `extract` (T2 — dream-side)

**Bi-directional fulfillment**: today's slider compresses tomes via density; `extract` is the inverse direction — yield the canonical tag set as a named output.

- Input: text OR `CanonicalBundle`.
- Output: tag set (sorted unique `(subject, predicate, object)` triples, JCS-canonicalised).
- Parameters: `{ontology, max_tags, multi_school}`. `multi_school` runs N extractors in tandem (Sieve + LLM + Wikidata + ontology-specific where applicable) and unions/clusters their outputs.
- LLM-mediated: configurable (Sieve-only by default; `multi_school` toggles LLM).
- UI: a "show me the tags" toggle on the slider product; density slider goes to 0 → tag-set view.

This closes the dream's bi-directional gap.

### 5.3 `compose` (T3 — library compression)

**Many-to-one fulfillment**: dream element "books or libraries… into a succinct summary."

- Input: array of `CanonicalBundle`.
- Output: one `CanonicalBundle` whose `state_integer` is the LCM of inputs, and whose `canonical_tome` is a deterministic merge with provenance markers per source.
- Parameters: `{merge_strategy}` ∈ `"lcm" | "intersect" | "diff"`.
- LLM-mediated: no — pure Gödel-state algebra.
- UI: drag-and-drop multiple bundle files into the demo; see merged SUM with provenance to each source.

### 5.4 `translate` (reserved)

Re-language a tome. Same trust-loop machinery. Defers behind T2 + T3 because it's not on the buyer-or-dream filter today.

### 5.5 `expand` (reserved)

Tags → narrative tome. The reverse of `extract`. Closes the round-trip dream: tags-from-tome AND tome-from-tags. Defers because once `extract` ships, the reverse is a smaller next step.

### 5.6 `restyle` (reserved)

Adjustable `register` axis (literal / poetic / revelatory), `genre` (academic / narrative / devotional), `cultural_frame`. The dream's "revelation, poetry, multi-style" devices. Defers because each axis is a Bench-cells-of-work investment per axis.

Reserved transforms have their entries pinned in the v1 registry (see `TRANSFORM_RECEIPT_FORMAT.md` §1.2) so verifiers don't reject them as adversarial when they ship.

---

## 6. Where the code lives

```
sum_engine_internal/transforms/
├── __init__.py              # registry exports + register() decorator
├── _base.py                 # Transform protocol, TransformEnv, TransformResult
├── slider.py                # T1 — migrated from ensemble/slider_renderer.py
├── extract.py               # T2
├── compose.py               # T3
└── ...                      # T4+ as they land

worker/src/transforms/
├── _base.ts                 # symmetric Transform interface
├── _registry.ts             # dispatch table + register() builder
├── slider.ts                # T1 — migrated from routes/render.ts
├── extract.ts               # T2
├── compose.ts               # T3
└── ...

worker/src/routes/
├── transform.ts             # POST /api/transform dispatch handler (T1)
└── render.ts                # legacy alias → transforms/slider via the registry (T1, until removed)
```

The Python and TS slider implementations of the canonicalisation methods stay in step via the existing K-matrix cross-runtime gate (extended in T1 to cover the new receipt shape).

---

## 7. What this design explicitly defers

1. **Programmatic transform registration via plugin entry points.** The v1 registry is a fixed enum in the Python + TS source. Third parties can't register transforms without forking. Defensible because the receipt-signing surface is the operator's trust anchor; a plugin model needs a separate signing-authority story (the slider product doesn't need it).

2. **Per-transform JWKS / kid.** All transforms share the operator's existing render-receipt signing kid. A future transform that needs a separate trust authority (e.g., a community-operated `translate` transform) would warrant its own kid and JWKS endpoint; that's a v2 design conversation.

3. **Transform composition** (one transform's output feeding another's input as a single signed pipeline). The first composition pattern is `extract → expand` (tags-from-tome → tome-from-tags = round-trip drift measurement). v1 ships each transform as standalone; chaining is application-layer for now. A "pipeline receipt" type would be v2 work.

4. **Streaming transforms.** v1 is request/response. Streaming token-by-token output with progressive signing is a real design problem (signature can't be computed until output is complete) and out of scope.

---

## 8. Cross-references

- [`TRANSFORM_RECEIPT_FORMAT.md`](TRANSFORM_RECEIPT_FORMAT.md) — wire spec for the receipt every transform produces.
- [`RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) — the prior-art receipt the slider currently produces; remains verifiable forever.
- [`SLIDER_CONTRACT.md`](SLIDER_CONTRACT.md) — slider product's fact-preservation contract. Stays canonical for `transform: "slider"`.
- [`AUDIT_LOG_FORMAT.md`](AUDIT_LOG_FORMAT.md) — audit-log rows every transform emits.
- [`MCP_INTEGRATION.md`](MCP_INTEGRATION.md) — MCP surface; gets a `transform` tool after T1.
- [`OPERATOR_AUDIT_2026-05-12.md`](OPERATOR_AUDIT_2026-05-12.md) — operator-side state at the time of this design.
