# NLI model registry (Phase E.1 v0.9.E)

**Status:** spec landing alongside the calibration fixture format. Implementation (actual ONNX model export + live-benchmark calibration) is operator-paced — runs after this spec lands and after the calibration set has at least 30 adjudicated rows. Same design-now / prototype-later pattern as R0.5.

The render-receipt verifier triangle (v0.9.B / v0.9.C / v0.9.D) closes the trust loop on the issuance side. The NLI audit layer is what closes the **measurement** side: every embedding-flagged "fact loss" cell on the long-doc bench gets a per-fact entailment judgement that distinguishes real fact loss from rephrased preservation. Today that judgement is rendered by an OpenAI LLM call; v0.9.E swaps in a local ONNX model running an MNLI/FEVER/ANLI-trained classifier.

## Why decouple from OpenAI

Three concerns, in order of severity:

1. **Cost coupling on the load-bearing rescue path.** The v0.7 long-doc bench made 654 NLI calls and rescued 653 of them — a 99.8 % rescue rate. That number is the centrepiece of the slider's verified-at-scale claim ([`docs/SLIDER_CONTRACT.md`](SLIDER_CONTRACT.md) §"Headline result"). At ~$0.30 per bench run with NLI audit, the cost is small but real, and it scales with corpus size.
2. **Provider-availability coupling.** SUM's claim about p10 fact preservation depends on an external API. Anthropic / OpenAI rate-limits, regional outages, or a model deprecation can break the bench reproducibly. Local inference removes this surface.
3. **Cross-runtime coverage.** ONNX Runtime is the unified inference API behind Node.js, Web (in-browser), and React Native bindings. Once a local NLI model exists, the v0.9.B browser receipt verifier can fall back to in-page entailment for offline audit cases — closing a gap the OpenAI path cannot close (no in-browser inference).

## What this spec defines

The contract a future implementation cycle delivers against:

- **Model class.** Candidate model class only; no specific HuggingFace slug pinned until benchmark validation.
- **Threshold contract.** How the local model's confidence float maps to the existing `EntailmentResponse.is_supported: bool` interface.
- **Calibration fixture format.** What's in `fixtures/nli_audit_calibration_v1.jsonl`, how it's generated, what the success criterion is.
- **Versioning policy.** The model is a kid-shaped pin; replays of historical bench runs use the model version recorded at run time.
- **Implementation gating.** What needs to be true before the implementation cycle starts.

## Candidate model class

**DeBERTa-family MNLI/FEVER/ANLI cross-encoder, exported to ONNX.**

Why this class:

- DeBERTa-v3 architecture is the strongest per-parameter NLI accuracy in publicly-available models as of 2025.
- MNLI + FEVER + ANLI is the standard NLI training set composition; FEVER specifically covers the "premise-supports-fact" judgement SUM's audit layer needs.
- Cross-encoder shape (premise + hypothesis concatenated, single classification head) is the right inductive bias for SUM's `(passage, fact)` pair shape.
- ONNX-exportable cleanly via `optimum.onnxruntime.ORTModelForSequenceClassification`.

**Model selection** is operator-paced and deferred to the implementation cycle. The implementation PR commits to ONE specific HuggingFace slug + revision SHA after benchmarking 2–3 candidates against the calibration set. Candidates worth validating include (but not limited to):

- `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- `cross-encoder/nli-deberta-v3-base`
- `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (if multilingual is wanted)

The pinned slug + revision SHA become the load-bearing contract for the bench's headline numbers. A future model bump is a deliberate calibration-rerun PR, not a silent dep update.

## Threshold contract

The OpenAI-backed `check_entailment` returns a `bool`. The local model returns a logits triple `(entailment, neutral, contradiction)` after softmax. The local checker's threshold rule:

```python
def is_supported(probs: tuple[float, float, float]) -> bool:
    """probs = (entailment, neutral, contradiction) summing to 1.0."""
    p_entailment = probs[0]
    return p_entailment >= ENTAILMENT_THRESHOLD
```

`ENTAILMENT_THRESHOLD` is calibrated against the adjudicated calibration set such that:

- **Rescue precision** (fraction of "supported" judgements where the adjudicated label is "entails") matches or beats the OpenAI path's measured ~99.8 % on the v0.7 long-doc bench.
- **Rescue recall** (fraction of adjudicated-entails cases the local model also marks supported) within ±2 percentage points of the OpenAI path.

If the threshold can't satisfy both, the implementation cycle reports the trade-off and the operator picks: high-precision (fewer rescues, lower median preservation) or high-recall (more rescues, higher false-positive risk on actual losses). The default is high-precision — SUM's contract is "facts are preserved" and a false positive on this layer is a worse failure than a false negative.

## Calibration fixture format (`fixtures/nli_audit_calibration_v1.jsonl`)

NDJSON, one row per audited fact. Authored by the operator running the existing OpenAI-backed bench, recording each NLI call and its outcome, then human-adjudicating the disagreement cases.

Each row:

```json
{
  "schema": "sum.nli_audit_calibration.v1",
  "fact": ["alice", "graduated", "2012"],
  "fact_key": "alice||graduated||2012",
  "passage": "<the rendered tome bytes the audit was performed against>",
  "embedding_score": 0.62,
  "openai_verdict": "entails",
  "local_model_verdict": "entails",
  "adjudicated_label": "entails",
  "rationale": "passage says 'Alice received her degree in 2012' which directly implies the fact",
  "source": "v0.7 long-doc bench, doc_long_relativity, formality=0.1"
}
```

| Field | Type | Meaning |
|---|---|---|
| `schema` | string | Always `"sum.nli_audit_calibration.v1"`. Lets format-validating tests reject mixed-version sets. |
| `fact` | `[s, p, o]` | The (subject, predicate, object) triple under audit. Stable ID for the fact across runs. |
| `fact_key` | string | Canonical `s||p||o` form for sorting / dedup. Derivable from `fact` but stored to keep the row self-describing. |
| `passage` | string | The rendered tome bytes the audit ran against. Real passage text from the bench corpus, NOT a synthetic example. |
| `embedding_score` | float | The embedding-similarity score that triggered the audit (the cell semantic-fact-preservation < `--audit-threshold` value). |
| `openai_verdict` | `"entails" \| "neutral" \| "contradicts"` | The OpenAI-backed `check_entailment` result, captured at audit time. The local model is NOT the oracle; OpenAI is a comparison backend. |
| `local_model_verdict` | `"entails" \| "neutral" \| "contradicts"` | The candidate local model's verdict. Populated by the implementation cycle once a specific model is chosen. |
| `adjudicated_label` | `"entails" \| "neutral" \| "contradicts"` | **The ground truth.** Human-reviewed with a documented rubric (see below). The threshold calibration optimises against this column, not against `openai_verdict`. |
| `rationale` | string | One-sentence justification. Helps audit tracing when verdicts disagree. |
| `source` | string | Where the row came from (bench run + cell ID). Lets a future review fetch the original render context. |

### Adjudication rubric

The human reviewer applies the same rule the OpenAI-backed system prompt uses:

> "Be strict: rephrasings that preserve meaning count as supported; loose associations or topic-similarity do not."

In the four cases:

| Case | Adjudicated label |
|---|---|
| Passage explicitly states the fact verbatim | `entails` |
| Passage rephrases the fact with preserved meaning (e.g. "Alice received her degree in 2012" for `alice / graduated / 2012`) | `entails` |
| Passage discusses related context but doesn't state the specific fact | `neutral` |
| Passage states something incompatible with the fact (e.g. polarity flip, wrong year) | `contradicts` |

The rubric is the contract; if a fixture row's `adjudicated_label` looks wrong on review, the right move is to amend the row + add a `rationale` note documenting the corner case, NOT to change the rubric.

### Calibration set sizing

Minimum **30 rows** for the implementation cycle to start. Drawn from real bench runs — the v0.4 / v0.6 / v0.7 audit JSONL artifacts are the seed source. Bias toward cases where `openai_verdict` and `local_model_verdict` would disagree, since those are the rows the threshold tuning depends on most.

The starter set committed alongside this spec is **demonstration-sized only** (3–5 rows) — enough to pin the format. The operator authoring the implementation PR adds the remaining ≥25 rows from real bench data.

## Versioning policy

The local NLI model is treated like the render-receipt signing key: **kid-shaped, immutable per-version, replay-stable**. The bench artifact records which model version produced each verdict, so a historical bench run replayed against a newer model version carries both verdicts side-by-side rather than being silently overwritten.

The kid format:

```
sum-nli-{model-slug-slugified}-{exported-revision-sha-short}-{export-date}
e.g. sum-nli-deberta-v3-base-mnli-fever-anli-7c3f2a1-2026-04-28
```

The exact slug + revision sha + export date land in the implementation cycle's PR + the model registry doc gets a row appended.

## Cross-runtime story

ONNX Runtime is the unified API behind Node.js / Web / React Native bindings. Once the model is exported for Python use, the SAME ONNX file works (with the same tokenizer JSON) in:

- **Node-side bench audit** (current bench cells delegate to Python; future Node-side bench could run the same model locally).
- **Browser-side spot-check** in the v0.9.B receipt verifier UI — fetch the ONNX model lazily on first verify, run inference in-page for a "this rendered tome's facts are preserved" advisory check.

The browser path is **explicitly out of scope for v0.9.E.** The v0.9.E implementation lands the Python side first; cross-runtime extension is a follow-on once the Python path's calibration is validated.

## Interface contract

The Python implementation exposes a class mirroring `LiveLLMAdapter.check_entailment`'s signature:

```python
class LocalNLIChecker:
    """Drop-in replacement for LiveLLMAdapter.check_entailment.

    Same input shape (fact, passage), same output shape (bool).
    Threshold is configured at construction time, not per-call.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        entailment_threshold: float = 0.85,
    ) -> None:
        ...

    async def check_entailment(
        self,
        fact: Tuple[str, str, str],
        passage: str,
    ) -> bool:
        ...
```

`async` for interface parity with the OpenAI path (callers already `await` it). The implementation can be sync internally — ONNX Runtime's `InferenceSession.run` is synchronous — but the wrapper presents the async surface so swapping is invisible to callers.

The bench harness wires this in via a `--nli-backend={openai,local}` CLI arg (default `openai` until the local path is validated). When `local`, the harness instantiates a `LocalNLIChecker` from a configured model path; when `openai`, the existing `LiveLLMAdapter.check_entailment` path runs unchanged.

## Implementation gating

This spec lands as `docs/NLI_MODEL_REGISTRY.md` (this file) + the calibration fixture starter set. Implementation does NOT land in the spec PR. It requires:

1. **Calibration set populated to ≥30 adjudicated rows.** Operator-paced; the starter set committed alongside this spec is 3–5 rows for format demonstration only.
2. **Model class validated against the calibration set.** Implementation cycle picks 2–3 specific HuggingFace slugs, exports each to ONNX, scores each against the calibration set, picks the best on a {precision, recall, latency, model-size} trade-off.
3. **A pinned model + tokenizer artifact published.** Either as a GitHub release asset, a PyPI sub-package, or a download-on-first-use script. The artifact's hash is named in this registry doc + the implementation PR + (eventually) the trust-root manifest's `algorithm_registry`.
4. **Bench harness wiring.** `--nli-backend=local` flag + per-cell verdict recording.

When all four are in place, the implementation cycle is one PR: the `LocalNLIChecker` class + the bench wiring + a test that replays the audited cells from the v0.7 long-doc bench and asserts rescue precision / recall match the calibration set's targets within tolerance.

## What this spec does NOT define

- **A specific model.** Pinned by the implementation cycle, not here. Picking a model before benchmarking it is the kind of premature commitment this spec exists to prevent.
- **The browser-side path.** Out of scope; the implementation cycle lands Python first.
- **Threshold values.** The threshold is calibration-set-derived, not author-asserted.
- **Latency / cost numbers.** Same — the implementation cycle measures and pins.

## Cross-references

- [`docs/SLIDER_CONTRACT.md`](SLIDER_CONTRACT.md) §"Headline result" — the median/p10 numbers the NLI audit layer rescues.
- [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) §2.6 — the v0.4 / v0.6 / v0.7 bench numbers + NLI rescue rate; the local-NLI swap target's contract is to match these within tolerance.
- [`docs/NEXT_SESSION_PLAYBOOK.md`](NEXT_SESSION_PLAYBOOK.md) v0.9.E — playbook entry that scoped this design.
- [`fixtures/nli_audit_calibration_v1.jsonl`](../fixtures/nli_audit_calibration_v1.jsonl) — calibration set; format defined in §"Calibration fixture format" above.
