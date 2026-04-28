# Model-call evidence format (`sum.model_call_evidence.v1`)

**Status:** design-now / prototype-later. Specifies an optional sidecar that records hash-only forensic binding to the model call that produced a render receipt. Implementation lands as its own cycle once the v0.9.B / v0.9.C / v0.9.D verifier triangle is fully closed and operators have a stable place to attach it.

The receipt format (`docs/RENDER_RECEIPT_FORMAT.md`) honestly admits that a verified receipt does NOT cryptographically bind the tome to the triples — it asserts "I observed this tome and these triples together" rather than "I produced this tome from these triples." The model-call evidence sidecar fills this gap by recording (hash-only) forensic provenance of what prompt actually went to the LLM and what response came back.

This is **process evidence, not semantic-truth evidence.** A signed model-call evidence record proves: "the issuer's Worker called this model with this prompt-template hash, this request-body hash, and got back this response, which produced this tome." It does NOT prove the model's output is factually correct, the prompt template is well-designed, or the response was not adversarially crafted.

## Why hash-only

Two separate concerns drove the hash-only choice:

1. **Privacy.** Raw prompts and raw model outputs may contain user-input prose that the user didn't expect to be permanently recorded. SUM's render path is agnostic about source content (it just signs whatever it produces), but the sidecar would otherwise create a high-value content log.
2. **Operational simplicity.** Hashes are forensic binding without storage cost. A sidecar record is bounded-size regardless of prompt length; raw prompt logging would scale with corpus size and require operator decisions about retention windows.

Verification of the sidecar still works because: the operator faced with a dispute ("did model X actually produce this tome?") rehashes the prompt template, axis prompt, request body, and response body from operator-side logs (which the operator chooses what to retain) and compares against the signed sidecar. The sidecar is proof; the operator-side logs are evidence; together they answer the dispute.

## Wire format

The sidecar is a JSON object emitted alongside the render receipt, with the same Ed25519/JCS/detached-JWS envelope shape:

```json
{
  "schema": "sum.model_call_evidence.v1",
  "kid": "sum-render-2026-04-27-1",
  "payload": {
    "render_id": "<sha256-trunc-16-hex matching the render_receipt's render_id>",
    "issued_at": "2026-04-27T18:00:00.000Z",
    "prompt_template_hash": "sha256-<hex>",
    "axis_prompt_hash": "sha256-<hex>",
    "request_body_hash": "sha256-<hex>",
    "provider_response_id_hash": "sha256-<hex>",
    "model_reported": "claude-haiku-4-5-20251001",
    "temperature": 0,
    "max_tokens": 1234,
    "output_tome_hash": "sha256-<hex>"
  },
  "jws": "<protected-b64>..<signature-b64>"
}
```

`render_id` ties the sidecar to a specific render receipt; the verifier asserts the two values match before trusting the binding.

The hash fields each cover a distinct surface of the model call:

| Field | Hashed bytes | Why |
|---|---|---|
| `prompt_template_hash` | The system-prompt template the Worker assembled (e.g. `worker/src/render/axis_prompts.ts::buildSystemPrompt(slider_position)`'s output). Pins the prompt-engineering layer's behaviour at issuance time. | Without this, an attacker who modifies the prompt-engineering code post-deploy could claim "the same template produced this tome"; with it, the sidecar fails verification because the rebuilt template no longer hashes the same. |
| `axis_prompt_hash` | The per-axis fragment selection (e.g. `formality=0.7` → "academic, third-person register"). Pins the slider-position-to-prompt mapping. | Mirror of the above for the per-axis prompt fragments. Lets a researcher reconstruct exactly which prompt fragments were applied. |
| `request_body_hash` | The full HTTP request body sent to the LLM provider (model, prompt, temperature, max_tokens, etc.). | Catches the case where the prompt is correct but a non-prompt parameter (temperature, response_format, tools) was changed silently. |
| `provider_response_id_hash` | The provider's response identifier (e.g. Anthropic's response `id`), hashed. | Lets a future audit cross-reference against the operator's provider-side billing log without exposing the raw ID. The hash is still a stable lookup key. |
| `model_reported` | (Plaintext, NOT hashed.) The model field the API actually echoed back. | Already in the render receipt's payload; duplicated here so the sidecar is self-contained. |
| `temperature`, `max_tokens` | (Plaintext, NOT hashed.) The non-prompt parameters that shape stochasticity / length. | Plaintext because they're already short scalars; a hash adds no privacy and loses readability. |
| `output_tome_hash` | sha256 of the rendered tome bytes. Same as the receipt's `tome_hash`. | Cross-reference. The sidecar's binding is `(prompt → response → tome)`; the receipt's binding is `(triples + sliders → tome)`. They share `output_tome_hash`. |

## Verification flow (when implemented)

A consumer with both a signed receipt and the matching signed model-call evidence:

```python
# 1. Verify the receipt.
receipt_result = verify_receipt(receipt, jwks)

# 2. Verify the sidecar (same JOSE-envelope core as receipt;
#    a future sum_engine_internal.model_call_evidence module will
#    expose verify_model_call_evidence with the same shape as
#    verify_receipt and verify_trust_manifest).
evidence_result = verify_model_call_evidence(evidence, jwks)

# 3. Cross-bind: the two MUST share render_id and tome_hash.
assert receipt_result.payload["render_id"] == evidence_result.payload["render_id"]
assert receipt_result.payload["tome_hash"] == evidence_result.payload["output_tome_hash"]

# 4. (Optional, dispute-time) The operator presents original prompt
#    template / axis prompt / request body / response ID from logs;
#    the consumer rehashes and compares against the sidecar's
#    *_hash fields. Match → process evidence intact; mismatch →
#    operator's logs and the signed sidecar disagree, the dispute
#    is real.
```

## Privacy guarantees

The sidecar is **safe to publish** alongside the receipt. None of its fields leak prompt content, response content, or user-input prose. Specifically:

- `prompt_template_hash` reveals only that "the issuer used template X" without revealing what X says — useful as a marker that a known-good template was applied, opaque otherwise.
- `axis_prompt_hash` is similarly opaque without operator-side knowledge of the axis-prompt table.
- `request_body_hash` covers the full request body INCLUDING any prompt content, which is opaque under sha256.
- `provider_response_id_hash` is the provider's internal ID hashed — useful for cross-referencing against billing without exposing the raw ID.
- `model_reported` and `temperature` / `max_tokens` are public anyway (they're already in the receipt's payload).
- `output_tome_hash` is the rendered tome's hash — the tome itself is what the user receives, so this isn't leaking new information.

The sidecar therefore can be:
- Logged to plaintext audit trails alongside receipts.
- Published in trust-root manifest extensions or anchored to Rekor.
- Archived long-term without exposing user content.

## Trust scope

A verified model-call evidence sidecar PROVES:

| Claim | Defence mechanism |
|---|---|
| The issuer attested to specific hash values for the prompt template, axis prompt, request body, response ID, and output tome at the time of the render. | Ed25519 signature; `verify_model_call_evidence` rejects on any signed-field tampering. |
| The sidecar binds to a specific render receipt (same render_id, same tome_hash). | Cross-binding asserted by the verifier in step 3 above. |
| The model parameters (model_reported, temperature, max_tokens) at render time were these values. | Plaintext signed fields. |

A verified sidecar DOES NOT PROVE:

| Non-claim | Why not |
|---|---|
| The prompt template is correct / well-designed / non-adversarial. | Hashes don't speak to content quality. |
| The model's output was deterministic. | Temperature is recorded; if ≠0, two calls with the same prompt produce different outputs (and different `output_tome_hash` values). |
| The provider returned the model the receipt claims. | The receipt's `model_reported` field is what the provider echoed back; the sidecar duplicates it but doesn't independently verify. |
| The operator's process around the call was correct (no human-in-the-loop intervention, no caching, etc.). | The sidecar binds the call's bytes; what happened around the call is the operator's claim. |

## Implementation gating

This design lands as `docs/MODEL_CALL_EVIDENCE_FORMAT.md` (this file). Implementation does NOT land in R0.5. It requires:

1. **v0.9.B / v0.9.C / v0.9.D verifiers all closed.** Don't ship a side-channel before the primary verification surface is fully closed — three of four are merged; v0.9.D is the most recent.
2. **Operator opt-in flag on `/api/render`.** The sidecar is opt-in because emitting it costs the Worker a few CPU-cycles + a JWS signature per call; not all operators want the per-call cost.
3. **A new `sum_engine_internal/model_call_evidence/` module** mirroring the trust_root and render_receipt module shapes — schema-aware wrapper over `verify_jose_envelope`, expected to be ~50 LOC since the shared core does the work.
4. **Worker-side hashing.** `worker/src/render/sign.ts` (or a new `worker/src/render/evidence.ts`) computes the sha256s and assembles the sidecar payload.

When all four are in place, the implementation cycle is one PR: signing on the Worker side + verifier on the Python / JS side + a small fixture set. The design above is the contract that PR implements.

## Cross-references

- [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) §5 — trust scope of the receipt; the sidecar fills the "binds tome to triples" gap §5 explicitly admits.
- [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) §1.8 — render receipt cryptographic binding; this sidecar extends the bound surface.
- [`docs/INCIDENT_RESPONSE.md`](INCIDENT_RESPONSE.md) case 7 — LLM provider silently changes served-model behavior; the sidecar's `request_body_hash` + `provider_response_id_hash` are the forensic binding that detects this case.
- [`docs/NEXT_SESSION_PLAYBOOK.md`](NEXT_SESSION_PLAYBOOK.md) R0.5 — playbook entry that scoped this design.
