# Algorithm registry (Phase E.1 G3)

**Status:** authoritative — every cryptographic algorithm SUM signs or verifies with appears here. Verifiers fail closed on any algorithm not listed under "current". Adding a new row is a deliberate spec PR followed by a deprecation cycle for any retiring row, not a silent dep update.

This registry is the operational counterpart to the trust-root manifest's `algorithm_registry` payload field ([`docs/TRUST_ROOT_FORMAT.md`](TRUST_ROOT_FORMAT.md) §1.1.3): the trust-root names the **currently-active** scheme; this doc is the **full lifecycle** record (deprecated rows, retired rows, planned rows).

## Why this registry exists

A verifier that accepts any algorithm advertised in a JWS protected header is one CVE away from "compromised". An attacker who can mutate the `alg` claim downgrades to a weaker scheme; if the verifier accepts the new alg, the attacker bypasses signature integrity entirely. This class of attack is the JWT/JWS history's "use of `alg: none`" / "RS256 → HS256 confusion" bug pattern.

Defence is to maintain an explicit allowlist — the registry — and **fail closed** on anything not on it. The verifier never trusts the JWS protected header's `alg` claim; it cross-checks it against the registry, and rejects with `unsupported_alg` if absent.

The registry also encodes deprecation cycles: an algorithm SUM is migrating away from sits in `deprecated` for a documented window (verifiers still accept it), then moves to `retired` (verifiers MUST reject), then drops out of the registry entirely. New algorithms enter as `planned` (verifiers don't accept yet; spec is in flight) → `current` (verifiers accept) → `deprecated` → `retired`.

## The registry

### Signature algorithms (JWS protected-header `alg` claim)

| `alg` | Status | Since | Deprecated | Retired | Notes |
|---|---|---|---|---|---|
| `EdDSA` | **current** | v0.9.A (2026-04) | — | — | Ed25519 over JCS-canonical bytes per RFC 8032; the only sig alg v1 receipts + trust-root manifests use. |

#### Verifier MUST behaviour

For every JWS protected header SUM verifies (render receipt, trust-root manifest, model-call evidence sidecar, etc.):

- Decode the protected header's `alg` claim.
- If `alg` is **not in the registry under the current row** → reject closed with `unsupported_alg`.
- If `alg` is **deprecated** → accept BUT log a warning. Operators should plan migration before the retired-on date.
- If `alg` is **retired** or **planned** → reject closed with `unsupported_alg`.
- If `alg` is missing or unparseable → reject closed (existing `header_invariant_violated` path covers this; no registry lookup needed).

### Prime schemes (CanonicalBundle prime derivation)

| Scheme | Status | Since | Deprecated | Retired | Notes |
|---|---|---|---|---|---|
| `sha256_64_v1` | **current** | v0.1.0 (2026-04-22) | — | — | The default prime scheme; SHA-256 of axiom key → first 8 bytes big-endian → next prime via 12-witness deterministic Miller-Rabin. Birthday-bound collision-safe < 2³² axioms per branch. |
| `sha256_128_v2` | **planned (cross-runtime byte-identity locked)** | — | — | — | 128-bit seed + Baillie-PSW primality. Implementation: `standalone_verifier/math.js` (Node) + `sum_engine_internal/algorithms/semantic_arithmetic.py::_deterministic_prime_v2` (Python). Cross-runtime byte-identity locked 2026-04-29 by `scripts/verify_godel_v2_cross_runtime.py` (K1-v2 + K2-v2 gate, runs on every PR). Default scheme stays `sha256_64_v1`; flipping the default is a separate operator decision requiring a `bundle_version` minor bump. Lifts the collision-safe ceiling from ~2³² to ~2⁶⁴ axioms. |

#### Migration policy (prime schemes)

- A prime scheme moves from `planned` → `current` only when a cross-runtime K-matrix gate (K1-v2, K2-v2, K3-v2, K4-v2 per Priority 3) is green.
- The previous `current` scheme moves to `deprecated` on the same release. **Both schemes remain accepted** during the deprecation window — verifiers that encounter a bundle under the deprecated scheme verify normally and surface a warning.
- The deprecation window is the longest of:
  - The render cache TTL (24 h per `worker/src/cache/bin_cache.ts::DEFAULT_TTL_SECONDS`).
  - The documented archival window for the most-recent release.
  - **One year** (the ecosystem migration time the playbook's G3 entry pins).
- After the deprecation window expires, the deprecated scheme moves to `retired` and verifiers MUST reject bundles under it. A retired row stays in the registry as historical record.

## The dual-sign migration pattern

When a sig alg in **current** moves to **deprecated** (because a successor is being introduced), the operator-side render path may emit BOTH signatures during the deprecation window:

```jsonc
{
  "schema": "sum.render_receipt.v1",
  "kid": "primary-kid",
  "payload": { ... },
  "jws": "<deprecated-alg detached jws>",          // existing verifiers
  "jws_alt": {                                     // OPTIONAL forward-compat
    "alg": "Ed25519-PostQuantum-2030",             // hypothetical successor
    "kid": "secondary-kid",
    "compact": "<successor-alg detached jws>"
  }
}
```

A v1 verifier ignores `jws_alt` (forward-compat tolerance per [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) §1.4) and verifies only `jws`. A v-next verifier verifies `jws_alt` (preferred) and falls back to `jws` (legacy). Both verifications use the registry's allowlist; the `jws_alt.alg` MUST be in the registry under `current` for the v-next verifier to accept it.

**This dual-sign envelope shape is documented but NOT YET implemented** — SUM's Worker today emits only the single-`jws` form. Dual-sign code lands in the implementation cycle that introduces the successor algorithm (no successor is currently in `planned`; that's a future decision).

## Operator-side: when to introduce a deprecation

The decision to deprecate a `current` algorithm is operator-paced and triggered by one of:

- **Public cryptanalysis advancement.** A practical attack on Ed25519's elliptic-curve foundation, or on SHA-256's collision resistance, moves the algorithm to `deprecated` immediately + sets a short retired-on date.
- **Quantum-break public demonstration.** Per the playbook G3 framing, "1.0 bundles verify forever" is undefended without a written migration story; a quantum-break public demo triggers the post-quantum migration cycle on a 1-year deprecation window.
- **Standards body recommendation.** RFC publication of a successor (e.g., RFC NNNN deprecating Ed25519 in favour of an algorithm name we don't know yet) — moves the current algorithm to `deprecated` aligned with the RFC's recommended timeline.

In each case the migration steps are:

1. Add a new row to the registry with status = `planned`. Specify the `alg` claim string + JOSE library support + key shape.
2. Add cross-runtime K-matrix gates for the new algorithm (K1-newalg, K2-newalg, etc.). Land them passing on every push.
3. Move the new row from `planned` → `current`.
4. Move the old row from `current` → `deprecated`. Set the `deprecated_on` date. Compute the retired-on date per the deprecation-window policy above.
5. (Optional) Begin emitting dual-sign envelopes per the pattern above. Lets v-next consumers preferentially verify under the new algorithm during the window.
6. After the deprecation window, move the old row to `retired`. Verifiers reject envelopes signed under the retired alg with `unsupported_alg`.

## Related design history (archived)

Two design documents previously stood as separate files in `docs/` but are now archived because their content is either subsumed by this registry or by shipping code:

- [`docs/archive/STAGE3_128BIT_DESIGN.md`](archive/STAGE3_128BIT_DESIGN.md) — full design rationale for `sha256_128_v2` (BPSW primality, 128-bit collision-resistance budget). The activation criteria are summarised in this registry; the full design history is preserved in the archive for any reader who needs the byte-level rationale.
- [`docs/archive/NLI_MODEL_REGISTRY.md`](archive/NLI_MODEL_REGISTRY.md) — the NLI model registry that pins which entailment models the slider bench accepts. Today's entailment-model contract lives at the top of `sum_engine_internal/ensemble/live_llm_adapter.py` (pinned-snapshot list the harness raises `SystemExit` on) and in `docs/SLIDER_CONTRACT.md` §"Layered fact-preservation metrics"; the archived doc has the longer-form rationale.

## Cross-references

- [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) §6 — key rotation; §6.1 — revocation surface (G3 revocation, complementary to crypto-agility).
- [`docs/TRUST_ROOT_FORMAT.md`](TRUST_ROOT_FORMAT.md) §1.1.3 — `algorithm_registry` payload field; mirrors this doc's "current" rows so a downstream consumer can verify what scheme a release used without hand-walking the registry.
- [`docs/PROOF_BOUNDARY.md`](PROOF_BOUNDARY.md) §1.8 — render-receipt cryptographic binding; the `alg` invariant the verifier enforces is what this registry feeds.
- [`docs/NEXT_SESSION_PLAYBOOK.md`](NEXT_SESSION_PLAYBOOK.md) G3, Priority 3, Phase D — the playbook entries that scope this registry. Phase D's "1.0 stability contract" pins crypto-agility as a load-bearing piece of the 10-year promise.
- [`docs/INCIDENT_RESPONSE.md`](INCIDENT_RESPONSE.md) — operator runbook for the corresponding crisis cases (cryptographic break, quantum demo).
