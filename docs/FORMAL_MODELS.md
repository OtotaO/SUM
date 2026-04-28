# Bounded formal modeling notes (R0.5)

**Status:** design-now / prototype-later. Names three small protocol surfaces in SUM where a one-page formal model in TLA+ or Alloy is high-leverage, sketches the model shape for each, and pins the strict guardrail that keeps this scoped.

## The guardrail

> "If the model becomes a research project, stop."

TLA+ and Alloy are tools for finding subtle protocol bugs in **small bounded state machines**, not for verifying SUM's prime algebra (which is already pinned by tests + Hypothesis property fuzz). Their value is exclusively in protocol surfaces where the wire-format / lifecycle decisions are easier to reason about formally than to debug after deploy.

Each candidate model below is **one page**, has **bounded state** (≤ ~10 actors, ≤ ~5 states each), and is **CI-optional** — the formal check is a discovery tool used at design time, not a continuous gate on every commit. If a model crosses one page or grows unbounded, that's the signal to put it down.

## Three candidates

### 1. JWKS rotation / cache TTL / revocation interaction

**Surface.** Render-receipt and trust-root manifests are signed with kids that rotate periodically. Verifiers cache JWKS responses for the `Cache-Control` window. G3's revocation surface (`/.well-known/revoked-kids.json`) adds a third state machine that interacts with both. The combinatorial space — kid in {current, deprecated, revoked} × verifier-cache state in {fresh, stale} × receipt age in {fresh, after-rotation, after-revocation} — has subtle correctness invariants that are easy to get wrong by hand.

**Concrete bug class this catches.** A receipt issued under a kid that is deprecated-but-not-yet-revoked, fetched by a verifier whose JWKS cache predates the deprecation: does the verifier accept or reject? What if the cache is later refreshed and the kid disappears (because the rotation grace window elapsed)? The right answer depends on the receipt's `signed_at` relative to the rotation timeline, but no one has formalized the timeline.

**Model shape.** TLA+ specification with three actors: `Issuer` (rotates kids on a configured cadence), `Verifier` (caches JWKS, periodically refreshes), `Adversary` (forges receipts under kids the issuer has deprecated). State variables: `current_kids`, `deprecated_kids`, `revoked_kids`, `verifier_cache`, `receipts_issued`, `receipts_accepted`. Invariants:

```
INV_NoAcceptanceAfterRevocation:
  \A r \in receipts_accepted :
    r.kid \notin revoked_kids \/ r.signed_at < revocation_time(r.kid)

INV_RotationGraceWindow:
  \A k \in deprecated_kids :
    rotated_at(k) + GRACE_WINDOW > now \/ k \in revoked_kids
```

**Estimated effort.** 2–4 hours to author + check via TLC for a bounded-state instance. Catches the lifecycle bug class before it ships in G3's implementation.

### 2. Delta-bundle composition semantics (Priority 6)

**Surface.** `bundle.is_delta` exists in the schema today, but what a verifier should DO when composing a delta bundle with its parent is under-specified. Priority 6 in the playbook calls for `docs/DELTA_SEMANTICS.md` to specify composition + a K5 cross-runtime harness; a formal model would surface composition bugs before they get implemented.

**Concrete bug class this catches.** Non-commutative composition: if delta A says "add fact X" and delta B says "add fact Y," does composing in order A→B vs B→A produce the same final state? (LCM is commutative, so yes — but the model verifies this rather than asserts it.) Non-associative composition: does (A→B)→C = A→(B→C)? Re-application: applying the same delta twice — does the bundle's idempotence guarantee hold? Stale parent: if a delta references parent P but the verifier has parent P', what's the failure mode?

**Model shape.** Alloy specification (Alloy is better than TLA+ for set-state systems like this; the prime algebra is naturally set-shaped). Signatures: `Bundle` (has `state_int`, `is_delta`, optional `parent_state_int`), `Verifier` (has a current `state_int`). Predicates:

```
pred composes [b: Bundle, v: Verifier] {
  b.is_delta implies b.parent_state_int = v.state_int
  v.state_int' = lcm[v.state_int, b.state_int]
}

assert ComposeIsCommutative {
  // For all bundles b1, b2 with no shared parent constraint,
  // (compose b1 then b2) yields the same state as (compose b2 then b1).
}
```

Run `check ComposeIsCommutative for 5 Bundle, 3 Verifier` to verify within bounded state.

**Estimated effort.** 4–6 hours including writing the spec, running the analyzer, interpreting findings.

### 3. Trust-root manifest update / rollback prevention (R0.2 lifecycle)

**Surface.** The trust-root manifest is published per-release. A consumer that has previously trusted manifest M_n SHOULD NOT silently accept manifest M_{n-1} that downgrades algorithm_registry.prime_scheme_current or removes a revoked kid. There's currently no formal lifecycle for this — every consumer makes its own decision about whether to trust an older manifest.

**Concrete bug class this catches.** Rollback attack: an attacker who can intercept a consumer's manifest fetch substitutes an older manifest that lists fewer revoked kids. The consumer accepts a receipt under what's now a revoked kid because the older manifest never marked it revoked.

**Model shape.** TLA+ with two actors: `Issuer` (publishes manifests M_1, M_2, ...) and `Consumer` (has accepted some prior manifest M_k, fetches new manifest M, decides whether to upgrade-to-M, ignore-M, or alert). State variables: per-manifest `version`, `revoked_kids_at_issuance`, `algorithm_registry`. Decision rules:

```
ConsumerDecision(M_old, M_new) ==
  IF M_new.version >= M_old.version
     /\ M_new.revoked_kids_at_issuance \supseteq M_old.revoked_kids_at_issuance
  THEN AcceptUpgrade
  ELSE Reject
```

The set-monotonicity rule for `revoked_kids` is the load-bearing invariant — formal modeling either proves it or finds the hole.

**Estimated effort.** 2–3 hours.

## When to implement

Each model is its own small cycle. The right ordering, in priority of leverage:

1. **Model 1 (JWKS lifecycle)** — most concrete bug class, directly informs G3 implementation. Should land before G3 ships.
2. **Model 3 (manifest rollback)** — directly informs the future v2-of-trust-root-manifest design. Should land before any consumer-side trust-store implementation.
3. **Model 2 (delta composition)** — directly informs Priority 6's `DELTA_SEMANTICS.md`. Should land at the start of P6's cycle, not after.

Each model lands as `docs/formal/<model-name>.tla` (or `.als`) + a one-paragraph addendum to the corresponding doc explaining what the model verified + the failure modes it surfaced. CI optional; the analyzer runs locally during model authoring, not in CI on every commit.

## What NOT to model

- **The prime algebra itself.** Already pinned by tests; formal modeling would be redundant.
- **JCS canonicalization.** Already pinned by Hypothesis property tests + cross-runtime fixtures; formal modeling would be redundant.
- **The full receipt-verifier algorithm.** Pinned by 15-fixture matrix + Hypothesis properties + cross-runtime smoke; formal modeling would be redundant and the state space is too large for one page.
- **Anything that requires modeling the LLM provider's behavior.** That's a research project.

The "research project" boundary is genuinely fuzzy — a model that takes a week to write or refuses to terminate within bounded state has crossed it. The three models above are deliberately scoped to surfaces where one page suffices; if the page bloats, the surface isn't appropriate for formal modeling.

## Cross-references

- [`docs/RENDER_RECEIPT_FORMAT.md`](RENDER_RECEIPT_FORMAT.md) §6 — JWKS rotation cadence; model 1 verifies the lifecycle.
- [`docs/TRUST_ROOT_FORMAT.md`](TRUST_ROOT_FORMAT.md) — trust-root manifest spec; model 3 verifies the rollback-prevention invariant.
- [`docs/NEXT_SESSION_PLAYBOOK.md`](NEXT_SESSION_PLAYBOOK.md) Priority 6 — delta-bundle semantics; model 2 informs `DELTA_SEMANTICS.md` design.
- [`docs/NEXT_SESSION_PLAYBOOK.md`](NEXT_SESSION_PLAYBOOK.md) R0.5 — playbook entry that scoped this design.
