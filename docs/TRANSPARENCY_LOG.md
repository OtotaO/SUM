# Transparency log — witnessed receipts (v1: public hash chain)

**Status: shipped, v1.** The log lives at
[`transparency/log.jsonl`](../transparency/log.jsonl); the tool is
[`scripts/witness_receipt.py`](../scripts/witness_receipt.py) (`append` /
`verify`); the chain is verified in CI by
`Tests/test_transparency_log.py` on every PR.

## The witness gap this closes (and how far)

A SUM receipt is self-signed: it proves the issuer's key signed exactly
these bytes, and its bound replays. What it cannot prove alone is **when it
existed**. An issuer could re-sign a doctored payload later and claim it was
always so; nothing in a detached JWS resists that. This is the audit-trail
paradox: self-signed ≠ witnessed. The field's answer (receiver attestation,
transparency logs) certifies *occurrence in time* — exactly the layer a
disputant checks first.

**v1 (this):** every issued receipt worth standing behind is appended to a
public, append-only, hash-chained log in this repository:

- Each entry commits the receipt's canonical hash (`sha256` over the RFC
  8785 JCS bytes of the full envelope) — so the entry pins the exact bytes,
  not a filename.
- Each entry commits the previous entry's hash — editing any historical
  line breaks every entry after it.
- Appends ride git commits to a public host — ordering and rough timing
  become publicly observable, and every clone is an independent replica of
  the history.

## What v1 proves, and what it does not (read both)

**Proves (against `verify`):** the log is internally consistent (chain +
per-entry hashes recompute); every witnessed receipt file still in the repo
hashes to its witnessed value; the sequence is append-only *within the file
you were handed*.

**Does NOT prove:** a trusted timestamp. A public git host is not a
timestamping authority; history can be force-pushed (detectably by clones
and caches, but not provably to a third party). v1 upgrades "trust my key"
to "trust my key, plus the public record of when I published, replicated in
every clone" — a real upgrade, honestly a **weak witness**.

**The next rung (deliberately not built yet):** mirror each entry to an
external transparency log — Sigstore **Rekor** is the natural choice
(hashedrekord entries over the same canonical hashes; independently
operated, inclusion-proof-verifiable, widely tooled). That makes the
timestamp third-party-attested and the witness strong. It is gated on one
external relying party existing, because a stronger witness with zero
relying parties is ceremony — the moment a disputant or adopter appears,
this is a one-afternoon addition (the entry hashes are already in the
exact shape Rekor wants).

## Operations

```bash
# Witness a newly issued receipt (then commit the log line):
python scripts/witness_receipt.py append path/to/receipt.json --note "..."

# Verify the whole chain + all witnessed in-repo files:
python scripts/witness_receipt.py verify
```

Rules: append-once per receipt (duplicate hashes are refused); never edit
or reorder existing lines (the chain exists to make that visible); the log
is public by design — do not witness receipts whose existence is itself
confidential.

Seeded 2026-07-10 with the two binding-gate goldens (BillSum CC0
compression, `risk_upper_bound` 0.645438 at δ=0.05, n=64; opus-100 EN→FR
translation, 0.412359 at δ=0.05, n=64) — the receipts Paper-1 stands on
are now witnessed, not merely committed.
