# verify-example — render once, verify in three runtimes

A minimal, copy-pasteable demonstration of SUM's cross-runtime trust
loop: render a signed bundle with the **published** `sum-engine` package,
then verify the *same bytes* under three independent runtimes. The point
is that the verdict does not depend on the runtime that produced it.

## What you need

- Python 3.10+ with the published package and its offline extractor:

  ```bash
  pip install 'sum-engine[sieve]'
  python -m spacy download en_core_web_sm   # one-time, for the [sieve] extractor
  ```

  `[sieve]` is the deterministic, offline (spaCy) extraction path — no
  API key and no network at render time.

- Node.js 16+ for the standalone verifier (`standalone_verifier/verify.js`
  has **zero npm dependencies** — BigInt + built-in `crypto` only). Node
  18.4+ additionally verifies Ed25519 signatures via WebCrypto when a
  bundle carries one; this example's bundle is unsigned, so any Node 16+
  reconstructs and checks the state integer.

## Run it

From the repository root:

```bash
bash examples/verify-example/run.sh
```

The script:

1. **Renders** `prose.txt` with the published `sum attest --extractor sieve`,
   writing a `CanonicalBundle` to `bundle.json`.
2. **Verifies in Python** with `sum verify --input bundle.json`.
3. **Verifies the same bytes in Node** with
   `node standalone_verifier/verify.js bundle.json`, which independently
   reconstructs the Gödel state integer from the canonical tome and
   checks it matches the Python-exported state.
4. **(Browser, manual)** the same `bundle.json` verifies under WebCrypto
   in `single_file_demo/index.html` (Chrome / Firefox / Safari) — open
   the page and paste the bundle.

`bundle.json` is a generated artifact; it is not checked in (the
timestamp differs on every run).

## What you'll see

Python (`sum verify`) prints, for this example's prose
(`Alice likes cats. Bob knows Python.`):

```
sum: ✓ verified 2 axiom(s), state integer matches (hmac=absent, ed25519=absent, extractor=sieve (verifiable))
```

Node (`standalone_verifier/verify.js`) prints `✅ WITNESS VERIFICATION
PASSED` and reports the reconstructed state integer matching the
Python-exported state digit-for-digit.

## What this proves (and what it does not)

- **Proven:** the canonical bundle's Gödel state integer reconstructs
  identically across Python and Node — the semantic content is
  runtime-independent, not a Python-specific artifact.
- **This example's bundle is unsigned** (no HMAC, no Ed25519), so the
  verifiers report `ed25519=absent` / `hmac=absent`. To exercise the
  public-key signature path, attest with an Ed25519 key:
  `sum attest --ed25519-key keys/issuer.pem < prose.txt`, then both the
  Python verifier and Node 18.4+ verify the signature too.
- A verified bundle attests *what was extracted and that the canonical
  state matches* — it does **not** assert the truth of the content. See
  [`docs/PROOF_BOUNDARY.md`](../../docs/PROOF_BOUNDARY.md) and
  [`docs/RENDER_RECEIPT_FORMAT.md`](../../docs/RENDER_RECEIPT_FORMAT.md)
  §5 for the exact trust scope.
