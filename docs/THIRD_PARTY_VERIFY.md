# Have someone verify a receipt you minted — the outreach that closes the gap

The one thing no code produces is a **second party**: someone who did not
author SUM, running the verifier on a receipt they did not make, and saying so.
That single act is the paper's stated open problem and the strongest possible
signal for a grant reviewer ("this went from credible to fund-now"). This doc is
(1) the exact thing to send a third party and (2) the two commands they run.

## Step 0 — mint a receipt over real text (you, once)

```bash
# the issuer script lives in the repo, not the wheel — clone it:
git clone https://github.com/OtotaO/SUM && cd SUM
pip install "sum-engine[research,judge]"
python examples/issue_meaning_receipt.py pairs.json --out out/ --scorer nli \
    --corpus-id my-corpus-v0 --transform "summarize:my-pipeline"
```

> The person *verifying* (Step 2) needs none of this — just
> `pip install "sum-engine[verify]"`, or `python -m sum_verify --demo` to
> dry-run the bundled golden first. Only minting (this step) needs the repo.

`pairs.json` is your own data — `[["original","the AI's rewrite"], …]`, ideally
n ≥ ~32 so the bound is meaningful. You now have `out/receipt.json`,
`out/jwks.json`, `out/losses.json` (and a private key you never share).

## Step 1 — the message to send

> **Subject: 90-second favor — independently verify a cryptographic receipt?**
>
> I've been building SUM, an open-source way to attach a *signed, replayable
> receipt* to AI-transformed text — it certifies, with a distribution-free
> statistical bound, how much meaning a transformation preserved, and anyone can
> check it **offline** with a tiny dependency. I'd value your independent eyes on
> whether the verification actually works on a machine that isn't mine.
>
> Three files are attached (`receipt.json`, `jwks.json`, `losses.json`). Two
> commands, no account, no data leaves your machine:
>
> ```bash
> pip install "sum-engine[verify]"        # cryptography + joserfc only — no ML stack
> python -m sum_verify receipt.json --jwks jwks.json --losses losses.json
> ```
>
> You should see `{"verified": true, "replayed": true, …}`. That means: the
> signature checked out **and** you independently re-ran the statistical bound and
> got the same number I committed. If you have 60 more seconds, try tampering —
> change a digit in `losses.json` and re-run; it should reject with a hash
> mismatch.
>
> If it works, a single line back — *"I verified it on my machine, signature and
> bound both check out"* — is all I need. If it doesn't, the error is exactly
> what I want to see. Thank you.

## What it honestly proves (say this, don't oversell)

- ✅ The receipt was signed by the key behind `jwks.json`, the envelope is
  well-formed, and the committed losses re-certify to the stated bound by exact
  integer equality — reproduced on *their* machine.
- ❌ It does **not** prove "meaning was preserved" — it bounds a *named proxy*,
  marginally, under exchangeability (the receipt says so). Don't claim more than
  the receipt does; the honesty is the moat.

## Who to ask (highest signal first)

A technically rigorous skeptic beats a friendly non-coder every time — the
adoption sims found the warmest prospects were the statistician, the security
engineer, the maintainer. A conformal-stats academic, an application-security
engineer, an EU AI-Act / provenance contact, or the NLnet reviewer thread are
ideal: the people most able to break it are the most convinced when they can't.
