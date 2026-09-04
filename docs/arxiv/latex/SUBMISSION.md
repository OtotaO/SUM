# arXiv submission checklist: Paper 1

Kit generated 2026-07-02 from `docs/arxiv/PAPER1_DRAFT.md` (final-form v0,
2026-06-09) and refreshed 2026-09-04 against the current draft.

The 2026-09-04 refresh folded in every draft change merged after the kit was
built (PRs #468, #474, #481), which the kit had been missing for seven weeks:
the Section 4 unit disclosure (the scorer's unit is a punctuation-or-newline
segment, not a linguistic sentence), the SynthID citation in full including
SynGuard, the SCITT / COSE Receipts positioning, the CARE and Kotte conformal
comparisons, a new Section 11 (Artifact availability), and four bibliography
entries. It also corrected one sentence that was wrong in the draft: the
artifact statement claimed "two independent reimplementations" of the meaning
receipt verifier and then named one. There is one (the JS verifier), exercised
in two runtimes; both files now say so.

The draft `PAPER1_DRAFT.md` was edited in the same pass to carry the two
reference titles this kit resolved in July, so the two files no longer
disagree. Direction matters here: `main.tex` was AHEAD of the Markdown draft
on those titles and on the author block. Do not regenerate this kit from the
Markdown wholesale; patch it.

## Title

Chain-of-Custody for AI-Transformed Text: Signed, Replayable, Distribution-Free Receipts for What a Transformation Preserved

## Categories

- **Primary: cs.CR** (Cryptography and Security). Sanity check against
  content: confirmed. The paper's spine is a signature scheme composition
  (Ed25519 / JCS / detached JWS / JWKS), an explicit threat model with
  attacker capabilities, and verification preconditions P1/P2. The statistics
  serve the attestation claim, not the other way around. cs.CR is right.
- **Cross-list: cs.LG** (the conformal / distribution-free bound machinery,
  per the outline's stated venue plan).
- Optional second cross-list to consider: **cs.CL** (the judges, corpora, and
  transformations are all NLP objects). The outline does not name it; operator
  call.

## Abstract (plain text, ready to paste into the arXiv abstract field)

Character count of the paragraph below: 1,589, which fits the 1,920-char
field. (The draft's own note says 1,426; that figure counts its Markdown
source differently. Both are under the limit.) The Contributions list stays
in the paper body, as the draft prescribes. Em dashes below are the draft's
own punctuation, kept verbatim.

Two questions about AI-transformed text lack a portable, offline-verifiable answer: who transformed this, and what did the transformation preserve? Provider disclosure (EU AI Act Article 50) and image-centric content provenance (C2PA, SynthID) do not cover text that has been paraphrased, summarized, or translated — a manifest detaches on copy, a watermark is defeated by rewriting. We present a receipt family that answers both questions for text. A signed, offline-verifiable receipt attests a transformation (Ed25519 over RFC 8785 JCS-canonical bytes, detached JWS, JWKS keys); on top of it, a distribution-free, replayable certificate bounds the expected meaning-loss of the transformation under a named judge. The certificate replays offline over a committed integer loss vector — a third party re-runs the conformal certifier and reproduces the bound to the bit — while the proof boundary stays explicit: it bounds a named proxy marginally, under exchangeability, never per-document truth and never "meaning" itself. We demonstrate on two public-domain corpora: certified expected meaning-loss <= 0.645 (95%) for abstractive summarization of US Congressional bills (BillSum, CC0; n=64) and <= 0.412 for EN->FR translation (opus-100; n=64), with 39/64 faithful translations scoring exactly zero meaning-loss (under a binary entailment judge at a 0.5 cut) despite near-zero lexical overlap — the property no watermark or lexical scheme can certify. The thesis is attest, don't detect: a signature survives an adversary with a thesaurus; a statistical "is-this-AI" classifier does not.

Note: arXiv's abstract field accepts inline TeX; if preferred, replace
`<=` with `$\le$` and `EN->FR` with `EN$\to$FR`.

## Suggested license

**arXiv.org perpetual, non-exclusive license 1.0** (the default). Rationale:
it keeps every later option open (journal or conference submission, or
relicensing) while satisfying arXiv. Choose **CC BY 4.0** only if maximal
downstream reuse matters more than venue flexibility; some journals dislike a
prior CC BY preprint. Do not choose CC0 for the paper itself unless that is a
deliberate decision, it is irrevocable.

## Compile status

**Recompiled successfully 2026-09-04 with tectonic** (`tectonic main.tex`,
exit 0). Output: a 10-page PDF (8 before the refresh), no TeX errors, no
undefined references or citations (log inspected), and exactly one overfull
box (10.98pt, at the Section 5 verification paragraph), which predates the
refresh. The long typewriter paths introduced in Section 11 needed
`\allowbreak` hints and a `\sloppy` scoped inside that one `itemize` to keep
them out of the margin; both are plain LaTeX, no new package. All 20
corrections were verified present in the rendered PDF text, not merely in the
source.

(Earlier: compiled 2026-07-12 with tectonic 0.16.9, 8-page PDF, same clean
log.) The earlier static self-check still corroborates the source (balanced
environments and braces, even math-dollar count, every `\cite` key has a
matching `\bibitem`, every `\ref`/`\eqref` has a matching `\label`, no bare `&`
outside tabulars, no unescaped `%`, 100% ASCII). All commands used are standard
LaTeX / amsmath / amsthm / booktabs / hyperref.

One packaging line was added for local compilation and is **arXiv-safe**:
`\usepackage{iftex}` followed by `\ifxetex\PassOptionsToPackage{xetex}{hyperref}\fi`
before `hyperref`. Tectonic's engine is XeTeX-based; with `\pdfoutput=1` set,
`iftex` misroutes hyperref to its pdfTeX driver, which references a PDF-version
primitive XeTeX lacks. The guard selects hyperref's xetex driver only under an
XeTeX engine. On arXiv's AutoTeX (real pdfTeX) `\ifxetex` is false, so the line
is a no-op and the pdfTeX driver loads exactly as before. Nothing scientific or
typographic in the rendered paper changes. `\pdfoutput=1` remains in the first
5 lines as arXiv expects.

To reproduce locally: `cd docs/arxiv/latex && tectonic main.tex` (BSD-licensed,
self-contained, downloads packages on first run). On a machine with TeXLive:
`pdflatex main.tex` twice, for the cross-references.

## Operator steps to submit

1. ~~Fill in the author block in `main.tex`.~~ **Done** (PR #405): Umar
   Syed, Independent Researcher, ototao@pm.me.
2. ~~Resolve the two unverified reference titles.~~ **Done** (PR #405,
   verified against the live arXiv abstract pages 2026-07-16).
3. Compile locally twice on any machine with TeX (or trust AutoTeX): check
   the rendered PDF once, end to end.
4. **Endorsement: still open, and it is the binding blocker.** See the
   status section below before spending effort here.
5. Run `./make_tarball.sh` and upload `paper1.tar.gz` (or upload `main.tex`
   alone, it is fully self-contained with no figures or .bib).
6. In the submission form: paste title and abstract from this file, set
   primary cs.CR, cross-list cs.LG, pick the license, submit, and review the
   AutoTeX-produced PDF before finalizing.
7. Timing note from the repo's own planning: an announced preprint helps the
   grant narrative (NLnet decision ~Sept); the outline flags timing as
   operator sub-decision 8.5.

## Endorsement status (as of 2026-09-04)

A first-time cs.CR submitter needs an endorsement. arXiv issues the submitter
a one-time endorsement code at submission start, which the submitter sends to
an eligible endorser; the endorser enters it at
`https://arxiv.org/auth/endorse`. Endorsement only attests that the
submission is legitimate research. It is not review or agreement.

**A code was issued to the operator in July 2026** (the value is not recorded
in this public file; it is in the operator's private notes). Codes expire, so
confirm it is still live before reusing it, and expect to request a fresh one.

**Two requests were sent on 2026-07-17.** Neither produced an endorsement.

- **Request 1 replied and is the useful data point.** The recipient read the
  draft, judged it relevant to cs.CR, and *attempted the endorsement*, but
  arXiv refused: the account did not meet the category's eligibility bar.
- **Request 2 has no recorded reply** as of 2026-09-04.

**The rule worth remembering, learned at the cost of one request:** being a
published author in the area is *not* sufficient. arXiv requires the endorser
to have a minimum number of the category's own submissions inside a trailing
time window (the refusal in request 1 was 2 qualifying papers against a
threshold of 3). Screen a candidate on *recent, repeated cs.CR submission
volume*, not on topical fit or seniority. A perfect topical match with two
cs.CR papers cannot endorse; a less perfect match with a steady cs.CR record
can.

Practical consequence: prefer someone who posts to cs.CR routinely. Ask one
person at a time, and ask them to check eligibility at
`https://arxiv.org/auth/endorse` before writing anything substantive.

## Open questions (not resolvable mechanically; answer before upload)

1. **Author name, affiliation, email: RESOLVED 2026-07-16** (PR #405). The
   Markdown draft still has no author block; `main.tex` carries the real one.
   This is a deliberate one-way difference, not drift.
2. **Two reference titles: RESOLVED 2026-07-16** (verified against the live
   arXiv abstract pages; bibliography updated, bracketed notes removed).
   arXiv:2604.23280 = Otsuka, Toyoda, Leung, "AI Identity: Standards, Gaps,
   and Research Directions for AI Agents". arXiv:2605.05503 = Ameen, Islam,
   Mahmud, Hamid, "Chainwash: Multi-Step Rewriting Attacks on Diffusion
   Language Model Watermarks".
3. **Bibliography completeness: RESOLVED 2026-09-04, and it was not
   cosmetic.** All eight author-less arXiv entries were resolved live against
   their abstract pages and now carry full author lists in printed order, plus
   the peer-reviewed venue where one exists (NeurIPS 2025 for
   `confsumm2025` and `advparaphrase2025`, COLM 2025 for
   `verifyingverifiers2025`). Every id resolved to the work described: there
   are no fabricated references.

   **Five of the eight titles were wrong.** They had been written as
   descriptive paraphrases rather than copied from the source, and four
   carried an invented subtitle:

   | key | printed | actual |
   | --- | --- | --- |
   | `toolreceipts2026` | "...signed tool-call receipts for agents" | "...Practical Hallucination Detection for AI Agents" |
   | `verifyingverifiers2025` | "...label noise in fact-verification benchmarks" | "...Unveiling Pitfalls and Potentials in Fact Verifiers" |
   | `eigenai2026` | "...deterministic-inference attestation" | "...Deterministic Inference, Verifiable Results" |
   | `advparaphrase2025` | "...universal training-free evasion of AI-text detectors" | "...A Universal Attack for Humanizing AI-Generated Text" |
   | `condfact2026` | title truncated, subtitle "via Conformal Sampling" dropped, spurious hyphen | full title restored |

   Three prose defects came out of the same pass and are fixed in Sections 9
   and 10:

   - **`toolreceipts2026` was cited on the wrong side of the argument.** It sat
     in the list of "signed-but-semantics-disclaiming" receipts. It is the
     opposite: it cross-references per-claim receipts specifically to detect
     hallucination and reports detection rates by error type, so as printed it
     undercut the very sentence it was supporting. Its receipts are also
     HMAC-authenticated, a symmetric MAC with no third-party verifiability,
     which should never have been grouped under an unqualified "signed"
     alongside Ed25519/JWS in a cs.CR venue. It now has its own bullet naming
     it as the closest counterexample and stating both distinctions.
   - **`verifyingverifiers2025` was cited for a claim it does not make.** The
     label-noise *ceiling on a bound* is our inference; Seo et al. report the
     empirical premise (roughly 16% ambiguous or mislabelled data shifts model
     rankings). Section 10 now attributes each half explicitly.
   - **"concurrent" was a priority claim we cannot support.** `condfact2026`
     is dated 28 March 2026, months before this arc. Section 9 now names the
     date instead.
4. **RFC 8032 authorship: CONFIRMED at source 2026-09-04.** rfc-editor.org
   lists exactly two authors, S. Josefsson (SJD AB) and I. Liusvaara
   (Independent), "Edwards-Curve Digital Signature Algorithm (EdDSA)",
   January 2017. The kit's correction stands; Bernstein et al. keep the
   parenthetical scheme credit. Also checked: RFC 8032 is IRTF Informational,
   NOT Standards Track, and the paper nowhere claims otherwise (it cites it
   only as "Ed25519 (RFC 8032)"). Nothing to change.
5. **Paladin-mini: RESOLVED 2026-09-04.** Now cited as arXiv:2506.20384,
   D. Ivry and O. Nahum, 25 June 2025. Verifying it changed the prose, which
   is why this mattered: the draft said lighter judges "match or surpass"
   MiniCheck, but the Paladin-mini abstract names no comparison at all, and
   the body compares against *Bespoke*-MiniCheck-7B on the *grounding
   benchmark the paper itself introduces*, winning 91.8% to 78.2% on average
   while LOSING the time-and-date subset 82.0% to 90.0%. Section 10 now
   states that scope instead of the bare comparative.
6. **Contributions placement.** The draft holds the Contributions list inside
   its Abstract section but its own drafting note says the abstract field is
   the prose only and "the Contributions list stays in the body". I placed it
   as an unnumbered paragraph at the end of Section 1. Confirm or move.
7. **Coverage table header "tl": RESOLVED 2026-09-04.** The caption of
   Table 2 now defines `tl` as the true loss rate of the generating
   distribution and *target* as the nominal coverage, and states the read:
   a method is valid where its row meets its target.
8. **Draft meta-commentary.** The draft's italic header note and the closing
   "Drafting notes for the operator" block are preserved as LaTeX comments at
   the top and bottom of `main.tex` (not rendered). The unfinished items in
   those notes remain open: coverage figure / system diagram (iii), a single
   running example (iv), first-person voice reconciliation (vi).
9. **Optional cs.CL cross-list** (see Categories above).

## Ambiguities in the Markdown and what the conversion did

- Markdown `**bold**` / `*italic*` mapped to `\textbf` / `\emph`; section
  cross-references like "§8" mapped to `\S\ref{...}` so they stay correct if
  sections renumber.
- The two Markdown tables became booktabs tables with captions and labels
  (the draft's tables had neither; captions were authored minimally and add
  no claims).
- Equation (1)'s `\tag{1}` is preserved so the §6 reference "certifies (1)"
  resolves identically.
- Algorithm 1's blockquote became an `amsthm` definition-style `Algorithm`
  environment; content unchanged.
- The one RFC 7515 + 7517 combined bullet became two bibitems.
- All honesty caveats and proof-boundary statements were carried over
  verbatim in meaning; nothing was strengthened, shortened, or reworded
  beyond LaTeX escaping.
