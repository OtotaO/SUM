# arXiv submission checklist: Paper 1

Kit generated 2026-07-02 from `docs/arxiv/PAPER1_DRAFT.md` (final-form v0,
2026-06-09). No repo files were modified.

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

No LaTeX toolchain exists on this machine (pdflatex, tectonic, xelatex,
latexmk, docker: all absent), so the kit was NOT compiled. A rigorous static
self-check was run instead and passed: balanced environments (stack-checked),
balanced braces, even math-dollar count, every `\cite` key has a matching
`\bibitem`, every `\ref`/`\eqref` has a matching `\label`, no bare `&` outside
tabulars, no bare `_`/`#` outside math or `\texttt`, no unescaped `%`, and the
file is 100% ASCII (no encoding surprises). All commands used are standard
LaTeX / amsmath / amsthm / booktabs. Estimated length when compiled: roughly
9 to 11 pages (11pt article, 1in margins, two tables, 27 references). Please
run one local compile (`pdflatex main.tex` twice, for the cross-references)
before upload; arXiv's AutoTeX will also compile it server-side.
`\pdfoutput=1` is already in the first 5 lines as arXiv expects.

## Operator steps to submit

1. Fill in the author block in `main.tex` (search for `[AUTHOR NAME]`).
2. Resolve the open questions below (especially the two unverified reference
   titles).
3. Compile locally twice on any machine with TeX (or trust AutoTeX): check
   the rendered PDF once, end to end.
4. arXiv account: create/log in at arxiv.org. First-time submitters to cs.CR
   typically need an **endorsement**; arXiv tells you at submission start and
   gives you an endorsement code to send to an endorser (someone with prior
   cs.CR submissions). Budget a few days for this if needed.
5. Run `./make_tarball.sh` and upload `paper1.tar.gz` (or upload `main.tex`
   alone, it is fully self-contained with no figures or .bib).
6. In the submission form: paste title and abstract from this file, set
   primary cs.CR, cross-list cs.LG, pick the license, submit, and review the
   AutoTeX-produced PDF before finalizing.
7. Timing note from the repo's own planning: an announced preprint helps the
   grant narrative (NLnet decision ~Sept); the outline flags timing as
   operator sub-decision 8.5.

## Open questions (not resolvable mechanically; answer before upload)

1. **Author name, affiliation, email.** The draft has no author block;
   `main.tex` carries a placeholder.
2. **Two reference titles are unverified.** The draft itself marks
   arXiv:2604.23280 (the provenance survey) and arXiv:2605.05503
   ("Chainwash") with "[exact title to verify before submission]". Those
   bracketed notes are preserved verbatim in the bibliography and must be
   replaced with the real titles/authors.
3. **Bibliography completeness.** The draft's reference list is explicitly
   "to be formatted"; I formatted it mechanically, but several arXiv-only
   entries lack author names, and no entry has a DOI. The draft's own note
   (v) asks for full bibliographic details before submission.
4. **RFC 8032 authorship correction.** The draft credits Ed25519's RFC to
   "D. J. Bernstein et al."; RFC 8032's actual listed authors are Josefsson
   and Liusvaara (Bernstein et al. designed the scheme). I used the real RFC
   authors and kept a parenthetical credit to Bernstein. Confirm you accept
   this factual correction.
5. **Paladin-mini has no reference entry.** Section 10 mentions it in prose
   ("e.g. Paladin-mini, 2025") and the draft's reference list omits it. I
   left it as a prose mention, faithful to the draft. Add a citation or leave
   as is.
6. **Contributions placement.** The draft holds the Contributions list inside
   its Abstract section but its own drafting note says the abstract field is
   the prose only and "the Contributions list stays in the body". I placed it
   as an unnumbered paragraph at the end of Section 1. Confirm or move.
7. **Coverage table header "tl".** Kept verbatim from the draft; presumably
   "true loss". A reviewer may ask; consider expanding it in the caption.
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
