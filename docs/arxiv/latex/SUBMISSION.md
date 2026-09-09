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
- **Cross-list: cs.LG** (the distribution-free bound machinery,
  per the outline's stated venue plan).
- **Second cross-list: cs.CL.** Added. The judges, corpora and transformations
  are all NLP objects: the paper's headline demonstrations are summarization
  and translation, its proxy is a sentence-entailment score, and Section 10's
  judge discussion is an NLP argument. The audience that cares whether a
  transformation preserved meaning reads cs.CL, and the paper's own
  positioning is against watermarking and detection work that lives there. The
  outline did not name it; this is the reversible, wider-reach choice, and a
  cross-list costs nothing but a checkbox.

## Abstract (plain text, ready to paste into the arXiv abstract field)

Character count of the paragraph below: 1,595, which fits the 1,920-char
field. (The Markdown draft's own drafting note still says 1,426. That figure
predates several revisions and no longer describes either abstract; treat the
count on this line as the only current one.) The Contributions list stays
in the paper body, as the draft prescribes. Em dashes below are the draft's
own punctuation, kept verbatim.

For transformed text, provenance and watermarking address attribution and detection, while preservation needs a separate measurement. We present a receipt family linking signed transformation records to a named meaning-loss proxy. A signed, offline-verifiable receipt attests a transformation (Ed25519 over RFC 8785 JCS-canonical bytes, detached JWS, JWKS keys); on top of it, a distribution-free, replayable certificate bounds the expected meaning-loss of the transformation under a named judge. The certificate replays offline over a committed integer loss vector — a third party re-runs the certifier and reproduces the bound to the bit — while the proof boundary stays explicit: it bounds a named proxy marginally, over an i.i.d. calibration sample and only where that sample matches deployment, never per-document truth and never "meaning" itself. We demonstrate on two public benchmark corpora — BillSum (CC0) and opus-100 (mixed-licence, raw text not redistributed) — over each corpus's own reference outputs rather than model outputs (the mechanism is producer-indifferent): certified expected meaning-loss <= 0.646 (95%) for abstractive summarization of US Congressional bills (BillSum, CC0; n=64) and <= 0.413 for EN->FR translation (opus-100; n=64), with 39/64 reference translations scoring exactly zero meaning-loss (under a binary entailment judge at a 0.5 cut) despite near-zero lexical overlap. The thesis is attest, don't detect: a receipt authenticates a recorded transformation. Its signature does not extend coverage to subsequent rewrites, which require new linked receipts.

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

**Recompiled successfully 2026-09-05 with tectonic** (`tectonic main.tex`,
exit 0). Output: a **13-page PDF, zero overfull boxes and zero undefined
references** (12 pages before Figure 1 was added). Seven
underfull hboxes remain and are cosmetic. Long typewriter paths in Section 11
needed `\allowbreak` hints and a `\sloppy` scoped inside that one `itemize`,
and the verify algorithm needed the same treatment; both are plain LaTeX, no
new package.

(Earlier in the same arc: 10 pages with exactly one overfull box at 2026-09-04,
and 8 pages at 2026-07-12 with tectonic 0.16.9.)

(Earlier: compiled 2026-07-12 with tectonic 0.16.9, 8-page PDF, same clean
log.) The earlier static self-check still corroborates the source (balanced
environments and braces, even math-dollar count, every `\cite` key has a
matching `\bibitem`, every `\ref`/`\eqref` has a matching `\label`, no bare `&`
outside tabulars, no unescaped `%`, 100% ASCII). All commands used are standard
LaTeX / amsmath / amsthm / booktabs / hyperref, plus **`tikz` with the
`positioning`, `arrows.meta` and `calc` libraries** (added 2026-09-05 for
Figure 1). No `\includegraphics`, no `\input`, no `\write18` or shell-escape,
and no `pgfplots`: the figure is drawn in the source, so the tarball still
ships `main.tex` alone.

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
3a. ~~Re-archive in Software Heritage and update the SWHID in Section 11.~~
   **Done 2026-09-09 for the corrections merged in PR #501.** Visit 10 is
   `full`; snapshot `swh:1:snp:503d73855f8b2829ea227865c61729aa16e79d60`
   captures `main` at `b64e57b43ed01da8b9dfe6ed4f40f6cb6d016efb`.
   Section 11 in both sources cites this verified snapshot. It contains the
   corrected paper before this archive-citation and checklist update, not
   byte-identical upload source. Rebuild after updating the citation.
   **If the paper's claims, evidence or implementation change again before
   upload, reopen this step:** save merged `main`, wait for a `full` visit,
   verify the snapshot's `refs/heads/main` target, update both citations and
   recompile. The origin SWHID does not change. See the verified archive
   record below for API links and the earlier failure history.
4. **Endorsement: still open, and it is the binding blocker.** See the
   status section below before spending effort here.
5. Run `./make_tarball.sh` and upload `paper1.tar.gz` (or upload `main.tex`
   alone: Figure 1 is drawn in TikZ inside the source, so there are no image
   files and no .bib to upload alongside it).
6. In the submission form: paste title and abstract from this file, set
   primary cs.CR, cross-list **cs.LG and cs.CL** (see Categories), pick the
   license, submit, and review the
   AutoTeX-produced PDF before finalizing.
7. Timing note from the repo's own planning: an announced preprint helps the
   grant narrative (NLnet decision ~Sept); the outline flags timing as
   operator sub-decision 8.5.

## Pre-submission audit (2026-09-05)

Six independent lenses over the paper, the draft and the repository, each
finding verified by two skeptics defaulting to refutation. 21 findings
survived. Every one was then re-checked by hand against the repo before it
became an edit. The corrections, grouped:

**Wrong against the committed artifacts.**
- `62/64 pairs fall on that grid` was false *and* self-contradictory (if the
  loss takes only five values, all 64 land on them by definition). The
  committed vector is 39/8/8/2/7 = 64. Found independently by five of the six
  lenses, and checkable by a reviewer in thirty seconds through the artifact
  link. Replaced with the histogram itself.
- The three printed upper bounds were rounded **down**: certified 0.645438 and
  0.412359 were printed as 0.6454, 0.645 and 0.412, each claiming a tighter
  bound than the receipt certifies. Upper bounds now round up.
- `python -m sum_verify <receipt>`, the paper's one printed verifier command,
  exits with a usage error. Replaced with `--demo` and the explicit-files form.
- "cryptography and joserfc only" is falsified by one `pip install`; `sympy` is
  a base dependency. The README already said so correctly; the paper did not.

**Claimed more than the mathematics or the code delivers.**
- Eq. (1) was stated under **exchangeability**, but Hoeffding, Clopper-Pearson
  and empirical-Bernstein all require **independence**. An exchangeable
  all-zeros-or-all-ones sequence has coverage 1/2 against a nominal 0.95. The
  two assumptions are now separated, the counterexample is given, and the fact
  that the signed receipts' own `disclosure` strings use the weaker word is
  disclosed rather than quietly re-signed.
- The cross-runtime claim implied full parity. The JS verifier does Stage A
  only; Stage B replay is Python-only. Now stated.
- **Neither demonstration certifies an AI transformation.** BillSum's outputs
  are the dataset's human-written reference summaries, opus-100's are its
  reference translations. The fixture code was honest about this
  (`TRANSFORM = "summarize:billsum-reference"`); the paper was not. Now in 7.4.
- The paper omitted the proxy-vs-human validity that its own verifier prints on
  every run, for the exact judge in the flagship demo (near zero on abstractive
  FRANK-XSum). Now in 7.4.

**Unverifiable from the page.**
- The loss weights were never given, so neither headline number nor 7.2's
  five-value grid was derivable. Both receipts use w_r = 0.6, w_f = 0.4,
  recorded in the signed `loss_definition`.
- The 0.958 joint-coverage figure shipped with no parameters. Reproduced
  2026-09-05 and now printed with them (Clopper-Pearson, G = 3, n = 60, rates
  0.10/0.20/0.30, delta 0.05, 2000 trials, seed 17).
- Section 11 promised "every number in Section 7" is reproducible from
  committed bytes. The first repair narrowed this by excluding 7.3, "whose
  generator is not committed". A pre-merge critic then falsified the
  *exclusion*: the coverage sweep does reproduce from the shipped certifier at
  the stated data-generating process and seed 11, and the 0.958 joint figure
  reproduces from a committed test. A later pass found that eleven of the
  twelve cells matched exactly and the twelfth, Hoeffding at tl = 0.5,
  delta = .05, printed 0.992 where the kernel returns 0.99145. The table now
  prints the reproducing 0.991 and Section 11 says so. The first pass missed
  this because it compared with a tolerance instead of comparing what is
  printed. The promise is restored and
  now says how to reproduce it, which is stronger than either the original
  claim or the narrowed one. Recorded because it is the instructive failure
  here: an honesty pass can over-correct into an underclaim, and an underclaim
  is falsifiable too.

**Caught by the pre-merge critic, after the first pass.** The markdown draft
received a partial copy of the corrections and had to be repaired: a literal
half-applied edit left the fragment "This / verdicts described in Section 5.";
one round-down bound survived in 7.3's own comparison; the precondition
paragraph, the receipts-say-exchangeability disclosure and the Stage A/B limit
had landed only in the LaTeX; and the abstract clause was ungrammatical. The
lesson is mechanical: the PDF verification pass reads `main.tex` only and
structurally cannot see the draft, so the draft needs its own check.

**Typesetting.** Both tables are now referenced from the prose; the loss
definition's two displays are contiguous again; the last overfull box is gone.
The paper compiles with zero overfull boxes and zero undefined references for
the first time.

## Cold read (2026-09-05, after the audit)

Every check before this one was *targeted* at a defect class. Nobody had read
the paper end to end since about forty edits landed across three revisions, so
two reviewers read the compiled PDF cold, with no edit history, reporting only
repetition, contradiction, promises the body walks back, and flow breaks. Five
findings survived unanimous refutation, covering **four distinct defects** (both
readers independently reported the broken sentence). **Three were caused by the
repairs themselves**:

- **Section 5 contradicted Section 11.** Section 5 said "verifiable in every
  runtime" means the signature *and the bound replay*; Section 11, corrected
  earlier the same day in #494, says the cross-runtime claim covers
  authenticity and
  disclosure and explicitly *not* bound reproduction. Both sentences defined
  the same phrase, oppositely. Section 5 now matches, and separates
  well-defined-in-any-runtime from implemented-in-one-runtime.
- **The abstract formed the assumption that Section 7.4 corrects.** The
  disclosure that neither output side was model-produced sat in 7.1, 7.2 and
  7.4 but never where a reader forms the belief. The abstract now carries it.
- **A sentence was left grammatically broken** by the previous commit: "…
  Separately, / and every verification claim …". The same half-edit failure
  mode as the markdown corruption, one commit later, in the first paragraph a
  reviewer reads for artifact credibility.
- Section 9 deferred the COSE_Sign1 re-envelope to Section 10, which never
  mentioned it. Section 10 now names it as future work.

The lesson: a targeted audit cannot see coherence damage it caused. A cold
read is a different instrument, and it should be the last gate before any
future submission, not the first.

## Software Heritage re-archive: verified 2026-09-09

After PR #501 merged, [save request 2467571](https://archive.softwareheritage.org/api/1/origin/save/2467571/)
was accepted and succeeded. Visit 10 is `full`, dated
`2026-09-09T04:05:48.751000+00:00`. The
[snapshot API](https://archive.softwareheritage.org/api/1/snapshot/503d73855f8b2829ea227865c61729aa16e79d60/?branches_from=refs/heads/main&branches_count=1)
confirms `refs/heads/main` targets
`b64e57b43ed01da8b9dfe6ed4f40f6cb6d016efb`, the merge containing the independently
reviewed paper corrections. Both Section 11 citations now use
`swh:1:snp:503d73855f8b2829ea227865c61729aa16e79d60`; the origin SWHID is
unchanged. This later citation and checklist update is outside that snapshot.

The request used the [documented save API](https://docs.softwareheritage.org/devel/swh-web/uri-scheme-api-request-archival.html).
One request sufficed. A successful request alone would not have established
archive parity: the `full` visit and exact branch target were checked as well.

### Earlier refresh and intermittent failures (2026-09-06, superseded)

The earlier snapshot `swh:1:snp:93b83ca29d9b468a9f034f0bc3d4259983d63339`
captured `main` at `996707c`. It predates PR #501 and is retained here only as
historical context, not the current submission archive.

**How it was resolved is the part worth keeping, because the first diagnosis
was wrong and the wrong diagnosis produced the wrong instruction.**

Six save requests failed across 2026-09-05 and 2026-09-06, all
`save_task_status: failed` with `visit_status: not_found`, while the origin was
verifiably fetchable. From that, this file previously concluded that the
failure was "specific to this origin rather than a general outage", and told
the operator that a seventh attempt would be "pointless and impolite" and that
the next step was to ask Software Heritage.

That inference was invalid and the conclusion was false. One origin's own
history cannot discriminate an origin-specific fault from an archive-wide one;
only a comparator can. Querying comparators settles it:

- `django/django`, all `git`-type scheduled visits: `not_found` at
  2026-09-03T18:54:35, **`full` at 18:56:16**, `not_found` again at 19:37:57.
  A success sandwiched between two failures inside 43 minutes.
- `torvalds/linux`: `not_found` on 2026-09-05 and 2026-09-03, `full` before.
- `rust-lang/rust`: `not_found` on 2026-09-05, `full` on 2026-09-03.

So the fault was archive-wide and **intermittent** in Software Heritage's `git`
loader. For an intermittent fault the correct action is precisely the one the
old note forbade: retry. The next retry succeeded on the first attempt.

The lesson, recorded because it cost a day: **a conclusion drawn from a single
series, with no comparator, is not evidence, and an operational instruction
derived from it can be exactly inverted.** If a save fails again, retry a few
times before concluding anything.

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
6. **Contributions placement: CONFIRMED as-is 2026-09-05.** The list sits as
   an unnumbered paragraph at the end of Section 1, which is what the draft's
   own note prescribes ("the Contributions list stays in the body") and what
   the plain-text abstract above assumes. No change; reopen only if you want
   it bulleted.
7. **Coverage table header "tl": RESOLVED 2026-09-04.** The caption of
   Table 2 now defines `tl` as the true loss rate of the generating
   distribution and *target* as the nominal coverage, and states the read:
   a method is valid where its row meets its target.
8. **Draft meta-commentary.** The draft's italic header note and the closing
   "Drafting notes for the operator" block are preserved as LaTeX comments at
   the top and bottom of `main.tex` (not rendered). Of the three unfinished
   items in those notes, as of 2026-09-05:
   - *(iii) system diagram*: **done.** Figure 1 in Section 3 draws the receipt
     chain and the two verification stages. Every box, arrow and label is a
     phrase taken from Sections 3, 4, 5 or 11, so the figure asserts nothing
     the text does not; the one thing it had to get right is that Stage A
     spans every runtime while Stage B is Python-only today, because a figure
     showing the JS verifier reaching Stage B would visually undo the
     correction made in #494. Drawn in TikZ with `positioning`, `arrows.meta`
     and `calc` only: no external files, no shell-escape, nothing AutoTeX
     cannot build. A coverage *plot* was deliberately not added: Table 2
     already carries those numbers, is referenced from the prose, and
     reproduces from the shipped certifier, so a plot would be redundant and
     would pull in `pgfplots`.
   - *(iv) a single running example*: **deliberately left.** Threading one
     example through Sections 3 to 7 is a structural rewrite of the paper, not
     a polish pass, and it is the author's call.
   - *(vi) first-person voice reconciliation*: **deliberately left.** The
     paper mixes "we" with passive constructions. It is consistent enough to
     read cleanly and a global voice edit is an authorial decision. Neither
     the audit nor the cold read raised it.
9. **cs.CL cross-list: DECIDED 2026-09-05** (added; see Categories above).

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
