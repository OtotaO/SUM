#!/usr/bin/env bash
# Build an arXiv submission package from the markdown preprint.
#
# Output: dist-arxiv/sum-detector-preprint.tar.gz containing
#   - main.tex       (pandoc-converted from docs/arxiv/sheaf-detector-note-v0.md)
#   - abstract.txt   (extracted from the preprint's ## Abstract section)
#   - metadata.yaml  (categories, comments, ACM/MSC classifications)
#   - README.md      (build provenance, SHA, reproduction recipe)
#
# Operator use:
#   1. Install pandoc:  brew install pandoc      (macOS)
#                       apt install pandoc        (Debian/Ubuntu)
#   2. Run:             ./scripts/arxiv/build_arxiv_package.sh
#   3. Upload to arXiv: tar.gz to the "Replace your submission with..."
#                       form during arXiv submit; categories per metadata.yaml.
#
# This script is idempotent — re-runs overwrite dist-arxiv/.
# It does NOT submit to arXiv (operator-only step).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$REPO_ROOT/docs/arxiv/sheaf-detector-note-v0.md"
OUT_DIR="$REPO_ROOT/dist-arxiv"
PACKAGE_NAME="sum-detector-preprint"

# Preflight checks.
if ! command -v pandoc >/dev/null 2>&1; then
    echo "ERROR: pandoc not found." >&2
    echo "Install: brew install pandoc  (macOS)" >&2
    echo "         apt install pandoc    (Debian/Ubuntu)" >&2
    exit 1
fi
if [ ! -f "$SRC" ]; then
    echo "ERROR: preprint source not found at $SRC" >&2
    exit 1
fi

# Capture build provenance.
BUILD_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Fresh dist-arxiv/.
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/$PACKAGE_NAME"

# Convert markdown → LaTeX.
# - --standalone produces a self-contained .tex
# - --resource-path lets pandoc find any local figures (none today)
# - --to=latex with the article class for arXiv compatibility
pandoc \
    --from=gfm+tex_math_dollars+pipe_tables \
    --to=latex \
    --standalone \
    --variable=documentclass:article \
    --variable=geometry:margin=1in \
    --variable=fontsize:11pt \
    --resource-path="$REPO_ROOT" \
    --output="$OUT_DIR/$PACKAGE_NAME/main.tex" \
    "$SRC"

# Extract abstract (lines between ## Abstract and the next ##).
awk '
    /^## Abstract$/ { in_abs=1; next }
    in_abs && /^## / { exit }
    in_abs { print }
' "$SRC" > "$OUT_DIR/$PACKAGE_NAME/abstract.txt"

# arXiv metadata YAML.
cat > "$OUT_DIR/$PACKAGE_NAME/metadata.yaml" <<EOF
# arXiv submission metadata for $(basename "$SRC")
# Generated: $BUILD_DATE
# Source SHA: $BUILD_SHA

title: |
  A cryptographically-anchored substrate for hallucination
  detection on signed render bundles

primary_category: cs.CR
secondary_categories:
  - cs.LG

# Operator fills in arXiv submit form:
authors:
  - name: ototao
    affiliation: SUM project
    email: ototao@pm.me

# arXiv "Comments" field (shown in listing):
comments: |
  Pre-circulation draft v0.1. 1500 lines. Anchored at SHA $BUILD_SHA.
  All bench digests reproducible across 3 LAPACK environments
  (Apple Accelerate / OpenBLAS Py 3.10 / OpenBLAS Py 3.12) per
  fixtures/bench_receipts/cross_machine_verification_*.json.

# MSC 2020 classification (math subject classification):
msc_class:
  - "68T50"   # Natural language processing
  - "55N31"   # Persistent homology and applications, sheaf cohomology
  - "94A60"   # Cryptography

# ACM 2012 Computing Classification:
acm_class:
  - "I.2.7"   # Natural Language Processing
  - "F.4.3"   # Formal Languages, sheaves
  - "K.6.5"   # Security and Protection
EOF

# README with build provenance + reproduction recipe.
cat > "$OUT_DIR/$PACKAGE_NAME/README.md" <<EOF
# arXiv submission package — sum-detector-preprint

Build provenance:
- Source: \`docs/arxiv/sheaf-detector-note-v0.md\` v0.1
- SHA at build time: $BUILD_SHA
- Build date (UTC): $BUILD_DATE
- pandoc version: $(pandoc --version | head -1)

## Files in this package

- \`main.tex\` — the preprint, pandoc-converted to standalone LaTeX
- \`abstract.txt\` — extracted abstract (for the arXiv abstract field)
- \`metadata.yaml\` — categories, MSC/ACM classifications, comments
- \`README.md\` — this file

## arXiv submission steps (operator)

1. Visit https://arxiv.org/submit
2. Category: Primary \`cs.CR\`, secondary \`cs.LG\`
3. Upload \`main.tex\` (or the .tar.gz of this directory if multi-file)
4. Paste contents of \`abstract.txt\` into the abstract field
5. Comments: paste from \`metadata.yaml\` \`comments:\` block
6. Submit

## Reproducibility for arXiv readers

The published bench digests are byte-stable across 3 LAPACK
environments at SHA $BUILD_SHA:

\`\`\`bash
git clone https://github.com/OtotaO/SUM.git
cd SUM
git checkout $BUILD_SHA   # or v0.6.0 tag once published
pip install -e '.[research,sieve]'
python -m spacy download en_core_web_sm
python3 -m pytest Tests/research/test_recovery_experiment_digests.py -v
\`\`\`

For cross-machine verification on Modal:

\`\`\`bash
pip install modal && modal token new
modal run scripts/research/cross_machine_verify_modal.py
\`\`\`

Expected outcome: \`BRANCH_A_THREE_ENVIRONMENTS_DIGESTS_MATCH\`.
EOF

# Tarball.
TAR="$OUT_DIR/${PACKAGE_NAME}.tar.gz"
( cd "$OUT_DIR" && tar czf "${PACKAGE_NAME}.tar.gz" "$PACKAGE_NAME" )

echo "→ wrote arXiv package: $TAR"
echo "  contents:"
tar tzf "$TAR" | sed 's/^/    /'
echo ""
echo "Operator next steps:"
echo "  1. Inspect $OUT_DIR/$PACKAGE_NAME/main.tex (pandoc output —"
echo "     confirm tables and math render correctly)"
echo "  2. Optional: run pdflatex $OUT_DIR/$PACKAGE_NAME/main.tex"
echo "     to confirm LaTeX builds locally before arXiv upload"
echo "  3. Upload .tar.gz to arXiv (https://arxiv.org/submit)"
