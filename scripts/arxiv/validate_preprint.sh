#!/usr/bin/env bash
# Validate the preprint at HEAD before invoking pandoc.
#
# Catches issues that would either fail the build or produce
# malformed output. Intended to run on every PR that touches
# docs/arxiv/sheaf-detector-note-v0.md, and as a preflight before
# the operator runs build_arxiv_package.sh.
#
# Validations:
#   1. Source file exists.
#   2. Has a `## Abstract` section.
#   3. Abstract is non-empty (≥ 200 chars between `## Abstract` and
#      the next `## `).
#   4. Has a `## References` section.
#   5. Every `fixtures/bench_receipts/*.json` reference resolves to
#      a file that exists in the repo.
#   6. The receipt-citation-chain verifier passes (every cited
#      bench_digest matches the receipt's actual digest).
#   7. No `## ` headings appear inside fenced code blocks (a known
#      pandoc-confusion source).
#
# Idempotent. No external network. No pandoc required.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$REPO_ROOT/docs/arxiv/sheaf-detector-note-v0.md"
PYTHON="${PYTHON:-python3}"

fail() { echo "FAIL: $1" >&2; exit 1; }
pass() { echo "  ok: $1"; }

echo "Validating preprint: $SRC"

# 1. Source exists.
[ -f "$SRC" ] || fail "preprint source not found at $SRC"
pass "source file exists"

# 2-3. Abstract section + non-empty.
abstract_text="$(awk '
    /^## Abstract$/ { in_abs=1; next }
    in_abs && /^## / { exit }
    in_abs { print }
' "$SRC")"
[ -n "$abstract_text" ] || fail "no ## Abstract section found"
pass "abstract section present"

abstract_chars=$(printf '%s' "$abstract_text" | wc -c | tr -d ' ')
if [ "$abstract_chars" -lt 200 ]; then
    fail "abstract is suspiciously short ($abstract_chars chars; expected ≥ 200)"
fi
pass "abstract is non-empty ($abstract_chars chars)"

# 4. References section.
if ! grep -q "^## References$" "$SRC"; then
    fail "no ## References section found"
fi
pass "references section present"

# 5. Every cited receipt resolves.
missing_receipts=0
while IFS= read -r ref; do
    rel="$ref"
    abs="$REPO_ROOT/$rel"
    if [ ! -f "$abs" ]; then
        echo "  MISSING: $rel" >&2
        missing_receipts=$((missing_receipts + 1))
    fi
done < <(grep -oE "fixtures/bench_receipts/[A-Za-z0-9_./-]+\.json" "$SRC" | sort -u)

if [ "$missing_receipts" -gt 0 ]; then
    fail "$missing_receipts receipt(s) referenced from preprint do not exist on disk"
fi
pass "all referenced receipts exist on disk"

# 6. Citation-chain verifier.
if ! $PYTHON -m scripts.research.verify_preprint_receipts >/dev/null 2>&1; then
    echo "FAIL: citation-chain verifier reports drift; run:" >&2
    echo "  $PYTHON -m scripts.research.verify_preprint_receipts" >&2
    exit 1
fi
pass "citation-chain verifier passes"

# 7. No ## headings inside fenced code blocks.
in_code=0
heading_in_code=0
while IFS= read -r line; do
    case "$line" in
        '```'*) in_code=$((1 - in_code)) ;;
        '## '*) [ $in_code -eq 1 ] && heading_in_code=$((heading_in_code + 1)) ;;
    esac
done < "$SRC"
if [ "$heading_in_code" -gt 0 ]; then
    fail "$heading_in_code '## ' heading(s) appear inside fenced code blocks (would confuse pandoc)"
fi
pass "no malformed headings inside code blocks"

echo ""
echo "Preprint validation: ALL CHECKS PASSED"
