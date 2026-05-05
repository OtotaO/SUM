"""scripts.research — research-bench harnesses + helpers.

Package-level side effect (loaded on first import of any
`scripts.research.*` submodule): force single-threaded BLAS via
`os.environ.setdefault` so bench bench_digests are byte-stable
across fresh Python processes. See `_deterministic_blas` for the
full rationale + scope. The setdefault is benign for non-bench
imports: production library code in `sum_engine_internal/research/`
doesn't import from this package, so production training still
gets multi-threaded BLAS by default.

If numpy is already imported in the parent process when this
package is first imported, the env-var setdefaults become no-ops
and a warning is emitted (see `_deterministic_blas`).
"""
from __future__ import annotations

# MUST be the first import in this package so env vars are set
# before any bench's `import numpy` chain.
from . import _deterministic_blas  # noqa: F401
