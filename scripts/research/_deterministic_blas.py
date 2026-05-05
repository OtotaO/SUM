"""
Force single-threaded BLAS at module-import time, before any numpy
or scipy import.

The v3.x research benches have a known LAPACK-threading reproducibility
issue: the same code on the same input produces slightly-different
floating-point results across fresh Python processes. The variance is
tiny (~1 ULP) but compounds across 200 SGD epochs in
`train_restriction_maps`, occasionally enough to shift bench AUCs
across the 3-decimal quantization boundary used by `bench_digest`.

Empirically: setting `VECLIB_MAXIMUM_THREADS=1` (Apple Accelerate)
+ `OPENBLAS_NUM_THREADS=1` + `MKL_NUM_THREADS=1` +
`OMP_NUM_THREADS=1` makes the v3.x benches' bench_digest **byte-
stable across fresh processes** on operator's Apple Silicon machine.
The bench `hybrid_comparison` was previously shape-pinned because of
this; the byte-digest pin can be re-instated once benches use this
helper.

Module-import side effect: this file MUST be imported before numpy.
The benches enforce this by importing this module as their FIRST
non-stdlib import. If numpy is imported first, BLAS picks up its
thread defaults from process env at numpy-import time and these
`setdefault` calls become no-ops.

Performance impact: single-threaded BLAS is slower on large matrix
ops. For our benches (16-doc corpus, ~7 triples/doc, stalk_dim=8)
the slowdown is negligible — entire bench runs in 1-2s. For
production training (which doesn't import this helper), parallel
BLAS remains the default.

Honest scope:
  - These env vars are read by numpy/scipy at import time. If
    numpy is already imported in this process, setting them here
    has no effect.
  - We use `setdefault` (not `set`) so operators who explicitly
    want multi-threaded BLAS can still get it by exporting the
    var in their shell.
"""
from __future__ import annotations

import os
import sys

_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",  # Apple Accelerate
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

for _var in _THREAD_ENV_VARS:
    os.environ.setdefault(_var, "1")

# Sanity check: warn if numpy is already imported (env vars are
# then no-ops). The benches should import this module FIRST.
if "numpy" in sys.modules:
    import warnings
    warnings.warn(
        "_deterministic_blas was imported AFTER numpy. The thread-count "
        "env vars set in this module will NOT take effect — numpy already "
        "read its BLAS thread config at import time. Move "
        "`import scripts.research._deterministic_blas` to the top of "
        "your bench script (before `import numpy`).",
        stacklevel=2,
    )
