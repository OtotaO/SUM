"""sum_engine_internal.research — research-grade modules.

Modules in this package are NOT part of the production install path.
They live behind the ``[research]`` extras flag in pyproject.toml so
that ``pip install sum-engine`` does not pull research dependencies
(numpy, scipy) by default. Use ``pip install 'sum-engine[research]'``.

Modules:

    sheaf_laplacian
        v1 sheaf-Laplacian hallucination detector. Implements the
        primitives specified in docs/SHEAF_HALLUCINATION_DETECTOR.md
        §3.2 (1-dim presence stalks). Math is grounded in Gebhart,
        Hansen & Schrater (2023, AISTATS, arXiv:2110.03789) Eq. 1
        and the sheaf-Laplacian theory of Hansen & Ghrist (2019).

Stability: research-grade. APIs may change between minor releases
without backwards-compatibility guarantees. Production-stable
counterparts will live elsewhere if/when the research artifacts
benchmark sufficiently.
"""
