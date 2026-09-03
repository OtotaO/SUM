import asyncio
import importlib.util
import inspect
import os

import pytest

# Third-party modules that whole test FILES gate on with a module-level
# ``pytest.importorskip``. When one of them is absent, pytest does not fail:
# it silently drops every test in those files from collection and the run
# still reports green. That is how the shipped ``sum_verify`` SDK suite, the
# five MCP server suites, the Hypothesis property suites and the verifier
# totality sweep (17 files, 574 tests after parametrization) went unrun in
# per-PR CI while the job stayed green at 2330 collected.
#
# Value = the extra that provides the module, for the failure message.
_REQUIRED_TEST_MODULES = {
    "joserfc": "[verify] / [receipt-verify]",
    "mcp": "[mcp]",
    "hypothesis": "[dev]",
}

# Deliberately NOT the extras' pins repeated by hand: the recipe below is the
# single place CI and a contributor both read, and it stays correct when
# pyproject's pins move.
_INSTALL_RECIPE = 'pip install -e ".[verify,mcp,dev]"'

# Escape hatch for a run that is knowingly partial (a bisect, a minimal
# container). Set it and the guard steps aside.
_OPT_OUT_ENV = "SUM_ALLOW_PARTIAL_TEST_ENV"


def _is_whole_tree_run(config) -> bool:
    """True when this invocation asks for directories, not named test files.

    The full-suite job runs ``pytest Tests/``; the targeted smoke jobs run
    ``pytest Tests/test_render_receipt_verifier.py`` and friends inside
    deliberately minimal venvs. Only the former promises to run everything,
    so only the former is held to the complete dependency set.
    """
    args = [str(a).split("::")[0] for a in config.args]
    if not args:  # bare ``pytest``: testpaths (= Tests) applies
        return True
    return all(os.path.isdir(a) for a in args)


def _missing_required_modules() -> list:
    missing = []
    for name in _REQUIRED_TEST_MODULES:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return missing


def _guard_full_collection(config) -> None:
    """Red the run when CI would silently under-collect.

    Keyed on ``CI`` (GitHub Actions sets it on every job) rather than on a
    bespoke env var, so the guard cannot be switched off by editing the
    workflow step that is exactly what regressed. Local runs are untouched.
    """
    if not os.environ.get("CI"):
        return
    if os.environ.get(_OPT_OUT_ENV):
        return
    if not _is_whole_tree_run(config):
        return
    missing = _missing_required_modules()
    if not missing:
        return
    detail = ", ".join(f"{n} ({_REQUIRED_TEST_MODULES[n]})" for n in missing)
    raise pytest.UsageError(
        "Whole-suite CI run is missing test-only dependencies: "
        f"{detail}.\n"
        "Those modules gate 17 test files behind a module-level "
        "pytest.importorskip, so without them the suite collects green while "
        "never running the shipped sum_verify SDK, the MCP server, the "
        "Hypothesis property suites or the verifier totality sweep.\n"
        f"Fix the CI install step: {_INSTALL_RECIPE}\n"
        "Do NOT add test dependencies to requirements-prod.txt; that file is "
        "the production floor.\n"
        f"For a knowingly partial run, set {_OPT_OUT_ENV}=1."
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as asynchronous")
    _guard_full_collection(config)


def pytest_pyfunc_call(pyfuncitem):
    test_func = pyfuncitem.obj
    if inspect.iscoroutinefunction(test_func):
        asyncio.run(test_func(**pyfuncitem.funcargs))
        return True
    return None
