"""Internal SUM engine modules — not a stable public API.

The CLI (``sum_cli.main``) imports from here; downstream integrations
should depend on the CLI contract (``sum attest`` / ``sum verify``
exit codes and the CanonicalBundle JSON schema) rather than import
these modules directly. Internal refactors may move or rename
submodules between minor versions.
"""
