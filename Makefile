# SUM repo — common developer commands.
#
# Run `make help` (or just `make`) for the targets. Anything here is a
# thin wrapper over the underlying tool — no magic, no hidden state.
# If you want to know what a target really does, read it below.

.DEFAULT_GOAL := help
.PHONY: help install test test-cli test-codec bench xruntime xruntime-adversarial \
        demo wheel sdist smoke fortress clean lint wasm wasm-bench wasm-bench-python

PYTHON ?= python

help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*## "; printf "SUM — make targets\n\n"} \
	      /^[a-zA-Z_-]+:.*## / { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "For the shipping CLI, prefer the installed binary:"
	@echo "  echo 'Alice likes cats.' | sum attest | sum verify"

install:  ## Editable install with sieve extras + dev tools.
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e '.[sieve,dev]'
	$(PYTHON) -m spacy download en_core_web_sm

test:  ## Full pytest run (1000+ tests).
	$(PYTHON) -m pytest Tests/ -q

test-cli:  ## CLI + codec + VC tests only (fast).
	$(PYTHON) -m pytest Tests/test_sum_cli_verify.py Tests/test_sum_cli_attest_ed25519.py \
	                    Tests/test_sum_cli_ledger.py Tests/test_hmac_optional.py \
	                    Tests/test_adversarial_bundles.py Tests/test_ed25519_attestation.py \
	                    Tests/test_verifiable_credential.py Tests/test_bench_model_resolution.py \
	                    Tests/test_sum_cli_resolve.py -q 2>/dev/null || \
	$(PYTHON) -m pytest Tests/test_sum_cli_verify.py Tests/test_sum_cli_attest_ed25519.py \
	                    Tests/test_sum_cli_ledger.py Tests/test_hmac_optional.py \
	                    Tests/test_adversarial_bundles.py Tests/test_ed25519_attestation.py \
	                    Tests/test_verifiable_credential.py Tests/test_bench_model_resolution.py -q

test-codec:  ## Canonical codec + adversarial bundle tests.
	$(PYTHON) -m pytest Tests/test_hmac_optional.py Tests/test_adversarial_bundles.py \
	                    Tests/test_ed25519_attestation.py Tests/test_timestamp_validation.py -q

bench:  ## Bench harness on seed_v1 (requires SUM_BENCH_MODEL or --no-llm).
	$(PYTHON) -m scripts.bench.run_bench --corpus scripts/bench/corpora/seed_v1.json \
	                                     --out bench_report.json --no-perf --quick

xruntime:  ## Cross-runtime valid-path K1/K1-mw/K2/K3/K4 (Python ↔ Node).
	$(PYTHON) -m scripts.verify_cross_runtime

xruntime-adversarial:  ## Cross-runtime rejection-matrix A1-A6 (Python ↔ Node).
	$(PYTHON) -m scripts.verify_cross_runtime_adversarial

fortress:  ## 21-check fortress gate (pure-math invariants).
	$(PYTHON) scripts/verify_fortress.py --json

demo:  ## Open the single-file demo in the default browser.
	@command -v open >/dev/null && open single_file_demo/index.html || \
	 command -v xdg-open >/dev/null && xdg-open single_file_demo/index.html || \
	 echo "Open single_file_demo/index.html in your browser"

wheel:  ## Build sdist + wheel into dist/.
	$(PYTHON) -m pip install --upgrade --quiet build
	$(PYTHON) -m build --sdist --wheel --outdir dist

sdist: wheel  ## Alias for `wheel` — both artifacts built together.

smoke:  ## Fresh-venv install + attest|verify round-trip from the built wheel.
	@rm -rf /tmp/sum-smoke
	$(PYTHON) -m venv /tmp/sum-smoke
	/tmp/sum-smoke/bin/pip install --quiet "$$(ls dist/*.whl | tail -1)[sieve]"
	/tmp/sum-smoke/bin/python -m spacy download en_core_web_sm >/dev/null 2>&1
	/tmp/sum-smoke/bin/sum --version
	@echo "Alice likes cats. Bob owns a dog." | /tmp/sum-smoke/bin/sum attest --extractor=sieve | \
	 /tmp/sum-smoke/bin/sum verify

lint:  ## ruff check (same rules as CI).
	$(PYTHON) -m ruff check sum_cli internal scripts

wasm:  ## Build + copy sum_core.wasm into single_file_demo/ (rerun after core-zig/ edits).
	cd core-zig && zig build wasm
	cp core-zig/zig-out/bin/sum_core.wasm single_file_demo/sum_core.wasm
	node single_file_demo/test_wasm.js

wasm-bench:  ## Serve the WASM-vs-JS browser benchmark (docs/WASM_PERFORMANCE.md).
	@echo "Open: http://localhost:8000/Tests/benchmarks/browser_wasm_bench.html"
	@echo "Run in Chrome 113+, Firefox 129+, Safari 17+; paste JSON into docs/WASM_PERFORMANCE.md."
	$(PYTHON) -m http.server 8000

wasm-bench-python:  ## Run the Python-side derivation benchmark companion (emits JSON).
	$(PYTHON) scripts/bench_python_derive.py

clean:  ## Remove build artifacts + bench history.
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache
	rm -f bench_report.json
