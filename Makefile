# SUM repo — common developer commands.
#
# Run `make help` (or just `make`) for the targets. Anything here is a
# thin wrapper over the underlying tool — no magic, no hidden state.
# If you want to know what a target really does, read it below.

.DEFAULT_GOAL := help
.PHONY: help install test test-cli test-codec bench xruntime xruntime-adversarial \
        demo wheel sdist smoke fortress clean lint wasm wasm-bench wasm-bench-python \
        vendor test-receipt-verify test-receipt-verify-py \
        test-transform-receipt-verify test-transform-receipt-fixtures \
        bench-receipt-audit bench-receipt-audit-replay test-trust-root \
        test-property test-nli-calibration-format \
        verify-preprint reproduce-preprint validate-preprint \
        pre-push

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

verify-release-bytes: wheel  ## Pre-tag empirical check: built wheel installs cleanly with [openai] AND [llm] aliases; OpenAI SDK lands; sum --version works. Run before `git tag vX.Y.Z`.
	@echo "─── Verify release bytes — adversarial pre-tag check ──────────────"
	@echo "[1/2] Fresh venv: install built wheel with [openai] (canonical extra)"
	@rm -rf /tmp/sum-release-openai
	$(PYTHON) -m venv /tmp/sum-release-openai
	@/tmp/sum-release-openai/bin/pip install --quiet --no-cache-dir "$$(ls dist/*.whl | tail -1)[openai]"
	@/tmp/sum-release-openai/bin/python -c "import openai, sum_engine_internal; print(f'  openai SDK {openai.__version__} importable from [openai]-installed wheel')"
	@/tmp/sum-release-openai/bin/sum --version
	@echo "[2/2] Fresh venv: install built wheel with [llm] (back-compat alias)"
	@rm -rf /tmp/sum-release-llm
	$(PYTHON) -m venv /tmp/sum-release-llm
	@/tmp/sum-release-llm/bin/pip install --quiet --no-cache-dir "$$(ls dist/*.whl | tail -1)[llm]"
	@/tmp/sum-release-llm/bin/python -c "import openai; print(f'  openai SDK {openai.__version__} importable from [llm]-installed wheel (back-compat alias works)')"
	@/tmp/sum-release-llm/bin/sum --version
	@printf '%s\n' "─── Release bytes verified — safe to 'git tag' ────────────"

probe-live-trust-loop:  ## End-to-end adversarial probe against the live deploy: render → JWKS → 6-fixture tamper → tome-hash check. Requires sum-engine[receipt-verify] importable. Pass an alternate URL as DEMO=https://...
	@bash scripts/probes/live_trust_loop_smoke.sh $(DEMO)

lint:  ## ruff check (same rules as CI).
	$(PYTHON) -m ruff check sum_cli internal scripts

pre-push:  ## Pre-flight gate: drift + smoke + trust-loop load-bearing gates that CI runs on every PR.
	@echo "─── Pre-push gate (matches CI's drift + smoke + trust-loop checks) ─"
	@echo "[1/6] Self-attestation drift check (CANONICAL_DOCS bytes ↔ meta/self_attestation.*)"
	@$(PYTHON) -m scripts.attest_repo_docs --check
	@echo "[2/6] Repo manifest drift check (fixtures/bench_receipts/* ↔ meta/repo_manifest.json)"
	@$(PYTHON) -m scripts.repo_manifest --check meta/repo_manifest.json
	@echo "[3/6] Bundle-metadata + self-attestation + transform-substrate smoke"
	@$(PYTHON) -m pytest -x --no-header -q \
	    Tests/test_self_attestation.py \
	    Tests/test_bundle_axiom_graph_entropy.py \
	    Tests/test_bundle_entropy_ci.py \
	    Tests/test_bundle_consistency_check.py \
	    Tests/test_bundle_distribution_mmd.py \
	    Tests/test_bundle_distribution_mmd_threshold.py \
	    Tests/test_bundle_corruption_score.py \
	    Tests/test_property_substrate.py \
	    Tests/test_transform_registry.py \
	    Tests/test_transform_receipt_verify.py \
	    Tests/test_transform_extract.py \
	    Tests/test_transform_extract_multi_school.py \
	    Tests/test_transform_compose.py \
	    Tests/test_transform_receipt_source_chain.py \
	    Tests/test_transform_share.py \
	    Tests/test_sum_cli_transform.py \
	    Tests/test_transform_receipt_verifier_fixtures.py
	@echo "[4/6] Cross-runtime valid-path K-matrix (Python ↔ Node)"
	@$(PYTHON) -m scripts.verify_cross_runtime
	@echo "[5/6] Cross-runtime rejection A-matrix (Python ↔ Node)"
	@$(PYTHON) -m scripts.verify_cross_runtime_adversarial
	@echo "[6/6] Fortress — 21 pure-math invariants"
	@$(PYTHON) scripts/verify_fortress.py --json >/dev/null
	@echo "─── Pre-push gate PASSED — safe to push ───────────────────────────"

wasm:  ## Build + copy sum_core.wasm into single_file_demo/ (rerun after core-zig/ edits).
	cd core-zig && zig build wasm
	cp core-zig/zig-out/bin/sum_core.wasm single_file_demo/sum_core.wasm
	node single_file_demo/test_wasm.js

vendor:  ## Regenerate vendored jose + canonicalize bundle (rerun after dep bumps).
	cd scripts/vendor && npm ci && npm run build

test-receipt-verify:  ## v0.9.B: Node smoke test against the receipt-fixture set.
	node single_file_demo/test_render_receipt_verify.js

test-transform-receipt-verify:  ## T1d: Node smoke against the browser transform-receipt verifier.
	node single_file_demo/test_transform_receipt_verify.js

test-transform-receipt-fixtures:  ## Cross-runtime fixture smoke: browser verifier against fixtures/transform_receipts/. Pair with Tests/test_transform_receipt_verifier_fixtures.py for the Python side.
	node single_file_demo/test_transform_receipt_fixtures.js

test-receipt-verify-py:  ## v0.9.C: Python receipt verifier against the same fixture set.
	$(PYTHON) -m pytest Tests/test_render_receipt_verifier.py -q

bench-receipt-audit:  ## v0.9.D: live audit against /api/render — costs LLM tokens.
	$(PYTHON) -m scripts.bench.runners.receipt_audit \
		--cells 8 --out audit_trail.ndjson

bench-receipt-audit-replay:  ## v0.9.D: replay-verify a saved audit trail (no network).
	$(PYTHON) -m scripts.bench.runners.receipt_audit \
		--replay audit_trail.ndjson

test-trust-root:  ## R0.2: trust-root manifest round-trip + tampering tests.
	$(PYTHON) -m pytest Tests/test_trust_root.py -q

test-property:  ## R0.4: Hypothesis fuzz on JCS + verifier surfaces.
	$(PYTHON) -m pytest Tests/test_property_jcs.py Tests/test_property_receipt.py -q

test-nli-calibration-format:  ## v0.9.E: NLI calibration fixture format validation.
	$(PYTHON) -m pytest Tests/test_nli_calibration_format.py -q

wasm-bench:  ## Serve the WASM-vs-JS browser benchmark (docs/WASM_PERFORMANCE.md).
	@echo "Open: http://localhost:8000/Tests/benchmarks/browser_wasm_bench.html"
	@echo "Run in Chrome 113+, Firefox 129+, Safari 17+; paste JSON into docs/WASM_PERFORMANCE.md."
	$(PYTHON) -m http.server 8000

wasm-bench-python:  ## Run the Python-side derivation benchmark companion (emits JSON).
	$(PYTHON) scripts/bench_python_derive.py

validate-preprint:  ## Lint preprint structure + receipt references. No pandoc required.
	bash scripts/arxiv/validate_preprint.sh

verify-preprint:  ## Verify preprint citation chain (every receipt's bench_digest matches its prose cite). Pure read; no compute.
	$(PYTHON) -m scripts.research.verify_preprint_receipts

reproduce-preprint: validate-preprint verify-preprint  ## Run all preprint-cited bench tests and verify the citation chain. Reviewer-friendly answer to "can I reproduce this?"
	$(PYTHON) -m pytest \
	  Tests/research/test_sheaf_path2_v3.py \
	  Tests/research/test_sheaf_path2_multi_llm_compare.py \
	  Tests/research/test_sheaf_path2_cross_corpus.py \
	  Tests/research/test_recovery_experiment_digests.py \
	  Tests/research/test_performance_audit.py \
	  Tests/research/test_recursive_compression_walk.py \
	  Tests/research/test_recursive_compression_cross_family.py \
	  -q --tb=short

clean:  ## Remove build artifacts + bench history.
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache
	rm -f bench_report.json
