SHELL := /bin/bash

.PHONY: all help lint test check

all: help

help:
	@echo "localllm - local compute engine (see README.md)"
	@echo ""
	@echo "Daily operations use ./serve directly:"
	@echo "  ./serve status              show running servers"
	@echo "  ./serve start [server...]   start server(s) (default: all defined)"
	@echo "  ./serve stop [server...]    stop server(s) (default: all running)"
	@echo ""
	@echo "Quality gates:"
	@echo "  make lint    shellcheck all shell scripts + python syntax"
	@echo "  make test    run validation tests in tests/"
	@echo "  make check   verify live setup (binary, servers.jsonc, models)"

lint:
	@echo "=== Lint ==="
	@command -v shellcheck > /dev/null 2>&1 || { echo "ERROR: shellcheck not found. Install via: brew install shellcheck"; exit 1; }
	@shellcheck -e SC1090 -e SC1091 serve snapshot-memory.sh tests/*.sh
	@python3 -m py_compile memory-budget.py tests/check_setup.py benchmarks/*.py && echo "Python syntax: OK"

test:
	@echo "=== Tests ==="
	@fail=0; \
	for t in tests/test-*.sh; do \
		if [ -f "$$t" ]; then \
			echo "Running $$t..."; \
			if bash "$$t"; then \
				echo "  PASS"; \
			else \
				echo "  FAIL"; \
				fail=1; \
			fi; \
		fi; \
	done; \
	if [ "$$fail" -ne 0 ]; then echo "Some tests failed"; exit 1; fi
	@echo "All tests passed."

check:
	@echo "=== Setup Check ==="
	@python3 tests/check_setup.py
