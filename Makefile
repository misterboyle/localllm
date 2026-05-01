SHELL := /bin/bash
HOME_DIR := $(HOME)
LOCALLLM_DIR := $(HOME_DIR)/.localllm
MODELS_DIR := $(LOCALLLM_DIR)/models
VENV_DIR := $(LOCALLLM_DIR)/venv
MLX_LM_DIR := $(HOME_DIR)/mlx-lm-turbo
TURBOQUANT_DIR := $(HOME_DIR)/turboquant-mlx
OPENCODE_CONFIG := $(HOME_DIR)/.config/opencode/opencode.jsonc
PYTHON := python3.14

.PHONY: all setup help deps models config start stop clean check lint test moe dense both

all: help

lint:
	@echo "=== Lint ==="
	@command -v shellcheck > /dev/null 2>&1 || { echo "ERROR: shellcheck not found. Install via: brew install shellcheck"; exit 1; }
	@shellcheck -e SC1090 -e SC1091 start-server.sh snapshot-memory.sh tests/*.sh
	@python3 -m py_compile memory-budget.py && echo "Python syntax: OK"

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

help:
	@echo "localllm - setup and management"
	@echo ""
	@echo "Targets:"
	@echo "  make setup       Full setup: deps, models, config"
	@echo "  make deps        Clone repos, create venv, install packages"
	@echo "  make models      Download MLX models from Hugging Face"
	@echo "  make config      Generate config files from examples"
	@echo "  make start       Start servers (delegates to start-server.sh)"
	@echo "  make stop        Stop all servers"
	@echo "  make check       Verify setup is complete"
	@echo "  make lint        Lint shell scripts and Python syntax"
	@echo "  make test        Run validation tests"
	@echo "  make clean       Remove venv and downloaded models"
	@echo ""
	@echo "Model sizes: dense 6-bit ~28 GB, moe 4-bit ~20 GB"

setup: deps models config
	@echo ""
	@echo "Setup complete. Next steps:"
	@echo "  1. Review ~/.localllm/models.jsonc (edit enabled servers)"
	@echo "  2. Review ~/.config/opencode/opencode.jsonc"
	@echo "  3. Run ./start-server.sh"

deps: $(VENV_DIR)/bin/python $(MLX_LM_DIR) $(TURBOQUANT_DIR)
	@echo "Installing mlx-lm-turbo (editable)..."
	@$(VENV_DIR)/bin/pip install -e $(MLX_LM_DIR) --quiet
	@echo "Installing turboquant-mlx (editable)..."
	@$(VENV_DIR)/bin/pip install -e $(TURBOQUANT_DIR) --quiet
	@echo "Dependencies installed."

$(VENV_DIR)/bin/python:
	@echo "Creating Python 3.14 venv at $(VENV_DIR)..."
	@if ! command -v $(PYTHON) > /dev/null 2>&1; then \
		echo "ERROR: $(PYTHON) not found. Install via: brew install python@3.14"; \
		exit 1; \
	fi
	@$(PYTHON) -m venv $(VENV_DIR)
	@$(VENV_DIR)/bin/pip install --upgrade pip --quiet
	@$(VENV_DIR)/bin/pip install huggingface-hub --quiet

$(MLX_LM_DIR):
	@echo "Cloning mlx-lm-turbo (arozanov/mlx-lm, feature/turboquant-kv-cache)..."
	@git clone --branch feature/turboquant-kv-cache --single-branch \
		https://github.com/arozanov/mlx-lm.git $(MLX_LM_DIR)

$(TURBOQUANT_DIR):
	@echo "Cloning turboquant-mlx (arozanov/turboquant-mlx)..."
	@git clone --single-branch https://github.com/arozanov/turboquant-mlx.git $(TURBOQUANT_DIR)

models: $(MODELS_DIR)/Qwen3.6-27B-UD-MLX-6bit $(MODELS_DIR)/Qwen3.6-35B-A3B-UD-MLX-4bit
	@echo "All models present."

$(MODELS_DIR)/Qwen3.6-27B-UD-MLX-6bit:
	@echo "Downloading Qwen3.6-27B-UD-MLX-6bit (~28 GB)..."
	@mkdir -p $(MODELS_DIR)
	@$(VENV_DIR)/bin/hf download unsloth/Qwen3.6-27B-UD-MLX-6bit \
		--local-dir $(MODELS_DIR)/Qwen3.6-27B-UD-MLX-6bit

$(MODELS_DIR)/Qwen3.6-35B-A3B-UD-MLX-4bit:
	@echo "Downloading Qwen3.6-35B-A3B-UD-MLX-4bit (~20 GB)..."
	@mkdir -p $(MODELS_DIR)
	@$(VENV_DIR)/bin/hf download unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit \
		--local-dir $(MODELS_DIR)/Qwen3.6-35B-A3B-UD-MLX-4bit

config:
	@echo "Generating config files..."
	@if [ ! -f $(HOME_DIR)/.localllm/models.jsonc ]; then \
		cp models.jsonc.example $(HOME_DIR)/.localllm/models.jsonc; \
		echo "  Created ~/.localllm/models.jsonc"; \
	else \
		echo "  ~/.localllm/models.jsonc already exists (skipped)"; \
	fi
	@mkdir -p $(HOME_DIR)/.config/opencode
	@if [ ! -f $(OPENCODE_CONFIG) ]; then \
		sed "s|\\$$HOME|$(HOME_DIR)|g" opencode.jsonc.example > $(OPENCODE_CONFIG); \
		echo "  Created ~/.config/opencode/opencode.jsonc"; \
	else \
		echo "  ~/.config/opencode/opencode.jsonc already exists (skipped)"; \
	fi

start:
	@./start-server.sh $(filter-out $@,$(MAKECMDGOALS))
%:
	@

stop:
	@for name in dense moe; do \
		pidfile="$(HOME_DIR)/.localllm/pids/$$name.pid"; \
		if [ -f "$$pidfile" ]; then \
			kill $$(cat "$$pidfile") 2>/dev/null && rm -f "$$pidfile" && echo "Stopped $$name"; \
		else \
			echo "$$name not running"; \
		fi; \
	done

check:
	@echo "=== Setup Check ==="
	@echo -n "Python 3.14: " && command -v python3.14 > /dev/null 2>&1 && echo "OK" || echo "MISSING (brew install python@3.14)"
	@echo -n "Venv: " && test -f $(VENV_DIR)/bin/python && echo "OK" || echo "MISSING (make deps)"
	@echo -n "mlx-lm-turbo: " && test -d $(MLX_LM_DIR) && echo "OK" || echo "MISSING (make deps)"
	@echo -n "turboquant-mlx: " && test -d $(TURBOQUANT_DIR) && echo "OK" || echo "MISSING (make deps)"
	@echo -n "Dense model: " && test -d $(MODELS_DIR)/Qwen3.6-27B-UD-MLX-6bit && echo "OK" || echo "MISSING (make models)"
	@echo -n "MoE model: " && test -d $(MODELS_DIR)/Qwen3.6-35B-A3B-UD-MLX-4bit && echo "OK" || echo "MISSING (make models)"
	@echo -n "Server config: " && test -f $(HOME_DIR)/.localllm/models.jsonc && echo "OK" || echo "MISSING (make config)"
	@echo -n "Opencode config: " && test -f $(OPENCODE_CONFIG) && echo "OK" || echo "MISSING (make config)"

clean:
	@echo "Removing venv..."
	@rm -rf $(VENV_DIR)
	@echo "Removing models..."
	@rm -rf $(MODELS_DIR)
	@echo "Clean complete."
