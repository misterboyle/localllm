# localllm — Style Guide

## Purpose

This repo is the **local compute engine for multi-agent software development factories**. It characterizes and optimizes running the best local models on Apple Silicon to power agent orchestration frameworks (e.g., gastown.ai, gas city) entirely on-device.

Everything in this repo serves that goal:
- **Server scripts** — launch and manage MLX inference servers as OpenAI-compatible endpoints
- **Benchmarking** — measure throughput, latency, and memory at various concurrency levels
- **Memory analysis** — understand KV cache costs to maximize concurrent agent conversations
- **Configuration** — tune models, quantization, and cache settings for optimal multi-agent performance

Code conventions for shell scripts, Python, and configuration files.

## Shell Scripts

### Structure

- Shebang: `#!/usr/bin/env bash`
- Strict mode: `set -euo pipefail` on line 2
- No trailing newline after strict mode — blank line before first comment/block

### Variables

- All variables double-quoted: `"$var"` never `$var`
- Uppercase for configuration/constants: `CONF_FILE`, `PID_DIR`
- Lowercase for local/function variables: `model`, `port`, `logf`
- Resolve paths with `SCRIPT_DIR` pattern: `SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"`

### Functions

- Use function name without parens: `log() {` not `log() () {`
- `local` for all function-scoped variables
- No `return` with a value for success — use `return 0` or omit
- Helper functions before main logic

### Logging

- Use `log()` helper with timestamped output: `echo "[$(date '+%H:%M:%S')] $*"`
- Use `tee -a` when logging to both stdout and file
- No `echo` without context — every output should be traceable

### Error Handling

- `set -e` for exit-on-error, `set -u` for unset variable errors
- Guard clauses at top: check prerequisites, fail early with `exit 1`
- Signal traps for cleanup: `trap cleanup EXIT INT TERM`
- ShellCheck annotations where needed: `# shellcheck disable=SC1090`

### Sourcing & Eval

- Prefer explicit variable resolution over `eval`
- When sourcing generated files, use ShellCheck disable: `# shellcheck disable=SC1090`
- Use `shlex.quote()` in Python-generated shell variables

### Arrays

- Bash arrays for argument lists: `args=(--model "$model" --port "$port")`
- Expansion: `"${args[@]}"` (always quoted)
- Append: `args+=(--flag "$value")`

## Python

### Structure

- Shebang: `#!/usr/bin/env python3`
- Module docstring on first line: triple-quoted, describes purpose and usage
- Standard library imports first, then third-party (none preferred)
- Constants in UPPERCASE at module level
- Scripts in this repo are stdlib-only — run with system `python3`
- Scripts in `mlx-lm-turbo/` require the venv — run with `mlx-lm-turbo/venv/bin/python3`

### Style

- Type hints where they add clarity, not required everywhere
- Docstrings for public functions: describe what, not how
- `argparse` for CLI interfaces
- No external dependencies unless absolutely necessary — standard library preferred

### Error Handling

- Fail fast with descriptive messages
- Use `sys.exit(1)` for CLI errors
- No bare `except:` — catch specific exceptions

## Configuration Files

### JSONC

- Use JSONC (JSON with comments) for human-readable configs
- `//` single-line comments for inline explanations
- Hierarchical structure: `defaults` section with per-item overrides
- `$HOME` placeholder for paths, expanded at runtime

### Example Files

- `.example` suffix: `models.jsonc.example`, `opencode.jsonc.example`
- `make config` copies examples to user locations, skips if exists
- Examples should be complete and functional as-is

### KV Cache Quantization Patterns

The MLX server supports two KV cache compression strategies. Agents should know the config patterns:

**TurboQuant** (`feature/turboquant-kv-cache` branch):
- `turboKvBits`: 1-4 (PolarQuant + Hadamard rotation, 3-bit recommended)
- `turboFp16Layers`: layers to keep FP16 (default: 1)
- `turboVBits`: optional V-bit width for standard affine quantization
- Config variant: `configs/turboQuant/models.jsonc`

**Mixed-Quant** (`feature/mixed-quant-kv-cache` branch):
- `kvCacheQuantization`: `"Kbits,Vbits"` string (e.g. `"8,4"`, `"8,2"`)
  - K is usually higher precision than V — key projections are more sensitive
  - `"8,4"` = balanced (good quality, ~2x compression)
  - `"8,2"` = aggressive (max compression, slight quality trade-off)
  - `"4,4"` = extreme (lowest memory, quality may suffer)
- `kvGroupSize`: quantization group size (default: 64)
- `quantizedKvStart`: token count to start quantizing (default: 5000)
- `kvCacheQuantizeAfterPrefill`: keep FP16 during prefill, quantize after (default: false)
- Config variant: `configs/mixedQuant/models.jsonc`

**Vanilla** (`official/main` branch):
- No KV compression — full FP16 KV cache
- Config variant: `configs/vanilla/models.jsonc`

To switch variants, copy the desired config into `~/.localllm/models.jsonc`:
```bash
cp configs/mixedQuant/models.jsonc ~/.localllm/models.jsonc
```

### Sampling Parameters

Sampling defaults are aligned with [Unsloth Qwen3.6 recommendations](https://unsloth.ai/docs/models/qwen3.6):

**Thinking mode (general tasks):**
- `temperature: 1.0`, `top_p: 0.95`, `top_k: 20`
- `presence_penalty: 1.5` (set per-request in opencode API body)

**Thinking mode (coding tasks):**
- `temperature: 0.6`, `top_p: 0.95`, `top_k: 20`
- `presence_penalty: 0.0`

**Instruct mode (non-thinking):**
- General: `temperature: 0.7`, `top_p: 0.8`, `top_k: 20`
- Reasoning: `temperature: 1.0`, `top_p: 0.95`, `top_k: 20`

**Config locations:**
- Server defaults: `models.jsonc` → `defaults.temperature`, `defaults.topP`, `defaults.topK`
- Opencode defaults: `opencode.jsonc` → `agent.temperature`, `agent.top_p` (applies to all agents)

**Notes:**
- `presence_penalty` and `frequency_penalty` are NOT server CLI args — they must be set per-request in the OpenAI API body
- Opencode does not currently support setting these in config; they default to `undefined` (disabled)
- `min_p: 0.0` is the default in both server and Unsloth recommendations

## Documentation

### Historical Docs

- Analysis documents from prior eras kept as reference
- Add "Historical note" header when context has changed
- Use markdown tables for comparisons and specifications

### Code Blocks

- Show full commands with context, not abbreviated
- Include shebangs and strict mode in shell examples
- Use language tags: ```bash, ```python, ```json

## Testing

### Validation Tests

- Shell scripts in `tests/` directory
- Name convention: `test-<component>.sh`
- Each test is self-contained, exits 0 on pass, 1 on failure
- Run all: `make test`

### Quality Gates

- `make lint` — shellcheck all scripts, validate Python syntax
- `make test` — run validation tests
- `make check` — verify setup completeness

## Git

### Commit Messages

- Imperative mood: "add", "fix", "update" (not "added", "fixed")
- First line: what and why, under 72 characters
- Body: context and rationale if needed

### Branching

- `main` branch for operational changes
- Feature branches for experiments, analysis, new tools
