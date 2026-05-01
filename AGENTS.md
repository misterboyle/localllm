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
