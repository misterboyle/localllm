# localllm — Agent Instructions

See [AGENTS.md](../AGENTS.md) for code style conventions.

## Purpose

This repo is the **local compute engine for multi-agent software development
factories**. It provides the inference layer that lets agent orchestration
frameworks (gastown.ai, gas city, etc.) run entirely on Apple Silicon — and
hosts the `mtp-vision` server that runs the opencode assistant itself.

**Your job:** characterize, optimize, and maintain the inference servers so
the daily driver (mtp-vision) stays fast and the parked layers (MLX) stay
workable.

## Configuration

All server settings are in **`servers.jsonc` in this repo** — the only file
`./serve` reads. **Read it** to get actual ports, model paths, and args.

`~/.localllm` is **data-only**: models (`models/`), pidfiles (`pids/`), logs,
prompt cache, venvs. Never look for or create config there — the old
`~/.localllm/models.jsonc` flow was removed 2026-08-21.

## Architecture

| Server | Type | Backend | Port | Status |
|--------|------|---------|------|--------|
| mtp-vision | Dense 27B, Qwen3.8 UD-Q4_K_XL + mmproj + MTP spec decode | llama.cpp (`~/llama.cpp-turbo`, `mtp-vanilla`) | 30082 | **Production** — opencode assistant |
| dense | Dense 27B, Qwen3.6 UD-Q6_K_XL | llama.cpp | 30080 | parked |
| moe | MoE 35B-A3B, Qwen3.6 UD-Q6_K_XL | llama.cpp | 30081 | parked — model not on disk |
| mlx-dense | Dense 27B, Qwen3.6 UD-MLX-4bit | MLX (`~/mlx-lm-turbo`) | 30090 | parked |
| mlx-moe | MoE 35B-A3B, Qwen3.6 UD-MLX-4bit | MLX (`~/mlx-lm-turbo`) | 30083 | parked |
| mlx-moe-8bit | MoE 35B-A3B, Qwen3.6 MLX-8bit | MLX (`~/mlx-lm-turbo`) | 30084 | parked |
| mlx-dense-mtp | Dense 27B, Qwen3.6 OptiQ-4bit, MTP | optiq (MLX) | 30091 | parked (model incomplete) |

Daily driver: mtp-vision. The MLX layer migrates to Qwen3.8 only if its use
resumes (decision 2026-08-21).

## Server Management

```bash
# Run the servers — this is the only way
./serve status
./serve start mtp-vision
./serve stop mtp-vision

# Health
curl http://localhost:30082/health

# Memory snapshot (llama.cpp + MLX processes, ports, prompt cache)
./snapshot-memory.sh running
```

**Footgun:** mtp-vision has no restart subcommand, and the opencode assistant
runs on it. Restarting (`./serve stop mtp-vision && ./serve start mtp-vision`)
kills the active session — do it at session end only.

## Benchmarking

Benchmarks hit live servers via
`~/mlx-lm-turbo/benchmarks/server_benchmark.py` (works against any
OpenAI-compatible endpoint). See `skills/benchmark/SKILL.md` for the full
playbook (context-length suite for dense, concurrency scaling for MoE).

**Footgun:** `mlx-lm-turbo/` is a sibling repo at `~/mlx-lm-turbo/`, NOT a
subdirectory of this repo. Use absolute paths — relative paths from this CWD
will fail. Run with the MLX venv python (`~/mlx-lm-turbo/venv/bin/python3`)
for its deps. Results go in `benchmarks/`.

## Memory

- `./snapshot-memory.sh <phase>` — live snapshot; RSS under-reports Metal
  memory (treat as a floor).
- `python3 memory-budget.py --model both` — KV-budget planning for the parked
  MLX models.

Reference numbers for the MLX layer (128GB M5 Max): MoE 35B-A3B fixed ~39 GB,
KV ~20 KB/token active; 27B dense fixed ~44 GB, KV ~64 KB/token active.
