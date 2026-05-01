# localllm — Agent Instructions

See [AGENTS.md](../AGENTS.md) for code style conventions.

## Purpose

This repo is the **local compute engine for multi-agent software development factories**. It provides the inference layer that lets agent orchestration frameworks (gastown.ai, gas city, etc.) run entirely on Apple Silicon.

**Your job:** help characterize, optimize, and maintain the MLX server stack so it can handle maximum concurrent agent conversations with acceptable latency and memory usage.

## Configuration

All server settings are in `~/.localllm/models.jsonc` (generated from `models.jsonc.example`). **Read this file** to get actual ports, model paths, and concurrency settings — the values below are from the default config and may differ.

## Architecture

MLX models running as independent `mlx_lm.server` processes on localhost. Default config:

| Server | Type | Quant | Default Port | Role |
|--------|------|-------|-------------|------|
| MoE | MoE (3B active) | 4-bit | 30083 | Primary — fast decode, worker tasks |
| Dense | Dense | 4-bit | 30090 | Disabled by default — reasoning comparison |

Both use TurboQuant KV cache compression (K8+V2 mixed precision).

## Server Management

```bash
# Start/stop servers
./start-server.sh

# Check health (ports from models.jsonc)
curl http://localhost:<moe-port>/health
curl http://localhost:<dense-port>/health

# Memory snapshot
./snapshot-memory.sh running
```

## Benchmarking

Run benchmarks against a live server. Read port and model path from `~/.localllm/models.jsonc`:

```bash
# Example with default config values
python3 mlx-lm-turbo/benchmarks/server_benchmark.py \
  --url http://localhost:30083/v1/chat/completions \
  --api-key sk-local \
  --model ~/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit \
  --max-tokens 256 --concurrency 1 --total-requests 5
```

## Memory Budget (MoE, default config on 128GB M5 Max)

These are reference numbers for the default setup. Actual values depend on model and machine.

- Fixed: 39 GB (weights 26 + compute 8 + overhead 5)
- KV budget: 74 GB of 128 GB system
- Active KV: 20 KB/token (fp16), 6.9 KB/token (K8,V2 cached)
- Max 13 concurrent conversations at 262K context
- Heavy workload (6 active @ 136K avg): 58 GB total, 70 GB margin

## Memory Budget (27B Dense, default config on 128GB M5 Max)

- Fixed: 44 GB (weights 31 + compute 8 + overhead 5)
- KV budget: 69 GB of 128 GB system
- Active KV: 64 KB/token (fp16), 22 KB/token (K8,V2 cached)
- Max 3 concurrent conversations at 262K context

## Key Files

- `~/.localllm/models.jsonc` — server config (enable/disable, ports, concurrency)
- `~/.config/opencode/opencode.jsonc` — opencode client config
- `mlx-lm-turbo/` — MLX server fork with TurboQuant support
- `turboquant-mlx/` — KV cache compression library
- `memory-budget.py` — memory calculator for both models
- `benchmarks/server_benchmark.py` — server benchmark tool

## Historical Docs

The repo contains detailed analysis from the prior llama.cpp era (KV cache mechanics, checkpoint behavior, memory profiling). See `MLX-CHARACTERIZATION-PLAN.md` for the current MLX evaluation plan.
