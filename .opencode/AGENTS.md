# localllm — Agent Instructions

## Architecture

Two MLX models running as independent `mlx_lm.server` processes on localhost:

| Model | Type | Quant | Port | Role |
|-------|------|-------|------|------|
| Qwen3.6-35B-A3B | MoE (3B active) | 4-bit | 30083 | Primary — fast decode, worker tasks |
| Qwen3.6-27B-UD | Dense | 4-bit | 30090 | Disabled by default — reasoning comparison |

Both use the same MLX stack with TurboQuant KV cache compression (K8+V2 mixed precision).

## Server Management

```bash
# Start/stop servers
./start-server.sh

# Check health
curl http://localhost:30083/health   # MoE
curl http://localhost:30090/health   # Dense

# Memory snapshot
./snapshot-memory.sh running
```

## Benchmarking

Run benchmarks against a live server:

```bash
# Single-threaded
python3 mlx-lm-turbo/benchmarks/server_benchmark.py \
  --url http://localhost:30083/v1/chat/completions \
  --api-key sk-local \
  --model /Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit \
  --max-tokens 256 --concurrency 1 --total-requests 5

# Multi-threaded
python3 mlx-lm-turbo/benchmarks/server_benchmark.py \
  --url http://localhost:30083/v1/chat/completions \
  --api-key sk-local \
  --model /Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit \
  --max-tokens 256 --concurrency 4 --total-requests 20 \
  --output results.json
```

## Memory Budget (MoE)

- Fixed: 39 GB (weights 26 + compute 8 + overhead 5)
- KV budget: 74 GB of 128 GB system
- Active KV: 20 KB/token (fp16), 6.9 KB/token (K8,V2 cached)
- Max 13 concurrent conversations at 262K context
- Heavy workload (6 active @ 136K avg): 58 GB total, 70 GB margin

## Memory Budget (27B Dense)

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
