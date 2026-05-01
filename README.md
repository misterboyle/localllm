# localllm

Local LLM inference on Apple Silicon using MLX. Two models, two servers, one config.

## Architecture

| Model | Type | Quant | Port | Role |
|-------|------|-------|------|------|
| Qwen3.6-27B-UD | Dense | 6-bit | 30090 | Main reasoner — complex reasoning, big context |
| Qwen3.6-35B-A3B-UD | MoE (3B active) | 4-bit | 30083 | Worker — tool use, code, fast decode |

Both run as independent `mlx_lm.server` processes on localhost. Configured via `~/.localllm/models.jsonc`. Connected to opencode via `~/.config/opencode/opencode.jsonc`.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.14
- 64 GB RAM minimum (128 GB recommended for running both models)
- ~50 GB disk for models

## Quick Setup

```bash
make setup
```

This clones dependencies, creates a Python venv, installs the MLX stack, downloads models, and generates config files. See `make help` for individual steps.

## Daily Use

```bash
# Start servers (foreground, Ctrl+C to stop)
./start-server.sh

# Check health
curl http://localhost:30090/health   # Dense
curl http://localhost:30083/health   # MoE

# Memory snapshot
./snapshot-memory.sh running
```

## Configuration

Edit `~/.localllm/models.jsonc` to enable/disable servers, tune KV quantization, adjust cache sizes, and set concurrency. See `models.jsonc.example` for all options.

The opencode client config lives at `~/.config/opencode/opencode.jsonc`. See `opencode.jsonc.example`.

## Components

| Repo | Branch | Purpose |
|------|--------|---------|
| `mlx-lm-turbo` | `feature/turboquant-kv-cache` | MLX server with prompt caching, KV quantization, disk cache |
| `turboquant-mlx` | `main` | TurboQuant KV cache compression (K8+V2 mixed precision) |

Both are installed editable in the venv so code changes take effect on next server restart.

## Models

Downloaded from Hugging Face via `huggingface-cli`:

- `unsloth/Qwen3.6-27B-UD-MLX-6bit` (~28 GB) — dense, 6-bit quantized
- `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit` (~20 GB) — MoE, 4-bit quantized

Stored in `~/.localllm/models/`.

## Benchmarking

The MLX server ships with a benchmark script at `mlx-lm-turbo/benchmarks/server_benchmark.py`. Run it against a running server:

```bash
# Single-threaded (1 concurrent request)
python3 mlx-lm-turbo/benchmarks/server_benchmark.py \
  --url http://localhost:30083/v1/chat/completions \
  --api-key sk-local \
  --model /Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit \
  --max-tokens 256 \
  --concurrency 1 \
  --total-requests 5

# Multi-threaded (4 concurrent requests)
python3 mlx-lm-turbo/benchmarks/server_benchmark.py \
  --url http://localhost:30083/v1/chat/completions \
  --api-key sk-local \
  --model /Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit \
  --max-tokens 256 \
  --concurrency 4 \
  --total-requests 20 \
  --output results.json
```

Outputs: time-to-first-token, tokens/sec (per-request and aggregate), and a per-second throughput chart. Results can be saved as JSON for comparison.

## Historical Docs

The repo contains detailed analysis from the prior llama.cpp era (KV cache mechanics, checkpoint behavior, memory profiling). Each document has a "Historical note" header explaining its relevance to the current MLX setup.
