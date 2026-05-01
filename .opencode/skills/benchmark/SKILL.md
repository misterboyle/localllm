---
name: benchmark
description: Run server benchmarks to measure MoE performance at different concurrency levels
---

Run the MLX server benchmark tool to measure decode speed and throughput at various concurrency levels.

Concurrency benchmarks directly measure how well the server handles multiple simultaneous agent conversations — the key metric for multi-agent factory performance.

## Steps

1. Read server config from `~/.localllm/models.jsonc` to get the actual port and model path.

2. Check the server is running:
   ```bash
   # Replace port with value from models.jsonc
   curl http://localhost:<port>/health
   ```

3. Run benchmarks at increasing concurrency levels:
   ```bash
   # Read these from ~/.localllm/models.jsonc
   PORT=30083
   MODEL=~/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit

    # Use venv python — mlx-lm-turbo scripts require aiohttp, numpy, etc.
    PY=mlx-lm-turbo/venv/bin/python3

    # Single-threaded baseline
    "$PY" mlx-lm-turbo/benchmarks/server_benchmark.py \
      --url "http://localhost:$PORT/v1/chat/completions" \
      --api-key sk-local \
      --model "$MODEL" \
      --max-tokens 256 --concurrency 1 --total-requests 5

    # Multi-threaded
    "$PY" mlx-lm-turbo/benchmarks/server_benchmark.py \
      --url "http://localhost:$PORT/v1/chat/completions" \
      --api-key sk-local \
      --model "$MODEL" \
      --max-tokens 256 --concurrency 4 --total-requests 20 \
      --output bench-conc4.json

    # High concurrency
    "$PY" mlx-lm-turbo/benchmarks/server_benchmark.py \
      --url "http://localhost:$PORT/v1/chat/completions" \
      --api-key sk-local \
      --model "$MODEL" \
      --max-tokens 256 --concurrency 8 --total-requests 32 \
      --output bench-conc8.json
   ```

4. Compare results — look at:
   - Aggregate tokens/sec (total throughput)
   - Per-request tokens/sec (per-conversation speed)
   - Time to first token (TTFT) — avg, min, max, p95
   - How throughput scales with concurrency

5. Save results to `benchmarks/` directory for comparison across runs.
