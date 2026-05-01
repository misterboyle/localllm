---
name: benchmark
description: Run server benchmarks to measure MoE performance at different concurrency levels
---

Run the MLX server benchmark tool to measure decode speed and throughput.

## Steps

1. Check the server is running: `curl http://localhost:30083/health`
2. Run benchmarks at increasing concurrency levels:

```bash
# Single-threaded baseline
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
  --output bench-conc4.json

# High concurrency
python3 mlx-lm-turbo/benchmarks/server_benchmark.py \
  --url http://localhost:30083/v1/chat/completions \
  --api-key sk-local \
  --model /Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit \
  --max-tokens 256 --concurrency 8 --total-requests 32 \
  --output bench-conc8.json
```

3. Compare results — look at:
   - Aggregate tokens/sec (total throughput)
   - Per-request tokens/sec (per-conversation speed)
   - Time to first token (TTFT) — avg, min, max, p95
   - How throughput scales with concurrency

4. Save results to `benchmarks/` directory for comparison across runs.
