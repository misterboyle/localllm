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

    # mlx-lm-turbo is a SIBLING repo at ~/mlx-lm-turbo/ (NOT a subdirectory)
    # Use absolute paths — relative paths from this CWD will fail
    PY=~/mlx-lm-turbo/venv/bin/python3
    BENCH=~/mlx-lm-turbo/benchmarks/server_benchmark.py

    # Single-threaded baseline
    "$PY" "$BENCH" \
      --url "http://localhost:$PORT/v1/chat/completions" \
      --api-key sk-local \
      --model "$MODEL" \
      --max-tokens 256 --concurrency 1 --total-requests 5

    # Multi-threaded
    "$PY" "$BENCH" \
      --url "http://localhost:$PORT/v1/chat/completions" \
      --api-key sk-local \
      --model "$MODEL" \
      --max-tokens 256 --concurrency 4 --total-requests 20 \
      --output bench-conc4.json

    # High concurrency
    "$PY" "$BENCH" \
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

## Context-Length Benchmarks

The default benchmark sends zero-context prompts (~10 tokens). To measure decode speed degradation with long conversations, run with synthetic context:

```bash
# Generate context-length test prompts (reuses workload estimates from memory-budget.py)
# Low: ~10K tokens (short conversation)
# Mid: ~50K tokens (typical agent session)
# High: ~100K tokens (long-running agent)

# Example: test with 50K context
python3 -c "
# Generate a prompt with ~50K tokens of synthetic context
context = ' '.join(['word'] * 40000)  # ~50K tokens
prompt = f'Continue this conversation: {context} What is the next step?'
with open('/tmp/prompt-50k.txt', 'w') as f:
    f.write(prompt)
"

# Run benchmark with context
"$PY" "$BENCH" \
  --url "http://localhost:$PORT/v1/chat/completions" \
  --api-key sk-local \
  --model "$MODEL" \
  --prompt-file /tmp/prompt-50k.txt \
  --max-tokens 256 --concurrency 1 --total-requests 5 \
  --output bench-50k-conc1.json
```

Workload context lengths (from memory-budget.py):
- Light: 30K active context
- Typical: 50K small + 150K large active
- Heavy: 80K small + 200K large active
- Worst: 128K small + 262K large

## Comparison Tool

Use `benchmarks/compare.py` to analyze results:

```bash
# Compare all results
python3 benchmarks/compare.py

# Filter by model
python3 benchmarks/compare.py dense
python3 benchmarks/compare.py moe
```

The comparison tool shows:
- Aggregate throughput scaling across concurrency levels
- Per-request decode speed degradation
- TTFT impact at higher concurrency
- Scaling efficiency (actual vs ideal linear scaling)
