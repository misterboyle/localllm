---
description: Run full benchmark suite (conc 1, 4, 8, 12) and save results
model: opencode/kimi-k2.5
---

Run the complete benchmark suite against the MoE server.

## Steps

1. Verify server is running: `curl http://localhost:30083/health`

2. Run benchmarks at all concurrency levels:

```bash
MODEL="/Users/michael/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit"
BASE="http://localhost:30083/v1/chat/completions"
API="sk-local"

for CONC in 1 4 8 12; do
  python3 mlx-lm-turbo/benchmarks/server_benchmark.py \
    --url "$BASE" \
    --api-key "$API" \
    --model "$MODEL" \
    --max-tokens 256 \
    --concurrency "$CONC" \
    --total-requests $((CONC * 4)) \
    --output "bench-moe-conc${CONC}.json"
  echo "---"
done
```

3. Summarize results:
   - Aggregate tokens/sec at each concurrency level
   - Per-request tokens/sec (how much does each conversation slow down?)
   - Time to first token (TTFT) scaling
   - Identify the sweet spot where throughput peaks but per-request speed is still acceptable

4. Save summary to `benchmarks/summary-<date>.md`
