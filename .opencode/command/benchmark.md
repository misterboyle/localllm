---
description: Run full benchmark suite (conc 1, 4, 8, 12) and save results
model: opencode/kimi-k2.5
---

Run the complete benchmark suite against the MoE server. Each concurrency level simulates a different number of simultaneous agent conversations in a factory.

## Steps

1. Read server config from `~/.localllm/models.jsonc` to get the actual port and model path.

2. Verify server is running:
   ```bash
   # Replace port with value from models.jsonc
   curl http://localhost:<port>/health
   ```

3. Run benchmarks at all concurrency levels:
   ```bash
   # Read these from ~/.localllm/models.jsonc
   PORT=30083
   MODEL=~/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit
   BASE="http://localhost:$PORT/v1/chat/completions"
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

4. Summarize results:
   - Aggregate tokens/sec at each concurrency level
   - Per-request tokens/sec (how much does each conversation slow down?)
   - Time to first token (TTFT) scaling
   - Identify the sweet spot where throughput peaks but per-request speed is still acceptable

5. Save summary to `benchmarks/summary-<date>.md`
