---
description: Characterize MoE model performance and memory
model: opencode/kimi-k2.5
---

Run a full characterization of the MoE model: memory, speed, and quality. This determines whether the model is suitable as a compute engine for multi-agent factories.

## Steps

1. **Memory analysis**
   ```bash
   python3 memory-budget.py --model moe
   python3 memory-budget.py --model both
   ./snapshot-memory.sh running
   ```

2. **Decode speed benchmarks**
   Read port and model path from `~/.localllm/models.jsonc`:
   ```bash
   PORT=30083  # from models.jsonc
   MODEL=~/.localllm/models/Qwen3.6-35B-A3B-UD-MLX-4bit  # from models.jsonc

   python3 mlx-lm-turbo/benchmarks/server_benchmark.py \
     --url "http://localhost:$PORT/v1/chat/completions" \
     --api-key sk-local \
     --model "$MODEL" \
     --max-tokens 256 --concurrency 1 --total-requests 5
   ```

3. **Multi-concurrency benchmarks**
   Run at concurrency 4, 8, and 12. Save results as JSON.

4. **Quality check**
   Send a few representative prompts and verify output quality is acceptable.

5. **Document findings**
   Save to `MOE-CHARACTERIZATION.md` with:
   - Memory profile (fixed, KV, headroom)
   - Decode speed (single and multi-concurrency)
   - TTFT at each concurrency level
   - Any issues or anomalies observed
